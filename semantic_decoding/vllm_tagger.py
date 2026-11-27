"""
semantic_decoding/vllm_tagger.py
=================================
Drop-in alternative to NOVICDecoder that uses a vLLM-served Vision-Language
Model (VLM) to assign a meaningful noun tag to each segmented object crop.

WHY vLLM (and not NOVIC directly)
-----------------------------------
NOVIC is a custom research model loaded via its own infer.py; it cannot be
served by vLLM.  vLLM targets standard HuggingFace-format LLMs/VLMs such as:
    - llava-hf/llava-1.5-7b-hf                  (LLaVA-1.5 7B)
    - llava-hf/llava-v1.6-mistral-7b-hf          (LLaVA-1.6 / LLaVA-NeXT)
    - Qwen/Qwen2-VL-7B-Instruct                  (Qwen2-VL)
    - microsoft/Phi-3.5-vision-instruct           (Phi-3.5-V)

vLLM brings:
    • PagedAttention — eliminates KV-cache fragmentation, highest GPU util
    • Continuous batching — all crops in one shot instead of serial NOVIC calls
    • FP16 / BF16 / AWQ / GPTQ quantisation out of the box
    • tensor-parallel multi-GPU support via tensor_parallel_size

API contract
------------
VLLMTagger.tag(crops_rgb: list[np.ndarray]) -> list[TagResult]

The returned TagResult is identical to the one NOVICDecoder produces so the
rest of main.py (saving, printing) is completely unchanged.

Usage in main.py
----------------
    from semantic_decoding.vllm_tagger import VLLMTagger

    tagger = VLLMTagger(
        model="llava-hf/llava-1.5-7b-hf",   # any vLLM-supported VLM
        device="cuda",
        dtype="float16",
        max_model_len=2048,
        tensor_parallel_size=1,
    )
    results = tagger.tag(crops)              # list[TagResult]
    tagger.close()                           # releases GPU memory

Install
-------
    pip install vllm                  # GPU wheel (CUDA 12.x)
    # or for older CUDA:
    pip install vllm --extra-index-url https://download.pytorch.org/whl/cu118
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

# Re-use the same TagResult dataclass as NOVICDecoder so the rest of the
# pipeline (save functions, print_results, …) never needs to change.
from semantic_decoding.novic.novic_decoder import TagResult

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Model-specific prompt templates
# Each entry: (chat_template_format, image_token, prompt_builder_fn)
# ─────────────────────────────────────────────────────────────────────────────

def _llava15_prompt(noun_hint: str | None = None) -> str:
    hint = f" Choose from: {noun_hint}." if noun_hint else ""
    return (
        "USER: <image>\n"
        "What is the single main object shown in this image? "
        f"Reply with ONE noun only — no articles, no punctuation.{hint}\n"
        "ASSISTANT:"
    )

def _llava16_prompt(noun_hint: str | None = None) -> str:
    hint = f" Choose from: {noun_hint}." if noun_hint else ""
    return (
        "[INST] <image>\n"
        "What is the single main object in this image? "
        f"Reply with ONE noun only — no articles, no punctuation.{hint} [/INST]"
    )

def _qwen2vl_prompt(noun_hint: str | None = None) -> str:
    hint = f" Choose from: {noun_hint}." if noun_hint else ""
    return (
        "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>\n"
        "What is the single main object shown in this image? "
        f"Reply with ONE noun only — no articles, no punctuation.{hint}"
        "<|im_end|>\n<|im_start|>assistant\n"
    )

def _phi35v_prompt(noun_hint: str | None = None) -> str:
    hint = f" Choose from: {noun_hint}." if noun_hint else ""
    return (
        "<|user|>\n<|image_1|>\n"
        "What is the single main object in this image? "
        f"Reply with ONE noun only — no articles, no punctuation.{hint}<|end|>\n"
        "<|assistant|>\n"
    )

# Model-name substring → prompt builder
_PROMPT_REGISTRY: list[tuple[str, callable]] = [
    ("llava-v1.6",   _llava16_prompt),
    ("llava-next",   _llava16_prompt),
    ("llava-1.5",    _llava15_prompt),
    ("llava",        _llava15_prompt),   # fallback for other llava variants
    ("qwen2-vl",     _qwen2vl_prompt),
    ("qwen2vl",      _qwen2vl_prompt),
    ("phi-3.5",      _phi35v_prompt),
    ("phi-3",        _phi35v_prompt),
]

def _get_prompt_fn(model_name: str) -> callable:
    lower = model_name.lower()
    for key, fn in _PROMPT_REGISTRY:
        if key in lower:
            return fn
    # Generic fallback: plain text, no special tokens
    logger.warning(
        f"Unknown model architecture '{model_name}'. "
        "Using generic single-noun prompt — may not work perfectly."
    )
    def _generic(noun_hint=None):
        hint = f" Choose from: {noun_hint}." if noun_hint else ""
        return (
            f"What is the single main object in this image? "
            f"Reply with ONE noun only.{hint}"
        )
    return _generic


# ─────────────────────────────────────────────────────────────────────────────
# VLLMTagger
# ─────────────────────────────────────────────────────────────────────────────

class VLLMTagger:
    """
    Uses a vLLM-served VLM to tag segmented object crops with a single noun.

    Parameters
    ----------
    model : str
        HuggingFace model ID or local path.  Examples:
            "llava-hf/llava-1.5-7b-hf"
            "llava-hf/llava-v1.6-mistral-7b-hf"
            "Qwen/Qwen2-VL-7B-Instruct"
            "microsoft/Phi-3.5-vision-instruct"
    dtype : str
        "float16" (default), "bfloat16", "float32", or "auto".
    max_model_len : int
        Maximum sequence length (tokens).  2048 is fine for single-image tagging.
    tensor_parallel_size : int
        Number of GPUs to shard the model across.  Default 1.
    max_new_tokens : int
        Max tokens to generate per tag.  10 is enough for a single noun.
    noun_hint : str | None
        Optional comma-separated hint to constrain the model's answer,
        e.g. "chair, table, bottle, plant".  Injected into the prompt.
    gpu_memory_utilization : float
        Fraction of GPU memory vLLM may use (0.0–1.0).  Default 0.90.
    """

    def __init__(
        self,
        model: str = "llava-hf/llava-1.5-7b-hf",
        dtype: str = "float16",
        max_model_len: int = 2048,
        tensor_parallel_size: int = 1,
        max_new_tokens: int = 10,
        noun_hint: str | None = None,
        gpu_memory_utilization: float = 0.90,
        device: str = "cuda",   # accepted for API symmetry; vLLM ignores it
    ):
        self.model_name = model
        self.max_new_tokens = max_new_tokens
        self.noun_hint = noun_hint
        self._prompt_fn = _get_prompt_fn(model)
        self._llm = None

        try:
            from vllm import LLM, SamplingParams

            logger.info(f"Loading VLM via vLLM: {model}  dtype={dtype}  "
                        f"tp={tensor_parallel_size}  max_len={max_model_len}")
            self._llm = LLM(
                model                  = model,
                dtype                  = dtype,
                max_model_len          = max_model_len,
                tensor_parallel_size   = tensor_parallel_size,
                gpu_memory_utilization = gpu_memory_utilization,
                trust_remote_code      = True,   # needed for Qwen2-VL, Phi-3.5
                limit_mm_per_prompt    = {"image": 1},
            )
            self._sampling = SamplingParams(
                max_tokens  = max_new_tokens,
                temperature = 0.0,   # greedy — deterministic single-noun answer
                stop        = ["\n", ".", ",", ";"],
            )
            logger.info(f"vLLM engine ready: {model}")

        except ImportError:
            logger.error(
                "vLLM is not installed.  Install it with:\n"
                "    pip install vllm\n"
                "Falling back to stub tagger (placeholder labels)."
            )
        except Exception as exc:
            logger.error(
                f"Failed to load VLM '{model}' via vLLM: {exc}\n"
                "Falling back to stub tagger."
            )

    # ── public API ────────────────────────────────────────────────────────

    @property
    def is_stub(self) -> bool:
        return self._llm is None

    def tag(self, crops_rgb: list[np.ndarray]) -> list[TagResult]:
        """
        Tag a batch of crops in ONE vLLM call (continuous batching).

        Parameters
        ----------
        crops_rgb : list of (H, W, 3) uint8 RGB arrays
            Segmented object crops as produced by extract_crop().

        Returns
        -------
        list[TagResult]
            One TagResult per crop, with .label = the predicted noun.
            .embedding is a zero vector (vLLM does not expose CLIP embeddings);
            use NOVICDecoder.embed_crops() separately if you need embeddings for
            scene-graph fusion.
        """
        if not crops_rgb:
            return []

        if self._llm is None:
            return [self._stub(i) for i in range(len(crops_rgb))]

        prompt_text = self._prompt_fn(self.noun_hint)
        pil_images  = [Image.fromarray(crop) for crop in crops_rgb]

        # Build one vLLM input dict per crop
        inputs = [
            {
                "prompt": prompt_text,
                "multi_modal_data": {"image": img},
            }
            for img in pil_images
        ]

        logger.info(
            f"  [vLLM] Running batch of {len(inputs)} crops through {self.model_name} …"
        )
        outputs = self._llm.generate(inputs, self._sampling)

        results: list[TagResult] = []
        for out in outputs:
            raw   = out.outputs[0].text.strip().lower()
            # strip leading articles and clean whitespace
            label = _clean_noun(raw)
            score = float(out.outputs[0].cumulative_logprob or 0.0)
            # vLLM gives log-prob; convert to a pseudo-probability for display
            pseudo_prob = float(np.exp(np.clip(score, -20, 0)))
            results.append(
                TagResult(
                    label     = label or "object",
                    score     = pseudo_prob,
                    embedding = np.zeros(512, dtype=np.float32),  # no CLIP emb
                    top_k     = [(label or "object", pseudo_prob)],
                )
            )
            logger.debug(f"    vLLM tagged: '{label}'  (logprob={score:.3f})")

        return results

    def close(self) -> None:
        """Release GPU memory held by the vLLM engine."""
        if self._llm is not None:
            try:
                import torch
                del self._llm
                self._llm = None
                torch.cuda.empty_cache()
                logger.info("vLLM engine released.")
            except Exception:
                pass

    # ── stub fallback ─────────────────────────────────────────────────────

    _STUB_LABELS = [
        "object", "container", "furniture", "device",
        "tool", "plant", "bottle", "cup", "chair", "book",
    ]

    def _stub(self, idx: int) -> TagResult:
        label = self._STUB_LABELS[idx % len(self._STUB_LABELS)]
        return TagResult(
            label     = label,
            score     = 0.3,
            embedding = np.zeros(512, dtype=np.float32),
            top_k     = [(label, 0.3)],
        )


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

_ARTICLES = {"a", "an", "the", "this", "that", "some"}

def _clean_noun(text: str) -> str:
    """Strip leading articles and return the first word."""
    words = text.strip().split()
    # drop leading articles
    while words and words[0].lower() in _ARTICLES:
        words.pop(0)
    return words[0].lower() if words else "object"
