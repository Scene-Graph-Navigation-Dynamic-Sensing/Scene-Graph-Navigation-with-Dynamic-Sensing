"""Open-vocabulary object tagging via NOVIC.

Wraps the real NOVICModel from /home/ravindu/SceneGraphNav/external/novic/infer.py
(the upstream NOVIC repo already checked out in this workspace, with a real
trained decoder checkpoint at novic/outputs/.../*.train — confirmed present,
42,919 target nouns). This is the single source of both the object's
embedding and its tag: NOVICModel.embed_images() returns the image
embedding (this is "the NOVIC embedding" attached to scene graph nodes),
and NOVICModel.classify_embeds() decodes it into free-form noun
predictions. No separate CLIP encoder module is needed — NOVIC embeds
images with its own built-in embedder.

Mirrors the working reference integration at
/home/ravindu/rashmika/Novic/YoloE-26/yoloe_novic_pipeline.py (YOLOE-26 +
NOVIC), adapted into a reusable class instead of a one-shot script.

Falls back to a deterministic stub decoder if the checkpoint can't be
loaded (missing file, import error, no GPU) so the rest of the pipeline
stays testable without it.
"""
from __future__ import annotations

import functools
import logging
import os
import threading
import sys
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class TagResult:
    label: str                          # top-1 decoded noun
    score: float                        # top-1 probability
    embedding: np.ndarray               # the NOVIC image embedding (F,)
    top_k: list[tuple[str, float]]      # beam alternatives, for debugging/visualization


def _serialized(method):
    """One model call at a time: the streaming pipeline embeds crops on its
    detection thread and decodes names on its fusion thread."""
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)
    return wrapper


class NOVICDecoder:
    def __init__(
        self,
        novic_dir: str = "/home/ravindu/SceneGraphNav/external/novic",
        checkpoint_path: str = "",
        gencfg: str = "beam_k10_vnone_gp_t1_a0",
        guide_targets: list[str] | None = None,
        device: str = "cuda",
        embedding_batch_size: int = 8,
    ):
        if isinstance(embedding_batch_size, bool) or not isinstance(embedding_batch_size, int):
            raise ValueError("embedding_batch_size must be an integer greater than zero")
        if embedding_batch_size < 1:
            raise ValueError("embedding_batch_size must be an integer greater than zero")
        self.device = device if torch.cuda.is_available() else "cpu"
        self.embedding_batch_size = embedding_batch_size
        self._model = None
        self._lock = threading.RLock()

        if checkpoint_path and os.path.exists(checkpoint_path):
            try:
                if novic_dir not in sys.path:
                    sys.path.insert(0, novic_dir)
                from infer import NOVICModel, utils

                utils.allow_tf32(enable=True)
                self._model = NOVICModel(
                    checkpoint=checkpoint_path,
                    gencfg=gencfg,
                    guide_targets=guide_targets,
                    device=self.device,
                )
                self._model.__enter__()  # keep embedder + decoder loaded for the pipeline's lifetime
                logger.info(
                    f"NOVIC decoder loaded from '{checkpoint_path}' "
                    f"({len(self._model.model_targets)} target nouns, gencfg={gencfg})."
                )
            except Exception as e:
                logger.error(f"Failed to load NOVIC model from '{checkpoint_path}': {e}. Using stub decoder.")
                self._model = None
        else:
            logger.warning(f"No NOVIC checkpoint at '{checkpoint_path}'. Using stub decoder.")

    def close(self):
        if self._model is not None:
            self._model.__exit__(None, None, None)

    @property
    def is_stub(self) -> bool:
        return self._model is None

    @_serialized
    @torch.no_grad()
    def embed_crops(self, crops_rgb: list[np.ndarray]) -> np.ndarray:
        """Cheap per-detection embedding step — safe to call every frame for
        every detection (used both for fusion-matching similarity in
        object_fusion, and as the running per-instance feature merged there).
        Returns (N, F) float32 array.
        """
        if not crops_rgb:
            return np.empty((0, 0), dtype=np.float32)

        if self._model is not None:
            batches: list[np.ndarray] = []
            for start in range(0, len(crops_rgb), self.embedding_batch_size):
                crop_batch = crops_rgb[start:start + self.embedding_batch_size]
                pil_images = [Image.fromarray(crop) for crop in crop_batch]
                embeds = self._model.embed_images(pil_images)
                batches.append(embeds.detach().cpu().numpy().astype(np.float32))
                del embeds, pil_images
                # Release per-batch activation memory before the next set of
                # crops. This matters for the large NOVIC image encoder when
                # a PF detector finds many objects in a single frame.
                if self.device == "cuda":
                    torch.cuda.empty_cache()
            return np.concatenate(batches, axis=0)

        return np.stack([self._embed_stub(crop) for crop in crops_rgb])

    def target_nouns(self) -> list[str]:
        """NOVIC's target vocabulary (42,919 nouns for the checkpoint used); empty for the stub decoder."""
        return list(self._model.model_targets) if self._model is not None else []

    @_serialized
    @torch.no_grad()
    def embed_text(self, texts: list[str]) -> np.ndarray:
        """Text-side embedding, in the same CLIP-family space as embed_crops()'s
        image embeddings (NOVIC's embedder is a CLIP model; this just calls its
        text tower instead of its vision tower). Used by
        task_planning/clip_tool.py to score candidate objects against a
        natural-language query via cosine similarity -- not used anywhere in
        the main detect/tag/fuse pipeline. Returns (N, F) float32 array.
        """
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        if self._model is not None:
            with self._model.embedder.inference_mode():
                tokens = self._model.embedder.tokenize(texts, output_dict=True)
                embeds = self._model.embedder.inference_tokens(tokens)
            return embeds.detach().cpu().numpy().astype(np.float32)

        return np.stack([self._embed_stub(np.frombuffer(t.encode(), dtype=np.uint8)) for t in texts])

    @_serialized
    @torch.no_grad()
    def decode_embeddings(self, embeddings: np.ndarray) -> list[TagResult]:
        """Expensive beam-search decode step — call once per finalized 3D
        instance (batched across all instances in a scene in one call), not
        once per frame. This is what makes tagging O(instances) instead of
        O(frames x detections).
        """
        if embeddings.size == 0:
            return []

        if self._model is not None:
            embeds_t = torch.from_numpy(embeddings).to(self.device)
            output = self._model.classify_embeds(embeds_t)
            results = []
            for i in range(embeddings.shape[0]):
                preds = output.preds[i]
                probs = output.probs[i]
                results.append(
                    TagResult(
                        label=preds[0] if preds else "unknown",
                        score=float(probs[0]) if probs else 0.0,
                        embedding=embeddings[i],
                        top_k=list(zip(preds, (float(p) for p in probs))),
                    )
                )
            return results

        return [self._decode_stub(embeddings[i]) for i in range(embeddings.shape[0])]

    @torch.no_grad()
    def tag(self, crops_rgb: list[np.ndarray]) -> list[TagResult]:
        """Convenience one-shot embed+decode for a small batch of crops
        (testing / single-frame use). The fusion pipeline uses embed_crops()
        and decode_embeddings() separately instead, for the reason above.
        """
        embeddings = self.embed_crops(crops_rgb)
        return self.decode_embeddings(embeddings)

    _STUB_EMBED_DIM = 512
    _STUB_CATEGORIES = [
        "object", "container", "furniture", "electronics", "tool",
        "plant", "book", "cup", "bottle", "chair",
    ]

    def _embed_stub(self, crop_rgb: np.ndarray) -> np.ndarray:
        """Deterministic placeholder embedding so downstream stages (fusion
        similarity, storage) stay testable without the real checkpoint."""
        h = abs(hash(crop_rgb.tobytes())) if crop_rgb.size else 0
        rng = np.random.default_rng(h % (2**32))
        embedding = rng.normal(size=self._STUB_EMBED_DIM).astype(np.float32)
        embedding /= np.linalg.norm(embedding) + 1e-8
        return embedding

    def _decode_stub(self, embedding: np.ndarray) -> TagResult:
        h = abs(hash(embedding.tobytes())) if embedding.size else 0
        idx = h % len(self._STUB_CATEGORIES)
        label = self._STUB_CATEGORIES[idx]
        score = 0.3
        logger.debug(f"Stub NOVIC decode: '{label}' (conf={score:.3f})")
        return TagResult(label=label, score=score, embedding=embedding, top_k=[(label, score)])
