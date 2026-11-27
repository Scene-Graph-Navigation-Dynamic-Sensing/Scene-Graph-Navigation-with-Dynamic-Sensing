"""
Step 5: NOVIC Generative Decoder Integration

Wraps the trained NOVIC decoder behind decode() / decode_batch() calls using
the beam configuration: beam_k10_vft2_gp_t1_a0.

NOVIC (Natural-language Open-Vocabulary Image Classifier) generates
free-form text labels from visual embeddings — unlike CLIP zero-shot
which requires a pre-defined label vocabulary.

decode_batch() exists because model.classify_embeds() already accepts a
(N, embed_dim) batch and returns per-sample predictions — decode() only ever
read row 0, so calling it in a loop meant N-1 forward passes were wasted
throwing away everything but the first result. The pipeline in this folder
batches every pending track's embedding into one classify_embeds() call per
frame instead of decoding one crop at a time.

NOTE: Replace the model loading stub with your actual NOVIC model class
      and load_state_dict() call. The interface (decode) stays the same.
"""

import torch
import logging
from typing import Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DecodeResult:
    """Result from a single NOVIC decode operation."""
    text: str
    score: float
    beam_alternatives: list[tuple[str, float]]  # top-k alternatives from beam search


class NOVICDecoder:
    """
    Open-vocabulary generative label decoder using NOVIC.

    Takes a CLIP/SigLIP image embedding and produces a free-form text label
    via beam search with vocabulary-free tempering.

    Args:
        checkpoint_path: Path to the NOVIC checkpoint file.
        beam_k: Beam width for beam search (default 10).
        vocab_free_temp: Vocabulary-free temperature (default 2.0).
        genre_penalty: Whether to apply genre penalty (default True).
        temperature: Softmax temperature (default 1.0).
        alpha: Interpolation weight (default 0.0).
        device: PyTorch device string.
    """

    def __init__(
        self,
        checkpoint_path: str,
        beam_k: int = 10,
        vocab_free_temp: float = 2.0,
        genre_penalty: bool = True,
        temperature: float = 1.0,
        alpha: float = 0.0,
        device: str = "cuda",
    ):
        self.device = device
        self.beam_k = beam_k
        self.vocab_free_temp = vocab_free_temp
        self.genre_penalty = genre_penalty
        self.temperature = temperature
        self.alpha = alpha

        # --- MODEL LOADING ---
        self.model = None
        self._checkpoint_path = checkpoint_path

        if checkpoint_path:
            try:
                import sys
                import os
                novic_path = os.path.abspath("/home/ravindu/SceneGraphNav/external/novic")
                if novic_path not in sys.path:
                    sys.path.append(novic_path)
                
                from infer import NOVICModel
                
                # Construct the gencfg name based on the params
                # e.g., 'beam_k10_vnone_gp_t1_a0'
                gp_flag = 'p' if self.genre_penalty else 'n'
                gencfg_name = f"beam_k{self.beam_k}_vnone_g{gp_flag}_t{int(self.temperature) if self.temperature.is_integer() else self.temperature}_a{int(self.alpha) if self.alpha.is_integer() else self.alpha}"
                
                logger.info(f"Initializing NOVICModel wrapper with gencfg={gencfg_name}")
                self.model = NOVICModel(
                    checkpoint=checkpoint_path,
                    gencfg=gencfg_name,
                    device=device,
                )
                self.model.load_decoder()
                logger.info(f"Real NOVIC model successfully loaded from {checkpoint_path}")
            except Exception as e:
                import traceback
                logger.error(f"Failed to load real NOVIC model: {e}")
                logger.error(traceback.format_exc())
                self.model = None

        if self.model is None:
            logger.warning(
                f"Using stub decoder — labels will be placeholders."
            )

    @torch.no_grad()
    def decode(self, image_embedding: torch.Tensor) -> tuple[str, float]:
        """
        Decode a visual embedding into an open-vocabulary text label.

        Args:
            image_embedding: (1, embed_dim) L2-normalized CLIP/SigLIP embedding.

        Returns:
            Tuple of (label_string, confidence_score).
        """
        if self.model is not None:
            return self._decode_real(image_embedding)
        else:
            return self._decode_stub(image_embedding)

    def _decode_real(self, image_embedding: torch.Tensor) -> tuple[str, float]:
        """Decode using the actual NOVIC model."""
        output = self.model.classify_embeds(image_embedding)
        return output.preds[0][0], output.probs[0][0]

    @torch.no_grad()
    def decode_batch(self, image_embeddings: torch.Tensor) -> list[tuple[str, float]]:
        """
        Decode a batch of embeddings in one NOVIC forward pass.

        Args:
            image_embeddings: (N, embed_dim) L2-normalized embeddings, one row
                per pending track this frame.

        Returns:
            List of N (label_string, confidence_score) tuples, same order as input.
        """
        n = image_embeddings.shape[0]
        if n == 0:
            return []
        if self.model is not None:
            output = self.model.classify_embeds(image_embeddings)
            return [(output.preds[i][0], output.probs[i][0]) for i in range(n)]
        return [self._decode_stub(image_embeddings[i : i + 1]) for i in range(n)]

    def _decode_stub(self, image_embedding: torch.Tensor) -> tuple[str, float]:
        """
        Stub decoder for development/testing without a real NOVIC checkpoint.
        Generates a deterministic placeholder label based on embedding statistics.
        """
        # Use embedding hash for deterministic pseudo-random label
        emb_sum = float(image_embedding.sum().cpu())
        emb_std = float(image_embedding.std().cpu())

        # Pseudo-categories based on embedding statistics
        categories = [
            "object", "person", "vehicle", "animal", "furniture",
            "electronics", "food", "tool", "container", "clothing",
            "plant", "building", "sports_equipment", "instrument", "toy",
        ]
        idx = int(abs(emb_sum * 1000)) % len(categories)
        confidence = min(0.95, max(0.1, 0.5 + emb_std * 2))

        label = categories[idx]
        logger.debug(f"Stub decode: '{label}' (conf={confidence:.3f})")

        return label, confidence

    @torch.no_grad()
    def decode_detailed(self, image_embedding: torch.Tensor) -> DecodeResult:
        """
        Extended decode that also returns beam-search alternatives.

        Args:
            image_embedding: (1, embed_dim) L2-normalized embedding.

        Returns:
            DecodeResult with top label, score, and beam alternatives.
        """
        if self.model is not None:
            output = self.model.classify_embeds(image_embedding)
            return DecodeResult(
                text=output.preds[0][0],
                score=output.probs[0][0],
                beam_alternatives=[(output.preds[0][i], output.probs[0][i]) for i in range(1, len(output.preds[0]))],
            )
        else:
            label, score = self.decode(image_embedding)
            return DecodeResult(text=label, score=score, beam_alternatives=[])
