"""
Step 4: Padded Cropping + CLIP/SigLIP Visual Encoder

Extracts object crops from video frames with contextual padding, then encodes
them into normalized visual embeddings using CLIP or SigLIP models.

The padded crop preserves surrounding context (important for small or
partially occluded objects), while the L2-normalized embeddings enable
cosine-similarity operations downstream (fusion, cycle-consistency, etc.).
"""

import cv2
import numpy as np
import torch
import logging
from typing import Optional
from PIL import Image

logger = logging.getLogger(__name__)


class CLIPEncoder:
    """
    Visual encoder using CLIP/SigLIP for generating object embeddings.

    Supports multiple model architectures through the open_clip library.
    The default model (ViT-H-14-378-quickgelu) is chosen for its strong
    open-vocabulary performance (DFN-5B model).

    Args:
        model_name: open_clip model architecture name.
        pretrained: Pretrained weights identifier.
        device: PyTorch device string.
        embed_dim: Expected embedding dimensionality (for validation).
    """

    def __init__(
        self,
        model_name: str = "ViT-H-14-378-quickgelu",
        pretrained: str = "dfn5b",
        device: str = "cuda",
        embed_dim: int = 1024,
        fp16: bool = False,
    ):
        try:
            import open_clip
        except ImportError:
            raise ImportError(
                "open_clip not installed. Run: pip install open_clip_torch"
            )

        self.device = device
        self.embed_dim = embed_dim
        self.fp16 = fp16

        logger.info(f"Loading CLIP model: {model_name} ({pretrained})...")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.model.to(device).eval()
        if fp16:
            self.model.half()

        # Validate embedding dimension
        with torch.no_grad():
            res = 378
            if hasattr(self.preprocess, 'transforms'):
                for t in self.preprocess.transforms:
                    if hasattr(t, 'size'):
                        res = t.size[0] if isinstance(t.size, (tuple, list)) else t.size
                        break
            dummy_pil = Image.new('RGB', (res, res))
            dummy = self.preprocess(dummy_pil).unsqueeze(0).to(device)
            if fp16:
                dummy = dummy.half()
            out = self.model.encode_image(dummy)
            actual_dim = out.shape[-1]
            if actual_dim != embed_dim:
                logger.warning(
                    f"Embedding dim mismatch: expected {embed_dim}, got {actual_dim}. "
                    f"Updating embed_dim to {actual_dim}."
                )
                self.embed_dim = actual_dim

        logger.info(
            f"CLIP encoder ready: {model_name}, embed_dim={self.embed_dim}, device={device}"
        )

    @staticmethod
    def pad_crop(
        frame: np.ndarray,
        bbox: np.ndarray,
        pad_frac: float = 0.15,
    ) -> np.ndarray:
        """
        Extract a padded crop from a frame given a bounding box.

        Adds contextual padding around the detection to preserve surrounding
        visual context (important for CLIP's global attention mechanism).

        Args:
            frame: Full video frame (H, W, 3), BGR format.
            bbox: Bounding box [x1, y1, x2, y2] in pixel coordinates.
            pad_frac: Fraction of box width/height to pad on each side (default 15%).

        Returns:
            Cropped image region as numpy array, BGR format.
        """
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox

        bw = x2 - x1
        bh = y2 - y1
        px = bw * pad_frac
        py = bh * pad_frac

        # Clamp to frame boundaries
        x1_pad = max(0, int(x1 - px))
        y1_pad = max(0, int(y1 - py))
        x2_pad = min(w, int(x2 + px))
        y2_pad = min(h, int(y2 + py))

        crop = frame[y1_pad:y2_pad, x1_pad:x2_pad]

        # Guard against degenerate crops
        if crop.size == 0:
            logger.warning(
                f"Empty crop for bbox {bbox}. Returning 1x1 fallback."
            )
            return np.zeros((1, 1, 3), dtype=np.uint8)

        return crop

    @torch.no_grad()
    def encode(
        self,
        frame: np.ndarray,
        bbox: np.ndarray,
        pad_frac: float = 0.15,
    ) -> torch.Tensor:
        """
        Extract a padded crop and encode it into a normalized CLIP embedding.

        Args:
            frame: Full video frame (H, W, 3), BGR format.
            bbox: Bounding box [x1, y1, x2, y2] in pixel coordinates.
            pad_frac: Contextual padding fraction.

        Returns:
            L2-normalized embedding tensor of shape (1, embed_dim).
        """
        crop = self.pad_crop(frame, bbox, pad_frac)
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(crop_rgb)
        tensor = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        if self.fp16:
            tensor = tensor.half()

        embedding = self.model.encode_image(tensor).float()
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)  # L2 normalize

        return embedding  # (1, embed_dim)

    @torch.no_grad()
    def encode_batch(
        self,
        frame: np.ndarray,
        bboxes: list[np.ndarray],
        pad_frac: float = 0.15,
    ) -> torch.Tensor:
        """
        Batch-encode multiple crops from a single frame.

        More efficient than calling encode() in a loop when there are
        many pending tracks to process in one frame.

        Args:
            frame: Full video frame (H, W, 3), BGR format.
            bboxes: List of bounding boxes [x1, y1, x2, y2].
            pad_frac: Contextual padding fraction.

        Returns:
            L2-normalized embedding tensor of shape (N, embed_dim).
        """
        if not bboxes:
            return torch.empty(0, self.embed_dim, device=self.device)

        tensors = []
        for bbox in bboxes:
            crop = self.pad_crop(frame, bbox, pad_frac)
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(crop_rgb)
            tensors.append(self.preprocess(pil_image))

        batch = torch.stack(tensors).to(self.device)
        if self.fp16:
            batch = batch.half()
        embeddings = self.model.encode_image(batch).float()
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

        return embeddings  # (N, embed_dim)

    @torch.no_grad()
    def encode_text(self, texts: list[str]) -> torch.Tensor:
        """
        Encode text prompts into CLIP text embeddings.
        Useful for open-vocabulary label grounding (Step 9.4).

        Args:
            texts: List of text strings to encode.

        Returns:
            L2-normalized text embedding tensor of shape (N, embed_dim).
        """
        import open_clip

        tokens = open_clip.tokenize(texts).to(self.device)
        embeddings = self.model.encode_text(tokens)
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        return embeddings
