"""Monocular metric depth via Depth Anything V3 (falls back to V2 metric-indoor).

Adapted from Scene-Graph-Generation/main_live.py's inline depth loop
(model load ~L779-791, per-frame inference ~L1145-1153) into a standalone,
swappable component. A future stereo depth source implements the same
DepthEstimator interface and drops in without touching downstream code.
"""
from __future__ import annotations

import logging
from typing import Protocol

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)


class DepthEstimator(Protocol):
    def estimate(self, rgb: np.ndarray, fx: float | None = None, fy: float | None = None) -> tuple[np.ndarray, bool]:
        """rgb: HxWx3 uint8 RGB image; fx/fy: focal length in pixels at rgb's
        resolution. Returns (depth_map HxW float32, is_metric)."""
        ...


# DA3's metric checkpoints predict depth for a canonical focal length of 300 px;
# metric depth = raw * focal / 300, with focal measured at the resolution the
# network actually ran at (depth_anything_3/utils/alignment.py::apply_metric_scaling).
# The upstream API only applies this inside the *nested* model, so a standalone
# DA3METRIC checkpoint returns unscaled canonical depth — measured on 3RScan
# b05fddcf as 1.50x too short before this conversion and 0.975x after.
DA3_CANONICAL_FOCAL_PX = 300.0


def da3_metric_scale(depth_hw: tuple[int, int], image_hw: tuple[int, int], fx: float, fy: float) -> float:
    """Factor converting DA3METRIC canonical depth at depth_hw to metres,
    given the focal length (fx, fy) of the image_hw input it was predicted from."""
    (dh, dw), (ih, iw) = depth_hw, image_hw
    focal_at_depth_res = 0.5 * (fx * dw / iw + fy * dh / ih)
    return focal_at_depth_res / DA3_CANONICAL_FOCAL_PX


class DepthAnythingEstimator:
    """Depth Anything V3 wrapper, with a V2 metric-indoor fallback."""

    def __init__(
        self,
        model_id: str = "depth-anything/DA3-BASE",
        fallback_model_id: str = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
        device: str = "cuda",
        min_valid_depth_m: float = 0.1,
        max_valid_depth_m: float = 10.0,
    ):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.min_valid_depth_m = min_valid_depth_m
        self.max_valid_depth_m = max_valid_depth_m
        self._backend = None
        self._is_metric = True
        self._needs_focal_scaling = False
        self._warned_missing_focal = False

        try:
            from depth_anything_3.api import DepthAnything3

            self._model = DepthAnything3.from_pretrained(model_id).to(self.device)
            self._model.eval()
            self._backend = "v3"
            # DA3-BASE/LARGE/GIANT are affine-invariant (relative depth only);
            # DA3METRIC-* are metric once scaled by focal length.
            self._needs_focal_scaling = "metric" in model_id.lower()
            self._is_metric = self._needs_focal_scaling
            logger.info(f"Loaded Depth Anything V3 model '{model_id}' on {self.device} "
                        f"({'metric' if self._is_metric else 'relative'} depth).")
        except Exception as e:
            logger.warning(
                f"Could not load Depth Anything V3 model '{model_id}' ({e}); "
                f"falling back to '{fallback_model_id}'."
            )
            from transformers import AutoModelForDepthEstimation, AutoImageProcessor

            self._processor = AutoImageProcessor.from_pretrained(fallback_model_id)
            self._model = AutoModelForDepthEstimation.from_pretrained(fallback_model_id).to(self.device)
            self._model.eval()
            self._backend = "v2"
            logger.info(f"Loaded Depth Anything V2 (metric-indoor) fallback model on {self.device}.")

    def estimate(self, rgb: np.ndarray, fx: float | None = None, fy: float | None = None) -> tuple[np.ndarray, bool]:
        """fx/fy are required for metric output from a DA3METRIC checkpoint.
        Without them the raw canonical depth is returned (is_metric=False) —
        callers that align to reference depth afterwards don't need them."""
        pil_img = Image.fromarray(rgb)
        with torch.no_grad():
            if self._backend == "v3":
                prediction = self._model.inference([pil_img])
                depth = prediction.depth[0]
                if isinstance(depth, torch.Tensor):
                    depth = depth.cpu().numpy()
            else:
                inputs = self._processor(images=pil_img, return_tensors="pt").to(self.device)
                outputs = self._model(**inputs)
                depth = outputs.predicted_depth.squeeze().cpu().numpy()

        depth = np.asarray(depth, dtype=np.float32)
        if not self._needs_focal_scaling:
            return depth, self._is_metric

        if fx is None or fy is None:
            if not self._warned_missing_focal:
                logger.warning("DA3 metric model called without fx/fy: returning canonical depth, "
                               "which is NOT in metres. Pass the camera focal length.")
                self._warned_missing_focal = True
            return depth, False

        depth *= da3_metric_scale(depth.shape, rgb.shape[:2], fx, fy)
        return depth, True

    def is_valid(self, depth_map: np.ndarray) -> bool:
        depth_max = float(np.nanmax(depth_map)) if depth_map.size else 0.0
        return np.isfinite(depth_max) and depth_max >= 1e-4

    def valid_mask(self, depth_map: np.ndarray) -> np.ndarray:
        return (
            (depth_map > self.min_valid_depth_m)
            & (depth_map < self.max_valid_depth_m)
            & np.isfinite(depth_map)
        )
