"""Padded object crop extraction, feeding NOVIC.

Two modes (mirrors /home/ravindu/rashmika/Novic/YoloE-26/yoloe_novic_pipeline.py):
  - "bbox": a padded box crop, keeping surrounding context.
  - "segmented": the same padded box crop with everything outside the
    instance mask zeroed out, isolating the object itself.
"""
from __future__ import annotations

import numpy as np


def extract_crop(
    image_rgb: np.ndarray,
    bbox: np.ndarray,
    mask: np.ndarray | None = None,
    mode: str = "segmented",
    padding_px: int = 15,
    padding_frac: float | None = None,
    blur_sigma_frac: float = 0.04,
) -> np.ndarray:
    """padding_frac, when given, pads by that fraction of the box's own size on
    each side instead of a fixed pixel count (dtd_pipeline's `pad_frac=0.15`).
    Scaling the context with the object keeps small far-away detections from
    being swamped by background.

    mode: "bbox" (the padded box), "segmented" (outside the mask set to black),
    or "blur" (outside the mask Gaussian-blurred, sigma = blur_sigma_frac of
    the crop's longer side)."""
    h, w = image_rgb.shape[:2]
    x1, y1, x2, y2 = bbox.astype(int)

    if padding_frac is not None:
        pad_x = int(round((x2 - x1) * padding_frac))
        pad_y = int(round((y2 - y1) * padding_frac))
    else:
        pad_x = pad_y = padding_px

    px1 = max(0, x1 - pad_x)
    py1 = max(0, y1 - pad_y)
    px2 = min(w, x2 + pad_x)
    py2 = min(h, y2 + pad_y)

    crop = image_rgb[py1:py2, px1:px2]
    if crop.size == 0:
        return np.zeros((1, 1, 3), dtype=np.uint8)

    if mode == "segmented" and mask is not None:
        mask_crop = mask[py1:py2, px1:px2]
        crop = crop.copy()
        crop[~mask_crop] = 0
    elif mode == "blur" and mask is not None:
        # object sharp, background blurred: the crop stays a natural photo
        # (what NOVIC was trained on) while the context carries little detail
        import cv2

        mask_crop = mask[py1:py2, px1:px2]
        sigma = max(3.0, blur_sigma_frac * max(crop.shape[:2]))
        blurred = cv2.GaussianBlur(crop, (0, 0), sigmaX=sigma, sigmaY=sigma)
        crop = np.where(mask_crop[..., None], crop, blurred)

    return crop
