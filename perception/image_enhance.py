"""Brightness / contrast correction for the detector and NOVIC crops.

Some captures are much darker than the phone videos the pipeline was tuned on
(Replica's office scenes average a V of ~75/255 against ~160 for its rooms),
and both YOLOE and NOVIC's CLIP encoder see less in them. The correction is
applied only to the image the detector and the crop encoder look at: depth,
poses and the voxel map's colours are untouched.

Configured under `detection.enhance` (all optional):

    method: none | gamma | clahe | gamma_clahe     (default none)
    target_mean: 0.45     gamma: brightness (HSV V, 0..1) the frame is lifted to
    max_gamma: 2.5        never lift harder than this (gamma = 1 / this at most)
    only_if_below: 0.40   gamma: leave frames already at least this bright alone
    clahe_clip: 2.0       CLAHE clip limit, applied to LAB lightness
    clahe_tiles: 8        CLAHE grid size
"""
from __future__ import annotations

import cv2
import numpy as np


def _gamma(image_rgb: np.ndarray, target: float, max_gamma: float, only_if_below: float) -> np.ndarray:
    v = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)[..., 2].astype(np.float32) / 255.0
    mean = float(v.mean())
    if mean >= only_if_below or mean <= 1e-3:
        return image_rgb
    # mean ** g = target  ->  g = log(target) / log(mean); g < 1 brightens
    g = float(np.clip(np.log(target) / np.log(mean), 1.0 / max_gamma, 1.0))
    lut = np.clip(((np.arange(256) / 255.0) ** g) * 255.0, 0, 255).astype(np.uint8)
    return cv2.LUT(image_rgb, lut)


def _clahe(image_rgb: np.ndarray, clip: float, tiles: int) -> np.ndarray:
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tiles, tiles)).apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def enhance(image_rgb: np.ndarray, cfg: dict | None) -> np.ndarray:
    cfg = cfg or {}
    method = cfg.get("method", "none")
    if method in (None, "none"):
        return image_rgb
    out = image_rgb
    if method in ("gamma", "gamma_clahe"):
        out = _gamma(out, cfg.get("target_mean", 0.45), cfg.get("max_gamma", 2.5), cfg.get("only_if_below", 0.40))
    if method in ("clahe", "gamma_clahe"):
        out = _clahe(out, cfg.get("clahe_clip", 2.0), int(cfg.get("clahe_tiles", 8)))
    if method not in ("gamma", "clahe", "gamma_clahe"):
        raise ValueError(f"detection.enhance.method: unknown '{method}'")
    return out
