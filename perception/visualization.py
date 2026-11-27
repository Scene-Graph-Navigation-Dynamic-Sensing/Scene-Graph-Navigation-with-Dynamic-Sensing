"""Shared per-frame debug-overlay helpers for detection+segmentation and
semantic-decoding visualizations.

Adapted from Scene-Graph-Generation/main_live.py's build_mask_overlay(),
build_mask_contour_overlay(), and save_perception_debug_outputs()
(~L174-276). The key change: those originals looked up a fixed class name
via `classes[class_ids[idx]]`. There is no class list in this pipeline —
detection() output carries no label at all, and only the semantic-decoding
call site has a real label (NOVIC's decoded name). Both call this module
with whatever `labels` list is appropriate for that stage.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def build_mask_overlay(image_rgb: np.ndarray, mask: np.ndarray, color=(0, 255, 255), alpha: float = 0.45) -> np.ndarray:
    overlay = image_rgb.copy()
    overlay[mask] = (overlay[mask] * (1.0 - alpha) + np.array(color, dtype=np.float32) * alpha).astype(np.uint8)
    return overlay


def build_mask_contour_overlay(image_rgb: np.ndarray, mask: np.ndarray, color=(255, 0, 0)) -> np.ndarray:
    overlay = image_rgb.copy()
    mask_u8 = (mask.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, color, 2)
    return overlay


def save_detection_debug_frame(
    out_dir: Path,
    frame_idx: int,
    image_rgb: np.ndarray,
    boxes: np.ndarray,
    confidences: np.ndarray,
    masks: list[np.ndarray],
    labels: list[str] | None = None,
) -> Path:
    """Saves one annotated frame: boxes + per-detection mask overlays + a
    combined overlay. If `labels` is None, boxes are labeled with just the
    objectness confidence (used for the raw detection stage). If `labels`
    is provided (NOVIC's decoded + canonicalized names), those are drawn
    instead — this is the only place a semantic name is shown.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    vis = image_rgb.copy()
    combined = image_rgb.copy()
    rng = np.random.default_rng(frame_idx)

    for idx, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        label = labels[idx] if labels else f"obj {confidences[idx]:.2f}"
        color = tuple(int(c) for c in rng.integers(0, 255, size=3))

        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(vis, label, (x1, max(y1 - 10, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if idx < len(masks):
            mask = masks[idx]
            combined[mask] = (combined[mask] * 0.5 + np.array(color) * 0.5).astype(np.uint8)
            cv2.rectangle(combined, (x1, y1), (x2, y2), color, 2)
            cv2.putText(combined, label, (x1, max(y1 - 10, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    frame_dir = out_dir / f"{frame_idx:06d}"
    frame_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(frame_dir / "00_original.jpg"), bgr)
    cv2.imwrite(str(frame_dir / "01_boxes.jpg"), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    if masks:
        cv2.imwrite(str(frame_dir / "02_masks_combined.jpg"), cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
        for idx, mask in enumerate(masks):
            overlay = build_mask_overlay(image_rgb, mask)
            contour = build_mask_contour_overlay(image_rgb, mask)
            cv2.imwrite(str(frame_dir / f"03_mask_{idx:02d}_overlay.jpg"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(frame_dir / f"03_mask_{idx:02d}_contour.jpg"), cv2.cvtColor(contour, cv2.COLOR_RGB2BGR))

    return frame_dir / "01_boxes.jpg"
