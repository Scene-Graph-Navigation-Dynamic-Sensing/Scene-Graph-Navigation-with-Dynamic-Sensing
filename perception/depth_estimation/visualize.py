"""Per-frame colorized depth visualization (outputs/visualizations/depth/)."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def save_depth_visualization(
    out_dir: Path,
    frame_idx: int,
    depth_map: np.ndarray,
    valid_mask: np.ndarray | None = None,
    max_depth_m: float = 10.0,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    clipped = np.clip(depth_map, 0.0, max_depth_m)
    normalized = (clipped / max_depth_m * 255.0).astype(np.uint8)
    colorized = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)

    if valid_mask is not None:
        colorized[~valid_mask] = 0

    out_path = out_dir / f"{frame_idx:06d}.jpg"
    cv2.imwrite(str(out_path), colorized)
    return out_path
