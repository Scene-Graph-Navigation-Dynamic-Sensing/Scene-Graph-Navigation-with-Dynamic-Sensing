"""Per-frame detection validation: reject non-objects and group boxes, and let each detection own its pixels.

All tests take one frame's detections (boxes, full-frame boolean masks) and are configured under
`detection.validation` (every part off by default):

  border_stuff       a large mask touching >= min_borders image borders is a wall / floor / ceiling candidate
                     (real objects usually fit in the frame); rejected
  label_stuff        prompt-free YOLOE's own class is a stuff / scene label ("wall", "floor", "living room", ...);
                     rejected (the label is otherwise ignored by the pipeline)
  group_test         the mask is "explained by other objects": >= min_children smaller detections lie inside it AND
                     (their masks cover >= child_coverage of it, OR it is >= 2 large disconnected blobs); rejected
  pixel_ownership    a mask >= containment inside a larger mask is subtracted from the larger one, so each pixel
                     belongs to the smaller, more specific detection; the larger detection's crop gets the removed
                     pixels filled (inpaint / blur / mean) so CLIP does not see the inner object
  purity_weighting   each detection's embedding weight = share of its box NOT covered by other detections' masks

The first three run inside the detector, before BoT-SORT (perception/detection/tracked_yoloe_detector.py);
ownership and purity run after it, before CLIP (object_fusion/streaming.detect_frame).
"""
from __future__ import annotations

import cv2
import numpy as np

# prompt-free YOLOE vocabulary entries that name a surface or a place, not an object
STUFF_LABELS = {"wall", "floor", "ceiling", "wallpaper", "tile wall", "wood wall", "glass wall", "hardwood floor",
                "wood floor", "tile flooring", "kitchen floor", "glass floor", "dance floor", "plaster", "surface",
                "corner", "hallway", "roof", "tile", "interior design", "home interior", "floor plan"}


def is_stuff_label(name: str) -> bool:
    n = name.lower().strip()
    return n in STUFF_LABELS or n == "room" or n.endswith(" room")


_NONOBJECT_SETS: dict[str, frozenset] = {}


def nonobject_labels(path: str) -> frozenset:
    """Prompt-free YOLOE tags that are not physical objects (one per line, '#' comments), for label_vocab;
    built by scripts/build_yoloe_object_vocab.py. Relative paths resolve against the repo root."""
    from pathlib import Path
    p = Path(path)
    if not p.is_absolute():
        p = Path(__file__).resolve().parents[2] / p
    key = str(p)
    if key not in _NONOBJECT_SETS:
        _NONOBJECT_SETS[key] = frozenset(l.strip().lower() for l in p.read_text().splitlines()
                                         if l.strip() and not l.startswith("#"))
    return _NONOBJECT_SETS[key]


def touches_borders(mask: np.ndarray) -> int:
    return int(mask[0].any()) + int(mask[-1].any()) + int(mask[:, 0].any()) + int(mask[:, -1].any())


def border_stuff(mask: np.ndarray, min_borders: int = 2, min_area_frac: float = 0.15) -> bool:
    return mask.mean() >= min_area_frac and touches_borders(mask) >= min_borders


def children_of(masks: list[np.ndarray], areas: np.ndarray, i: int, containment: float) -> list[int]:
    """Detections j whose mask lies >= containment inside mask i and is smaller than it."""
    out = []
    for j, m in enumerate(masks):
        if j == i or areas[j] == 0 or areas[j] >= 0.8 * areas[i]:
            continue
        if (m & masks[i]).sum() >= containment * areas[j]:
            out.append(j)
    return out


def group_flags(masks: list[np.ndarray], min_children: int = 2, child_coverage: float = 0.6,
                fragment_min_frac: float = 0.15, containment: float = 0.85) -> tuple[np.ndarray, list[str]]:
    """True where a detection is a group box; reason per detection."""
    areas = np.array([m.sum() for m in masks], dtype=np.int64)
    flags, reasons = np.zeros(len(masks), bool), [""] * len(masks)
    for i, m in enumerate(masks):
        if areas[i] == 0:
            continue
        kids = children_of(masks, areas, i, containment)
        if len(kids) < min_children:
            continue
        union = np.zeros_like(m)
        for j in kids:
            union |= masks[j]
        coverage = (union & m).sum() / areas[i]
        n, _, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
        big = int((stats[1:, cv2.CC_STAT_AREA] >= fragment_min_frac * areas[i]).sum()) if n > 1 else 0
        if coverage >= child_coverage:
            flags[i], reasons[i] = True, f"children cover {coverage:.0%}"
        elif big >= 2:
            flags[i], reasons[i] = True, f"{big} disconnected blobs"
    return flags, reasons


def pixel_ownership(masks: list[np.ndarray], containment: float = 0.8, min_owned_frac: float = 0.1
                    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """(owned masks, removed-pixel masks): each larger mask minus the smaller masks lying inside it. If almost
    nothing would remain (< min_owned_frac), the detection keeps its own mask (it is then mostly its children,
    which the group test handles)."""
    areas = np.array([m.sum() for m in masks], dtype=np.int64)
    owned, removed = [], []
    for i, m in enumerate(masks):
        kids = children_of(masks, areas, i, containment)
        if not kids:
            owned.append(m)
            removed.append(np.zeros_like(m))
            continue
        cut = np.zeros_like(m)
        for j in kids:
            cut |= masks[j]
        o = m & ~cut
        if o.sum() < min_owned_frac * areas[i]:
            owned.append(m)
            removed.append(np.zeros_like(m))
        else:
            owned.append(o)
            removed.append(cut & m)
    return owned, removed


def fill_pixels(image_rgb: np.ndarray, region: np.ndarray, method: str = "inpaint") -> np.ndarray:
    """Image with `region` replaced by its surroundings, so an inner object no longer shows in a crop."""
    if not region.any() or method == "none":
        return image_rgb
    if method == "inpaint":
        return cv2.inpaint(image_rgb, (region.astype(np.uint8) * 255), 5, cv2.INPAINT_TELEA)
    out = image_rgb.copy()
    if method == "blur":
        blurred = cv2.GaussianBlur(image_rgb, (0, 0), sigmaX=15)
        out[region] = blurred[region]
    else:                                                   # "mean": average colour of the ring around the region
        ring = cv2.dilate(region.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool) & ~region
        out[region] = image_rgb[ring].mean(0).astype(np.uint8) if ring.any() else 127
    return out


def purity_weights(boxes: np.ndarray, masks: list[np.ndarray], floor: float = 0.1) -> np.ndarray:
    """Share of each detection's box not covered by the OTHER detections' masks (>= floor)."""
    if not len(masks):
        return np.zeros(0)
    h, w = masks[0].shape
    weights = np.ones(len(masks))
    for i, b in enumerate(boxes):
        x1, y1 = max(int(b[0]), 0), max(int(b[1]), 0)
        x2, y2 = min(int(np.ceil(b[2])), w), min(int(np.ceil(b[3])), h)
        area = max((x2 - x1) * (y2 - y1), 1)
        others = np.zeros((y2 - y1, x2 - x1), bool)
        for j, m in enumerate(masks):
            if j != i:
                others |= m[y1:y2, x1:x2]
        weights[i] = max(1.0 - others.sum() / area, floor)
    return weights
