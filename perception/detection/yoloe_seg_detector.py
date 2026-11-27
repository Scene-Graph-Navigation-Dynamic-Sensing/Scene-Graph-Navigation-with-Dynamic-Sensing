"""Class-agnostic object detection + segmentation via YOLOE-seg-pf (prompt-free).

Run with no text prompt / class list (model.predict() only — no
model.set_classes() call, unlike a text-prompted YOLOE/YOLOWorld setup).
The "-pf" checkpoint is a distilled, re-parameterized YOLOE that no longer
accepts prompts; it still returns an internal class id per detection, but
that id is deliberately discarded here — this module returns geometry
(box + mask) and objectness confidence only, never a semantic label. The
object's tag comes exclusively from NOVIC downstream
(semantic_decoding/novic), per the pipeline design.

(A YOLOE+NOVIC confidence-fusion variant was tried and reverted — YOLOE-pf's
own class guess turned out too poorly calibrated to be useful even just as
corroboration evidence, e.g. its single highest-confidence prediction across
a full test set, 0.75, was "American shorthair" for a pillow.)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    bbox: np.ndarray       # [x1, y1, x2, y2] in pixel coords
    mask: np.ndarray       # HxW bool
    conf: float            # objectness confidence, no class semantics attached


class YOLOESegDetector:
    def __init__(
        self,
        weights_path: str,
        conf_thresh: float = 0.25,
        iou_thresh: float = 0.5,
        img_size: int = 640,
        device: str = "cuda",
        fix_channel_order: bool = True,
    ):
        from ultralytics import YOLOE

        self.model = YOLOE(weights_path)
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.img_size = img_size
        self.device = device
        # Ultralytics reads a numpy image as BGR; the pipeline's frames are RGB. Until 2026-10-14 the RGB array
        # was passed as is, so YOLOE saw red and blue swapped (detection.fix_channel_order: false reproduces it).
        self.fix_channel_order = bool(fix_channel_order)
        logger.info(f"Loaded YOLOE prompt-free detector from '{weights_path}'.")

    def detect(self, image_rgb: np.ndarray) -> list[Detection]:
        """image_rgb: HxWx3 uint8 RGB image. No prompt/classes are ever set."""
        results = self.model.predict(
            source=np.ascontiguousarray(image_rgb[..., ::-1]) if self.fix_channel_order else image_rgb,
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            imgsz=self.img_size,
            device=self.device,
            verbose=False,
        )
        result = results[0]

        detections: list[Detection] = []
        if result.masks is None or result.boxes is None:
            return detections

        h, w = image_rgb.shape[:2]
        masks_data = result.masks.data.cpu().numpy()  # (N, mh, mw)
        boxes_xyxy = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()

        for i in range(len(boxes_xyxy)):
            mask = masks_data[i]
            if mask.shape != (h, w):
                import cv2

                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            detections.append(
                Detection(
                    bbox=boxes_xyxy[i],
                    mask=mask.astype(bool),
                    conf=float(confs[i]),
                )
            )
        return detections
