"""
Step 2: Localization Backbone — YOLO26 Tracker

Wraps the YOLO26 detector (from Ultralytics) so it always returns class-agnostic
boxes with persistent IDs. This isolates the choice of backbone behind one interface,
so Steps 3+ never need to know which detector you picked.

Uses YOLO26's native tracker (BoT-SORT/ByteTrack) in class-agnostic mode.
All 80 COCO classes are collapsed into a single foreground class — the actual
labeling is deferred to the NOVIC decoder (Step 5).
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)


class YoloBotsortTracker:
    """
    Class-agnostic object detector + tracker using Ultralytics YOLO26.

    Args:
        weights_path: Path to YOLO26 .pt weights file.
        conf_thresh: Minimum confidence for detections (default 0.35).
        iou_thresh: IoU threshold for NMS (default 0.5).
        device: PyTorch device string ('cuda', 'cpu', 'mps').
        tracker_type: Tracker config ('botsort.yaml' or 'bytetrack.yaml').
        img_size: Input image size for the model.
    """

    def __init__(
        self,
        weights_path: str,
        conf_thresh: float = 0.35,
        iou_thresh: float = 0.5,
        device: str = "cuda",
        tracker_type: str = "botsort.yaml",
        img_size: int = 640,
    ):
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "Ultralytics not installed. Run: pip install ultralytics"
            )

        self.model = YOLO(weights_path)
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.device = device
        self.tracker_type = tracker_type
        self.img_size = img_size
        self._warmup_done = False

        logger.info(
            f"YoloBotsortTracker initialized: weights={weights_path}, "
            f"conf={conf_thresh}, device={device}, tracker={tracker_type}"
        )

    def warmup(self, frame: np.ndarray):
        """Run a single inference pass to warm up the model and CUDA kernels."""
        if not self._warmup_done:
            logger.info("Warming up YOLO26 model...")
            self.model.track(
                frame,
                persist=True,
                conf=self.conf_thresh,
                iou=self.iou_thresh,
                classes=None,
                agnostic_nms=True,
                device=self.device,
                imgsz=self.img_size,
                tracker=self.tracker_type,
                verbose=False,
            )
            self._warmup_done = True
            logger.info("YOLO26 warmup complete.")

    def update(self, frame: np.ndarray) -> list[dict]:
        """
        Run detection + tracking on a single frame.

        Args:
            frame: BGR image as numpy array (H, W, 3).

        Returns:
            List of detection dicts, each containing:
                - track_id (int): Persistent tracker-assigned ID.
                - bbox (np.ndarray): [x1, y1, x2, y2] in pixel coords.
                - conf (float): Detection confidence score.
        """
        results = self.model.track(
            frame,
            persist=True,
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            classes=None,          # ignore the 80 COCO classes — treat as one foreground class
            agnostic_nms=True,
            device=self.device,
            imgsz=self.img_size,
            tracker=self.tracker_type,
            verbose=False,
        )[0]

        detections = []
        if results.boxes is not None and results.boxes.id is not None:
            # If classes=None was used, it still detects the class index it thinks it is.
            for box, tid, conf, cls in zip(
                results.boxes.xyxy.cpu().numpy(),
                results.boxes.id.cpu().numpy(),
                results.boxes.conf.cpu().numpy(),
                results.boxes.cls.cpu().numpy(),
            ):
                detections.append({
                    "track_id": int(tid),
                    "bbox": box.astype(np.float32),
                    "conf": float(conf),
                    "label": results.names.get(int(cls), "unknown")
                })

        return detections

    def reset(self):
        """Reset the tracker state (e.g., when switching video sources)."""
        self.model.predictor = None
        self._warmup_done = False
        logger.info("YoloBotsortTracker reset.")
