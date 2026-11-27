"""Panoptic segmentation as the pipeline's mask source, with BoT-SORT track ids (drop-in for TrackedYOLOEDetector).

Mask2Former (Swin-L, ADE20K panoptic) assigns every pixel to exactly one segment, so detections never contain one
another (no group boxes, no table-plus-laptop overlap), and wall / floor / ceiling arrive labelled as stuff and are
dropped here. The "thing" segments go through Ultralytics' BoT-SORT exactly as YOLOE's detections do (same tracker
config, same contract: when the tracker returns tracks only the tracked detections are kept; when it returns none,
the frame's detections pass on without ids).

Optional union with YOLOE (detection.panoptic.yoloe_union): panoptic models miss small objects (wall plugs,
switches, vents, small items on desks). YOLOE prompt-free runs on the same frame and its SMALL masks
(<= max_area_frac of the image) that do not lie on a panoptic thing segment (<= max_thing_overlap of the mask) are
added. Only these YOLOE-sourced masks go through detection.validation (label_stuff / border_stuff / group_test);
panoptic segments cannot nest, so they need none of it.

Benchmarked frame by frame in scripts/eval_frontend_variants.py (results/frontend_tests).
"""
from __future__ import annotations

import logging

import numpy as np

from perception.detection.tracked_yoloe_detector import TrackedDetection, TrackStats

logger = logging.getLogger(__name__)

ADE_STUFF = {"wall", "floor", "ceiling", "building", "sky", "tree", "road", "grass", "sidewalk", "earth", "mountain",
             "water", "sea", "field", "sand", "river", "hill", "land", "skyscraper", "path", "runway", "dirt track",
             "lake", "waterfall", "house", "hovel", "tower", "bridge", "pier"}


class TrackedPanopticDetector:
    def __init__(self, model: str = "facebook/mask2former-swin-large-ade-panoptic", threshold: float = 0.5,
                 stuff_labels: list[str] | None = None, tracker: str = "botsort.yaml", device: str = "cuda",
                 yoloe_union: dict | None = None, validation: dict | None = None, fix_channel_order: bool = True):
        import torch
        from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

        self.torch = torch
        self.device = device
        self.proc = AutoImageProcessor.from_pretrained(model)
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(model).to(device).eval()
        self.names = {int(k): v.strip() for k, v in self.model.config.id2label.items()}
        stuff = set(stuff_labels) if stuff_labels else ADE_STUFF
        self.stuff_ids = {k for k, v in self.names.items() if v in stuff}
        self.threshold = threshold
        self.tracker_cfg = tracker
        self.tracker = None
        self.stats = TrackStats()
        self._seen: set[int] = set()
        u = yoloe_union or {}
        self.union = u if u.get("enabled") else None
        self.yoloe = None
        if self.union:
            from ultralytics import YOLOE
            self.yoloe = YOLOE(u["weights"])
            self.fix_channel_order = fix_channel_order
        v = validation or {}
        self.validation = {k: c for k, c in v.items() if isinstance(c, dict) and c.get("enabled")
                           and k in ("border_stuff", "label_stuff", "group_test")}
        logger.info("Loaded panoptic detector '%s' (threshold %.2f, %d stuff classes dropped%s) with tracker '%s'.",
                    model, threshold, len(self.stuff_ids),
                    f", + YOLOE small objects from {u['weights']}" if self.union else "", tracker)

    # ── tracker ──
    def _make_tracker(self):
        from ultralytics.trackers.track import TRACKER_MAP
        from ultralytics.utils import YAML, IterableSimpleNamespace
        from ultralytics.utils.checks import check_yaml
        cfg = IterableSimpleNamespace(**YAML.load(check_yaml(self.tracker_cfg)))
        cfg.device = self.device
        return TRACKER_MAP[cfg.tracker_type](args=cfg)

    def reset(self) -> None:
        self._seen.clear()
        self.stats = TrackStats()
        self.tracker = None

    # ── masks ──
    def _panoptic(self, image_rgb):
        torch = self.torch
        h, w = image_rgb.shape[:2]
        inp = self.proc(images=image_rgb, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model(**inp)
        pan = self.proc.post_process_panoptic_segmentation(out, threshold=self.threshold, target_sizes=[(h, w)],
                                                           label_ids_to_fuse=self.stuff_ids)[0]
        seg = pan["segmentation"].cpu().numpy()
        masks, confs, labels = [], [], []
        for s in pan["segments_info"]:
            if s["label_id"] in self.stuff_ids:
                continue
            m = seg == s["id"]
            if m.sum() < 30:
                continue
            masks.append(m)
            confs.append(float(s["score"]))
            labels.append(self.names[s["label_id"]])
        return masks, confs, labels

    def _yoloe_small(self, image_rgb, things):
        import cv2
        from perception.detection import detection_validation as dv
        u = self.union
        h, w = image_rgb.shape[:2]
        img = np.ascontiguousarray(image_rgb[..., ::-1]) if self.fix_channel_order else image_rgb
        r = self.yoloe.predict(img, conf=u.get("conf_thresh", 0.10), iou=u.get("iou_thresh", 0.5),
                               imgsz=u.get("img_size", 1024), verbose=False, retina_masks=True)[0]
        if r.masks is None or len(r.boxes) == 0:
            return [], [], []
        ms = r.masks.data.cpu().numpy().astype(bool)
        if ms.shape[1:] != (h, w):
            ms = np.stack([cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) for m in ms]) > 0
        conf = r.boxes.conf.cpu().numpy()
        names = [str(r.names[int(c)]) for c in r.boxes.cls.cpu().numpy()]
        # validation on the YOLOE-sourced masks only (all of YOLOE's masks, so group_test sees the children)
        reject = np.zeros(len(ms), bool)
        vb, vl, vg = (self.validation.get(k) for k in ("border_stuff", "label_stuff", "group_test"))
        for j in range(len(ms)):
            if vl and dv.is_stuff_label(names[j]):
                reject[j] = True
            elif vb and dv.border_stuff(ms[j], vb.get("min_borders", 2), vb.get("min_area_frac", 0.15)):
                reject[j] = True
        if vg and len(ms):
            flags, _ = dv.group_flags(list(ms), vg.get("min_children", 2), vg.get("child_coverage", 0.6),
                                      vg.get("fragment_min_frac", 0.15), vg.get("containment", 0.85))
            reject |= flags
        dropped = self.stats.dropped_by_validation
        dropped["yoloe_union_rejected"] = dropped.get("yoloe_union_rejected", 0) + int(reject.sum())
        thing_union = np.zeros((h, w), bool)
        for m in things:
            thing_union |= m
        kept, out_m, out_c, out_l = np.zeros((h, w), bool), [], [], []
        for j in np.argsort(-conf):
            m = ms[j]
            a = m.sum()
            if reject[j] or a == 0 or a > u.get("max_area_frac", 0.02) * h * w:
                continue
            if (m & thing_union).sum() > u.get("max_thing_overlap", 0.5) * a:
                continue
            if (m & kept).sum() > 0.5 * a:                     # duplicate of a small mask already added
                continue
            m = m & ~thing_union & ~kept                       # keep the masks disjoint, like the panoptic ones
            if m.sum() < 0.5 * a:
                continue
            kept |= m
            out_m.append(m)
            out_c.append(float(conf[j]))
            out_l.append(names[j])
        return out_m, out_c, out_l

    # ── one frame ──
    def track(self, image_rgb: np.ndarray) -> list[TrackedDetection]:
        import torch
        from ultralytics.engine.results import Boxes

        self.stats.frames += 1
        h, w = image_rgb.shape[:2]
        masks, confs, labels = self._panoptic(image_rgb)
        src = ["panoptic"] * len(masks)
        if self.union:
            m2, c2, l2 = self._yoloe_small(image_rgb, masks)
            masks, confs, labels, src = masks + m2, confs + c2, labels + l2, src + ["yoloe"] * len(m2)
            self.stats.dropped_by_validation["yoloe_union_added"] = (
                self.stats.dropped_by_validation.get("yoloe_union_added", 0) + len(m2))
        if not masks:
            return []
        boxes = []
        for m in masks:
            ys, xs = np.nonzero(m)
            boxes.append([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1])
        boxes = np.asarray(boxes, np.float32)
        data = np.concatenate([boxes, np.asarray(confs, np.float32)[:, None], np.zeros((len(boxes), 1), np.float32)], 1)
        if self.tracker is None:
            self.tracker = self._make_tracker()
        det = Boxes(torch.as_tensor(data), (h, w)).cpu().numpy()
        tracks = self.tracker.update(det, np.ascontiguousarray(image_rgb[..., ::-1]))
        if len(tracks):
            idx, ids = tracks[:, -1].astype(int), tracks[:, 4].astype(int)
        else:
            idx, ids = np.arange(len(masks)), np.full(len(masks), -1)
        out = []
        for i, tid in zip(idx, ids):
            if tid < 0:
                self.stats.untracked += 1
                is_new = True
            else:
                is_new = tid not in self._seen
                if is_new:
                    self._seen.add(tid)
                    self.stats.new_tracks += 1
                self.stats.track_lengths[tid] = self.stats.track_lengths.get(tid, 0) + 1
            self.stats.detections += 1
            out.append(TrackedDetection(bbox=boxes[i], mask=masks[i], conf=float(confs[i]), track_id=int(tid),
                                        is_new_track=is_new, label=f"{labels[i]}" + ("" if src[i] == "panoptic" else "*")))
        return out


def from_config(det_cfg: dict) -> TrackedPanopticDetector:
    """detection.backend: panoptic -> the detector built from detection.panoptic (+ detection.validation for the
    YOLOE union masks)."""
    p = dict(det_cfg.get("panoptic") or {})
    u = dict(p.pop("yoloe_union", None) or {})
    if u.get("enabled"):
        u.setdefault("weights", det_cfg["weights"])
        u.setdefault("conf_thresh", det_cfg["conf_thresh"])
        u.setdefault("img_size", det_cfg["img_size"])
    return TrackedPanopticDetector(tracker=det_cfg.get("tracker", "botsort.yaml"), device=det_cfg["device"],
                                   yoloe_union=u, validation=det_cfg.get("validation"),
                                   fix_channel_order=det_cfg.get("fix_channel_order", True), **p)
