"""YOLOE-seg-pf detection with BoT-SORT track identity across frames.

Why this exists: the static pipeline embeds every mask in every frame and lets
the fusion stage recover object identity purely from cosine similarity between
those embeddings. Within one continuous video that is redundant work -- the
tracker already tells us that the box in frame k is the same physical object it
was in frame k-1, from motion and appearance, for free.

So identity comes from the tracker, and the image encoder runs only when a
track id is seen for the first time. The embedding is cached against that id
and reused for every later frame of the same track. Cosine similarity is then
needed only to decide whether a *new* track is a re-appearance of an object
already in the map (the tracker loses ids through occlusion and view changes),
which is a per-new-track cost rather than a per-detection-per-frame one.

Ultralytics ships the BoT-SORT config (botsort.yaml); `persist=True` keeps the
tracker state alive across successive model.track() calls, which is what makes
ids stable over a sequence rather than being reset each call.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TrackedDetection:
    bbox: np.ndarray          # [x1, y1, x2, y2] pixel coords
    mask: np.ndarray          # HxW bool
    conf: float               # objectness confidence, no class semantics
    track_id: int             # BoT-SORT id, -1 when the tracker gave none
    is_new_track: bool        # first frame this id has ever been seen
    embedding: np.ndarray | None = None   # filled by the embedding cache
    label: str = ""           # YOLOE's class name (prompt-free vocabulary or the text prompts); unused by fusion


@dataclass
class TrackStats:
    frames: int = 0
    detections: int = 0
    new_tracks: int = 0
    embeddings_computed: int = 0
    untracked: int = 0
    dropped_by_mask_box: int = 0      # detection.mask_box_filter
    dropped_by_validation: dict = field(default_factory=dict)   # detection.validation: rule -> count
    boxes_tightened: int = 0          # detection.tighten_box_to_mask
    track_lengths: dict = field(default_factory=dict)

    @property
    def embeddings_saved(self) -> int:
        """How many image-encoder calls the tracker avoided."""
        return self.detections - self.embeddings_computed

    def as_dict(self) -> dict:
        lens = list(self.track_lengths.values())
        return {
            "frames": self.frames,
            "detections": self.detections,
            "unique_tracks": len(self.track_lengths),
            "new_tracks": self.new_tracks,
            "untracked_detections": self.untracked,
            "dropped_by_mask_box": self.dropped_by_mask_box,
            "dropped_by_validation": dict(self.dropped_by_validation),
            "boxes_tightened": self.boxes_tightened,
            "embeddings_computed": self.embeddings_computed,
            "embeddings_saved": self.embeddings_saved,
            "embedding_call_reduction": (
                round(1.0 - self.embeddings_computed / self.detections, 4)
                if self.detections else 0.0),
            "mean_track_length": round(float(np.mean(lens)), 2) if lens else 0.0,
            "max_track_length": int(max(lens)) if lens else 0,
        }


class TrackedYOLOEDetector:
    """YOLOE-seg-pf + BoT-SORT. Returns geometry and a persistent track id."""

    def __init__(
        self,
        weights_path: str,
        conf_thresh: float = 0.25,
        iou_thresh: float = 0.5,
        img_size: int = 640,
        device: str = "cuda",
        tracker: str = "botsort.yaml",
        mask_box_filter: dict | None = None,
        tighten_box_to_mask: bool = False,
        validation: dict | None = None,
        fix_channel_order: bool = True,
    ):
        from ultralytics import YOLOE

        self.model = YOLOE(weights_path)
        # Mask-box alignment, applied to each frame's detections BEFORE BoT-SORT (and so before CLIP):
        #  - mask_box_filter {enabled, min_box_frac, min_fill, min_contained}: drop a group / area detection:
        #    a box covering more than min_box_frac of the image that also (if set) has a mask filling less than
        #    min_fill of it and (if set) contains at least min_contained smaller detections. Fill alone (0.5,
        #    0.2) cost room_0 its sofa (cushion holes lower a sofa's fill); "box > 30% and contains >= 3
        #    detections" spares furniture (results/mask_box_fill/summary.md, results/replica_align/).
        #  - tighten_box_to_mask: shrink each box to its mask's bounding rectangle (never grow it), so the
        #    tracker and the NOVIC crop see the object rather than the box's slack.
        # Registered here so it runs ahead of the tracker callback Ultralytics adds on the first track() call.
        self.mask_box_filter = mask_box_filter if (mask_box_filter or {}).get("enabled") else None
        self.tighten_box_to_mask = bool(tighten_box_to_mask)
        # detection.validation (perception/detection/detection_validation.py): border_stuff, label_stuff,
        # label_vocab and group_test reject a detection here, before BoT-SORT. label_vocab drops a detection whose
        # own prompt-free YOLOE tag is not a physical object (configs/vocab/yoloe_pf_nonobject.txt: wall, room,
        # office, abstract tags, ...); the tag is used only for this decision and is not passed on as a name.
        v = validation or {}
        self.validation = {k: c for k, c in v.items() if isinstance(c, dict) and c.get("enabled")
                           and k in ("border_stuff", "label_stuff", "label_vocab", "group_test")}
        if self.mask_box_filter or self.tighten_box_to_mask or self.validation:
            self.model.add_callback("on_predict_postprocess_end", self._align_masks_and_boxes)
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.img_size = img_size
        self.device = device
        # Ultralytics reads a numpy image as BGR; the pipeline's frames are RGB. Until 2026-10-14 the RGB array
        # was passed as is, so YOLOE saw red and blue swapped (detection.fix_channel_order: false reproduces it).
        self.fix_channel_order = bool(fix_channel_order)
        self.tracker = tracker
        self._seen_track_ids: set[int] = set()
        self.stats = TrackStats()
        logger.info("Loaded YOLOE '%s' with tracker '%s'.", weights_path, tracker)

    def _align_masks_and_boxes(self, predictor) -> None:
        import cv2
        import torch

        for i, result in enumerate(predictor.results):
            if result.masks is None or result.boxes is None or len(result.boxes) == 0:
                continue
            h, w = result.orig_shape
            masks = result.masks.data.cpu().numpy()
            data = result.boxes.data.clone()
            boxes = data[:, :4].cpu().numpy()
            keep = []
            mbf = self.mask_box_filter or {}
            full = []                                   # full-frame boolean masks, for the validation tests
            for j in range(len(boxes)):
                m = masks[j]
                if m.shape != (h, w):
                    m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                full.append(m > 0.5)
            reject = {}
            if self.validation:
                from perception.detection import detection_validation as dv
                vb, vl, vg = (self.validation.get(k) for k in ("border_stuff", "label_stuff", "group_test"))
                vv = self.validation.get("label_vocab")
                nonobj = dv.nonobject_labels(vv.get("nonobject_file", "configs/vocab/yoloe_pf_nonobject.txt")) if vv else ()
                names = result.names
                cls = result.boxes.cls.cpu().numpy().astype(int)
                if vg:
                    flags, _ = dv.group_flags(full, vg.get("min_children", 2), vg.get("child_coverage", 0.6),
                                              vg.get("fragment_min_frac", 0.15), vg.get("containment", 0.85))
                for j in range(len(boxes)):
                    if vv and str(names.get(int(cls[j]), "")).lower().strip() in nonobj:
                        reject[j] = "label_vocab"
                    elif vl and dv.is_stuff_label(str(names.get(int(cls[j]), ""))):
                        reject[j] = "label_stuff"
                    elif vb and dv.border_stuff(full[j], vb.get("min_borders", 2), vb.get("min_area_frac", 0.15)):
                        reject[j] = "border_stuff"
                    elif vg and flags[j]:
                        reject[j] = "group_test"
            areas = np.clip(boxes[:, 2] - boxes[:, 0], 0, None) * np.clip(boxes[:, 3] - boxes[:, 1], 0, None)
            for j in range(len(boxes)):
                if j in reject:
                    self.stats.dropped_by_validation[reject[j]] = self.stats.dropped_by_validation.get(reject[j], 0) + 1
                    continue
                m = full[j]
                x1, y1 = max(int(np.floor(boxes[j, 0])), 0), max(int(np.floor(boxes[j, 1])), 0)
                x2, y2 = min(int(np.ceil(boxes[j, 2])), w), min(int(np.ceil(boxes[j, 3])), h)
                area = max((x2 - x1) * (y2 - y1), 1)
                inside = m[y1:y2, x1:x2]
                if self.mask_box_filter:
                    # drop when every active test says "group / area": box larger than min_box_frac of the
                    # image, AND (if set) mask filling less than min_fill of the box, AND (if set) at least
                    # min_contained smaller detections of this frame lying (>= 85%) inside the box
                    is_group = area / (h * w) > mbf.get("min_box_frac", 0.2)
                    if is_group and mbf.get("min_fill") is not None:
                        is_group = float(inside.sum()) / area < mbf["min_fill"]
                    if is_group and mbf.get("min_contained"):
                        bx = boxes[j]
                        iw = np.clip(np.minimum(bx[2], boxes[:, 2]) - np.maximum(bx[0], boxes[:, 0]), 0, None)
                        ih = np.clip(np.minimum(bx[3], boxes[:, 3]) - np.maximum(bx[1], boxes[:, 1]), 0, None)
                        contained = (iw * ih >= 0.85 * areas) & (areas < 0.8 * areas[j]) & (areas > 0)
                        contained[j] = False
                        is_group = int(contained.sum()) >= mbf["min_contained"]
                    if is_group:
                        self.stats.dropped_by_mask_box += 1
                        continue
                if self.tighten_box_to_mask and inside.any():
                    ys, xs = np.nonzero(inside)
                    tight = [x1 + xs.min(), y1 + ys.min(), x1 + xs.max() + 1, y1 + ys.max() + 1]
                    new = [max(boxes[j, 0], tight[0]), max(boxes[j, 1], tight[1]),
                           min(boxes[j, 2], tight[2]), min(boxes[j, 3], tight[3])]
                    if not np.allclose(new, boxes[j]):
                        data[j, :4] = torch.as_tensor(new, dtype=data.dtype)
                        self.stats.boxes_tightened += 1
                keep.append(j)
            if len(keep) < len(boxes) or self.tighten_box_to_mask:
                if not keep:
                    predictor.results[i] = result[torch.zeros(0, dtype=torch.long)]
                    continue
                idx = torch.as_tensor(keep, dtype=torch.long)
                new_result = result[idx]
                new_result.update(boxes=data[idx.to(data.device)])
                predictor.results[i] = new_result

    def reset(self) -> None:
        """Drop tracker state and id history -- call between sequences."""
        self._seen_track_ids.clear()
        self.stats = TrackStats()
        if hasattr(self.model, "predictor") and getattr(self.model.predictor, "trackers", None):
            for t in self.model.predictor.trackers:
                t.reset()

    def track(self, image_rgb: np.ndarray) -> list[TrackedDetection]:
        results = self.model.track(
            source=np.ascontiguousarray(image_rgb[..., ::-1]) if self.fix_channel_order else image_rgb,
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            imgsz=self.img_size,
            device=self.device,
            tracker=self.tracker,
            persist=True,
            verbose=False,
        )
        result = results[0]
        self.stats.frames += 1

        detections: list[TrackedDetection] = []
        if result.masks is None or result.boxes is None or len(result.boxes) == 0:
            return detections

        h, w = image_rgb.shape[:2]
        masks_data = result.masks.data.cpu().numpy()
        boxes_xyxy = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        cls = result.boxes.cls.cpu().numpy().astype(int)
        names = result.names or {}
        if result.boxes.id is not None:
            ids = result.boxes.id.cpu().numpy().astype(int)
        else:
            # tracker produced no ids this frame (e.g. first frame, or every
            # detection below the tracker's own init threshold)
            ids = np.full(len(boxes_xyxy), -1, dtype=int)

        for i in range(len(boxes_xyxy)):
            mask = masks_data[i]
            if mask.shape != (h, w):
                import cv2
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

            tid = int(ids[i])
            if tid < 0:
                self.stats.untracked += 1
                is_new = True          # no identity, so treat as its own object
            else:
                is_new = tid not in self._seen_track_ids
                if is_new:
                    self._seen_track_ids.add(tid)
                    self.stats.new_tracks += 1
                self.stats.track_lengths[tid] = self.stats.track_lengths.get(tid, 0) + 1

            self.stats.detections += 1
            detections.append(TrackedDetection(
                bbox=boxes_xyxy[i],
                mask=mask.astype(bool),
                conf=float(confs[i]),
                track_id=tid,
                is_new_track=is_new,
                label=str(names.get(int(cls[i]), "")),
            ))
        return detections


class TrackEmbeddingCache:
    """Runs the image encoder once per track id, reuses it for later frames.

    embed_fn takes a list of crops and returns an (N, D) array -- the same
    contract as NOVICDecoder.embed_crops, so the existing encoder drops in.
    Untracked detections (id < 0) are always embedded, since without an id
    there is nothing to reuse.
    """

    def __init__(self, embed_fn, stats: TrackStats | None = None):
        self.embed_fn = embed_fn
        self.cache: dict[int, np.ndarray] = {}
        self.stats = stats

    def embed(self, detections: list[TrackedDetection], crops: list[np.ndarray]) -> np.ndarray | None:
        """Fills det.embedding for every detection; returns them stacked."""
        if not detections:
            return None

        todo = [i for i, d in enumerate(detections)
                if d.track_id < 0 or d.track_id not in self.cache]

        if todo:
            new_embs = self.embed_fn([crops[i] for i in todo])
            new_embs = np.asarray(new_embs)
            for slot, i in enumerate(todo):
                emb = new_embs[slot]
                detections[i].embedding = emb
                if detections[i].track_id >= 0:
                    self.cache[detections[i].track_id] = emb
            if self.stats is not None:
                self.stats.embeddings_computed += len(todo)

        for d in detections:
            if d.embedding is None:
                d.embedding = self.cache[d.track_id]

        return np.stack([d.embedding for d in detections])


def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity between two (N, D) / (M, D) embedding sets."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    a = a / np.clip(np.linalg.norm(a, axis=1, keepdims=True), 1e-8, None)
    b = b / np.clip(np.linalg.norm(b, axis=1, keepdims=True), 1e-8, None)
    return a @ b.T
