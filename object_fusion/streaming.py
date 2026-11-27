"""Streaming object fusion: detection, fusion and the scene graph run in
parallel, and the scene graph is rebuilt every time a track ends.

Threads, connected by queues:

  detection  YOLOE-seg-pf + BoT-SORT on each frame and the padded crops. Needs
             only the RGB frames, so it starts at once, while the voxel map
             (or, in a scene update, the relocalization) is still computed.
  embedding  one worker per GPU, each with its own CLIP (NOVIC image encoder):
             frames are embedded on whichever GPU is free, and handed to
             fusion in frame order.
  fusion     waits for the frame's voxels, lifts each detection into the voxel
             grid and feeds it to InstanceFusion.add_frame(). When a track has
             not been seen for track_end_frames frames it is committed: matched
             against every object in the map (saved objects and this run's),
             merged or added as new, and named by NOVIC.
  graph      (the calling thread) whenever a commit happened, takes a snapshot
             of the fused objects and hands it to `on_update`, which builds and
             writes the live scene graph. While it builds, fusion goes on.

Depth (DA3) and camera poses (DROID-SLAM) are computed before this starts:
DROID-SLAM optimizes the whole trajectory at once.
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_END = object()


def prepare_native_libraries() -> None:
    """Call on the main thread before any worker thread starts.

    DoVSG's DBSCAN denoising calls scikit-learn's kneighbors(). Its first
    call builds a threadpoolctl controller, which scans the loaded libraries
    with dl_iterate_phdr() and runs a Python callback while holding the
    dynamic loader's lock. If another thread holds the GIL at that moment and
    needs the loader lock (a first-time symbol lookup in alphashape/scipy, or
    OpenCV in BoT-SORT), the process deadlocks with every thread asleep --
    which is what a scene update hit. scikit-learn caches the controller, so
    building it once up front removes the scan from the threaded part.
    """
    try:
        from sklearn.utils.parallel import _get_threadpool_controller
        _get_threadpool_controller()
    except Exception:  # older scikit-learn: nothing cached, nothing to warm
        logger.debug("scikit-learn threadpool controller not warmed up", exc_info=True)


def detect_frame(cfg: dict, image_rgb: np.ndarray, detector, tracked: bool):
    """One frame: YOLOE (+ BoT-SORT) boxes, masks, confidences, track ids, and
    the padded crop of each detection. Detector and crops see the frame after
    the optional brightness correction (detection.enhance); the map does not."""
    from perception.image_enhance import enhance
    from semantic_decoding.novic.crop_utils import extract_crop

    sem_cfg = cfg["semantic_decoding"]
    image_rgb = enhance(image_rgb, cfg["detection"].get("enhance"))
    h, w = image_rgb.shape[:2]
    detections = detector.track(image_rgb) if tracked else detector.detect(image_rgb)
    if detections:
        boxes = np.stack([d.bbox for d in detections])
        masks = np.stack([d.mask for d in detections])
        confs = np.array([d.conf for d in detections])
        track_ids = np.array([d.track_id for d in detections]) if tracked else np.full(len(detections), -1)
    else:
        boxes, masks = np.zeros((0, 4)), np.zeros((0, h, w), dtype=bool)
        confs, track_ids = np.zeros((0,)), np.zeros((0,), dtype=int)
    # detection.validation.pixel_ownership / purity_weighting (perception/detection/detection_validation.py):
    # a mask lying inside a larger one is subtracted from it (for the voxels AND the crop, whose removed pixels
    # are filled so CLIP does not see the inner object); each embedding is weighted by how much of its box is
    # its own. Masks of the original detections set the weights; the owned masks go on to fusion.
    val = (cfg["detection"].get("validation") or {})
    own, pur = val.get("pixel_ownership") or {}, val.get("purity_weighting") or {}
    images = [image_rgb] * len(boxes)
    weights = np.ones(len(boxes))
    if len(boxes) and (own.get("enabled") or pur.get("enabled")):
        from perception.detection import detection_validation as dv
        mlist = [m.astype(bool) for m in masks]
        if pur.get("enabled"):
            weights = dv.purity_weights(boxes, mlist, pur.get("floor", 0.1))
        if own.get("enabled"):
            owned, removed = dv.pixel_ownership(mlist, own.get("containment", 0.8), own.get("min_owned_frac", 0.1))
            masks = np.stack(owned)
            images = [dv.fill_pixels(image_rgb, r, own.get("crop_fill", "inpaint")) if r.any() else image_rgb
                      for r in removed]
    crops = [extract_crop(images[i], boxes[i], masks[i], mode=sem_cfg["crop_mode"],
                          padding_px=sem_cfg["crop_padding_px"], padding_frac=sem_cfg.get("crop_padding_frac"))
             for i in range(len(boxes))]
    return boxes, masks, confs, track_ids, crops, weights


def weight_embeddings(emb: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Scale each detection's embedding by its weight: fusion's running average (count-weighted sum of unit
    vectors, then normalised) then counts a half-covered view about half as much as a clear one."""
    if emb is None or not len(emb) or np.allclose(weights, 1.0):
        return emb
    return emb * np.asarray(weights, dtype=emb.dtype)[:, None]


def detect_and_embed(cfg: dict, image_rgb: np.ndarray, detector, novic, tracked: bool):
    """detect_frame() plus the NOVIC embedding of each crop."""
    boxes, masks, confs, track_ids, crops, weights = detect_frame(cfg, image_rgb, detector, tracked)
    return boxes, masks, confs, track_ids, weight_embeddings(novic.embed_crops(crops), weights)


class StreamingFusion:
    """Runs detection -> fusion -> live scene graph in parallel over a sequence.

    frame_names    one name per frame, in order ("000000", ...)
    load_image     idx -> RGB uint8 image
    frame_voxels   idx -> (pixel_index_mapping, pixel_index_mask); may block
                   until that frame's voxels exist
    prepare        called once on the fusion thread before the first frame
                   (starts the fusion stream, once its voxel map exists)
    on_update      snapshot (list of object copies), info -> None; builds the
                   live scene graph. Runs on the calling thread, at most every
                   live_interval_s seconds.
    extra_embedders  callables, each returning another embedder (an object
                   with embed_crops(), e.g. NOVICDecoder on a second GPU). Each
                   is built on its own thread, then embeds frames alongside
                   `novic`; frames reach fusion in order whichever GPU did them.
    """

    def __init__(self, cfg: dict, fusion, detector, novic, frame_names: list[str], load_image, frame_voxels,
                 prepare=None, on_update=None, det_viz_dir: Path | None = None, obj_min_detections: int = 1,
                 keep_frame_records: bool = True, queue_size: int = 64, extra_embedders=(),
                 live_interval_s: float = 0.0):
        self.cfg, self.fusion, self.detector, self.novic = cfg, fusion, detector, novic
        self.frame_names, self.load_image, self.frame_voxels = frame_names, load_image, frame_voxels
        self.prepare, self.on_update = prepare, on_update
        self.det_viz_dir, self.obj_min_detections = det_viz_dir, obj_min_detections
        self.keep_frame_records = keep_frame_records
        self.tracked = callable(getattr(detector, "track", None))
        self.extra_embedders, self.live_interval_s = list(extra_embedders), live_interval_s
        self._to_embed = queue.Queue(maxsize=queue_size)
        self._embedded: dict[int, tuple] = {}
        self._embedded_cv = threading.Condition()
        self._n_frames = len(frame_names)
        self.frames_per_embedder: dict[str, int] = {}
        self._changed = threading.Event()
        self._done = threading.Event()
        self._errors: list[BaseException] = []
        self.frame_records: list[dict] = []
        self.commits: list[dict] = []
        self.live_updates = 0
        self.timings: dict[str, float] = {}

    # ── threads ──

    def _detection_loop(self):
        from perception.visualization import save_detection_debug_frame

        t0 = time.time()
        try:
            for idx, name in enumerate(self.frame_names):
                if self._errors:
                    break
                image = self.load_image(idx)
                boxes, masks, confs, track_ids, crops, weights = detect_frame(self.cfg, image, self.detector,
                                                                              self.tracked)
                if self.det_viz_dir is not None:
                    save_detection_debug_frame(self.det_viz_dir, idx, image, boxes, confs, list(masks))
                if self.keep_frame_records:
                    self.frame_records.append({"name": name, "image_rgb": image, "boxes": boxes, "masks": masks,
                                               "confs": confs})
                self._to_embed.put((idx, name, boxes, masks, confs, track_ids, crops, weights))
                if idx % 20 == 0 or idx == len(self.frame_names) - 1:
                    extra = f", {len(set(track_ids[track_ids >= 0]))} tracks" if self.tracked else ""
                    logger.info(f"[detection] frame {idx + 1}/{len(self.frame_names)}: {len(boxes)} detections{extra}")
        except BaseException as e:  # noqa: BLE001 -- re-raised on the calling thread
            self._errors.append(e)
        finally:
            self.timings["detection_s"] = round(time.time() - t0, 1)
            for _ in range(1 + len(self.extra_embedders)):
                self._to_embed.put(_END)

    def _embed_loop(self, make_embedder, label: str):
        """Embeds queued frames on one GPU; results are handed to fusion by index."""
        t0 = time.time()
        done = 0
        try:
            try:
                embedder = make_embedder()
            except Exception:
                if label == "gpu0":
                    raise
                logger.exception(f"Embedder {label} could not be loaded; continuing without it.")
                return
            while True:
                item = self._to_embed.get()
                if item is _END or self._errors:
                    break
                idx, name, boxes, masks, confs, track_ids, crops, weights = item
                emb = weight_embeddings(embedder.embed_crops(crops), weights)
                with self._embedded_cv:
                    self._embedded[idx] = (idx, name, boxes, masks, confs, track_ids, emb)
                    self._embedded_cv.notify_all()
                done += 1
        except BaseException as e:  # noqa: BLE001
            self._errors.append(e)
            with self._embedded_cv:
                self._embedded_cv.notify_all()
        finally:
            self.frames_per_embedder[label] = done
            self.timings[f"embedding_{label}_s"] = round(time.time() - t0, 1)

    def _next_frame(self, idx: int):
        """Frame idx once embedded; None when the sequence (or a worker) ended."""
        with self._embedded_cv:
            while idx not in self._embedded:
                if self._errors or idx >= self._n_frames:
                    return None
                self._embedded_cv.wait(timeout=1.0)
            return self._embedded.pop(idx)

    def _fusion_loop(self):
        t0 = time.time()
        waited = 0.0
        try:
            if self.prepare is not None:
                w = time.time()
                self.prepare()
                waited += time.time() - w
            last = -1
            for next_idx in range(self._n_frames):
                item = self._next_frame(next_idx)
                if item is None or self._errors:
                    break
                idx, name, boxes, masks, confs, track_ids, emb = item
                w = time.time()
                pixel_indexes, pixel_mask = self.frame_voxels(idx)
                waited += time.time() - w
                detections = self.fusion.build_detection_list_from_frame(
                    boxes=boxes, masks=masks, confs=confs, embeddings=emb, pixel_indexes=pixel_indexes,
                    pixel_indexes_mask=pixel_mask, image_name=name, track_ids=track_ids)
                events = self.fusion.add_frame(idx, detections)
                last = idx
                if events:
                    self.commits += events
                    self._changed.set()
            if not self._errors:
                events = self.fusion.finish(last + 1)
                self.commits += events
        except BaseException as e:  # noqa: BLE001
            self._errors.append(e)
        finally:
            self.timings["fusion_s"] = round(time.time() - t0, 1)
            self.timings["fusion_waiting_for_voxels_s"] = round(waited, 1)
            self._done.set()
            self._changed.set()

    # ── driver ──

    def run(self) -> list[dict]:
        """Blocks until every frame is fused; returns the commit log."""
        t0 = time.time()
        prepare_native_libraries()
        threads = [threading.Thread(target=self._detection_loop, name="detection", daemon=True),
                   threading.Thread(target=self._embed_loop, args=(lambda: self.novic, "gpu0"), name="embed-0",
                                    daemon=True),
                   threading.Thread(target=self._fusion_loop, name="fusion", daemon=True)]
        threads += [threading.Thread(target=self._embed_loop, args=(make, f"gpu{k + 1}"), name=f"embed-{k + 1}",
                                     daemon=True) for k, make in enumerate(self.extra_embedders)]
        for t in threads:
            t.start()
        graph_time = 0.0
        last_update = 0.0
        while True:
            self._changed.wait()
            if not self._done.is_set():  # at most one live rebuild per live_interval_s
                wait = self.live_interval_s - (time.time() - last_update)
                if wait > 0:
                    self._done.wait(wait)
            self._changed.clear()
            if self._errors or self._done.is_set():
                break
            if self.on_update is None:
                continue
            snapshot = self.fusion.snapshot(self.obj_min_detections)
            if not snapshot:
                continue
            g = time.time()
            try:
                self.on_update(snapshot, {"update": self.live_updates + 1, "commits": len(self.commits),
                                          "frames_fused": 1 + max((c["committed_at"] for c in self.commits),
                                                                  default=0)})
                self.live_updates += 1
            except Exception:  # the live graph is a preview; the final graph is built after fusion
                logger.exception("Live scene-graph update failed; fusion continues.")
            graph_time += time.time() - g
            last_update = time.time()
        for t in threads:
            t.join()
        if self._errors:
            raise self._errors[0]
        self.timings["live_scene_graph_s"] = round(graph_time, 1)
        self.timings["total_s"] = round(time.time() - t0, 1)
        logger.info(f"Streaming fusion: {len(self.commits)} track commits, {self.live_updates} live scene-graph "
                    f"updates; frames embedded per GPU {self.frames_per_embedder}; {self.timings}")
        return self.commits


class quiet_loggers:
    """Silences INFO logging of the scene-graph building modules while a live
    update runs, so the log shows track commits and one line per update."""

    NAMES = ("object_fusion.postprocess", "object_fusion.instance_tagger", "scene_graph.builder.dovsg_relations",
             "scene_graph.builder.graph_builder")

    def __enter__(self):
        self._levels = {n: logging.getLogger(n).level for n in self.NAMES}
        for n in self.NAMES:
            logging.getLogger(n).setLevel(logging.WARNING)

    def __exit__(self, *exc):
        for n, level in self._levels.items():
            logging.getLogger(n).setLevel(level)


def tree_lines(graph, marks: dict | None = None) -> str:
    marks = marks or {}
    lines = []

    def walk(node, depth):
        rel = f" ({node.parent_relation})" if node.parent_relation else ""
        lines.append("  " * depth + node.node_id + rel + (f"   <-- {marks[node.node_id]}" if node.node_id in marks
                                                           else ""))
        for child in node.children.values():
            walk(child, depth + 1)

    walk(graph.root, 0)
    return "\n".join(lines)


def write_live_graph(path: Path, graph, objects, view_dataset, info: dict) -> None:
    """The live scene graph as JSON (same tree as the final .json, plus the
    update counter), written atomically so a reader never sees half a file."""
    import json

    from scene_graph.storage.serializer import compute_node_centroids, node_naming_info, scene_graph_to_tree

    embeddings = {o["class_id"]: True for o in objects}
    tree = scene_graph_to_tree(graph, embeddings, compute_node_centroids(objects, view_dataset),
                               node_naming_info(objects))
    tmp = Path(path).with_suffix(".tmp")
    tmp.write_text(json.dumps({**info, "time": time.strftime("%H:%M:%S"), "scene_graph": tree}, indent=2))
    tmp.replace(path)
