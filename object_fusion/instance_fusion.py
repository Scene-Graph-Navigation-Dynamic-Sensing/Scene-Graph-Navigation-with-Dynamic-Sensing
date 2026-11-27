"""Multi-view 3D instance fusion — clusters per-frame YOLOE-seg-pf detections
into persistent 3D object instances inside DoVSG's global voxel map.

Subclasses DoVSG's InstanceProcess (dovsg/memory/instances/instance_process.py)
and reuses almost all of it unchanged: compute_spatial_similarities,
compute_visual_similarities, aggregate_similarities, merge_detections_to_objects,
denoise_objects, merge_objects, filter_objects, indexes_align_objects,
change_objects, create_object, get_bounding_box, mask_subtract_contained,
get_object_id. Only two things are replaced:

  1. gsam2_obs_to_detection_list() -> build_detection_list_from_frame(), since
     that's the only method coupled to GroundingDINO+SAM2's pickle format and
     a fixed text class list. YOLOE-seg-pf detections carry no class at all
     (see perception/detection/yoloe_seg_detector.py), so every detection's
     class_name is a placeholder "object" during fusion — matching is driven
     entirely by spatial IoU + NOVIC-embedding cosine similarity
     (compute_text_similarities is kept wired in for tensor-shape
     compatibility but its weight is fixed to 0, so it has zero effect).

  2. get_instances() -> a version that reads pre-built per-frame detection
     lists instead of loading semantic_memory/*.pkl files from disk.

Every detection's NOVIC embedding (computed fresh on every frame) arrives as
its clip_ft. DoVSG's merge_obj_to_obj() keeps a detection-count-weighted
running average of it on each object; that average drives matching, and it is
what gets named. When a BoT-SORT track appears for the last time, the running
average of the object it fed is decoded by NOVIC and the result is appended to
that object's 'track_names'. instance_tagger.py turns the latest still-valid
record into the final class_name.

Matching level (object_fusion.match_level):

  "track" (default)  Each BoT-SORT track is accumulated on its own: its
                     detections' voxels are merged and its embeddings kept as
                     a detection-count-weighted running average. Only when the
                     track ends (not seen for track_end_frames frames) is the
                     whole track compared with every object in the map --
                     saved objects of an earlier run and objects already
                     committed in this run alike -- by 0.3 geometric + 0.7
                     cosine similarity. A match merges into that object
                     (embeddings weighted by detection count); otherwise the
                     track becomes a new object. So a split track, or an
                     object that leaves the view and comes back under a new
                     id, is re-identified at its second track's end. See
                     start_stream() / add_frame() / finish().
  "detection"        DoVSG's own method: every detection is matched against
                     the map as it arrives; a continuing track follows the
                     object its first frame matched.
"""
from __future__ import annotations

import hashlib
import logging
import threading

import cv2
import numpy as np
import torch
from tqdm import trange

from object_fusion._dovsg_import import ensure_dovsg_importable

logger = logging.getLogger(__name__)


def build_instance_fusion_class(scene_graph_generation_root: str):
    """Factory: dovsg's InstanceProcess is only importable once
    scene_graph_generation_root is on sys.path, so the subclass is built
    lazily behind this function rather than at module import time.
    """
    ensure_dovsg_importable(scene_graph_generation_root)
    from dovsg.memory.instances.instance_process import InstanceProcess
    from dovsg.memory.instances.instance_utils import DetectionList, MapObjectList

    # Same CPU-safe fallback Scene-Graph-Generation/main.py applies: accurate
    # 3D IoU needs pytorch3d, which isn't installed here (confirmed missing
    # in the `scene_graph` conda env) — fall back to axis-aligned IoU instead
    # of hard-failing every similarity computation.
    def _patched_compute_3d_iou_accuracte_batch(self, bbox1, bbox2):
        try:
            bbox1_expanded = self.expand_3d_box(bbox1)
            bbox2_expanded = self.expand_3d_box(bbox2)
            import pytorch3d.ops as ops

            bbox1_expanded = bbox1_expanded[:, [0, 2, 5, 3, 1, 7, 4, 6]]
            bbox2_expanded = bbox2_expanded[:, [0, 2, 5, 3, 1, 7, 4, 6]]
            _, iou = ops.box3d_overlap(bbox1_expanded.float(), bbox2_expanded.float())
            return iou
        except (ImportError, ModuleNotFoundError):
            return self.compute_iou_batch(bbox1, bbox2)

    InstanceProcess.compute_3d_iou_accuracte_batch = _patched_compute_3d_iou_accuracte_batch

    class InstanceFusion(InstanceProcess):
        def __init__(
            self,
            downsample_voxel_size: float = 0.01,
            spatial_weight: float = 0.3,
            vis_weight: float = 0.7,
            sim_threshold: float = 0.55,
        ):
            # part_level_classes keeps DoVSG's default (handle/drawer/door/...) —
            # NOVIC can legitimately decode these as part-level nouns later in
            # instance_tagger.py, and indexes_align_objects() should still treat
            # them specially when that happens.
            super().__init__(downsample_voxel_size=downsample_voxel_size, part_level_classes=None)
            # No text similarity in this pipeline (no fixed class list) —
            # keep the key present (dummy text_ft) for tensor-shape
            # compatibility with the reused aggregate_similarities(), but
            # zero its contribution.
            self.spatial_weight = spatial_weight
            self.vis_weight = vis_weight
            self.text_weight = 0.0
            self.sim_threshold = sim_threshold
            assert abs(self.spatial_weight + self.vis_weight + self.text_weight - 1.0) < 1e-5
            self._overlap_index_cache: dict[bytes, object] = {}
            self._overlap_index_hits = 0
            self._overlap_index_misses = 0
            # track_id -> index into the running `objects` list, for the
            # continuing-track fast path in get_instances() below.
            self._track_to_object_index: dict[int, int] = {}
            self.track_name_log: list[dict] = []
            self.commit_log: list[dict] = []
            self._unnamed: list = []  # deferred track-end decodes, see name_ended_tracks(defer=True)
            self.objects = None
            self._lock = threading.RLock()
            # pixels peeled off each mask edge before it is lifted into the voxel map
            # (object_fusion.mask_erode_px); edge pixels often carry the depth of
            # whatever is behind the object, e.g. the wall around a screen
            self.mask_erode_px = 0
            self.min_merge_cosine = 0.0   # object_fusion.min_merge_cosine; 0 = off
            # object_fusion.rematch_min_containment: a re-match merge is for a fragment that
            # lies mostly inside its target (a laptop screen joining its laptop). None = same
            # rule as a track commit (min_merge_overlap), which on Replica office_0 let objects
            # of 42 and 73 detections be absorbed at containment 0.31-0.49.
            self.rematch_min_containment = None
            # object_fusion.track_split_cosine: a detection whose embedding is this far from
            # its BoT-SORT track's running embedding is not the same object (the tracker
            # handed the id over between sparse frames); the track is closed and a new one
            # starts under the same id. 0 = off (tracker trusted).
            self.track_split_cosine = 0.0
            self.small_merge_ratio = 0.0      # object_fusion.small_merge_ratio; 0 = off
            self.track_validation = {}        # object_fusion.track_validation (see _reject_track)
            self._stuff_probe = None
            self.small_merge_cosine = 0.9     # object_fusion.small_merge_cosine
            self.track_splits = 0

        def compute_overlap_matrix_2set(self, objects_map, objects_new, bbox_map, bbox_new) -> np.ndarray:
            """Share of each new object's voxels that lie within
            downsample_voxel_size of some voxel of each map object, for the
            pairs whose boxes overlap (upstream's IoU gate and ratio).

            Upstream answers "within" with a brute-force FAISS search of every
            point against every point (IndexFlatL2), so one pair costs
            |points_map| x |points_new| and the sequence cost grows with the
            square of the map: 336 s of a 240-frame lap_in run. All points sit
            on the voxel grid, so "within d" is the voxel itself or a grid
            neighbour less than d away: with d = one voxel, the 6 face
            neighbours. That is looked up by voxel index in a sorted array
            instead. The float search counted a face neighbour 97-99% of the
            time (its squared distance rounds to either side of d**2); here
            it is always counted.
            """
            m, n = len(objects_map), len(objects_new)
            overlap_matrix = np.zeros((m, n))
            if m == 0 or n == 0:
                return overlap_matrix

            sorted_map = []
            for obj in objects_map:
                indexes = np.asarray(obj["indexes"], dtype=np.int64)
                key = hashlib.blake2b(indexes.tobytes(), digest_size=16).digest()
                cached = self._overlap_index_cache.get(key)
                if cached is None:
                    cached = np.unique(indexes)
                    self._overlap_index_cache[key] = cached
                    self._overlap_index_misses += 1
                else:
                    self._overlap_index_hits += 1
                sorted_map.append(cached)

            try:
                iou = self.compute_3d_iou_accuracte_batch(bbox_map, bbox_new)
            except ValueError:
                iou = self.compute_iou_batch(bbox_map, bbox_new)

            for j in range(n):
                rows = [i for i in range(m) if iou[i, j] >= 1e-6]
                if not rows:
                    continue
                neighbours = self._grid_neighbourhood(np.asarray(objects_new[j]["indexes"], dtype=np.int64))
                for i in rows:
                    hit = np.zeros(neighbours.shape[1], dtype=bool)
                    for cand in neighbours:  # the voxel itself, then each neighbour
                        valid = cand >= 0
                        pos = np.searchsorted(sorted_map[i], cand[valid])
                        pos[pos == len(sorted_map[i])] = 0
                        hit[valid] |= sorted_map[i][pos] == cand[valid]
                    overlap_matrix[i, j] = hit.mean()

            # merges retire old voxel sets; keep the cache from growing forever
            if len(self._overlap_index_cache) > 4096:
                self._overlap_index_cache.clear()

            return overlap_matrix

        def _grid_neighbourhood(self, indexes: np.ndarray) -> np.ndarray:
            """(k, len(indexes)) voxel indexes of each voxel and of its grid
            neighbours closer than downsample_voxel_size (-1 outside the grid)."""
            vd = self.view_dataset
            ratio = self.downsample_voxel_size / vd.resolution
            r = int(np.ceil(ratio))
            offsets = [np.array(o) for o in np.ndindex(2 * r + 1, 2 * r + 1, 2 * r + 1)]
            offsets = [o - r for o in offsets if np.linalg.norm(o - r) <= ratio + 1e-6]
            offsets.sort(key=lambda o: np.abs(o).sum())  # the voxel itself first
            voxels = vd.index_to_voxel(indexes)
            limit = np.asarray(vd.voxel_num)
            out = np.empty((len(offsets), len(indexes)), dtype=np.int64)
            for k, o in enumerate(offsets):
                if not o.any():  # the voxel itself: its own index, even on the grid's upper edge
                    out[k] = indexes
                    continue
                v = voxels + o
                inside = np.all((v >= 0) & (v < limit), axis=1)
                out[k] = np.where(inside, vd.voxel_to_index(v), -1)
            return out

        def start_new_scan(self, view_dataset):
            """Resets per-scan accumulators. Call this once, before building
            any per-frame detection lists — build_detection_list_from_frame()
            appends to self.objects_original_indexes as it runs, so unlike
            upstream DoVSG (which does detection-list-building and fusion in
            one interleaved loop inside get_instances()), here the reset must
            happen before that loop starts, not inside get_instances() itself
            (which now only does the fusion/merge pass over already-built
            lists and must NOT clear what this accumulated).
            """
            self.objects_original_indexes = []
            self.class_id_counts = {}
            self.view_dataset = view_dataset

        def build_detection_list_from_frame(
            self,
            boxes: np.ndarray,          # (N, 4) xyxy
            masks: np.ndarray,          # (N, H, W) bool
            confs: np.ndarray,          # (N,)
            embeddings: np.ndarray,     # (N, F) NOVIC embeddings, same order
            pixel_indexes: np.ndarray,  # (H, W) int32 global voxel index per pixel, -1 = invalid
            pixel_indexes_mask: np.ndarray,  # (H, W) bool, valid-depth pixels
            image_name: str,
            track_ids: np.ndarray | None = None,  # (N,) BoT-SORT ids, -1 when untracked
        ):
            fg_detection_list = DetectionList()
            if len(boxes) == 0:
                return fg_detection_list

            h, w = pixel_indexes.shape[:2]
            image_area = h * w
            keep = []
            for i in range(len(boxes)):
                area = masks[i].sum()
                if area < max(self.mask_area_threshold, 10):
                    continue
                if area > self.max_bbox_area_ratio * image_area:
                    continue
                if confs[i] < self.mask_conf_threshold:
                    continue
                keep.append(i)
            if not keep:
                return fg_detection_list

            boxes = boxes[keep]
            masks = masks[keep]
            confs = confs[keep]
            embeddings = embeddings[keep]
            track_ids = np.asarray(track_ids)[keep] if track_ids is not None else np.full(len(keep), -1)

            masks = self.mask_subtract_contained(boxes, masks)

            for i in range(len(boxes)):
                det_mask = masks[i]
                lift_mask = det_mask
                if self.mask_erode_px > 0:
                    k = 2 * self.mask_erode_px + 1
                    eroded = cv2.erode(det_mask.astype(np.uint8), np.ones((k, k), np.uint8)).astype(bool)
                    if eroded.sum() >= max(self.mask_area_threshold, 10):   # thin objects keep their mask
                        lift_mask = eroded
                global_object_indexes = self.create_object(pixel_indexes, lift_mask, pixel_indexes_mask)
                self.objects_original_indexes.extend(global_object_indexes.tolist())
                if len(global_object_indexes) < max(self.min_points_threshold, 5):
                    continue

                global_object_indexes = self.process_indexes(global_object_indexes)
                pcd_bbox = self.get_bounding_box(global_object_indexes)
                pcd_bbox.color = [0, 1, 0]
                if pcd_bbox.volume() < 1e-6:
                    continue

                # Placeholder — the real open-vocabulary label is assigned
                # once per finalized instance in instance_tagger.py, using
                # NOVIC. Fusion-time matching never looks at this string.
                class_name = "object"
                class_id = self.get_object_id(class_name=class_name)
                clip_ft = torch.from_numpy(np.asarray(embeddings[i], dtype=np.float32))

                fg_detection_list.append({
                    "mask_idx": [i],
                    "image_name": [image_name],
                    "class_name": class_name,
                    "class_id": class_id,
                    "conf": float(confs[i]),
                    "num_detections": 1,
                    "mask": [det_mask],
                    "xyxy": [boxes[i]],
                    "inst_color": np.random.rand(3),
                    "indexes": global_object_indexes,
                    "bbox": pcd_bbox,
                    "clip_ft": clip_ft,
                    "text_ft": clip_ft,  # dummy; text_weight=0 so never actually used
                    # List-typed because merge_obj_to_obj() can only combine
                    # list and int fields (it concatenates lists and raises on
                    # anything else), so every merge carries these along.
                    "track_ids": [int(track_ids[i])],
                    "track_names": [],
                    # class_ids of memory objects this detection was fused
                    # with (scripts/update_scene_graph.py); lists concatenate
                    # on merge, so an empty list means a new object
                    "memory_ids": [],
                })

            return fg_detection_list

        @staticmethod
        def last_seen_frames(per_frame_detection_lists: list) -> dict[int, int]:
            """track id -> index of the last frame it appears in, i.e. the
            frame after which that track has ended."""
            last_seen = {}
            for idx, detections in enumerate(per_frame_detection_lists):
                for det in detections:
                    tid = det["track_ids"][0]
                    if tid >= 0:
                        last_seen[tid] = idx
            return last_seen

        def name_ended_tracks(self, objects, ended_track_ids, frame_idx: int, decode_fn, defer: bool = False,
                              events: list | None = None) -> None:
            """Decodes the running-average embedding of each object an ended
            track fed (one decode per object, batched), and appends the result
            to that object's 'track_names'. 'num_detections' is recorded so
            instance_tagger.py can tell whether the object changed afterwards.

            defer=True (streaming): the record is appended now with the
            embedding as it is at this track end, and decoded later together
            with every other deferred record (flush_track_names()). A NOVIC
            beam search per track end, inside the fusion lock, took 204 s of a
            240-frame run; one batch at the end gives the same names.
            `events` are commit-log entries that get the name when it is known.
            """
            if defer:
                for tid in ended_track_ids:
                    obj_idx = self._track_to_object_index[tid]
                    obj = objects[obj_idx]
                    record = {"track_id": int(tid), "frame": frame_idx, "label": None, "score": None,
                              "top_k": None, "num_detections": int(obj["num_detections"])}
                    log_entry = {**record, "object_index": obj_idx}
                    obj["track_names"].append(record)
                    self.track_name_log.append(log_entry)
                    embedding = np.asarray(obj["clip_ft"].detach().cpu() if torch.is_tensor(obj["clip_ft"])
                                           else obj["clip_ft"], dtype=np.float32).ravel().copy()
                    self._unnamed.append((embedding, [record, log_entry], list(events or [])))
                return

            tracks_by_object: dict[int, list[int]] = {}
            for tid in ended_track_ids:
                tracks_by_object.setdefault(self._track_to_object_index[tid], []).append(tid)

            obj_indexes = list(tracks_by_object)
            embeddings = np.stack([np.asarray(objects[i]["clip_ft"], dtype=np.float32).ravel()
                                   for i in obj_indexes])
            for obj_idx, result in zip(obj_indexes, decode_fn(embeddings)):
                obj = objects[obj_idx]
                for tid in tracks_by_object[obj_idx]:
                    record = {
                        "track_id": int(tid),
                        "frame": frame_idx,
                        "label": result.label,
                        "score": float(result.score),
                        "top_k": result.top_k,
                        "num_detections": int(obj["num_detections"]),
                    }
                    obj["track_names"].append(record)
                    self.track_name_log.append({**record, "object_index": obj_idx})

        def flush_track_names(self, batch_size: int = 64) -> None:
            """Decodes every deferred track-end record (see name_ended_tracks)."""
            with self._lock:
                if not self._unnamed or self._decode_fn is None:
                    return
                pending, self._unnamed = self._unnamed, []
            embeddings = np.stack([p[0] for p in pending])
            results = []
            for a in range(0, len(embeddings), batch_size):
                results += list(self._decode_fn(embeddings[a:a + batch_size]))
            with self._lock:
                for (_, records, events), result in zip(pending, results):
                    for record in records:
                        record.update(label=result.label, score=float(result.score), top_k=result.top_k)
                    for event in events:
                        event["name"] = result.label
            logger.info(f"Named {len(pending)} track ends with NOVIC in one batch.")

        def merge_obj_to_obj(self, obj1, obj2, run_dbscan=True):
            # 'novelty' is a float DoVSG's merge cannot combine; it is
            # recomputed for the merged object at its next commit
            obj1.pop("novelty", None)
            obj2.pop("novelty", None)
            return super().merge_obj_to_obj(obj1, obj2, run_dbscan=run_dbscan)

        def memory_seed(self, obj) -> dict:
            """A saved instance as a fusion object, for get_instances(initial_objects=).
            Only the fields merge_obj_to_obj() can combine are carried over."""
            indexes = self.process_indexes(np.asarray(obj["indexes"]), run_dbscan=False)
            clip_ft = torch.from_numpy(np.asarray(obj["clip_ft"], dtype=np.float32).ravel())
            bbox = self.get_bounding_box(indexes)
            bbox.color = [0, 1, 0]
            return {"mask_idx": [], "image_name": [], "class_name": obj["class_name"], "class_id": obj["class_id"],
                    "conf": float(obj.get("conf", 1.0)), "num_detections": int(obj.get("num_detections", 1)),
                    "mask": [], "xyxy": [], "inst_color": np.random.rand(3), "indexes": indexes, "bbox": bbox,
                    "clip_ft": clip_ft, "text_ft": clip_ft, "track_ids": [], "track_names": [],
                    "memory_ids": [obj["class_id"]]}

        # ─── track-level matching, streamed frame by frame ───

        def start_stream(self, view_dataset, initial_objects=None, decode_fn=None, track_end_frames: int = 10,
                         memory_sim_threshold: float | None = None, memory_points=None,
                         novelty_distance_m: float = 0.05, min_merge_overlap: float = 0.3):
            """Starts track-level fusion. initial_objects are saved instances of
            an earlier run (in this view_dataset's voxel grid); a track matches
            one of them at memory_sim_threshold (default: sim_threshold) and an
            object of this run at sim_threshold. decode_fn names an object each
            time a track is committed to it.

            A track only merges into an object when one of the two lies largely
            inside the other (containment >= min_merge_overlap, see
            _similarity()): in a static scene an object that reappears is where
            it was. Without it, cosine alone (0.7 x 0.83 = 0.58 >= 0.55) merged
            a chair into a table across the room, and objects standing on a
            table into the table.

            memory_points (the saved map's voxel centres) turns on the
            new-object check: each object of this run gets 'novelty', the share
            of its voxels farther than novelty_distance_m from anything in the
            saved map, updated every time a track is committed to it."""
            with self._lock:
                self.view_dataset = view_dataset
                self.objects = MapObjectList([self.memory_seed(o) for o in (initial_objects or [])])
                self.memory_sim_threshold = memory_sim_threshold or self.sim_threshold
                self.track_end_frames = track_end_frames
                self.min_merge_overlap = min_merge_overlap
                self._decode_fn = decode_fn
                self.novelty_distance_m = novelty_distance_m
                self._memory_tree = None
                if memory_points is not None:
                    from scipy.spatial import cKDTree
                    self._memory_tree = cKDTree(np.asarray(memory_points))
                self._pending: dict[int, dict] = {}
                self._first_seen: dict[int, int] = {}
                self._last_seen: dict[int, int] = {}
                self._untracked = 0
                self._frames_added = 0
                self._uids: dict[int, int] = {}  # id(object) -> stable number for logs
                self._track_to_object_index = {}
                self.track_name_log = []
                self.commit_log = []
                self._unnamed = []

        def add_frame(self, idx: int, detections) -> list[dict]:
            """Adds one frame's DetectionList; returns the commits it caused."""
            with self._lock:
                events = []
                for det in detections:
                    tid = det["track_ids"][0]
                    if tid < 0:  # no identity: a one-detection track
                        self._untracked += 1
                        key = -self._untracked
                        self._pending[key], self._first_seen[key], self._last_seen[key] = det, idx, idx
                        events.append(self._commit(key, idx))
                        continue
                    if tid in self._pending and self.track_split_cosine > 0:
                        e_new = np.asarray(det["clip_ft"], dtype=np.float32).ravel()
                        e_old = np.asarray(self._pending[tid]["clip_ft"], dtype=np.float32).ravel()
                        cos = float(e_new @ e_old / (np.linalg.norm(e_new) * np.linalg.norm(e_old) + 1e-12))
                        if cos < self.track_split_cosine:
                            self.track_splits += 1
                            logger.info(f"track {tid} split at frame {idx}: detection cos {cos:.3f} "
                                        f"< {self.track_split_cosine} to the track so far")
                            events.append(self._commit(tid, idx))
                    if tid in self._pending:
                        self._pending[tid] = self.merge_obj_to_obj(self._pending[tid], det, run_dbscan=False)
                    else:
                        self._pending[tid], self._first_seen[tid] = det, idx
                    self._last_seen[tid] = idx
                ended = [t for t in self._pending if idx - self._last_seen[t] >= self.track_end_frames]
                events += [self._commit(t, idx) for t in sorted(ended, key=self._last_seen.get)]
                self._frames_added += 1
                if self._frames_added % self.denoise_interval == 0:
                    # saved objects are already clean: DBSCAN over an 85k-voxel
                    # table top every 20 frames cost a scene update ~100 s
                    own = [k for k, o in enumerate(self.objects) if not o.get("memory_ids")]
                    for k, obj in zip(own, self.denoise_objects(MapObjectList([self.objects[k] for k in own]))):
                        self.objects[k] = obj
                return [e for e in events if e is not None]

        def finish(self, idx: int) -> list[dict]:
            """Commits the tracks still open when the video ends."""
            with self._lock:
                events = [self._commit(t, idx) for t in sorted(self._pending, key=self._last_seen.get)]
                events += self._rematch(idx)
            self.flush_track_names()
            return [e for e in events if e is not None]

        def _similarity(self, obj, candidates: list[int]):
            """DoVSG's score, 0.3 x geometric + 0.7 x cosine, where geometric is
            the share of obj's voxels within a voxel of the candidate. A merge
            also needs one of the two to lie largely inside the other
            (containment, the larger of the two directions, >=
            min_merge_overlap): an object standing on a table touches it only
            where it stands (bottle 0.19, laptop 0.06, watch 0.11 of their
            voxels), while a second view of the same object, or a fragment of
            it, is largely inside it (0.4-1.0) -- and a table crop, which shows
            what stands on it, has a cosine of 0.7-0.8 to those objects."""
            one = DetectionList([obj])
            others = MapObjectList([self.objects[j] for j in candidates])
            geometric = self.compute_spatial_similarities(one, others)[0].numpy()
            reverse = self.compute_spatial_similarities(others, MapObjectList([obj]))[:, 0].numpy()
            containment = np.maximum(geometric, reverse)
            cosine = self.compute_visual_similarities(one, others)[0].numpy()
            agg = self.spatial_weight * geometric + self.vis_weight * cosine
            threshold = np.array([self.memory_sim_threshold if self.objects[j].get("memory_ids")
                                  else self.sim_threshold for j in candidates])
            # object_fusion.min_merge_cosine: geometry alone cannot force a merge. Without it, an object
            # flush against a large neighbour (a plug on a wall, a tablet on a table) scores geo ~1.0 and
            # merges at cosine ~0.55; on Replica office_0, 8 of 16 re-match merges had cosine < 0.75.
            passing = np.where((agg >= threshold) & (containment >= self.min_merge_overlap)
                               & (cosine >= self.min_merge_cosine), agg, -np.inf)
            # object_fusion.small_merge_ratio / small_merge_cosine: an object much smaller than the
            # candidate (a plug or switch on a wall node, a tablet on a table) overlaps it fully and
            # its crops resemble the larger object's, which include it; it only merges if it looks
            # almost the same (a fragment of that object), not merely similar. On Replica office_0 such
            # merges left 11 of 14 missed objects inside wall and table nodes.
            if self.small_merge_ratio > 0:
                n_obj = len(obj["indexes"])
                ratio = np.array([n_obj / max(len(self.objects[j]["indexes"]), 1) for j in candidates])
                small = ratio < self.small_merge_ratio
                passing = np.where(small & (cosine < self.small_merge_cosine), -np.inf, passing)
            self._last_containment = containment
            return geometric, cosine, agg, threshold, passing

        def _rematch(self, idx: int) -> list[dict]:
            """Object fusion among this run's objects: an object committed
            before its neighbour grew (a laptop's screen track ending before
            the laptop's main track) is compared again, smallest first, with
            the same rule as a track, and merged if it now matches. Saved
            objects are never merged into each other."""
            events, changed = [], True
            while changed:
                changed = False
                own = [k for k, o in enumerate(self.objects) if not o.get("memory_ids")]
                for k in sorted(own, key=lambda k: len(self.objects[k]["indexes"])):
                    candidates = [j for j in range(len(self.objects)) if j != k]
                    if not candidates:
                        break
                    geometric, cosine, agg, threshold, passing = self._similarity(self.objects[k], candidates)
                    if self.rematch_min_containment is not None:
                        passing = np.where(self._last_containment >= self.rematch_min_containment, passing, -np.inf)
                    if not np.isfinite(passing.max()):
                        continue
                    b = int(np.argmax(passing))
                    target = candidates[b]
                    event = {"track_id": None, "rematch_of": self.object_label(k), "committed_at": idx,
                             "detections": int(self.objects[k]["num_detections"]),
                             "cosine": round(float(cosine[b]), 3), "geometric": round(float(geometric[b]), 3),
                             "containment": round(float(self._last_containment[b]), 3),
                             "aggregate": round(float(agg[b]), 3), "threshold": float(threshold[b]),
                             "decision": f"merged into {self.object_label(target)}"}
                    logger.info(f"re-match {event['rematch_of']} ({event['detections']} detections) -> "
                                f"{event['decision']} [cos {event['cosine']}, geo {event['geometric']}, "
                                f"contain {event['containment']}, "
                                f"agg {event['aggregate']} vs {event['threshold']}]")
                    self.objects[target] = self.merge_obj_to_obj(self.objects[target], self.objects[k],
                                                                 run_dbscan=False)
                    del self.objects[k]
                    self.commit_log.append(event)
                    events.append(event)
                    changed = True
                    break
            return events

        def _commit(self, key: int, idx: int) -> dict | None:
            track = self._pending.pop(key)
            event = {"track_id": int(key) if key >= 0 else -1, "first_frame": self._first_seen.pop(key),
                     "last_frame": self._last_seen.pop(key), "committed_at": idx,
                     "detections": int(track["num_detections"])}
            track["indexes"] = self.process_indexes(track["indexes"], run_dbscan=True)
            if len(track["indexes"]) < max(self.min_points_threshold, 5):
                return None
            track["bbox"] = self.get_bounding_box(track["indexes"])
            track["bbox"].color = [0, 1, 0]
            event["centroid"] = np.round(self.view_dataset.index_to_point(np.asarray(track["indexes"])).mean(0),
                                         2).tolist()
            reason = self._reject_track(track)
            if reason:
                event["decision"] = f"rejected: {reason}"
                self.commit_log.append(event)
                logger.info(f"track {event['track_id']} (frames {event['first_frame']}-{event['last_frame']}, "
                            f"{event['detections']} detections, at {event['centroid']}) -> {event['decision']}")
                return event

            best = None
            if len(self.objects):
                geometric, cosine, agg, threshold, passing = self._similarity(track, list(range(len(self.objects))))
                closest = int(np.argmax(agg))
                event.update(closest=self.object_label(closest), cosine=round(float(cosine[closest]), 3),
                             geometric=round(float(geometric[closest]), 3),
                             containment=round(float(self._last_containment[closest]), 3),
                             aggregate=round(float(agg[closest]), 3), threshold=float(threshold[closest]))
                if np.isfinite(passing.max()):
                    best = int(np.argmax(passing))

            if best is not None:
                self.objects[best] = self.merge_obj_to_obj(self.objects[best], track, run_dbscan=False)
                k = best
                event["decision"] = f"merged into {self.object_label(k)}"
            else:
                self.objects.append(track)
                k = len(self.objects) - 1
                self._uids[id(track)] = len(self._uids)
                event["decision"] = f"new object {self.object_label(k)}"
            event["object_index"] = k

            obj = self.objects[k]
            if self._memory_tree is not None and not obj.get("memory_ids"):
                d, _ = self._memory_tree.query(self.view_dataset.index_to_point(np.asarray(obj["indexes"])), k=1,
                                               distance_upper_bound=self.novelty_distance_m)
                obj["novelty"] = event["novelty"] = round(float(np.isinf(d).mean()), 3)

            if self._decode_fn is not None:
                self._track_to_object_index[key] = k
                # named later, in one batch (flush_track_names)
                self.name_ended_tracks(self.objects, [key], idx, self._decode_fn, defer=True, events=[event])
            self.commit_log.append(event)
            logger.info(f"track {event['track_id']} (frames {event['first_frame']}-{event['last_frame']}, "
                        f"{event['detections']} detections, at {event['centroid']}) -> {event['decision']}"
                        + (f" [closest {event['closest']}: cos {event['cosine']}, geo {event['geometric']}, "
                           f"contain {event['containment']}, "
                           f"agg {event['aggregate']} vs {event['threshold']}]" if "closest" in event else "")
                        + (f" '{event['name']}'" if "name" in event else "")
                        + (f", new space {event['novelty']}" if "novelty" in event else ""))
            if best is not None:  # the object grew: what did not match it before may now
                self._rematch(idx)
            return event

        def _reject_track(self, track) -> str:
            """object_fusion.track_validation, at commit, on the whole track (3D voxels + averaged embedding):
              structural_plane  the track's voxels form one thin, large, horizontal or vertical slab: a piece of
                                floor / ceiling / wall that the detector boxed like an object
                                (thickness = smallest PCA std < max_thickness_m, extent along the largest axis
                                > min_extent_m, normal within axis_tol of vertical or of horizontal)
              stuff_probe       a logistic "stuff vs thing" probe on the averaged CLIP embedding, trained on other
                                scenes' nodes (scripts/train_stuff_probe.py), says stuff with p > threshold
            Returns the reason, or "" to keep the track."""
            tv = self.track_validation or {}
            sp = tv.get("structural_plane") or {}
            if sp.get("enabled"):
                P = self.view_dataset.index_to_point(np.asarray(track["indexes"]))
                if len(P) >= 50:
                    c = P - P.mean(0)
                    ev, evec = np.linalg.eigh(np.cov(c.T))
                    thick, normal = float(np.sqrt(max(ev[0], 0))), evec[:, 0]
                    proj = c @ evec[:, 2]
                    extent = float(np.percentile(proj, 97.5) - np.percentile(proj, 2.5))
                    nz, tol = abs(float(normal[2])), sp.get("axis_tol", 0.2)
                    if (thick < sp.get("max_thickness_m", 0.03) and extent > sp.get("min_extent_m", 1.2)
                            and (nz > 1 - tol or nz < tol)):
                        return (f"structural plane ({'horizontal' if nz > 0.5 else 'vertical'}, "
                                f"{thick * 100:.1f} cm thick, {extent:.1f} m)")
            pr = tv.get("stuff_probe") or {}
            if pr.get("enabled"):
                if self._stuff_probe is None:
                    from pathlib import Path as _P
                    path = _P(pr["path"])
                    d = np.load(path if path.is_absolute() else _P(__file__).resolve().parents[1] / path)
                    self._stuff_probe = (d["w"].astype(np.float32), float(d["b"]))
                w, b = self._stuff_probe
                e = np.asarray(track["clip_ft"], dtype=np.float32).ravel()
                e = e / (np.linalg.norm(e) + 1e-12)
                p = 1.0 / (1.0 + np.exp(-(float(e @ w) + b)))
                if p > pr.get("threshold", 0.9):
                    return f"stuff probe p={p:.2f}"
            return ""

        def object_label(self, k: int) -> str:
            obj = self.objects[k]
            return obj["memory_ids"][0] if obj.get("memory_ids") else f"new#{self._uids.get(id(obj), k)}"

        def snapshot(self, obj_min_detections: int = 1) -> list[dict]:
            """Light copies of the committed objects (no per-frame masks), for
            building a live scene graph while fusion goes on."""
            self.flush_track_names()  # the live graph reads the track-end names
            with self._lock:
                out = []
                for k, obj in enumerate(self.objects or []):
                    if obj["num_detections"] < obj_min_detections:
                        continue
                    c = {key: (list(v) if isinstance(v, list) else v) for key, v in obj.items()
                         if key not in ("mask", "bbox")}
                    c["mask"] = []
                    c["indexes"] = list(np.asarray(obj["indexes"]))
                    c["clip_ft"] = np.asarray(obj["clip_ft"].detach().cpu() if torch.is_tensor(obj["clip_ft"])
                                              else obj["clip_ft"], dtype=np.float32).copy()
                    c["text_ft"] = c["clip_ft"]
                    c["fusion_index"] = k
                    out.append(c)
                return out

        def finalize(self, obj_min_detections: int = 3, merge_overlapping: bool | str = True,
                     merge_sim_threshold: float | None = None):
            """The end-of-scan clean-up of get_instances(), on the streamed objects."""
            self.flush_track_names()
            with self._lock:
                return self._finalize(self.objects, obj_min_detections, merge_overlapping, merge_sim_threshold)

        def get_instances(self, view_dataset, per_frame_detection_lists: list, obj_min_detections: int = 3,
                          decode_fn=None, initial_objects=None, merge_overlapping: bool | str = True,
                          merge_sim_threshold: float | None = None, match_level: str = "track",
                          track_end_frames: int = 10, memory_sim_threshold: float | None = None):
            """Fuses pre-built per-frame detection lists, at match_level
            "track" (streamed through add_frame()) or "detection"."""
            if match_level == "detection":
                return self._get_instances_by_detection(
                    view_dataset, per_frame_detection_lists, obj_min_detections, decode_fn, initial_objects,
                    merge_overlapping, merge_sim_threshold)
            if match_level != "track":
                raise ValueError(f"match_level must be 'track' or 'detection', got {match_level!r}")
            self.start_stream(view_dataset, initial_objects, decode_fn, track_end_frames, memory_sim_threshold)
            for idx in trange(len(per_frame_detection_lists), desc="instance fusion (track level)"):
                self.add_frame(idx, per_frame_detection_lists[idx])
            self.finish(len(per_frame_detection_lists))
            return self.finalize(obj_min_detections, merge_overlapping, merge_sim_threshold)

        def _get_instances_by_detection(self, view_dataset, per_frame_detection_lists: list,
                                        obj_min_detections: int = 3, decode_fn=None, initial_objects=None,
                                        merge_overlapping: bool | str = True,
                                        merge_sim_threshold: float | None = None):
            """Assumes start_new_scan(view_dataset) was already called before
            per_frame_detection_lists was built (see class docstring).

            A continuing track merges straight into the object its id was
            resolved to, with no re-matching; the tracker is trusted for
            identity. Untracked detections, and a track id's first frame, are
            scored against the whole map (0.3 spatial + 0.7 visual, merged if
            >= sim_threshold, otherwise a new object), because a new id may be
            an object the tracker lost through occlusion.

            With decode_fn (NOVICDecoder.decode_embeddings), each object is
            named when a track that fed it ends: see name_ended_tracks().

            With initial_objects (saved instances of an earlier run, in this
            view_dataset's voxel grid), fusion starts from them instead of an
            empty map, as DoVSG's get_instances(objects=...) does for a scene
            update: a detection that matches one merges into it, anything else
            becomes a new object. Their 'memory_ids' say which is which.
            Pass merge_overlapping="new_only" then: the final overlap merge
            judges containment by nearest points, and against a completed table
            top everything standing on it counts as contained, so over all
            objects it merges memory objects with each other and with what is
            new. "new_only" runs it among the new objects alone (a bottle and
            its cap), at merge_sim_threshold (default: sim_threshold).
            """
            self.view_dataset = view_dataset

            names = view_dataset.names
            assert len(per_frame_detection_lists) == len(names), (
                f"Expected one detection list per frame ({len(names)}), got {len(per_frame_detection_lists)}"
            )

            objects = MapObjectList([self.memory_seed(o) for o in (initial_objects or [])])
            self._track_to_object_index = {}
            self.track_name_log = []
            last_seen = self.last_seen_frames(per_frame_detection_lists)

            for idx in trange(len(names), desc="instance fusion"):
                fg_detection_list = per_frame_detection_lists[idx]
                if len(fg_detection_list) == 0:
                    continue

                if len(objects) == 0:
                    for det in fg_detection_list:
                        obj_idx = len(objects)
                        objects.append(det)
                        tid = det["track_ids"][0]
                        if tid >= 0:
                            self._track_to_object_index[tid] = obj_idx
                else:
                    known, unknown = [], []
                    for det in fg_detection_list:
                        tid = det["track_ids"][0]
                        (known if tid >= 0 and tid in self._track_to_object_index else unknown).append(det)

                    for det in known:
                        obj_idx = self._track_to_object_index[det["track_ids"][0]]
                        objects[obj_idx] = self.merge_obj_to_obj(objects[obj_idx], det, run_dbscan=False)

                    if unknown:
                        unknown_list = DetectionList(unknown)
                        spatial_sim = self.compute_spatial_similarities(unknown_list, objects)
                        visual_sim = self.compute_visual_similarities(unknown_list, objects)
                        text_sim = self.compute_text_similarities(unknown_list, objects)
                        agg_sim = self.aggregate_similarities(spatial_sim, visual_sim, text_sim)
                        agg_sim[agg_sim < self.sim_threshold] = float("-inf")

                        for i, det in enumerate(unknown_list):
                            if agg_sim[i].max() == float("-inf"):
                                obj_idx = len(objects)
                                objects.append(det)
                            else:
                                obj_idx = int(agg_sim[i].argmax())
                                objects[obj_idx] = self.merge_obj_to_obj(objects[obj_idx], det, run_dbscan=False)
                            tid = det["track_ids"][0]
                            if tid >= 0:
                                self._track_to_object_index[tid] = obj_idx

                if (idx + 1) % self.denoise_interval == 0:
                    objects = self.denoise_objects(objects)

                if decode_fn is not None:
                    ended = sorted({det["track_ids"][0] for det in fg_detection_list
                                    if det["track_ids"][0] >= 0 and last_seen[det["track_ids"][0]] == idx})
                    if ended:
                        self.name_ended_tracks(objects, ended, idx, decode_fn)

            return self._finalize(objects, obj_min_detections, merge_overlapping, merge_sim_threshold)

        def _finalize(self, objects, obj_min_detections, merge_overlapping, merge_sim_threshold):
            logger.info("Denoising fused instances...")
            objects = self.denoise_objects(objects)
            if merge_overlapping == "new_only":
                logger.info("Merging overlapping new instances...")
                seeded = [o for o in objects if o.get("memory_ids")]
                fresh = MapObjectList([o for o in objects if not o.get("memory_ids")])
                threshold, self.sim_threshold = self.sim_threshold, merge_sim_threshold or self.sim_threshold
                fresh = self.merge_objects(fresh) if len(fresh) > 1 else fresh
                self.sim_threshold = threshold
                objects = MapObjectList(seeded + list(fresh))
            elif merge_overlapping:
                logger.info("Merging overlapping instances...")
                objects = self.merge_objects(objects)
            objects = self.filter_objects(objects, obj_min_detections=obj_min_detections)
            objects = self.indexes_align_objects(objects)
            objects = self.filter_objects(objects, obj_min_detections=obj_min_detections)
            objects = self.change_objects(objects)

            objects_original_indexes = np.unique(self.objects_original_indexes)
            if len(objects) == 0:
                objects_indexes = np.array([])
            else:
                objects_indexes = np.concatenate(objects.get_values("indexes"))
            object_filter_indexes = np.setdiff1d(objects_original_indexes, objects_indexes)

            return objects, object_filter_indexes

    return InstanceFusion
