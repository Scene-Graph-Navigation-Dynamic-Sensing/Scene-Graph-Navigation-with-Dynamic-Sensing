#!/usr/bin/env python3
"""Update a saved scene graph with a new video of the same scene: find what was
added or removed and change only those nodes.

The new video goes through the same object fusion as the initial run
(object_fusion/instance_fusion.py, match_level "track", streamed by
object_fusion/streaming.py); the only difference is that fusion starts from
the saved objects instead of an empty map. Each BoT-SORT track is accumulated
while it lasts, and when it ends it is compared with every object in the map
-- the saved ones and those already found in this video -- by 0.3 geometric +
0.7 cosine similarity. A match merges into that object (embeddings weighted
by detection count); otherwise the track is a new object, which is checked
for new space and named. Detection, relocalization, fusion and the scene
graph run in parallel, and the scene graph is rebuilt every time a track
ends (outputs/scene_graphs/<scene_name>_live.json).

The steps follow DoVSG's scene update (external/DovSG/dovsg/controller.py,
update_scene()), with the same names:

  get_align_observations       relocalize the new video in the memory's world.
                               DoVSG: ACE rough pose + LightGlue + ICP. Here:
                               SuperPoint + LightGlue matches between keyframes
                               of both runs, lifted to 3D, RANSAC Sim3 (two
                               runs' monocular depth need not share a scale),
                               ICP on the downsampled maps.
  find_need_to_delete_indexes  a memory voxel is deleted when the new video
                               sees through it (observed depth > memory depth +
                               margin, most views agreeing); the margin is sized
                               for monocular depth, not DoVSG's 2 voxels.
  update_view_dataset          the frames are voxelized in the memory's own
                               voxel grid, so a voxel index means the same place
                               in both runs; deleted voxels leave the map.
  update_instance_objects      objects with > delete_rate of their voxels
                               deleted are removed; fusion as above. A saved
                               object is matched at scene_update.sim_threshold
                               (DoVSG's 0.75), an object of this video at
                               object_fusion.sim_threshold (0.55). A matched
                               saved object keeps its node, name and geometry;
                               its embedding becomes the detection-weighted
                               average of both runs. A new object is kept if
                               at least min_novelty of it is space the saved map
                               had empty, and NOVIC names it confidently.
  update_scene_graph           removed nodes go (their dependants are placed
                               again); new objects are placed onto the existing
                               graph with the on / inside / belongs rules;
                               every other node stays as it was.

object_fusion.match_level: detection switches to DoVSG's own per-detection
matching (sequential, on every observation_stride-th frame).

Usage:
    python scripts/update_scene_graph.py --config configs/pipeline_fyp_videos.yaml \\
        --memory fyp_initial_v2 --source data/updated_same_angle.mp4 --scene_name fyp_update_same_angle

    # reuse depth + poses already computed for the new video
    ... --dataset_dir data/processed/<processed run of the new video>

Outputs:
  outputs/scene_graphs/<scene_name>.pkl / .json    the updated scene graph
  outputs/scene_graphs/<scene_name>_live.json      the scene graph as it was
                                                   rebuilt at each track end
  outputs/scene_graphs/<scene_name>_changes.json   relocalization quality, every
                                                   track commit with its cosine /
                                                   geometric similarity, what was
                                                   added / removed and why
  outputs/visualizations/scene_update/<scene_name>/  changes.png, scene_graph.pdf
"""
from __future__ import annotations

import argparse
import copy
import faulthandler
import json
import logging
import multiprocessing
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from navigation.voxel_map import PersistedVoxelMap  # noqa: E402
from object_fusion._dovsg_import import ensure_dovsg_importable  # noqa: E402
from object_fusion.instance_fusion import build_instance_fusion_class  # noqa: E402
from object_fusion.instance_tagger import tag_instances  # noqa: E402
from object_fusion.view_dataset import voxelize_frame_in_grid  # noqa: E402
from object_fusion.streaming import (StreamingFusion, prepare_native_libraries, quiet_loggers,  # noqa: E402
                                     tree_lines, write_live_graph)
from perception.detection.tracked_yoloe_detector import TrackedYOLOEDetector  # noqa: E402
from perception.detection.yoloe_seg_detector import YOLOESegDetector  # noqa: E402
from scene_graph.builder.dovsg_relations import build_scene_graph_dovsg, remove_nodes  # noqa: E402
from scene_graph.storage.serializer import load_scene_graph, save_scene_graph  # noqa: E402
from scene_graph.storage.visualizer import save_scene_graph_visualization  # noqa: E402
from semantic_decoding.novic.novic_decoder import NOVICDecoder  # noqa: E402
from scripts.run_static_scene_graph import (load_config, load_everyday_names,  # noqa: E402
                                            run_perception_and_semantic_decoding)

logger = logging.getLogger("update_scene_graph")

DEFAULT_UPDATE_CFG = {
    "observation_stride": 1,          # every Nth frame of the new video; tracking needs consecutive frames
    # get_align_observations
    "memory_keyframe_stride": 8,      # memory frames matched against ...
    "observation_keyframe_stride": 10,  # ... every Nth frame of the new video
    "ransac_thresh_m": 0.05,
    "registration_voxel_m": 0.02,     # maps downsampled to this for ICP
    "min_registration_fscore": 0.3,   # symmetric F-score @2cm below this = relocalization failed
    "relocalization_device": "cuda",  # e.g. the second GPU, so LightGlue does not share one with detection
    "voxelize_workers": 6,            # processes voxelizing the new video's frames
    # find_need_to_delete_indexes / update_instance_objects
    "delete_frame_stride": 5,         # frames used by the see-through test
    "delete_depth_m": 0.10,           # DoVSG: 2 voxels; monocular depth needs more
    "delete_depth_rel": 0.08,         # ... or this share of the depth, whichever is larger
    "delete_min_votes": 2,            # frames that must see through a voxel
    "delete_vote_share": 0.6,         # of the frames that look at it
    "delete_rate": 0.5,               # DoVSG: object removed when this share of its voxels is deleted
    # a "removed" object is kept if the new video still detects it this many
    # times (matched to it at sim_threshold): depth noise must not delete
    # something that is plainly visible
    "redetect_min_detections": 3,
    # DoVSG InstanceProcess.sim_threshold, for matching a saved object: a clear
    # bottle's depth is the laptop behind it (geometric ~1.0, cosine 0.57 ->
    # 0.70), so saved objects are matched more strictly than this run's own.
    "sim_threshold": 0.75,
    # new objects
    "match_distance_m": 0.05,         # geometric similarity radius (report)
    "novelty_distance_m": 0.05,       # voxel farther than this from every memory voxel = new space
    # share of a new object's voxels that must be new space. Depth through a
    # clear bottle lands on what is behind it, so about half its voxels are
    # old surfaces (0.49 measured); fragments of saved objects score ~0.0
    "min_novelty": 0.4,
}


class ObservationMap:
    """The new video's frames voxelized in the memory's voxel grid (DoVSG
    ViewDataset.voxelize() with the memory's bounds), with the ViewDataset API
    the fusion and relation code read. Frames are voxelized in parallel in
    worker processes (object_fusion.view_dataset.voxelize_frame_in_grid;
    threads would fight detection for the GIL); frame(k) waits for frame k
    only, so fusion starts on the first frames while the rest are still being
    voxelized. wait_all() collects every frame for the removal check."""

    def __init__(self, dataset_dir: Path, frames: list[int], memory_map: PersistedVoxelMap, T_to_memory: np.ndarray,
                 pool, scene_graph_generation_root: str, nb_neighbors: int = 30, std_ratio: float = 1.5):
        from scripts.compare_video_poses import decompose_sim3

        self.dataset_dir, self.frames, self.memory = dataset_dir, frames, memory_map
        self.resolution = memory_map.resolution
        self.bounds = memory_map.bounds
        self.voxel_num = memory_map.voxel_num
        k = json.loads((dataset_dir / "metadata.json").read_text())["K"]
        self.K = np.array([[k[0], 0, k[2]], [0, k[4], k[5]], [0, 0, 1]], dtype=np.float64)
        self.names = [f"{i:06d}" for i in frames]
        self.length = len(frames)
        self.indexes_colors_mapping_dict = dict(memory_map.indexes_colors_mapping_dict)
        self._lock = threading.Lock()
        scale, R, t = decompose_sim3(T_to_memory)
        grid = (tuple(self.bounds.lower_bound.tolist()), tuple(self.bounds.higher_bound.tolist()),
                float(self.resolution), tuple(int(v) for v in self.voxel_num))
        self._futures = [pool.submit(voxelize_frame_in_grid, str(dataset_dir), i, (float(scale), R, t), *grid,
                                     scene_graph_generation_root, nb_neighbors, std_ratio) for i in frames]
        self._complete = False

    def frame(self, k: int):
        r = self._futures[k].result()
        return r["mapping"], r["valid"]

    def wait_all(self) -> "ObservationMap":
        with self._lock:
            if self._complete:
                return self
            rs = [f.result() for f in self._futures]
            self.images = [r["image"] for r in rs]
            self.masks = [r["mask"] for r in rs]
            self.poses = [r["pose"] for r in rs]
            self.depths = [r["depth"] for r in rs]
            self.pixel_index_mappings = [r["mapping"] for r in rs]
            self.pixel_index_masks = [r["valid"] for r in rs]
            ix = np.concatenate([r["mapping"][r["valid"]] for r in rs])
            rgb = np.concatenate([r["image"][r["valid"]] / 255.0 for r in rs])
            idx, inv = np.unique(ix, return_inverse=True)
            colour = np.zeros((len(idx), 3))
            np.add.at(colour, inv, rgb)
            colour /= np.bincount(inv, minlength=len(idx))[:, None]
            self.observed = dict(zip(idx.tolist(), colour))
            self.indexes_colors_mapping_dict.update(self.observed)
            self._complete = True
            logger.info(f"{self.length} frames voxelized: {len(self.observed)} voxels in the memory grid "
                        f"({len(self.observed.keys() - self.memory.indexes_colors_mapping_dict.keys())} "
                        "not in the memory).")
            return self

    def index_to_point(self, indexes):
        return self.memory.index_to_point(indexes)

    def point_to_index(self, points):
        return self.memory.point_to_index(points)

    # used by fusion's voxel-overlap lookup (instance_fusion._grid_neighbourhood)
    def index_to_voxel(self, indexes):
        return self.memory.index_to_voxel(indexes)

    def voxel_to_index(self, voxels):
        return self.memory.voxel_to_index(voxels)

    def index_to_pcd(self, indexes):
        import open3d as o3d

        indexes = np.asarray(indexes)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.index_to_point(indexes))
        pcd.colors = o3d.utility.Vector3dVector(
            np.array([self.indexes_colors_mapping_dict.get(ix, (0.5, 0.5, 0.5)) for ix in indexes.tolist()]))
        return pcd


# ─── get_align_observations ───

def _cloud(points, colours, voxel):
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(np.clip(colours, 0, 1))
    return pcd.voxel_down_sample(voxel)


def get_align_observations(memory_dir: Path, new_dir: Path, memory_map: PersistedVoxelMap, frames: list[int],
                           ucfg: dict) -> tuple[np.ndarray, dict]:
    """Sim3 taking the new video's world to the memory's."""
    from scripts.compare_video_poses import (KeyframeFeatures, Matcher, Sequence, collect_matches,
                                             decompose_sim3, register_improved, symmetric_fscore)

    t0 = time.time()
    seq_m, seq_n = Sequence(memory_dir), Sequence(new_dir)
    voxel = ucfg["registration_voxel_m"]
    idx = np.fromiter(memory_map.indexes_colors_mapping_dict.keys(), dtype=np.int64)
    cloud_m = _cloud(memory_map.index_to_point(idx), np.array(list(memory_map.indexes_colors_mapping_dict.values())),
                     voxel)
    pts, cols = [], []
    for i in frames[::2]:
        p, m = seq_n.points(i)
        T = seq_n.poses[i]
        pts.append(p[m] @ T[:3, :3].T + T[:3, 3])
        cols.append(seq_n.rgb(i)[m] / 255.0)
    cloud_n = _cloud(np.concatenate(pts), np.concatenate(cols), voxel)

    matcher = Matcher("lightglue", device=ucfg["relocalization_device"])
    fm = KeyframeFeatures(seq_m, ucfg["memory_keyframe_stride"], matcher)
    fn = KeyframeFeatures(seq_n, ucfg["observation_keyframe_stride"], matcher)
    pairs = collect_matches(fm, fn, matcher)
    logger.info(f"[relocalization] {len(pairs)} of {len(fm.frames)} x {len(fn.frames)} keyframe pairs verified.")
    reg = register_improved(fm, fn, pairs, cloud_m, cloud_n, ucfg["ransac_thresh_m"])
    T = reg["T"]
    s, R, t = decompose_sim3(T)
    b = copy.deepcopy(cloud_n).transform(T)
    d = np.asarray(b.compute_point_cloud_distance(cloud_m))
    info = {
        "image_pairs_verified": len(pairs),
        "matches_3d": reg["matches"],
        "sim3_ransac_inliers": reg["match_inliers"],
        "refinement_steps_fscore@2cm": reg["refinement_steps"],
        "scale": round(s, 4),
        "rotation_deg": round(float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))), 2),
        "translation_m": round(float(np.linalg.norm(t)), 3),
        "transform_new_to_memory": np.round(T, 6).tolist(),
        "fscore@2cm": round(symmetric_fscore(cloud_m, cloud_n, T, 0.02), 4),
        "median_distance_mm": round(float(np.median(d[d < 0.10])) * 1000, 2),
        "seconds": round(time.time() - t0, 1),
    }
    logger.info(f"[relocalization] scale {info['scale']}, {info['rotation_deg']} deg, {info['translation_m']} m; "
                f"F-score@2cm {info['fscore@2cm']}, median distance {info['median_distance_mm']} mm "
                f"({info['seconds']}s, in parallel with detection).")
    return T, info


# ─── find_need_to_delete_indexes ───

def find_need_to_delete_indexes(memory_map: PersistedVoxelMap, obs: ObservationMap, ucfg: dict) -> np.ndarray:
    obs.wait_all()
    idx = np.fromiter(memory_map.indexes_colors_mapping_dict.keys(), dtype=np.int64)
    pts = memory_map.index_to_point(idx)
    looked = np.zeros(len(idx), np.int32)
    through = np.zeros(len(idx), np.int32)
    fx, fy, cx, cy = obs.K[0, 0], obs.K[1, 1], obs.K[0, 2], obs.K[1, 2]
    step = ucfg["delete_frame_stride"]
    for pose, depth, valid in zip(obs.poses[::step], obs.depths[::step], obs.masks[::step]):
        h, w = depth.shape
        inv = np.linalg.inv(pose)
        pc = pts @ inv[:3, :3].T + inv[:3, 3]
        # only voxels in front of the camera and inside the image (DoVSG)
        front = pc[:, 2] > 0.1
        u = np.full(len(pc), -1, np.int64)
        v = np.full(len(pc), -1, np.int64)
        u[front] = np.round(fx * pc[front, 0] / pc[front, 2] + cx).astype(np.int64)
        v[front] = np.round(fy * pc[front, 1] / pc[front, 2] + cy).astype(np.int64)
        m = front & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        m[m] = valid[v[m], u[m]]  # invalid depth can't give reference
        # nearest memory surface per pixel, compared with the observed depth
        zbuf = np.full((h, w), np.inf)
        np.minimum.at(zbuf, (v[m], u[m]), pc[m, 2])
        diff = depth[v[m], u[m]] - zbuf[v[m], u[m]]
        margin = np.maximum(ucfg["delete_depth_m"], ucfg["delete_depth_rel"] * zbuf[v[m], u[m]])
        looked[m] += 1
        through[np.where(m)[0][diff > margin]] += 1
    deleted = (through >= ucfg["delete_min_votes"]) & (through >= ucfg["delete_vote_share"] * np.maximum(looked, 1))
    logger.info(f"[removal] {int((looked > 0).sum())} memory voxels in view, {int(deleted.sum())} seen through.")
    return idx[deleted]


def removed_objects(memory_objects, need_delete_indexes, ucfg) -> tuple[list[str], dict]:
    """DoVSG: an object is removed when more than delete_rate of its voxels were seen through."""
    share = {o["class_id"]: round(float(np.isin(np.asarray(o["indexes"]), need_delete_indexes).mean()), 3)
             for o in memory_objects}
    return [cid for cid, rate in share.items() if rate > ucfg["delete_rate"]], share


def keep_redetected(removed_ids, fused, memory_objects, ucfg) -> list[str]:
    """Safety check on the see-through test: an object it wants to remove is
    kept when the new video detected it again (a track matched it, adding at
    least redetect_min_detections detections). Seen through by depth but seen
    by the detector = the depth was wrong, the object is still there."""
    old = {o["class_id"]: int(o["num_detections"]) for o in memory_objects}
    seen_again = {o["memory_ids"][0]: int(o["num_detections"]) - old.get(o["memory_ids"][0], 0)
                  for o in fused if o.get("memory_ids")}
    kept = [cid for cid in removed_ids if seen_again.get(cid, 0) >= ucfg["redetect_min_detections"]]
    if kept:
        logger.info(f"[removal] kept {kept}: seen through by depth but detected again in the new video.")
    return [cid for cid in removed_ids if cid not in kept]


# ─── update_view_dataset ───

def update_view_dataset(obs: ObservationMap, need_delete_indexes: np.ndarray, keep_indexes: set) -> None:
    """Deleted voxels leave the map (except those of objects that stay)."""
    deleted = set(need_delete_indexes.tolist()) - keep_indexes
    obs.indexes_colors_mapping_dict = {ix: c for ix, c in obs.indexes_colors_mapping_dict.items()
                                       if ix not in deleted}


# ─── update_instance_objects ───

def _embedding(obj) -> np.ndarray:
    ft = obj["clip_ft"]
    ft = ft.detach().cpu().numpy() if hasattr(ft, "detach") else np.asarray(ft)
    return ft.astype(np.float32).ravel()


def update_instance_objects(fused, memory_objects, removed_ids, obs, ctx) -> tuple[list, list, list, dict]:
    """Splits fused objects into saved and new ones. A saved object keeps its
    node, name and geometry; if this video matched tracks to it, its
    embedding becomes the detection-weighted average of both runs. A new
    object is kept if it is mostly new space and NOVIC names it confidently.
    Returns (kept, added, report, reobserved)."""
    cfg, ucfg = ctx["cfg"], ctx["ucfg"]
    min_det = cfg["object_fusion"]["obj_min_detections"]
    matched = {o["memory_ids"][0]: o for o in fused if o.get("memory_ids")}
    kept, reobserved = [], {}
    for mo in memory_objects:
        if mo["class_id"] in removed_ids:
            continue
        m = dict(mo)
        f = matched.get(mo["class_id"])
        if f is not None and int(f["num_detections"]) > int(mo["num_detections"]):
            reobserved[mo["class_id"]] = int(f["num_detections"]) - int(mo["num_detections"])
            m["clip_ft"] = m["semantic_embedding"] = _embedding(f)
            m["num_detections"] = int(f["num_detections"])
            m["reobserved_detections"] = reobserved[mo["class_id"]]
        kept.append(m)

    new = [o for o in fused if not o.get("memory_ids") and int(o["num_detections"]) >= min_det]
    for o in new:
        o["clip_ft"] = _embedding(o)
    report = describe_new_objects(new, kept, obs, ucfg, ctx["memory_tree"])
    candidates = [new[r["new_index"]] for r in report if r["novelty"] >= ucfg["min_novelty"]]
    label_cfg = cfg["label_mapper"]
    added = list(tag_instances(
        list(candidates), ctx["novic"], merge_similarity_threshold=label_cfg["merge_similarity_threshold"],
        min_name_score=label_cfg.get("min_name_score", 0.15),
        synonym_similarity=label_cfg.get("synonym_similarity", 0.8), everyday_names=ctx["everyday"],
        everyday_name_weight=label_cfg.get("everyday_name_weight", 0.2), low_score_action="drop",
        manual_labels=label_cfg.get("manual_labels"))) if candidates else []
    renumber(added, {o["class_id"] for o in memory_objects})
    for r in report:
        obj = new[r["new_index"]]
        if r["novelty"] < ucfg["min_novelty"]:
            r["decision"] = "already in the memory map (not new space)"
        elif any(obj is a for a in added):
            r["decision"] = f"ADDED as {obj['class_id']}"
            r["name_confidence"] = round(float(obj["name_confidence"]), 3)
            r["novic_label"] = obj.get("novic_label") or obj["class_name"]
        else:
            r["decision"] = "new space, but NOVIC could not name it confidently (dropped)"
    return kept, added, report, reobserved


def describe_new_objects(new_objects, memory_objects, obs, ucfg, memory_tree) -> list[dict]:
    """Per new object: the closest saved object by cosine + geometric
    similarity (share of its voxels within match_distance_m), and its novelty
    (share of its voxels farther than novelty_distance_m from every memory
    voxel)."""
    from scipy.spatial import cKDTree

    mem_trees = [cKDTree(obs.index_to_point(np.asarray(o["indexes"]))) for o in memory_objects]
    mem_emb = np.stack([_embedding(o) for o in memory_objects])
    mem_emb /= np.linalg.norm(mem_emb, axis=1, keepdims=True)
    out = []
    for k, obj in enumerate(new_objects):
        pts = obs.index_to_point(np.asarray(obj["indexes"]))
        emb = _embedding(obj)
        cosine = mem_emb @ (emb / np.linalg.norm(emb))
        geometric = np.array([np.isfinite(t.query(pts, k=1, distance_upper_bound=ucfg["match_distance_m"])[0]).mean()
                              for t in mem_trees])
        best = int(np.argmax(0.3 * geometric + 0.7 * cosine))
        d, _ = memory_tree.query(pts, k=1, distance_upper_bound=ucfg["novelty_distance_m"])
        out.append({"new_index": k, "points": len(pts), "detections": int(obj["num_detections"]),
                    "centroid": np.round(pts.mean(0), 3).tolist(),
                    "best_memory": memory_objects[best]["class_id"], "cosine": round(float(cosine[best]), 3),
                    "geometric": round(float(geometric[best]), 3), "novelty": round(float(np.isinf(d).mean()), 3)})
    return out


def renumber(objects, taken: set[str]) -> None:
    """tag_instances() numbers each name from _0; keep the memory's ids unique."""
    for obj in objects:
        n = 0
        while f"{obj['class_name']}_{n}" in taken:
            n += 1
        obj["class_id"] = f"{obj['class_name']}_{n}"
        taken.add(obj["class_id"])


# ─── update_scene_graph ───

def update_scene_graph(history_graph, removed_ids, objects, obs, cfg):
    graph = copy.deepcopy(history_graph)
    orphans = remove_nodes(graph, list(removed_ids))
    sg = cfg["scene_graph"]
    graph = build_scene_graph_dovsg(
        view_dataset=obs, instance_objects=objects,
        scene_graph_generation_root=cfg["external"]["scene_graph_generation_root"],
        resolution=sg["resolution"], stand_floor_threshold=sg["stand_floor_threshold"],
        alphashape_alpha=sg["alphashape_alpha"],
        alpha_shape_overleaf_rate_threshold=sg["alpha_shape_overleaf_rate_threshold"],
        inside_threshold=sg["inside_threshold"], part_level_classes=sg.get("part_level_classes"),
        support_max_gap_m=sg.get("support_max_gap_m", 0.10),
        support_max_penetration_m=sg.get("support_max_penetration_m", 0.15),
        next_to_max_gap_m=sg.get("next_to_max_gap_m", 0.30),
        next_to_min_vertical_overlap=sg.get("next_to_min_vertical_overlap", 0.5), history_graph=graph)
    return graph, orphans


# ─── Timing report ───

# (timings key, step, runs in parallel with the main flow?)
TIMING_ROWS = [
    ("load_memory_s", "load the initial scene graph + map", False),
    ("depth_pose_s", "new video: depth (DA3) + camera poses (DROID-SLAM)", False),
    ("load_models_s", "load detector + NOVIC", False),
    ("relocalization_s", "1. get_align_observations (relocalize new video)", True),
    ("removal_check_s", "2. find_need_to_delete_indexes (see-through test)", True),
    ("streaming_s", "3-4. detect + track + embed + fuse objects (streaming)", False),
    ("finalize_fusion_s", "4. finalize fusion (merge new objects)", False),
    ("decide_objects_s", "4. update_instance_objects (removed / kept / added)", False),
    ("update_scene_graph_s", "5. update_view_dataset + update_scene_graph (nodes removed / added)", False),
    ("save_and_visualize_s", "save graph + plots", False),
]


def timing_report(timings: dict, header: str) -> str:
    """Markdown table of the time each step took. Steps marked 'parallel' run
    alongside detection, so their times overlap and do not add up to the total."""
    total = max(float(timings.get("total_s", 0.0)), 1e-9)
    lines = [f"# Scene update timing report: {header}", "",
             "| step | seconds | share of total | note |", "|---|---:|---:|---|"]
    for key, name, parallel in TIMING_ROWS:
        if key in timings:
            lines.append(f"| {name} | {timings[key]:.1f} | {100 * timings[key] / total:.0f}% | "
                         f"{'in parallel with detection' if parallel else ''} |")
    lines += [f"| **total (wall clock)** | **{total:.1f}** | 100% | |", ""]
    stream = {k[len("streaming_"):-2]: v for k, v in timings.items()
              if k.startswith("streaming_") and k != "streaming_s"}
    if stream:
        lines += ["Inside the streaming step (threads overlap):", ""]
        lines += [f"- {k.replace('_', ' ')}: {v:.1f} s" for k, v in stream.items()]
        lines.append("")
    main_steps = sum(timings.get(k, 0.0) for k, _, par in TIMING_ROWS if not par)
    slowest = max(((timings[k], n) for k, n, par in TIMING_ROWS if k in timings and not par), default=None)
    lines.append(f"Sequential steps add up to {main_steps:.1f} s of {total:.1f} s.")
    if slowest:
        lines.append(f"Slowest sequential step: {slowest[1]} ({slowest[0]:.1f} s, {100 * slowest[0] / total:.0f}%).")
    return "\n".join(lines)


# ─── Report ───

def plot_changes(path: Path, obs: ObservationMap, kept, added, removed):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    idx = np.fromiter(obs.memory.indexes_colors_mapping_dict.keys(), dtype=np.int64)
    pts = obs.index_to_point(idx)
    sel = np.random.default_rng(0).choice(len(pts), min(80000, len(pts)), replace=False)
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.scatter(pts[sel, 0], pts[sel, 1], s=0.2, c="lightgrey", zorder=0)
    for group, colour in ((kept, "tab:blue"), (removed, "tab:red"), (added, "tab:green")):
        for obj in group:
            p = obs.index_to_point(np.asarray(obj["indexes"]))
            ax.scatter(p[:, 0], p[:, 1], s=0.5, c=colour, alpha=0.5 if group is kept else 1.0)
            ax.annotate(obj["class_id"], p.mean(0)[:2], fontsize=9 if group is kept else 12,
                        color="black" if group is kept else colour, weight="normal" if group is kept else "bold")
    for colour, label in (("tab:blue", "unchanged"), ("tab:red", "removed"), ("tab:green", "added")):
        ax.scatter([], [], c=colour, label=label)
    ax.legend(loc="upper left")
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Scene update, top-down (memory world, z up)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ─── update_scene ───

def main():
    # `kill -USR1 <pid>` prints every thread's Python stack (e.g. a hung stream)
    faulthandler.register(signal.SIGUSR1, all_threads=True)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/pipeline_fyp_videos.yaml")
    ap.add_argument("--memory", required=True,
                    help="scene name of the initial run: outputs/scene_graphs/<memory>.pkl + data/processed/<memory>")
    ap.add_argument("--source", help="the new video (runs depth + DROID-SLAM on it first)")
    ap.add_argument("--dataset_dir", help="an already processed run of the new video (rgb/point/mask/poses)")
    ap.add_argument("--scene_name", required=True, help="name of the updated scene graph")
    args = ap.parse_args()
    if not (args.source or args.dataset_dir):
        ap.error("give the new video (--source) or its processed dataset (--dataset_dir)")

    cfg = load_config(REPO_ROOT / args.config)
    ucfg = {**DEFAULT_UPDATE_CFG, **(cfg.get("scene_update") or {})}
    fusion_cfg, det_cfg, sem_cfg = cfg["object_fusion"], cfg["detection"], cfg["semantic_decoding"]
    match_level = fusion_cfg.get("match_level", "track")
    out_graphs = REPO_ROOT / cfg["outputs"]["scene_graphs_dir"]
    viz_dir = REPO_ROOT / cfg["outputs"]["visualizations_dir"] / "scene_update" / args.scene_name
    viz_dir.mkdir(parents=True, exist_ok=True)
    log_path = REPO_ROOT / cfg["outputs"]["logs_dir"] / f"{args.scene_name}.log"
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(log_path)])
    ensure_dovsg_importable(cfg["external"]["scene_graph_generation_root"])
    t_start = time.time()
    timings = {}

    # memory: the initial run
    memory_dir = REPO_ROOT / cfg["outputs"]["data_processed_dir"] / args.memory
    t = time.time()
    memory = load_scene_graph(out_graphs / f"{args.memory}.pkl")
    memory_map = PersistedVoxelMap.from_dict(memory["voxel_map"])
    memory_objects = memory["instance_objects"]
    timings["load_memory_s"] = round(time.time() - t, 1)
    logger.info(f"[time] load memory: {timings['load_memory_s']}s")
    logger.info(f"Memory '{args.memory}': {[o['class_id'] for o in memory_objects]}.")

    # new observation: depth + poses
    if args.dataset_dir:
        new_dir = Path(args.dataset_dir)
        new_dir = new_dir if new_dir.is_absolute() else REPO_ROOT / new_dir
    else:
        t = time.time()
        new_dir = REPO_ROOT / cfg["outputs"]["data_processed_dir"] / args.scene_name
        subprocess.run([sys.executable, str(REPO_ROOT / "scripts" / "run_static_scene_graph.py"),
                        "--config", str(REPO_ROOT / args.config), "--source", args.source,
                        "--scene_name", args.scene_name, "--pose_only"], cwd=REPO_ROOT, check=True)
        timings["depth_pose_s"] = round(time.time() - t, 1)
        logger.info(f"[time] new video depth + poses: {timings['depth_pose_s']}s")
    length = json.loads((new_dir / "metadata.json").read_text())["length"]
    frames = list(range(0, length, ucfg["observation_stride"]))

    # worker processes for voxelizing the new video, started now so their
    # start-up overlaps with model loading
    pool = ProcessPoolExecutor(max_workers=ucfg["voxelize_workers"],
                               mp_context=multiprocessing.get_context("spawn"))
    pool.submit(int, 0).result()

    # models
    t_models = time.time()
    detector_kwargs = dict(weights_path=det_cfg["weights"], conf_thresh=det_cfg["conf_thresh"],
                           iou_thresh=det_cfg["iou_thresh"], img_size=det_cfg["img_size"], device=det_cfg["device"],
                           fix_channel_order=det_cfg.get("fix_channel_order", True))
    if det_cfg.get("backend") == "panoptic" and match_level == "track":    # perception/detection/panoptic_detector.py
        from perception.detection.panoptic_detector import from_config as panoptic_from_config
        detector = panoptic_from_config(det_cfg)
    else:
        detector = (TrackedYOLOEDetector(tracker=det_cfg.get("tracker", "botsort.yaml"),
                                            mask_box_filter=det_cfg.get("mask_box_filter"),
                                            tighten_box_to_mask=det_cfg.get("tighten_box_to_mask", False),
                                            validation=det_cfg.get("validation"),
                                            **detector_kwargs)
                    if match_level == "track" else YOLOESegDetector(**detector_kwargs))
    novic = NOVICDecoder(novic_dir=sem_cfg["novic_dir"], checkpoint_path=sem_cfg["novic_checkpoint"],
                         gencfg=sem_cfg["gencfg"], guide_targets=sem_cfg["guide_targets"], device=sem_cfg["device"],
                         embedding_batch_size=sem_cfg.get("embedding_batch_size", 8))
    extra_embedders = [
        (lambda device=device: NOVICDecoder(
            novic_dir=sem_cfg["novic_dir"], checkpoint_path=sem_cfg["novic_checkpoint"], gencfg=sem_cfg["gencfg"],
            guide_targets=sem_cfg["guide_targets"], device=device,
            embedding_batch_size=sem_cfg.get("embedding_batch_size", 8)))
        for device in sem_cfg.get("extra_embedding_devices") or []]
    fusion = build_instance_fusion_class(cfg["external"]["scene_graph_generation_root"])(
        downsample_voxel_size=fusion_cfg["downsample_voxel_size"], spatial_weight=fusion_cfg["spatial_weight"],
        vis_weight=fusion_cfg["vis_weight"],
        sim_threshold=fusion_cfg["sim_threshold"] if match_level == "track" else ucfg["sim_threshold"])
    timings["load_models_s"] = round(time.time() - t_models, 1)
    logger.info(f"[time] load models: {timings['load_models_s']}s")
    from scipy.spatial import cKDTree
    memory_points = memory_map.index_to_point(
        np.fromiter(memory_map.indexes_colors_mapping_dict.keys(), dtype=np.int64))
    ctx = {"cfg": cfg, "ucfg": ucfg, "novic": novic, "everyday": load_everyday_names(cfg),
           "memory_tree": cKDTree(memory_points)}
    logger.info(f"Object fusion: {match_level}-level matching on {len(frames)} frames; saved objects matched at "
                f"{ucfg['sim_threshold']}, this video's objects at {fusion.sim_threshold}.")

    prepare_native_libraries()  # before any thread starts: see its docstring
    side = ThreadPoolExecutor(max_workers=2)
    # 1. relocalization starts at once, in parallel with detection
    reloc = side.submit(get_align_observations, memory_dir, new_dir, memory_map, frames, ucfg)
    state = {}

    def prepare():
        T, reg_info = reloc.result()
        if reg_info["fscore@2cm"] < ucfg["min_registration_fscore"]:
            raise RuntimeError(f"Relocalization failed (F-score@2cm {reg_info['fscore@2cm']}): "
                               "the new video may not show the memory's scene.")
        obs = ObservationMap(new_dir, frames, memory_map, T, pool, cfg["external"]["scene_graph_generation_root"])
        state.update(obs=obs, reg=reg_info)
        # 2. removal check, as soon as every frame is voxelized
        def timed_removal():
            t0 = time.time()
            out = find_need_to_delete_indexes(memory_map, obs, ucfg)
            timings["removal_check_s"] = round(time.time() - t0, 1)
            logger.info(f"[time] see-through test: {timings['removal_check_s']}s")
            return out

        state["removal"] = side.submit(timed_removal)
        return obs

    if match_level == "track":
        live_path = out_graphs / f"{args.scene_name}_live.json"

        def prepare_stream():
            obs = prepare()
            fusion.start_new_scan(obs)
            fusion.start_stream(obs, initial_objects=memory_objects, decode_fn=novic.decode_embeddings,
                                track_end_frames=fusion_cfg.get("track_end_frames", 10),
                                memory_sim_threshold=ucfg["sim_threshold"], memory_points=memory_points,
                                novelty_distance_m=ucfg["novelty_distance_m"],
                                min_merge_overlap=fusion_cfg.get("min_merge_overlap", 0.3))

        def on_update(snapshot, info):
            t = time.time()
            obs = state["obs"]
            removal = state.get("removal")
            removed_ids = (keep_redetected(removed_objects(memory_objects, removal.result(), ucfg)[0],
                                           snapshot, memory_objects, ucfg)
                           if removal is not None and removal.done() else [])
            with quiet_loggers():
                kept, added, _, _ = update_instance_objects(snapshot, memory_objects, removed_ids, obs, ctx)
                graph, _ = update_scene_graph(memory["scene_graph"], removed_ids, kept + added, obs, cfg)
            marks = {o["class_id"]: "added" for o in added}
            write_live_graph(live_path, graph, kept + added, obs,
                             {**info, "added": [o["class_id"] for o in added], "removed": removed_ids})
            logger.info(f"[live scene graph #{info['update']}] after {info['frames_fused']}/{len(frames)} frames, "
                        f"{info['commits']} tracks ({time.time() - t:.1f}s): added {list(marks) or 'nothing'}, "
                        f"removed {removed_ids or 'nothing'}\n" + tree_lines(graph, marks))

        stream = StreamingFusion(
            cfg, fusion, detector, novic, frame_names=[f"{i:06d}" for i in frames],
            load_image=lambda k: np.asarray(Image.open(new_dir / "rgb" / f"{frames[k]:06d}.jpg"), dtype=np.uint8),
            frame_voxels=lambda k: state["obs"].frame(k), prepare=prepare_stream,
            on_update=on_update if fusion_cfg.get("live_scene_graph", False) else None,
            det_viz_dir=None, obj_min_detections=fusion_cfg["obj_min_detections"], keep_frame_records=False,
            extra_embedders=extra_embedders, live_interval_s=fusion_cfg.get("live_scene_graph_interval_s", 0.0))
        t = time.time()
        stream.run()
        commits = fusion.commit_log
        timings["streaming_s"] = round(time.time() - t, 1)
        timings.update({f"streaming_{k}": v for k, v in stream.timings.items()})
        t = time.time()
        # DoVSG's final overlap merge, among this video's objects only (a
        # bottle and its cap): over all objects it would glue everything on
        # the completed table top together
        fused, _ = fusion.finalize(obj_min_detections=fusion_cfg["obj_min_detections"], merge_overlapping="new_only",
                                   merge_sim_threshold=fusion_cfg["sim_threshold"])
        timings["finalize_fusion_s"] = round(time.time() - t, 1)
    else:
        # DoVSG's per-detection matching, sequential
        obs = prepare().wait_all()
        per_frame, _ = run_perception_and_semantic_decoding(cfg, obs, detector, novic, fusion)
        fused, _ = fusion.get_instances(view_dataset=obs, per_frame_detection_lists=per_frame,
                                        obj_min_detections=fusion_cfg["obj_min_detections"],
                                        initial_objects=memory_objects, merge_overlapping="new_only",
                                        merge_sim_threshold=fusion_cfg["sim_threshold"], match_level="detection")
        commits = []

    obs = state["obs"].wait_all()
    timings["relocalization_s"] = state["reg"]["seconds"]
    need_delete_indexes = state["removal"].result()
    t = time.time()
    removed_ids, delete_share = removed_objects(memory_objects, need_delete_indexes, ucfg)
    removed_ids = keep_redetected(removed_ids, fused, memory_objects, ucfg)
    kept, added, report, reobserved = update_instance_objects(fused, memory_objects, removed_ids, obs, ctx)
    for r in report:
        logger.info(f"new object #{r['new_index']} ({r['detections']} detections, {r['points']} voxels at "
                    f"{r['centroid']}): closest saved object {r['best_memory']} (cosine {r['cosine']}, geometric "
                    f"{r['geometric']}), new space {r['novelty']} -> {r['decision']}")
    objects = kept + added
    timings["decide_objects_s"] = round(time.time() - t, 1)
    logger.info(f"[time] decide removed / kept / added objects: {timings['decide_objects_s']}s")
    t = time.time()
    update_view_dataset(obs, need_delete_indexes,
                        set(np.concatenate([np.asarray(o["indexes"]) for o in objects]).tolist()))
    graph, orphans = update_scene_graph(memory["scene_graph"], removed_ids, objects, obs, cfg)
    timings["update_scene_graph_s"] = round(time.time() - t, 1)
    logger.info(f"[time] update scene graph: {timings['update_scene_graph_s']}s")
    novic.close()
    pool.shutdown()
    side.shutdown()

    t = time.time()
    for o in objects:
        o.pop("mask", None)
    embeddings = {o["class_id"]: np.asarray(o["semantic_embedding"], np.float32)
                  for o in objects if "semantic_embedding" in o}
    pkl_path, json_path = save_scene_graph(out_graphs, args.scene_name, graph, objects, embeddings, view_dataset=obs)
    save_scene_graph_visualization(graph, viz_dir)
    removed = [o for o in memory_objects if o["class_id"] in removed_ids]
    plot_changes(viz_dir / "changes.png", obs, kept, added, removed)
    timings["save_and_visualize_s"] = round(time.time() - t, 1)
    timings["total_s"] = round(time.time() - t_start, 1)
    report_path = out_graphs / f"{args.scene_name}_timing_report.md"
    report_path.write_text(timing_report(timings, f"{args.memory} -> {args.scene_name}"))
    logger.info("Timing report:\n" + report_path.read_text())

    added_ids = [o["class_id"] for o in added]
    summary = {
        "memory": args.memory,
        "new_observation": str(new_dir),
        "match_level": match_level,
        "frames": len(frames),
        "added": [{"node_id": o["class_id"], "parent": graph.object_nodes[o["class_id"]].parent.node_id,
                   "relation": graph.object_nodes[o["class_id"]].parent_relation,
                   "centroid": np.round(obs.index_to_point(np.asarray(o["indexes"])).mean(0), 3).tolist(),
                   "name_confidence": round(float(o["name_confidence"]), 3),
                   "novic_label": o.get("novic_label") or o["class_name"]} for o in added],
        "removed": removed_ids,
        "re_placed_after_removal": orphans,
        "unchanged": [o["class_id"] for o in kept],
        "reobserved_detections_per_saved_object": reobserved,
        "relocalization": state["reg"],
        "deleted_voxels": int(len(need_delete_indexes)),
        "deleted_share_per_object": delete_share,
        "new_objects": report,
        "track_commits": commits,
        "settings": {**ucfg, "match_level": match_level, "this_run_sim_threshold": fusion.sim_threshold,
                     "track_end_frames": fusion_cfg.get("track_end_frames", 10)},
        "timings_s": timings,
    }
    changes_path = out_graphs / f"{args.scene_name}_changes.json"
    changes_path.write_text(json.dumps(summary, indent=2, default=str))
    logger.info("Updated scene graph:\n" + tree_lines(graph, {cid: "added" for cid in added_ids}))
    logger.info(f"Added: {added_ids or 'nothing'}; removed: {removed_ids or 'nothing'}. "
                f"Time: {timings['total_s']} s {timings}")
    logger.info(f"Saved {pkl_path}, {json_path}, {changes_path}, {viz_dir}")


if __name__ == "__main__":
    main()
