#!/usr/bin/env python3
"""DoVSG dynamic update method on this project's scene graphs.

Updates a saved scene graph (this project's pipeline: YOLOE + NOVIC + DoVSG
fusion and relations) with a new video of the same scene, using DoVSG's
update logic (external/DovSG/dovsg/controller.py, run_tasks ->
get_align_observations -> update_scene):

  1. ACE                ace_utils.ace_poses()        rough pose of every new frame in the memory map
  2. LightGlue anchor   RGBFeatureMatch, anchor_poses() most similar memory frame per new frame; the frame
                                                     with most matches keeps its ACE pose, the rest follow
                                                     through the video's own poses (b2w @ c2b)
     (scale)            --estimate_scale             optional, not DoVSG: Sim3 from the LightGlue matches
  3. colored ICP        calculate_alignment_colored_icp()  one rigid correction for all frames; fails -> stop
  4. changes            find_need_to_delete_indexes() memory voxels the new depth sees through
  5. update             split_removed() (> 50% deleted -> removed), voxel map, YOLOE + NOVIC on the new
                        frames, fusion from the surviving objects, remove_nodes + build_scene_graph_dovsg

Every new frame is used (--observation_stride 1). Your safeguards from
scripts/update_scene_graph.py (re-detection guard, novelty check, merge
among new objects only) are off unless --safeguards. --robust_changes (not
DoVSG) replaces step 4 with a voted test for estimated depth: see
find_seen_through_voted().

Run in the scene_graph env, after train_ace.py:

    python dynamic_updates/dovsg_method/run_dovsg_update.py --config configs/pipeline_fyp_videos.yaml \\
        --memory fyp_initial_v2 --dataset_dir data/processed/fyp_updated_same_angle \\
        --scene_name fyp_update_same_angle_dovsg

Outputs:
  outputs/dovsg_method/<scene_name>/      every step: images, numbers (summary.json, report.md)
  outputs/scene_graphs/<scene_name>.pkl/.json   the updated scene graph (same format as the initial run)
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import open3d as o3d
from PIL import Image

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
for p in (str(REPO_ROOT), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import viz  # noqa: E402
from ace_utils import ace_poses, read_K  # noqa: E402
from dovsg_core import (RGBFeatureMatch, anchor_poses, calculate_alignment_colored_icp,  # noqa: E402
                        find_need_to_delete_indexes, find_seen_through_voted, split_removed,
                        update_voxel_colors)
from navigation.voxel_map import PersistedVoxelMap  # noqa: E402
from object_fusion._dovsg_import import ensure_dovsg_importable  # noqa: E402
from scene_graph.storage.serializer import load_scene_graph, save_scene_graph  # noqa: E402
from scene_graph.storage.visualizer import save_scene_graph_visualization  # noqa: E402

logger = logging.getLogger("dovsg_update")

DEFAULT_CFG = {
    "dovsg_root": "/home/ravindu/SceneGraphNav/external/DovSG",
    "observation_stride": 1,        # every new frame: where the change happened is not known in advance
    "delete_rate": 0.5,             # DoVSG: removed when more than this share of an object's voxels is deleted
    "sim_threshold": 0.75,          # DoVSG InstanceProcess: a detection joins an object at 0.3 overlap + 0.7 cosine >= this
    "obj_min_detections": 2,        # DoVSG update_instance_objects(): get_instances(obj_min_detections=2)
    "lightglue_max_keypoints": 1024,  # DoVSG RGBFeatureMatch
    "viz_frames": 8,                # frames with per-frame pictures (evenly spaced); numbers cover all frames
    "scale_ransac_thresh_m": 0.05,  # --estimate_scale only
    # --robust_changes only: tuned on lap_in_fast (its own video deletes 6%, DoVSG's test 96%) and lap_remove
    "robust_rel_tol": 0.15,         # tolerance = max(abs, rel x voxel depth); monocular depth is off by ~5-10%
    "robust_abs_tol": 0.02,
    "robust_vote_share": 0.7,       # deleted when through votes > this share of the frames that see the voxel
    "robust_min_votes": 3,
    # --safeguards only (scripts/update_scene_graph.py)
    "redetect_min_detections": 3,
    "match_distance_m": 0.05,
    "novelty_distance_m": 0.05,
    "min_novelty": 0.4,
}


class UpdateViewDataset:
    """The new frames in the memory's voxel grid, with the ViewDataset API
    the fusion, relation and storage code read. Built like DoVSG's
    update_view_dataset(): each frame voxelized with its final pose and the
    memory's bounds (points outside the bounds are dropped, as DoVSG cannot
    grow its bounds either)."""

    def __init__(self, memory_map: PersistedVoxelMap, observations: dict):
        self.memory = memory_map
        self.resolution, self.bounds, self.voxel_num = memory_map.resolution, memory_map.bounds, memory_map.voxel_num
        self.names = [o["frame"] for o in observations.values()]
        self.length = len(self.names)
        self.append_length_log = [self.length]
        self.images, self.masks, self.pixel_index_mappings, self.pixel_index_masks = [], [], [], []
        lo, hi = self.bounds.lower_bound, self.bounds.higher_bound
        ix_all, rgb_all = [], []
        for o in observations.values():
            gpoint = o["point"] @ o["pose"][:3, :3].T + o["pose"][:3, 3]
            valid = o["mask"] & np.all((gpoint >= lo) & (gpoint < hi), axis=-1)
            mapping = -np.ones(o["mask"].shape, dtype=np.int64)
            mapping[valid] = memory_map.point_to_index(gpoint[valid])
            self.images.append((o["rgb"] * 255).astype(np.uint8))
            self.masks.append(o["mask"])
            self.pixel_index_mappings.append(mapping)
            self.pixel_index_masks.append(valid)
            ix_all.append(mapping[valid])
            rgb_all.append(o["rgb"][valid])
        ix, inv = np.unique(np.concatenate(ix_all), return_inverse=True)
        colour = np.zeros((len(ix), 3))
        np.add.at(colour, inv, np.concatenate(rgb_all))
        colour /= np.bincount(inv, minlength=len(ix))[:, None]
        self.observed = dict(zip(ix.tolist(), colour))
        self.indexes_colors_mapping_dict = dict(memory_map.indexes_colors_mapping_dict)

    def index_to_point(self, indexes):
        return self.memory.index_to_point(indexes)

    def point_to_index(self, points):
        return self.memory.point_to_index(points)

    def index_to_voxel(self, indexes):
        return self.memory.index_to_voxel(indexes)

    def voxel_to_index(self, voxels):
        return self.memory.voxel_to_index(voxels)

    def index_to_pcd(self, indexes):
        indexes = np.asarray(indexes)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.index_to_point(indexes))
        pcd.colors = o3d.utility.Vector3dVector(
            np.array([self.indexes_colors_mapping_dict.get(ix, (0.5, 0.5, 0.5)) for ix in indexes.tolist()]))
        return pcd


# ─── helpers ───

def rot_deg(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R[:3, :3]) - 1) / 2, -1, 1))))


def pose_delta(A, B):
    return round(rot_deg(np.linalg.inv(A) @ B), 2), round(float(np.linalg.norm(A[:3, 3] - B[:3, 3])), 3)


def frame_cloud(point, rgb, mask, pose, voxel=0.01):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector((point[mask] @ pose[:3, :3].T + pose[:3, 3]).astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(rgb[mask].astype(np.float64))
    return pcd.voxel_down_sample(voxel)


def load_observations(new_dir: Path, stride: int, scene_graph_generation_root: str) -> dict:
    """The new video's frames as DoVSG observations: rgb in [0, 1], depth (m),
    point (camera frame, m), mask, intrinsic, c2b = the video's own pose.
    The mask gets DoVSG's statistical outlier removal (use_inlier_mask=True)."""
    ensure_dovsg_importable(scene_graph_generation_root)
    from dovsg.utils.utils import get_inlier_mask

    K = read_K(new_dir)
    names = sorted(p.stem for p in (new_dir / "poses").glob("*.txt"))[::stride]
    observations = {}
    t0 = time.time()
    logger.info(f"[load] {len(names)} new frames from {new_dir} (outlier removal per frame)...")
    for k, name in enumerate(names):
        if k and k % 20 == 0:
            eta = (time.time() - t0) / k * (len(names) - k)
            logger.info(f"[load] {k}/{len(names)} frames, ~{eta:.0f} s left")
        rgb = np.asarray(Image.open(new_dir / "rgb" / f"{name}.jpg"), dtype=np.float32) / 255.0
        point = np.load(new_dir / "point" / f"{name}.npy").astype(np.float32)
        mask = np.load(new_dir / "mask" / f"{name}.npy").astype(bool)
        mask &= get_inlier_mask(point=point, color=rgb, mask=mask)
        observations[k] = {"frame": name, "rgb": rgb, "point": point, "depth": point[..., 2].astype(np.float64),
                           "mask": mask, "intrinsic": K, "c2b": np.loadtxt(new_dir / "poses" / f"{name}.txt")}
    return observations


def estimate_scale(observations, lg, memory_dir, memory_names, thresh):
    """Not DoVSG: a Sim3 (scale too) from the LightGlue matches between each
    new frame and its best memory frame, lifted to 3D with both depths."""
    from scripts.compare_video_poses import decompose_sim3, ransac_sim3

    src, dst = [], []
    for k, o in observations.items():
        b = lg[k]
        if b["matches"] < 8:
            continue
        name = memory_names[b["index"]]
        mp = np.load(memory_dir / "point" / f"{name}.npy").astype(np.float64)
        mm = np.load(memory_dir / "mask" / f"{name}.npy").astype(bool)
        mpose = np.loadtxt(memory_dir / "poses" / f"{name}.txt")
        un, vn = np.round(b["kp_new"]).astype(int).T
        um, vm = np.round(b["kp_memory"]).astype(int).T
        h, w = o["mask"].shape
        hm, wm = mm.shape
        ok = (un >= 0) & (un < w) & (vn >= 0) & (vn < h) & (um >= 0) & (um < wm) & (vm >= 0) & (vm < hm)
        un, vn, um, vm = un[ok], vn[ok], um[ok], vm[ok]
        ok = o["mask"][vn, un] & mm[vm, um]
        pn = o["point"][vn[ok], un[ok]].astype(np.float64)
        pm = mp[vm[ok], um[ok]]
        src.append(pn @ o["pose"][:3, :3].T + o["pose"][:3, 3])
        dst.append(pm @ mpose[:3, :3].T + mpose[:3, 3])
    src, dst = np.concatenate(src), np.concatenate(dst)
    T, inl = ransac_sim3(src, dst, thresh)
    s, R, t = decompose_sim3(T)
    for o in observations.values():
        o["point"] = o["point"] * s
        o["depth"] = o["depth"] * s
        pose = o["pose"].copy()
        o["pose"][:3, :3] = R @ pose[:3, :3]
        o["pose"][:3, 3] = s * R @ pose[:3, 3] + t
    return {"scale": round(s, 4), "rotation_deg": round(rot_deg(R), 2), "translation_m": round(float(np.linalg.norm(t)), 3),
            "matches_3d": int(len(src)), "inliers": int(inl.sum())}


# ─── main ───

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/pipeline_fyp_videos.yaml")
    ap.add_argument("--memory", required=True, help="initial run: outputs/scene_graphs/<memory>.pkl + data/processed/<memory>")
    ap.add_argument("--dataset_dir", required=True,
                    help="the new video processed with run_static_scene_graph.py --pose_only (rgb/point/mask/poses)")
    ap.add_argument("--scene_name", required=True, help="name of the updated scene graph")
    ap.add_argument("--observation_stride", type=int, default=None, help="use every Nth new frame (default 1: all)")
    ap.add_argument("--estimate_scale", action="store_true",
                    help="not DoVSG: correct the scale between the two runs before colored ICP")
    ap.add_argument("--safeguards", action="store_true",
                    help="not DoVSG: re-detection guard, novelty check, final merge among new objects only")
    ap.add_argument("--robust_changes", action="store_true",
                    help="not DoVSG: voted change test with a depth-relative tolerance (for estimated depth)")
    ap.add_argument("--force_update", action="store_true", help="continue when colored ICP fails (DoVSG stops)")
    args = ap.parse_args()

    from scripts.run_static_scene_graph import load_config, load_everyday_names

    cfg = load_config(REPO_ROOT / args.config)
    ucfg = {**DEFAULT_CFG, **(cfg.get("dovsg_update") or {})}
    if args.observation_stride:
        ucfg["observation_stride"] = args.observation_stride
    sgg_root = cfg["external"]["scene_graph_generation_root"]
    dovsg_root = Path(ucfg["dovsg_root"])
    out = REPO_ROOT / "outputs" / "dovsg_method" / args.scene_name
    if out.exists():
        shutil.rmtree(out)
    dirs = {k: out / k for k in ("1_ace", "2_lightglue", "3_colored_icp", "4_changes", "5_update")}
    for d in dirs.values():
        d.mkdir(parents=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(out / "run.log")])
    ensure_dovsg_importable(sgg_root)
    t_start, timings = time.time(), {}
    summary = {"memory": args.memory, "new_video": args.dataset_dir, "scene_name": args.scene_name,
               "estimate_scale": args.estimate_scale, "safeguards": args.safeguards,
               "robust_changes": args.robust_changes, "settings": ucfg, "steps": {}}

    def finish(stopped=None):
        if stopped:
            summary["stopped"] = stopped
            logger.error(f"Stopped: {stopped}")
        timings["total_s"] = round(time.time() - t_start, 1)
        summary["timings_s"] = timings
        (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
        (out / "report.md").write_text(report(summary))
        logger.info(f"Outputs: {out}")

    # ── memory ──
    t = time.time()
    memory_dir = REPO_ROOT / cfg["outputs"]["data_processed_dir"] / args.memory
    ace_model = memory_dir / "ace" / "ace.pt"
    if not ace_model.exists():
        sys.exit(f"{ace_model} not found: train it first with dynamic_updates/dovsg_method/train_ace.py --memory {args.memory}")
    memory = load_scene_graph(REPO_ROOT / cfg["outputs"]["scene_graphs_dir"] / f"{args.memory}.pkl")
    memory_map = PersistedVoxelMap.from_dict(memory["voxel_map"])
    memory_objects = memory["instance_objects"]
    mem_idx = np.fromiter(memory_map.indexes_colors_mapping_dict.keys(), dtype=np.int64)
    mem_pts = memory_map.index_to_point(mem_idx)
    mem_cols = np.array([memory_map.indexes_colors_mapping_dict[i] for i in mem_idx.tolist()], dtype=np.float64)
    memory_names = sorted(p.stem for p in (memory_dir / "rgb").glob("*.jpg"))
    new_dir = Path(args.dataset_dir)
    new_dir = new_dir if new_dir.is_absolute() else REPO_ROOT / new_dir
    observations = load_observations(new_dir, ucfg["observation_stride"], sgg_root)
    keys = list(observations)
    viz_keys = sorted({keys[i] for i in np.linspace(0, len(keys) - 1, min(ucfg["viz_frames"], len(keys))).round().astype(int)})
    timings["load_s"] = round(time.time() - t, 1)
    summary["memory_map"] = {"voxels": int(len(mem_idx)), "frames": len(memory_names), "objects": len(memory_objects)}
    logger.info(f"Memory '{args.memory}': {len(mem_idx)} voxels, {len(memory_objects)} objects; "
                f"{len(observations)} new frames from {new_dir}")

    # ── 1. ACE ──
    logger.info(f"[1 ACE] relocalizing {len(observations)} frames with {ace_model}...")
    t = time.time()
    for o, (pose, inliers) in zip(observations.values(), ace_poses(dovsg_root, ace_model, observations)):
        o["pose"], o["ace_inliers"], o["ace_pose"] = pose, inliers, pose.copy()
    timings["ace_s"] = round(time.time() - t, 1)
    np.save(dirs["1_ace"] / "ace_poses.npy", np.stack([o["ace_pose"] for o in observations.values()]))
    viz.bar(dirs["1_ace"] / "ace_inliers.png", [o["frame"] for o in observations.values()],
            [o["ace_inliers"] for o in observations.values()], None, "ACE: RANSAC inliers per new frame", "inliers")
    logger.info(f"[1 ACE] {timings['ace_s']}s; inliers median "
                f"{int(np.median([o['ace_inliers'] for o in observations.values()]))}")

    # ── 2. LightGlue anchor ──
    t = time.time()
    matcher = RGBFeatureMatch(str(dovsg_root / "third_party" / "LightGlue"), ucfg["lightglue_max_keypoints"])
    memory_images = [np.asarray(Image.open(memory_dir / "rgb" / f"{n}.jpg")) for n in memory_names]
    memory_features = [matcher.extract_feature(im) for im in memory_images]
    logger.info(f"[2 LightGlue] matching {len(observations)} new frames against {len(memory_images)} memory "
                f"frames ({len(observations) * len(memory_images)} pairs)...")
    lg, t_lg = {}, time.time()
    for n_done, (k, o) in enumerate(observations.items()):
        if n_done and n_done % 10 == 0:
            eta = (time.time() - t_lg) / n_done * (len(observations) - n_done)
            logger.info(f"[2 LightGlue] {n_done}/{len(observations)} frames, ~{eta:.0f} s left")
        lg[k] = matcher.find_most_similar_image((o["rgb"] * 255).astype(np.uint8), memory_features)
    matches = [lg[k]["matches"] for k in keys]
    anchor, b2w = anchor_poses(observations, matches)
    for o in observations.values():
        o["anchor_pose"] = o["pose"].copy()
    timings["lightglue_s"] = round(time.time() - t, 1)
    viz.bar(dirs["2_lightglue"] / "matches_per_frame.png", [o["frame"] for o in observations.values()], matches,
            keys.index(anchor), "LightGlue matches with the most similar memory frame (orange = anchor)", "matches")
    for k in sorted(set(viz_keys) | {anchor}):
        viz.match_pair(dirs["2_lightglue"] / f"frame_{observations[k]['frame']}{'_anchor' if k == anchor else ''}.png",
                       (observations[k]["rgb"] * 255).astype(np.uint8), memory_images[lg[k]["index"]],
                       lg[k]["kp_new"], lg[k]["kp_memory"],
                       f"new {observations[k]['frame']} vs memory {memory_names[lg[k]['index']]}")
    summary["steps"]["2_lightglue"] = {
        "anchor_frame": observations[anchor]["frame"], "anchor_matches": int(lg[anchor]["matches"]),
        "anchor_ace_inliers": observations[anchor]["ace_inliers"], "b2w": np.round(b2w, 6).tolist(),
        "frames": [{"frame": observations[k]["frame"], "best_memory_frame": memory_names[lg[k]["index"]],
                    "matches": int(lg[k]["matches"]), "ace_inliers": observations[k]["ace_inliers"]} for k in keys]}
    logger.info(f"[2 LightGlue] {timings['lightglue_s']}s; anchor frame {observations[anchor]['frame']} "
                f"({lg[anchor]['matches']} matches)")
    del memory_features

    # ── optional scale (not DoVSG) ──
    if args.estimate_scale:
        sc = estimate_scale(observations, lg, memory_dir, memory_names, ucfg["scale_ransac_thresh_m"])
        summary["steps"]["scale"] = sc
        logger.info(f"[scale] new video x {sc['scale']} to match the memory ({sc['inliers']}/{sc['matches_3d']} inliers)")

    # ── 3. colored ICP ──
    t = time.time()
    source, target = o3d.geometry.PointCloud(), o3d.geometry.PointCloud()
    for o in observations.values():
        source += frame_cloud(o["point"], o["rgb"], o["mask"], o["pose"])
    # DoVSG adds each frame's matched memory frame (repeats included) at full
    # resolution; with every new frame that is tens of millions of points, so
    # each matched memory frame is added once, at 1 cm (ICP's finest radius).
    for i in sorted({lg[k]["index"] for k in keys}):
        n = memory_names[i]
        target += frame_cloud(np.load(memory_dir / "point" / f"{n}.npy").astype(np.float32),
                              memory_images[i].astype(np.float32) / 255.0,
                              np.load(memory_dir / "mask" / f"{n}.npy").astype(bool),
                              np.loadtxt(memory_dir / "poses" / f"{n}.txt"))
    is_success, ref, icp_log = calculate_alignment_colored_icp(source, target)

    def fit(T, r):
        e = o3d.pipelines.registration.evaluate_registration(source.voxel_down_sample(r), target.voxel_down_sample(r), r, T)
        return {"fitness": round(e.fitness, 4), "inlier_rmse_mm": round(e.inlier_rmse * 1000, 2)}

    icp = {"success": bool(is_success), "correction_deg": round(rot_deg(ref), 2),
           "correction_m": round(float(np.linalg.norm(ref[:3, 3])), 3), "scales": icp_log,
           "before": {f"{r * 100:g}cm": fit(np.eye(4), r) for r in (0.02, 0.01)},
           "after": {f"{r * 100:g}cm": fit(ref, r) for r in (0.02, 0.01)}, "transform": np.round(ref, 6).tolist()}
    timings["colored_icp_s"] = round(time.time() - t, 1)
    viz.top_down_clouds(dirs["3_colored_icp"] / "top_down_before_after.png", np.asarray(target.points),
                        np.asarray(source.points), np.asarray(copy.deepcopy(source).transform(ref).points),
                        f"before ICP (ACE + anchor), fitness@2cm {icp['before']['2cm']['fitness']}",
                        f"after colored ICP, fitness@2cm {icp['after']['2cm']['fitness']}")
    if is_success:
        for o in observations.values():
            o["pose"] = ref @ o["pose"]
    for k in keys:
        o = observations[k]
        o["ace_error"] = pose_delta(o["ace_pose"], o["pose"])
    np.save(dirs["3_colored_icp"] / "final_poses.npy", np.stack([o["pose"] for o in observations.values()]))
    for k in viz_keys:
        o = observations[k]
        h, w = o["mask"].shape
        renders = [viz.render_voxels(mem_pts, mem_cols, o[p], o["intrinsic"], h, w) for p in ("ace_pose", "anchor_pose", "pose")]
        viz.pose_check(dirs["3_colored_icp"] / f"pose_check_{o['frame']}.png", o["rgb"], renders,
                       ["memory from ACE pose", "after LightGlue anchor", "after colored ICP"], o["frame"])
    summary["steps"]["1_ace"] = {"frames": [{"frame": o["frame"], "inliers": o["ace_inliers"],
                                             "error_vs_final_deg": o["ace_error"][0], "error_vs_final_m": o["ace_error"][1]}
                                            for o in observations.values()]}
    summary["steps"]["3_colored_icp"] = icp
    logger.info(f"[3 colored ICP] success={is_success}, correction {icp['correction_deg']} deg / {icp['correction_m']} m, "
                f"fitness@2cm {icp['before']['2cm']['fitness']} -> {icp['after']['2cm']['fitness']}")
    if not is_success and not args.force_update:
        return finish("colored ICP failed (DoVSG: 'relocalize error'); --force_update continues anyway")

    # ── 4. changes ──
    t = time.time()
    if args.robust_changes:
        deleted, through, confirm = find_seen_through_voted(
            mem_idx, mem_pts, observations, ucfg["robust_rel_tol"], ucfg["robust_abs_tol"],
            ucfg["robust_vote_share"], ucfg["robust_min_votes"])
    else:
        deleted, through = find_need_to_delete_indexes(mem_idx, mem_pts, mem_cols, observations, memory_map.resolution)
    survivors, removed_by_depth, share = split_removed(memory_objects, deleted, ucfg["delete_rate"])
    timings["changes_s"] = round(time.time() - t, 1)
    is_deleted = np.isin(mem_idx, deleted)
    through_of = dict(zip(mem_idx.tolist(), through.tolist()))
    objects_report = {}
    for obj in memory_objects:
        idx = np.asarray(obj["indexes"])
        frames_through = [through_of.get(i, 0) for i in idx.tolist()]
        objects_report[obj["class_id"]] = {
            "voxels": int(len(idx)), "deleted_share": share[obj["class_id"]],
            "removed": obj["class_id"] in removed_by_depth,
            # how many new frames saw through its deleted voxels (1 = a single frame was enough)
            "median_frames_through_deleted": float(np.median(
                [f for f, d in zip(frames_through, np.isin(idx, deleted)) if d] or [0])),
            "centroid": np.round(memory_map.index_to_point(idx).mean(0), 3).tolist()}
    viz.seen_through_map(dirs["4_changes"] / "seen_through_top_down.png", mem_pts, mem_cols, is_deleted, objects_report,
                         f"memory from above: {int(is_deleted.sum())} voxels seen through (red); label = share deleted")
    for k in viz_keys:
        o = observations[k]
        h, w = o["mask"].shape
        hit = viz.render_voxels(mem_pts[is_deleted], np.ones((int(is_deleted.sum()), 3)), o["pose"], o["intrinsic"], h, w)
        viz.frame_overlay(dirs["4_changes"] / f"seen_through_{o['frame']}.png", o["rgb"], hit.sum(2) > 0)
    summary["steps"]["4_changes"] = {"voxels_seen_through": int(is_deleted.sum()),
                                     "voxels_by_frames_through": {str(n): int((through[is_deleted] == n).sum()) for n in range(1, 6)},
                                     "voxels_seen_through_by_6_or_more_frames": int((through[is_deleted] >= 6).sum()),
                                     "removed_by_depth": removed_by_depth, "objects": objects_report}
    if args.robust_changes:
        seen = (through + confirm) > 0
        summary["steps"]["4_changes"]["robust"] = {
            "voxels_seen": int(seen.sum()), "voxels_with_a_through_vote": int((through > 0).sum()),
            "median_through_share_of_deleted": round(float(np.median(
                (through / np.maximum(through + confirm, 1))[is_deleted])) if is_deleted.any() else 0.0, 3)}
    logger.info(f"[4 changes] {int(is_deleted.sum())} memory voxels seen through; over delete_rate "
                f"{ucfg['delete_rate']}: {removed_by_depth or 'none'}")

    # ── 5. update: voxel map, objects, scene graph ──
    t = time.time()
    obs = UpdateViewDataset(memory_map, observations)
    obs.indexes_colors_mapping_dict = update_voxel_colors(memory_map.indexes_colors_mapping_dict, deleted, obs.observed)
    timings["update_voxel_map_s"] = round(time.time() - t, 1)

    t = time.time()
    from object_fusion.instance_fusion import build_instance_fusion_class
    from object_fusion.instance_tagger import tag_instances
    from object_fusion.streaming import detect_and_embed
    from perception.detection.yoloe_seg_detector import YOLOESegDetector
    from perception.visualization import save_detection_debug_frame
    from scripts.update_scene_graph import _embedding, describe_new_objects, keep_redetected, renumber, update_scene_graph
    from semantic_decoding.novic.novic_decoder import NOVICDecoder

    det_cfg, sem_cfg, fusion_cfg, label_cfg = cfg["detection"], cfg["semantic_decoding"], cfg["object_fusion"], cfg["label_mapper"]
    detector = YOLOESegDetector(weights_path=det_cfg["weights"], conf_thresh=det_cfg["conf_thresh"],
                                iou_thresh=det_cfg["iou_thresh"], img_size=det_cfg["img_size"], device=det_cfg["device"])
    novic = NOVICDecoder(novic_dir=sem_cfg["novic_dir"], checkpoint_path=sem_cfg["novic_checkpoint"],
                         gencfg=sem_cfg["gencfg"], guide_targets=sem_cfg["guide_targets"], device=sem_cfg["device"],
                         embedding_batch_size=sem_cfg.get("embedding_batch_size", 8))
    fusion = build_instance_fusion_class(sgg_root)(
        downsample_voxel_size=fusion_cfg["downsample_voxel_size"], spatial_weight=fusion_cfg["spatial_weight"],
        vis_weight=fusion_cfg["vis_weight"], sim_threshold=ucfg["sim_threshold"])
    # DoVSG: get_semantic_memory() on the new frames (here YOLOE + NOVIC), then
    # get_instances(objects=<surviving objects>) fuses them into the map
    fusion.start_new_scan(obs)
    per_frame = []
    for i, name in enumerate(obs.names):
        boxes, masks, confs, track_ids, embeddings = detect_and_embed(cfg, obs.images[i], detector, novic, False)
        save_detection_debug_frame(dirs["5_update"] / "detections", i, obs.images[i], boxes, confs, list(masks))
        per_frame.append(fusion.build_detection_list_from_frame(
            boxes=boxes, masks=masks, confs=confs, embeddings=embeddings, pixel_indexes=obs.pixel_index_mappings[i],
            pixel_indexes_mask=obs.pixel_index_masks[i], image_name=name, track_ids=track_ids))
    timings["detect_embed_s"] = round(time.time() - t, 1)

    t = time.time()
    if args.safeguards:  # removal candidates stay in, so the re-detection guard can see them
        removal_candidates = {o["class_id"]: o for o in memory_objects if o["class_id"] in removed_by_depth}
        initial = survivors + list(removal_candidates.values())
    else:
        initial = survivors
    fused, _ = fusion.get_instances(
        view_dataset=obs, per_frame_detection_lists=per_frame, obj_min_detections=ucfg["obj_min_detections"],
        initial_objects=initial, merge_overlapping="new_only" if args.safeguards else True,
        merge_sim_threshold=fusion_cfg["sim_threshold"] if args.safeguards else None, match_level="detection")
    removed_ids = list(removed_by_depth)
    if args.safeguards:
        removed_ids = keep_redetected(removed_ids, fused, memory_objects, {**ucfg})
        fused = [f for f in fused if not (f.get("memory_ids") and f["memory_ids"][0] in removed_ids)]

    by_id = {o["class_id"]: o for o in memory_objects}
    kept, merged_away, reobserved = [], {}, {}
    for f in fused:
        if not f.get("memory_ids"):
            continue
        mid = f["memory_ids"][0]
        m = dict(by_id[mid])
        m["indexes"] = list(np.asarray(f["indexes"]).tolist())  # DoVSG: geometry grows with the new detections
        new_det = int(f["num_detections"]) - int(by_id[mid].get("num_detections", 0))
        if new_det > 0:
            reobserved[mid] = new_det
            m["clip_ft"] = m["semantic_embedding"] = _embedding(f)
            m["num_detections"] = int(f["num_detections"])
        kept.append(m)
        for other in f["memory_ids"][1:]:
            merged_away[other] = mid
    new = [f for f in fused if not f.get("memory_ids") and int(f["num_detections"]) >= ucfg["obj_min_detections"]]
    for f in new:
        f["clip_ft"] = _embedding(f)
    new_report = []
    if args.safeguards and new:
        from scipy.spatial import cKDTree

        new_report = describe_new_objects(new, kept, obs, ucfg, cKDTree(mem_pts))
        new = [new[r["new_index"]] for r in new_report if r["novelty"] >= ucfg["min_novelty"]]
    added = list(tag_instances(
        list(new), novic, merge_similarity_threshold=label_cfg["merge_similarity_threshold"],
        min_name_score=label_cfg.get("min_name_score", 0.15), synonym_similarity=label_cfg.get("synonym_similarity", 0.8),
        everyday_names=load_everyday_names(cfg), everyday_name_weight=label_cfg.get("everyday_name_weight", 0.2),
        low_score_action="drop", manual_labels=label_cfg.get("manual_labels"))) if new else []
    renumber(added, set(by_id))
    kept_ids = {o["class_id"] for o in kept}
    gone = sorted(set(by_id) - kept_ids)  # removed by depth, merged into another object, or dropped by fusion
    objects = kept + added
    timings["fuse_objects_s"] = round(time.time() - t, 1)

    t = time.time()
    graph, orphans = update_scene_graph(memory["scene_graph"], gone, objects, obs, cfg)
    timings["update_scene_graph_s"] = round(time.time() - t, 1)
    novic.close()

    # ── save ──
    t = time.time()
    for o in objects:
        o.pop("mask", None)
    embeddings = {o["class_id"]: np.asarray(o["semantic_embedding"], np.float32) for o in objects if "semantic_embedding" in o}
    pkl_path, json_path = save_scene_graph(REPO_ROOT / cfg["outputs"]["scene_graphs_dir"], args.scene_name, graph,
                                           objects, embeddings, view_dataset=obs)
    save_scene_graph_visualization(memory["scene_graph"], dirs["5_update"] / "scene_graph_before")
    save_scene_graph_visualization(graph, dirs["5_update"] / "scene_graph_after")
    now_idx = np.fromiter(obs.indexes_colors_mapping_dict.keys(), dtype=np.int64)
    viz.changes_map(dirs["5_update"] / "changes_top_down.png", memory_map.index_to_point(now_idx),
                    np.array([obs.indexes_colors_mapping_dict[i] for i in now_idx.tolist()]),
                    {c: objects_report[c]["centroid"] for c in gone},
                    {o["class_id"]: memory_map.index_to_point(np.asarray(o["indexes"])).mean(0).tolist() for o in added},
                    f"updated map: {len(gone)} removed (red), {len(added)} added (green)")
    timings["save_s"] = round(time.time() - t, 1)

    summary["steps"]["5_update"] = {
        "removed": {c: ("seen through" if c in removed_by_depth else
                        f"merged into {merged_away[c]}" if c in merged_away else "dropped by fusion") for c in gone},
        "kept_after_redetection": sorted(set(removed_by_depth) - set(removed_ids)) if args.safeguards else [],
        "added": [{"node_id": o["class_id"], "parent": graph.object_nodes[o["class_id"]].parent.node_id,
                   "relation": graph.object_nodes[o["class_id"]].parent_relation,
                   "name_confidence": round(float(o.get("name_confidence", 0)), 3)} for o in added
                  if o["class_id"] in graph.object_nodes],
        "re_placed_after_removal": orphans, "unchanged": sorted(kept_ids), "reobserved_detections": reobserved,
        "new_object_candidates": new_report, "voxels_before": int(len(mem_idx)),
        "voxels_after": int(len(obs.indexes_colors_mapping_dict)), "saved": [str(pkl_path), str(json_path)]}
    logger.info(f"[5 update] removed {gone or 'nothing'}; added {[o['class_id'] for o in added] or 'nothing'}")
    finish()


def report(s: dict) -> str:
    st = s["steps"]
    L = [f"# DoVSG dynamic update: {s['scene_name']}", "",
         f"Memory `{s['memory']}`, new video `{s['new_video']}`. Scale step: {'on' if s['estimate_scale'] else 'off (DoVSG)'}; "
         f"safeguards: {'on' if s['safeguards'] else 'off (DoVSG)'}; change test: "
         f"{'voted, depth-relative (not DoVSG)' if s.get('robust_changes') else 'DoVSG (2 cm, one frame is enough)'}.", ""]
    if "1_ace" in st:
        f = st["1_ace"]["frames"]
        err_m = [x["error_vs_final_m"] for x in f]
        err_d = [x["error_vs_final_deg"] for x in f]
        L += ["## 1. ACE", "", f"{len(f)} frames. Inliers: median {int(np.median([x['inliers'] for x in f]))}, "
              f"min {min(x['inliers'] for x in f)}. ACE pose vs final pose: median {np.median(err_d):.1f}° / "
              f"{np.median(err_m):.3f} m, worst {max(err_d):.1f}° / {max(err_m):.3f} m.", "",
              "Pictures: `1_ace/ace_inliers.png`; `3_colored_icp/pose_check_*.png` (2nd panel = ACE pose).", ""]
    if "2_lightglue" in st:
        a = st["2_lightglue"]
        L += ["## 2. LightGlue anchor", "", f"Anchor frame `{a['anchor_frame']}`: {a['anchor_matches']} matches, "
              f"{a['anchor_ace_inliers']} ACE inliers. All other frames are placed from it with `b2w @ c2b`.", "",
              "Pictures: `2_lightglue/matches_per_frame.png`, `frame_*.png`.", ""]
    if "scale" in st:
        c = st["scale"]
        L += ["## Scale (not DoVSG)", "", f"New video x **{c['scale']}** ({c['inliers']}/{c['matches_3d']} inliers).", ""]
    if "3_colored_icp" in st:
        c = st["3_colored_icp"]
        L += ["## 3. Colored ICP", "", f"Success: **{c['success']}**. Correction {c['correction_deg']}° / {c['correction_m']} m.", "",
              "| radius | accepted | correspondences | needed |", "|---|---|---:|---:|"]
        L += [f"| {x['radius_m'] * 100:g} cm | {x['accepted']} | {x.get('correspondences', '-')} | {x.get('needed', '-')} |"
              for x in c["scales"]]
        L += ["", "| radius | fitness before | after | RMSE before (mm) | after |", "|---|---:|---:|---:|---:|"]
        L += [f"| {r} | {c['before'][r]['fitness']} | {c['after'][r]['fitness']} | {c['before'][r]['inlier_rmse_mm']} | "
              f"{c['after'][r]['inlier_rmse_mm']} |" for r in c["before"]]
        L += ["", "Pictures: `3_colored_icp/top_down_before_after.png`, `pose_check_*.png`.", ""]
    if "4_changes" in st:
        c = st["4_changes"]
        L += ["## 4. Detected changes", "", f"{c['voxels_seen_through']} memory voxels seen through "
              f"(by 1 frame: {c['voxels_by_frames_through']['1']}, by 6+ frames: {c['voxels_seen_through_by_6_or_more_frames']}).", "",
              "| object | voxels | deleted | frames through (median) | removed |", "|---|---:|---:|---:|---|"]
        if "robust" in c:
            r, cs = c["robust"], s["settings"]
            L[-3:-3] = [f"Voted test: tolerance max({cs['robust_abs_tol'] * 100:g} cm, {cs['robust_rel_tol']:.0%} of depth), "
                        f"deleted when ≥ {cs['robust_min_votes']} frames and > {cs['robust_vote_share']:.0%} of the frames "
                        f"that see a voxel see through it. {r['voxels_with_a_through_vote']} of {r['voxels_seen']} seen voxels "
                        f"got at least one through vote (DoVSG's test would delete all of those).", ""]
            L.pop(-3)  # the blank line before the inserted note
        for cid, v in sorted(c["objects"].items(), key=lambda x: -x[1]["deleted_share"]):
            L.append(f"| {cid} | {v['voxels']} | {v['deleted_share']:.0%} | {v['median_frames_through_deleted']:g} | "
                     f"{'**yes**' if v['removed'] else ''} |")
        L += ["", "Pictures: `4_changes/seen_through_top_down.png`, `seen_through_*.png`.", ""]
    if "5_update" in st:
        c = st["5_update"]
        L += ["## 5. Updated scene graph", ""]
        L += [f"- Removed `{k}`: {v}" for k, v in c["removed"].items()] or ["- Removed: nothing"]
        L += [f"- Added `{a['node_id']}` {a['relation']} `{a['parent']}` (name confidence {a['name_confidence']})"
              for a in c["added"]] or ["- Added: nothing"]
        if c["re_placed_after_removal"]:
            L.append(f"- Placed again (their support was removed): {', '.join(c['re_placed_after_removal'])}")
        L += [f"- Voxels: {c['voxels_before']} → {c['voxels_after']}", "",
              "Pictures: `5_update/changes_top_down.png`, `scene_graph_before/`, `scene_graph_after/`, `detections/`.",
              f"Saved: {', '.join(f'`{p}`' for p in c['saved'])}.", ""]
    if "stopped" in s:
        L += [f"**Stopped:** {s['stopped']}.", ""]
    L += ["## Time", "", "| step | s |", "|---|---:|"] + [f"| {k} | {v} |" for k, v in s["timings_s"].items()]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
