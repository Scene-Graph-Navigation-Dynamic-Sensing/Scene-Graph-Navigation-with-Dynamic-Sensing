#!/usr/bin/env python3
"""Scene update: the new video registered to the memory map, the memory's voxels
projected into every new frame to find what changed, and the new video's
objects found with the same pipeline that built the memory.

  1. ACE                 rough pose of every new frame in the memory map (DoVSG's ACE)
  2. LightGlue           each new frame matched against the memory frames nearest its ACE pose;
                         every match lifted to 3D with both depths -> one Sim3 (scale too) for the
                         whole new video (its own DROID-SLAM trajectory is kept)
  3. colored ICP         one rigid correction against the memory voxel map; kept only if it
                         improves the overlap. Frames that still do not fit are left out of step 4
  4. changes             memory voxels projected into every new frame, judged by depth and
                         colour (change_detection.py); an object most of whose seen voxels
                         changed is removed
  5. new video's objects scripts/run_static_scene_graph.py on the new video with its registered
                         poses: same detector, tracker, fusion, clean-up and naming as the memory
  6. update              each new object matched to a memory object (same place + same look),
                         or added (new space), or a removed object found elsewhere (moved);
                         voxel map and scene graph updated

Run in the scene_graph env, after dynamic_updates/dovsg_method/train_ace.py:

    CUDA_VISIBLE_DEVICES=1,0 python dynamic_updates/projection_method/run_projection_update.py \\
        --config configs/pipeline_lap_videos.yaml --memory lap_in_fast \\
        --dataset_dir data/processed/lap_remove --scene_name lap_remove_projection

Outputs:
  outputs/projection_method/<scene_name>/        every step: pictures, numbers, report.md, changes.mp4
  outputs/scene_graphs/<scene_name>.pkl/.json    the updated scene graph
  outputs/scene_graphs/<scene_name>_observation.pkl/.json   the new video's own scene graph (step 5)
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import multiprocessing
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import yaml
from PIL import Image

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
DOVSG_METHOD = HERE.parent / "dovsg_method"
for p in (str(REPO_ROOT), str(DOVSG_METHOD), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import viz  # noqa: E402
from ace_utils import ace_poses, read_K  # noqa: E402
from change_detection import (cluster_changes, object_status, project, stable_voxels,  # noqa: E402
                              vote_changes)

logger = logging.getLogger("projection_update")

DEFAULT_CFG = {
    "dovsg_root": str(REPO_ROOT / "external" / "DovSG"),
    "load_workers": 8,
    # 2. LightGlue
    "lightglue_max_keypoints": 1024,
    "lightglue_candidates": 8,          # memory frames matched per new frame: the nearest to its ACE pose
    "candidate_angle_weight_m": 0.5,    # nearness = centre distance (m) + this x viewing-direction angle (rad)
    "min_pair_matches": 30,             # a pair with fewer LightGlue matches is not used
    "sim3_ransac_thresh_m": 0.05,
    # 3. colored ICP and per-frame fit
    "frame_fit_distance_m": 0.03,       # a frame point fits when a memory voxel is this close
    "min_frame_fit": 0.3,               # frames that fit worse are left out of the change test
    # 4. changes (change_detection.py)
    "abs_tol_m": 0.02,
    "rel_tol": 0.15,                    # monocular depth is off by ~5-10% of the distance
    "colour_thresh": 0.2,               # RGB distance (0-1 scale, after the per-frame gain)
    "min_votes": 3,
    "vote_share": 0.7,
    "stable_confirm_share": 0.7,        # judged only where the memory's own frames confirm the voxel this often
    "min_seen_share": 0.2,              # an object the video saw less of than this is "not seen" and stays
    "remove_share": 0.5,                # removed when more than this share of its seen voxels changed
    "region_eps_m": 0.03,               # changed voxels grouped into regions (DBSCAN)
    "region_min_voxels": 100,
    # 6. update
    "match_threshold": 0.75,            # new object = memory object at 0.3 geometric + 0.7 cosine >= this
    "min_geometric": 0.2,               # ... and at least this share of it within match_distance_m
    "match_distance_m": 0.05,
    "novelty_distance_m": 0.05,
    "min_novelty": 0.4,                 # share of a new object in space the memory had empty (or changed)
    "moved_cosine": 0.85,               # a new object this similar to a removed one is that object, moved
    "min_saw_empty": 0.3,               # an added object is "new" when the memory saw this share of it as empty space
    "viz_frames": 8,
}


# ─── loading ───

def _load_frame(new_dir: str, name: str, sgg_root: str):
    """One new frame with DoVSG's outlier mask and an edge-free mask (process pool worker)."""
    from object_fusion._dovsg_import import ensure_dovsg_importable
    from scripts.compare_video_poses import edge_free_mask

    ensure_dovsg_importable(sgg_root)
    from dovsg.utils.utils import get_inlier_mask

    d = Path(new_dir)
    rgb = np.asarray(Image.open(d / "rgb" / f"{name}.jpg"), dtype=np.uint8)
    point = np.load(d / "point" / f"{name}.npy").astype(np.float32)
    mask = np.load(d / "mask" / f"{name}.npy").astype(bool)
    mask &= get_inlier_mask(point=point, color=rgb / 255.0, mask=mask)
    return rgb, point, mask, edge_free_mask(point, mask), np.loadtxt(d / "poses" / f"{name}.txt")


def load_new_frames(new_dir: Path, sgg_root: str, workers: int) -> dict:
    names = sorted(p.stem for p in (new_dir / "poses").glob("*.txt"))
    K = read_K(new_dir)
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        loaded = list(pool.map(_load_frame, [str(new_dir)] * len(names), names, [sgg_root] * len(names)))
    obs = {}
    for k, (name, (rgb, point, mask, valid, c2b)) in enumerate(zip(names, loaded)):
        obs[k] = {"frame": name, "rgb": rgb.astype(np.float32) / 255.0, "point": point, "mask": mask, "valid": valid,
                  "depth": point[..., 2].astype(np.float64), "intrinsic": K, "c2b": c2b}
    return obs


class MemoryFrames:
    """The memory video's frames (rgb, depth points, masks, poses in the map's world), loaded on demand."""

    def __init__(self, memory_dir: Path):
        self.dir = memory_dir
        self.names = sorted(p.stem for p in (memory_dir / "rgb").glob("*.jpg"))
        self.poses = np.stack([np.loadtxt(memory_dir / "poses" / f"{n}.txt") for n in self.names])
        self._geo = {}

    def rgb(self, j: int) -> np.ndarray:
        return np.asarray(Image.open(self.dir / "rgb" / f"{self.names[j]}.jpg"))

    def geometry(self, j: int):
        from scripts.compare_video_poses import edge_free_mask

        if j not in self._geo:
            point = np.load(self.dir / "point" / f"{self.names[j]}.npy").astype(np.float32)
            mask = np.load(self.dir / "mask" / f"{self.names[j]}.npy").astype(bool)
            self._geo[j] = (point, edge_free_mask(point, mask))
        return self._geo[j]


# ─── 2. LightGlue + Sim3 ───

def nearest_memory_frames(pose: np.ndarray, memory_poses: np.ndarray, k: int, angle_weight: float) -> np.ndarray:
    dist = np.linalg.norm(memory_poses[:, :3, 3] - pose[:3, 3], axis=1)
    cos = np.clip(memory_poses[:, :3, 2] @ pose[:3, 2], -1, 1)
    return np.argsort(dist + angle_weight * np.arccos(cos))[:k]


def lift(kp: np.ndarray, point: np.ndarray, valid: np.ndarray):
    """Keypoints (x, y) -> camera-frame 3D points where the depth is trustworthy."""
    h, w = valid.shape
    u = np.clip(np.round(kp[:, 0]).astype(int), 0, w - 1)
    v = np.clip(np.round(kp[:, 1]).astype(int), 0, h - 1)
    return point[v, u].astype(np.float64), valid[v, u]


def ransac_sim3_fast(src: np.ndarray, dst: np.ndarray, thresh: float, iters: int = 5000, score_n: int = 20000,
                     seed: int = 0):
    """compare_video_poses.ransac_sim3, scoring hypotheses on a random subset
    (with 100k+ matches, scoring all of them 20000 times took most of a minute);
    the final fit and inliers use every match."""
    from scripts.compare_video_poses import apply, umeyama

    rng = np.random.default_rng(seed)
    n = len(src)
    sub = rng.choice(n, min(n, score_n), replace=False)
    best, best_count = None, -1
    for _ in range(iters):
        idx = rng.choice(n, 3, replace=False)
        if np.linalg.norm(np.cross(src[idx[1]] - src[idx[0]], src[idx[2]] - src[idx[0]])) < 1e-4:
            continue
        T = umeyama(src[idx], dst[idx])
        if not 0.66 < np.cbrt(np.linalg.det(T[:3, :3])) < 1.5:
            continue
        count = int((np.linalg.norm(apply(T, src[sub]) - dst[sub], axis=1) < thresh).sum())
        if count > best_count:
            best, best_count = T, count
    T = best
    for _ in range(3):
        inl = np.linalg.norm(apply(T, src) - dst, axis=1) < thresh
        T = umeyama(src[inl], dst[inl])
    inl = np.linalg.norm(apply(T, src) - dst, axis=1) < thresh
    return T, inl


def lightglue_sim3(observations: dict, memory: MemoryFrames, ucfg: dict, dirs: dict, viz_keys: list) -> dict:
    from dovsg_core import RGBFeatureMatch
    from scripts.compare_video_poses import apply, decompose_sim3

    matcher = RGBFeatureMatch(str(Path(ucfg["dovsg_root"]) / "third_party" / "LightGlue"),
                              ucfg["lightglue_max_keypoints"])
    t = time.time()
    mem_feats = {}
    src, dst, owner, frames = [], [], [], []
    for n_done, (k, o) in enumerate(observations.items()):
        if n_done and n_done % 25 == 0:
            logger.info(f"[2 LightGlue] {n_done}/{len(observations)} frames ({time.time() - t:.0f}s)")
        f = matcher.extract_feature((o["rgb"] * 255).astype(np.uint8))
        rec = {"frame": o["frame"], "pairs": [], "matches_3d": 0}
        best = None
        for j in nearest_memory_frames(o["ace_pose"], memory.poses, ucfg["lightglue_candidates"],
                                       ucfg["candidate_angle_weight_m"]):
            j = int(j)
            if j not in mem_feats:
                mem_feats[j] = matcher.extract_feature(memory.rgb(j))
            r0, r1, m = matcher.match_features(f, mem_feats[j])
            pairs = m["matches"].cpu().numpy()
            rec["pairs"].append((memory.names[j], int(len(pairs))))
            if len(pairs) < ucfg["min_pair_matches"]:
                continue
            kp_new = r0["keypoints"].cpu().numpy()[pairs[:, 0]]
            kp_mem = r1["keypoints"].cpu().numpy()[pairs[:, 1]]
            if best is None or len(pairs) > len(best[1]):
                best = (j, kp_new, kp_mem)
            pn, okn = lift(kp_new, o["point"], o["valid"])
            mpt, mvalid = memory.geometry(j)
            pm, okm = lift(kp_mem, mpt, mvalid)
            ok = okn & okm
            if ok.sum() < 8:
                continue
            c2b, mpose = o["c2b"], memory.poses[j]
            src.append(pn[ok] @ c2b[:3, :3].T + c2b[:3, 3])
            dst.append(pm[ok] @ mpose[:3, :3].T + mpose[:3, 3])
            owner.append(np.full(int(ok.sum()), k))
            rec["matches_3d"] += int(ok.sum())
        rec["best_memory_frame"] = memory.names[best[0]] if best else None
        rec["best_matches"] = int(len(best[1])) if best else 0
        frames.append(rec)
        if k in viz_keys and best is not None:
            viz.match_pair(dirs["reloc"] / f"lightglue_{o['frame']}.png", (o["rgb"] * 255).astype(np.uint8),
                           memory.rgb(best[0]), best[1], best[2],
                           f"new {o['frame']} vs memory {memory.names[best[0]]}")
    lg_s = time.time() - t
    del matcher, mem_feats
    src, dst, owner = np.concatenate(src), np.concatenate(dst), np.concatenate(owner)
    t = time.time()
    T, inl = ransac_sim3_fast(src, dst, ucfg["sim3_ransac_thresh_m"])
    s, R, tr = decompose_sim3(T)
    err = np.linalg.norm(apply(T, src) - dst, axis=1)
    for rec, k in zip(frames, observations):
        mine = owner == k
        rec["sim3_inlier_share"] = round(float(inl[mine].mean()), 3) if mine.any() else None
        rec["sim3_median_error_m"] = round(float(np.median(err[mine])), 4) if mine.any() else None
    for o in observations.values():  # camera -> memory world, points scaled
        o["point"] = o["point"] * s
        o["depth"] = o["depth"] * s
        pose = np.eye(4)
        pose[:3, :3] = R @ o["c2b"][:3, :3]
        pose[:3, 3] = s * R @ o["c2b"][:3, 3] + tr
        o["pose"] = o["sim3_pose"] = pose
    return {"lightglue_s": round(lg_s, 1), "sim3_s": round(time.time() - t, 1),
            "pairs_matched": int(sum(len(r["pairs"]) for r in frames)), "matches_3d": int(len(src)),
            "inliers": int(inl.sum()), "scale": round(s, 4),
            "rotation_deg": round(float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))), 2),
            "translation_m": round(float(np.linalg.norm(tr)), 3),
            "median_error_inliers_mm": round(float(np.median(err[inl])) * 1000, 1), "frames": frames}


# ─── 3. colored ICP + per-frame fit ───

def frame_points_world(o: dict, n: int, rng) -> np.ndarray:
    pts = o["point"][o["valid"]]
    if len(pts) > n:
        pts = pts[rng.choice(len(pts), n, replace=False)]
    return pts.astype(np.float64) @ o["pose"][:3, :3].T + o["pose"][:3, 3]


def frame_fit(observations: dict, memory_tree, dist: float, n: int = 3000) -> dict:
    rng = np.random.default_rng(0)
    return {k: float(np.isfinite(memory_tree.query(frame_points_world(o, n, rng), k=1,
                                                   distance_upper_bound=dist)[0]).mean())
            for k, o in observations.items()}


def colored_icp(observations: dict, mem_pts: np.ndarray, mem_cols: np.ndarray):
    from dovsg_core import calculate_alignment_colored_icp
    from run_dovsg_update import frame_cloud

    source = o3d.geometry.PointCloud()
    for o in observations.values():
        source += frame_cloud(o["point"], o["rgb"], o["valid"], o["pose"])
    source = source.voxel_down_sample(0.01)
    target = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(mem_pts.astype(np.float64)))
    target.colors = o3d.utility.Vector3dVector(mem_cols.astype(np.float64))

    def fit(T):
        e = o3d.pipelines.registration.evaluate_registration(source, target, 0.02, T)
        return round(e.fitness, 4), round(e.inlier_rmse * 1000, 2)

    ok, T, log = calculate_alignment_colored_icp(source, target)
    before, after = fit(np.eye(4)), fit(T)
    used = bool(ok and after[0] > before[0])
    return (T if used else np.eye(4)), {"icp_success": bool(ok), "used": used, "fitness@2cm_before": before[0],
                                        "fitness@2cm_after": after[0], "rmse_mm_before": before[1],
                                        "rmse_mm_after": after[1], "scales": log}, source


# ─── 4. the memory's own frames: which voxels can be judged ───

def memory_stable_voxels(memory_dir: Path, mem_pts: np.ndarray, mem_cols: np.ndarray, ucfg: dict, sgg_root: str,
                         batch: int = 48) -> np.ndarray:
    """The change test run with the memory's own frames and poses; the voxels
    they confirm are the ones a new video can be judged against. Cached in
    <memory_dir>/projection_self_test.npz (keyed by the map size and settings)."""
    keys = ("abs_tol_m", "rel_tol", "colour_thresh", "min_votes", "stable_confirm_share")
    tag = json.dumps({"voxels": len(mem_pts), **{k: ucfg[k] for k in keys}}, sort_keys=True)
    cache = memory_dir / "projection_self_test.npz"
    if cache.exists():
        c = np.load(cache)
        if str(c["tag"]) == tag:
            logger.info(f"[4 changes] memory self-test reused from {cache}")
            return c["stable"]
    names = sorted(p.stem for p in (memory_dir / "poses").glob("*.txt"))
    K = read_K(memory_dir)
    total = {"through": 0, "colour": 0, "confirm": 0}
    with ProcessPoolExecutor(max_workers=ucfg["load_workers"], mp_context=multiprocessing.get_context("spawn")) as pool:
        for b in range(0, len(names), batch):
            chunk = names[b:b + batch]
            loaded = pool.map(_load_frame, [str(memory_dir)] * len(chunk), chunk, [sgg_root] * len(chunk))
            frames = [{"frame": n, "rgb": rgb.astype(np.float32) / 255.0, "depth": point[..., 2].astype(np.float64),
                       "valid": valid, "pose": pose, "K": K} for n, (rgb, point, _, valid, pose) in zip(chunk, loaded)]
            v = vote_changes(mem_pts, mem_cols, frames, ucfg)
            for k in total:
                total[k] = total[k] + v[k]
            logger.info(f"[4 changes] memory self-test: {min(b + batch, len(names))}/{len(names)} memory frames")
    stable = stable_voxels(total, ucfg["min_votes"], ucfg["stable_confirm_share"])
    np.savez_compressed(cache, stable=stable, tag=tag, **total)
    return stable


# ─── 4. pictures of the changes ───

def changed_in_frame(points, changed_kind, o, ccfg):
    """Changed voxels this frame sees (not hidden), as an overlay mask: 1 = empty now, 2 = colour changed."""
    h, w = o["valid"].shape
    ii, r, c, z = project(points, o["pose"], o["intrinsic"], h, w)
    keep = o["valid"][r, c] & (o["depth"][r, c] - z >= -np.maximum(ccfg["abs_tol_m"], ccfg["rel_tol"] * z))
    out = np.zeros((h, w), np.uint8)
    out[r[keep], c[keep]] = changed_kind[ii[keep]]
    return cv2.dilate(out, np.ones((3, 3), np.uint8))


def overlay(rgb: np.ndarray, kind: np.ndarray) -> np.ndarray:
    out = rgb.copy()
    for value, colour in ((1, (1.0, 0.0, 0.0)), (2, (1.0, 0.6, 0.0))):
        m = kind == value
        out[m] = 0.35 * out[m] + 0.65 * np.array(colour)
    return out


def label(img: np.ndarray, text: str) -> np.ndarray:
    img = (np.clip(img, 0, 1) * 255).astype(np.uint8).copy()
    cv2.rectangle(img, (0, 0), (img.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(img, text, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def changes_top_down(path, points, colours, kind, status, centroids, title):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 8))
    same = kind == 0
    ax.scatter(points[same, 0], points[same, 1], s=0.5, c=colours[same], linewidths=0)
    for value, colour, name in ((1, "red", "empty now"), (2, "orange", "colour changed")):
        m = kind == value
        ax.scatter(points[m, 0], points[m, 1], s=1.2, c=colour, linewidths=0, label=f"{name} ({int(m.sum())})")
    for cid, v in status.items():
        ax.annotate(f"{cid}\n{v['status']} {v['changed_share_of_seen']:.0%}", centroids[cid][:2], fontsize=7,
                    ha="center", color="red" if v["status"] == "removed" else "black")
    ax.set_aspect("equal")
    ax.legend(loc="upper right", markerscale=8, fontsize=8)
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ─── 5. the new video through the memory's pipeline ───

def prepare_observation_dataset(new_dir: Path, obs_dir: Path, observations: dict, scale: float) -> None:
    """The new video with its registered poses (memory world) and scaled depth,
    laid out for run_static_scene_graph.py --dataset_dir."""
    if obs_dir.exists():
        shutil.rmtree(obs_dir)
    for sub in ("rgb", "mask", "point", "poses"):
        (obs_dir / sub).mkdir(parents=True)
    for i, o in enumerate(observations.values()):
        name = o["frame"]
        assert name == f"{i:06d}", "the static pipeline reads frames 000000, 000001, ..."
        os.symlink((new_dir / "rgb" / f"{name}.jpg").resolve(), obs_dir / "rgb" / f"{name}.jpg")
        os.symlink((new_dir / "mask" / f"{name}.npy").resolve(), obs_dir / "mask" / f"{name}.npy")
        if abs(scale - 1.0) < 1e-6:
            os.symlink((new_dir / "point" / f"{name}.npy").resolve(), obs_dir / "point" / f"{name}.npy")
        else:
            np.save(obs_dir / "point" / f"{name}.npy",
                    (np.load(new_dir / "point" / f"{name}.npy") * scale).astype(np.float32))
        np.savetxt(obs_dir / "poses" / f"{name}.txt", o["pose"])
    shutil.copy2(new_dir / "metadata.json", obs_dir / "metadata.json")


def run_memory_pipeline(cfg: dict, config_out: Path, obs_dir: Path, obs_name: str, viz_dir: Path) -> None:
    run_cfg = copy.deepcopy(cfg)
    run_cfg["outputs"]["visualizations_dir"] = str(viz_dir)  # not the shared outputs/visualizations
    import torch

    sem = run_cfg["semantic_decoding"]  # extra embedders only on GPUs this process can see
    sem["extra_embedding_devices"] = [d for d in sem.get("extra_embedding_devices") or []
                                      if int(d.split(":")[1]) < torch.cuda.device_count()]
    config_out.write_text(yaml.safe_dump(run_cfg, sort_keys=False))
    cmd = [sys.executable, str(REPO_ROOT / "scripts" / "run_static_scene_graph.py"), "--config", str(config_out),
           "--dataset_dir", str(obs_dir), "--scene_name", obs_name, "--skip_gravity_alignment"]
    logger.info("[5 objects] " + " ".join(cmd))
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


# ─── 6. decisions ───

def _embedding(obj) -> np.ndarray:
    ft = obj.get("semantic_embedding", obj.get("clip_ft"))
    ft = ft.detach().cpu().numpy() if hasattr(ft, "detach") else np.asarray(ft)
    ft = ft.astype(np.float32).ravel()
    return ft / max(np.linalg.norm(ft), 1e-8)


def memory_saw_empty(points: np.ndarray, memory: "MemoryFrames", K: np.ndarray, ucfg: dict, stride: int = 2) -> float:
    """Share of `points` that the memory's own frames looked through (their
    depth more than tau behind the point in at least 2 frames): space the
    memory saw empty. The rest was hidden, out of view, or occupied."""
    through = np.zeros(len(points), np.int32)
    for j in range(0, len(memory.names), stride):
        mpt, mvalid = memory.geometry(j)
        h, w = mvalid.shape
        ii, r, c, z = project(points, memory.poses[j], K, h, w)
        ok = mvalid[r, c]
        ii, r, c, z = ii[ok], r[ok], c[ok], z[ok]
        tau = np.maximum(ucfg["abs_tol_m"], ucfg["rel_tol"] * z)
        through[ii[mpt[r, c, 2] - z > tau]] += 1
    return float((through >= 2).mean())


def decide(obs_objects, obs_pts, memory_objects, mem_obj_pts, status, remaining_tree, ucfg,
           saw_empty=lambda pts: 1.0) -> list[dict]:
    """Each object of the new video: matched to a memory object, new, moved, or already in the map."""
    from scipy.spatial import cKDTree

    trees = [cKDTree(p) for p in mem_obj_pts]
    mem_emb = np.stack([_embedding(o) for o in memory_objects])
    rows = []
    for k, (o, pts) in enumerate(zip(obs_objects, obs_pts)):
        cos = mem_emb @ _embedding(o)
        geo = np.array([np.isfinite(t.query(pts, k=1, distance_upper_bound=ucfg["match_distance_m"])[0]).mean()
                        for t in trees])
        score = 0.3 * geo + 0.7 * cos
        score_geo = np.where(geo >= ucfg["min_geometric"], score, -1)
        j = int(np.argmax(score_geo))
        d, _ = remaining_tree.query(pts, k=1, distance_upper_bound=ucfg["novelty_distance_m"])
        novelty = float(np.isinf(d).mean())
        row = {"index": k, "object": o["class_id"], "voxels": int(len(pts)), "detections": int(o["num_detections"]),
               "centroid": np.round(pts.mean(0), 3).tolist(), "novelty": round(novelty, 3),
               "best_memory": memory_objects[j]["class_id"], "cosine": round(float(cos[j]), 3),
               "geometric": round(float(geo[j]), 3), "score": round(float(score[j]), 3)}
        if score_geo[j] >= ucfg["match_threshold"]:
            row["decision"] = "matched"
            row["memory_id"] = memory_objects[j]["class_id"]
        elif novelty >= ucfg["min_novelty"]:
            removed = [i for i, m in enumerate(memory_objects) if status[m["class_id"]]["status"] == "removed"]
            best_removed = max(removed, key=lambda i: cos[i], default=None)
            if best_removed is not None and cos[best_removed] >= ucfg["moved_cosine"]:
                row["decision"] = "moved"
                row["memory_id"] = memory_objects[best_removed]["class_id"]
                row["cosine_to_removed"] = round(float(cos[best_removed]), 3)
            else:
                row["decision"] = "added"
                row["memory_saw_empty"] = round(saw_empty(pts), 3)
                # the memory looked through this space: the object is new; otherwise
                # the space was hidden (or out of view) in the memory video
                row["kind"] = ("new: the memory saw this space empty" if row["memory_saw_empty"] >= ucfg["min_saw_empty"]
                               else "newly seen: hidden or out of view in the memory video")
        else:
            row["decision"] = "in the memory map already"
            row["inside"] = memory_objects[int(np.argmax(geo))]["class_id"] if geo.max() > 0 else "background"
        rows.append(row)
    return rows


# ─── main ───

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/pipeline_lap_videos.yaml")
    ap.add_argument("--memory", required=True, help="initial run: outputs/scene_graphs/<memory>.pkl + data/processed/<memory>")
    ap.add_argument("--dataset_dir", required=True,
                    help="the new video processed with run_static_scene_graph.py --pose_only (rgb/point/mask/poses)")
    ap.add_argument("--scene_name", required=True, help="name of the updated scene graph")
    ap.add_argument("--reuse_observation", action="store_true",
                    help="skip step 5 when outputs/scene_graphs/<scene_name>_observation.pkl exists (same poses)")
    args = ap.parse_args()

    from scripts.run_static_scene_graph import load_config

    cfg = load_config(REPO_ROOT / args.config)
    ucfg = {**DEFAULT_CFG, **(cfg.get("projection_update") or {})}
    sgg_root = cfg["external"]["scene_graph_generation_root"]
    out = REPO_ROOT / "outputs" / "projection_method" / args.scene_name
    keep_obs = args.reuse_observation and (out / "5_observation").exists()
    if out.exists():
        for child in out.iterdir():
            if not (keep_obs and child.name == "5_observation"):
                shutil.rmtree(child) if child.is_dir() else child.unlink()
    dirs = {"reloc": out / "1-3_relocalization", "changes": out / "4_changes", "obs": out / "5_observation",
            "update": out / "6_update"}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(out / "run.log")])
    from object_fusion._dovsg_import import ensure_dovsg_importable

    ensure_dovsg_importable(sgg_root)
    from scipy.spatial import cKDTree

    from navigation.voxel_map import PersistedVoxelMap
    from scene_graph.storage.serializer import load_scene_graph, save_scene_graph
    from scene_graph.storage.visualizer import save_scene_graph_visualization

    t_start, timings = time.time(), {}
    summary = {"memory": args.memory, "new_video": args.dataset_dir, "scene_name": args.scene_name,
               "settings": ucfg, "steps": {}}

    def write_outputs():
        timings["total_s"] = round(time.time() - t_start, 1)
        summary["timings_s"] = timings
        (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
        (out / "report.md").write_text(report(summary))

    # ── memory and new video ──
    t = time.time()
    memory_dir = REPO_ROOT / cfg["outputs"]["data_processed_dir"] / args.memory
    ace_model = memory_dir / "ace" / "ace.pt"
    if not ace_model.exists():
        sys.exit(f"{ace_model} not found: train it first with dynamic_updates/dovsg_method/train_ace.py --memory {args.memory}")
    memory = load_scene_graph(REPO_ROOT / cfg["outputs"]["scene_graphs_dir"] / f"{args.memory}.pkl")
    memory_map = PersistedVoxelMap.from_dict(memory["voxel_map"])
    memory_objects = memory["instance_objects"]
    res = memory_map.resolution
    mem_idx = np.fromiter(memory_map.indexes_colors_mapping_dict.keys(), dtype=np.int64)
    mem_pts = memory_map.index_to_point(mem_idx) + res / 2  # voxel centres
    mem_cols = np.array([memory_map.indexes_colors_mapping_dict[i] for i in mem_idx.tolist()], dtype=np.float64)
    row_of = dict(zip(mem_idx.tolist(), range(len(mem_idx))))
    memory_frames = MemoryFrames(memory_dir)
    new_dir = Path(args.dataset_dir)
    new_dir = new_dir if new_dir.is_absolute() else REPO_ROOT / new_dir
    observations = load_new_frames(new_dir, sgg_root, ucfg["load_workers"])
    keys = list(observations)
    viz_keys = sorted({keys[i] for i in np.linspace(0, len(keys) - 1, min(ucfg["viz_frames"], len(keys))).round().astype(int)})
    timings["load_s"] = round(time.time() - t, 1)
    summary["memory_map"] = {"voxels": int(len(mem_idx)), "frames": len(memory_frames.names), "objects": len(memory_objects)}
    logger.info(f"Memory '{args.memory}': {len(mem_idx)} voxels, {len(memory_objects)} objects, "
                f"{len(memory_frames.names)} frames; {len(observations)} new frames ({timings['load_s']}s)")

    # ── 1. ACE ──
    t = time.time()
    for o, (pose, inliers) in zip(observations.values(), ace_poses(Path(ucfg["dovsg_root"]), ace_model, observations)):
        o["ace_pose"], o["ace_inliers"] = pose, inliers
    timings["ace_s"] = round(time.time() - t, 1)
    viz.bar(dirs["reloc"] / "1_ace_inliers.png", [o["frame"] for o in observations.values()],
            [o["ace_inliers"] for o in observations.values()], None, "ACE: RANSAC inliers per new frame", "inliers")
    logger.info(f"[1 ACE] {timings['ace_s']}s; inliers median {int(np.median([o['ace_inliers'] for o in observations.values()]))}")

    # ── 2. LightGlue -> Sim3 ──
    lg = lightglue_sim3(observations, memory_frames, ucfg, dirs, viz_keys)
    timings["lightglue_s"], timings["sim3_s"] = lg.pop("lightglue_s"), lg.pop("sim3_s")
    import torch

    torch.cuda.empty_cache()
    summary["steps"]["2_lightglue_sim3"] = lg
    viz.bar(dirs["reloc"] / "2_lightglue_matches.png", [r["frame"] for r in lg["frames"]],
            [r["best_matches"] for r in lg["frames"]], None,
            f"LightGlue: matches with the best of the {ucfg['lightglue_candidates']} nearest memory frames", "matches")
    logger.info(f"[2 LightGlue] {lg['pairs_matched']} pairs in {timings['lightglue_s']}s; Sim3 from {lg['inliers']}/"
                f"{lg['matches_3d']} 3D matches: scale {lg['scale']}, {lg['rotation_deg']} deg, {lg['translation_m']} m, "
                f"median error {lg['median_error_inliers_mm']} mm ({timings['sim3_s']}s)")

    # ── 3. colored ICP, per-frame fit ──
    t = time.time()
    mem_tree = cKDTree(mem_pts)
    T_icp, icp, source = colored_icp(observations, mem_pts, mem_cols)
    for o in observations.values():
        o["pose"] = T_icp @ o["pose"]
    fit = frame_fit(observations, mem_tree, ucfg["frame_fit_distance_m"])
    good = [k for k in keys if fit[k] >= ucfg["min_frame_fit"]]
    timings["colored_icp_s"] = round(time.time() - t, 1)
    icp.update({"correction_deg": round(float(np.degrees(np.arccos(np.clip((np.trace(T_icp[:3, :3]) - 1) / 2, -1, 1)))), 2),
                "correction_m": round(float(np.linalg.norm(T_icp[:3, 3])), 3),
                "frames_fit_median": round(float(np.median(list(fit.values()))), 3),
                "frames_used_for_changes": len(good), "frames_left_out": [observations[k]["frame"] for k in keys if k not in good]})
    summary["steps"]["3_colored_icp"] = icp
    for rec, k in zip(lg["frames"], keys):
        rec["fit_after_icp"] = round(fit[k], 3)
        rec["ace_vs_final_deg"] = round(float(np.degrees(np.arccos(np.clip(
            (np.trace(observations[k]["ace_pose"][:3, :3].T @ observations[k]["pose"][:3, :3]) - 1) / 2, -1, 1)))), 2)
        rec["ace_vs_final_m"] = round(float(np.linalg.norm(observations[k]["ace_pose"][:3, 3] - observations[k]["pose"][:3, 3])), 3)
    viz.top_down_clouds(dirs["reloc"] / "3_top_down_sim3_vs_icp.png", mem_pts[::3], np.asarray(source.points)[::3],
                        np.asarray(copy.deepcopy(source).transform(T_icp).points)[::3],
                        f"after LightGlue Sim3, fitness@2cm {icp['fitness@2cm_before']}",
                        f"after colored ICP ({'used' if icp['used'] else 'not used'}), fitness@2cm {icp['fitness@2cm_after']}")
    viz.bar(dirs["reloc"] / "3_frame_fit.png", [observations[k]["frame"] for k in keys], [fit[k] for k in keys], None,
            f"share of each frame's points within {ucfg['frame_fit_distance_m'] * 100:g} cm of the memory map "
            f"(< {ucfg['min_frame_fit']} left out of the change test)", "fit")
    for k in viz_keys:
        o = observations[k]
        h, w = o["valid"].shape
        renders = [viz.render_voxels(mem_pts, mem_cols, o[p], o["intrinsic"], h, w) for p in ("ace_pose", "sim3_pose", "pose")]
        viz.pose_check(dirs["reloc"] / f"pose_check_{o['frame']}.png", o["rgb"], renders,
                       ["memory from ACE pose", "after LightGlue Sim3", "after colored ICP"], o["frame"])
    logger.info(f"[3 colored ICP] used={icp['used']}, correction {icp['correction_deg']} deg / {icp['correction_m']} m, "
                f"fitness@2cm {icp['fitness@2cm_before']} -> {icp['fitness@2cm_after']}; {len(good)}/{len(keys)} frames "
                f"fit >= {ucfg['min_frame_fit']} (median {icp['frames_fit_median']})")

    np.save(dirs["reloc"] / "final_poses.npy", np.stack([observations[k]["pose"] for k in keys]))
    (dirs["reloc"] / "final_poses_frames.json").write_text(json.dumps([observations[k]["frame"] for k in keys]))

    # ── 4. changes: memory voxels projected into every new frame ──
    t = time.time()
    stable = memory_stable_voxels(memory_dir, mem_pts, mem_cols, ucfg, sgg_root)
    timings["memory_self_test_s"] = round(time.time() - t, 1)
    t = time.time()
    frames = [{"frame": observations[k]["frame"], "rgb": observations[k]["rgb"], "depth": observations[k]["depth"],
               "valid": observations[k]["valid"], "pose": observations[k]["pose"], "K": observations[k]["intrinsic"]}
              for k in good]
    votes = vote_changes(mem_pts, mem_cols, frames, ucfg)
    unstable_changed = int((votes["changed"] & ~stable).sum())
    votes["changed"] &= stable
    changed = votes["changed"]
    kind = np.where(changed, np.where(votes["through"] >= votes["colour"], 1, 2), 0).astype(np.uint8)
    status = object_status(memory_objects, row_of, votes, stable, ucfg)
    obj_of_row = np.full(len(mem_idx), -1)
    for n, obj in enumerate(memory_objects):
        obj_of_row[[row_of[i] for i in obj["indexes"] if i in row_of]] = n
    regions = []
    for members in cluster_changes(mem_pts, changed, ucfg["region_eps_m"], 10, ucfg["region_min_voxels"]):
        owners = obj_of_row[members]
        parts = {(memory_objects[o]["class_id"] if o >= 0 else "background"): int((owners == o).sum())
                 for o in np.unique(owners)}
        regions.append({"voxels": int(len(members)), "centroid": np.round(mem_pts[members].mean(0), 3).tolist(),
                        "extent_m": np.round(np.ptp(mem_pts[members], axis=0), 3).tolist(),
                        "empty_now_share": round(float((kind[members] == 1).mean()), 3),
                        "in": dict(sorted(parts.items(), key=lambda kv: -kv[1]))})
    timings["changes_s"] = round(time.time() - t, 1)
    centroids = {o["class_id"]: memory_map.index_to_point(np.asarray(o["indexes"])).mean(0) + res / 2 for o in memory_objects}
    seen = (votes["through"] + votes["colour"] + votes["confirm"]) > 0
    summary["steps"]["4_changes"] = {
        "voxels_stable_in_memory": int(stable.sum()), "voxels_in_memory": int(len(stable)),
        "changed_but_unstable_in_memory_ignored": unstable_changed,
        "voxels_seen": int(seen.sum()), "voxels_seen_and_stable": int((seen & stable).sum()),
        "voxels_changed": int(changed.sum()),
        "voxels_empty_now": int((kind == 1).sum()), "voxels_colour_changed": int((kind == 2).sum()),
        "objects": status, "changed_regions": regions, "per_frame": votes["per_frame"]}
    changes_top_down(dirs["changes"] / "changes_top_down.png", mem_pts, mem_cols, kind, status, centroids,
                     f"memory from above: {int((kind == 1).sum())} voxels empty now (red), {int((kind == 2).sum())} "
                     f"colour changed (orange), of {int((seen & stable).sum())} seen and stable")
    writer, video_path = None, dirs["changes"] / "changes.mp4"
    for k in keys:
        o = observations[k]
        h, w = o["valid"].shape
        mask = changed_in_frame(mem_pts[changed], kind[changed], o, ucfg)
        render = viz.render_voxels(mem_pts, mem_cols, o["pose"], o["intrinsic"], h, w)
        left = label(overlay(o["rgb"], mask), f"new {o['frame']}  red: empty now  orange: colour changed")
        right = label(render, f"memory from this pose (fit {fit[k]:.2f}{'' if k in good else ', not used'})")
        panel = np.concatenate([left, right], axis=1)
        if writer is None:
            writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 8, (panel.shape[1], panel.shape[0]))
        writer.write(cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
        if k in viz_keys:
            Image.fromarray(panel).save(dirs["changes"] / f"frame_{o['frame']}.png")
    writer.release()
    logger.info(f"[4 changes] {int(stable.sum())}/{len(stable)} memory voxels stable in the memory's own frames; "
                f"{int(changed.sum())} of {int((seen & stable).sum())} seen stable voxels changed "
                f"({int((kind == 1).sum())} empty now, {int((kind == 2).sum())} colour); {len(regions)} regions; "
                f"removed: {[c for c, v in status.items() if v['status'] == 'removed'] or 'nothing'} ({timings['changes_s']}s)")

    # ── 5. the new video's objects, same pipeline as the memory ──
    t = time.time()
    obs_name = f"{args.scene_name}_observation"
    obs_dir = REPO_ROOT / cfg["outputs"]["data_processed_dir"] / obs_name
    obs_pkl = REPO_ROOT / cfg["outputs"]["scene_graphs_dir"] / f"{obs_name}.pkl"
    if not (keep_obs and obs_pkl.exists()):
        prepare_observation_dataset(new_dir, obs_dir, observations, lg["scale"])
        run_memory_pipeline(cfg, dirs["obs"] / "config.yaml", obs_dir, obs_name, dirs["obs"] / "visualizations")
        log = REPO_ROOT / cfg["outputs"]["logs_dir"] / f"{obs_name}.log"
        if log.exists():
            shutil.copy2(log, dirs["obs"] / "pipeline.log")
    timings["observation_pipeline_s"] = round(time.time() - t, 1)
    observed = load_scene_graph(obs_pkl)
    obs_map = PersistedVoxelMap.from_dict(observed["voxel_map"])
    obs_objects = observed["instance_objects"]
    obs_pts = [obs_map.index_to_point(np.asarray(o["indexes"])) + obs_map.resolution / 2 for o in obs_objects]
    logger.info(f"[5 objects] the new video's own scene graph: {[o['class_id'] for o in obs_objects]} "
                f"({timings['observation_pipeline_s']}s)")

    # ── 6. update ──
    t = time.time()
    from run_dovsg_update import UpdateViewDataset

    from dovsg_core import update_voxel_colors
    from scripts.update_scene_graph import renumber, update_scene_graph

    # space the memory had occupied and still is: not changed, not part of a removed object
    removed_rows = np.zeros(len(mem_idx), bool)
    for obj in memory_objects:
        if status[obj["class_id"]]["status"] == "removed":
            removed_rows[[row_of[i] for i in obj["indexes"] if i in row_of]] = True
    remaining_tree = cKDTree(mem_pts[~changed & ~removed_rows])
    mem_obj_pts = [memory_map.index_to_point(np.asarray(o["indexes"])) + res / 2 for o in memory_objects]
    rows = decide(obs_objects, obs_pts, memory_objects, mem_obj_pts, status, remaining_tree, ucfg,
                  saw_empty=lambda pts: memory_saw_empty(pts, memory_frames, read_K(memory_dir), ucfg))
    redetected = {r["memory_id"] for r in rows if r["decision"] == "matched" and status[r["memory_id"]]["status"] == "removed"}
    moved = {}
    for r in rows:
        if r["decision"] == "moved" and r["memory_id"] not in moved:
            moved[r["memory_id"]] = r
    removed = [c for c, v in status.items() if v["status"] == "removed" and c not in redetected and c not in moved]

    def in_grid(pts):
        lo, hi = memory_map.bounds.lower_bound, memory_map.bounds.higher_bound
        ok = np.all((pts >= lo) & (pts < hi), axis=1)
        return np.unique(memory_map.point_to_index(pts[ok])).tolist(), int((~ok).sum())

    changed_idx = set(mem_idx[changed].tolist())
    objects, dropped_outside = [], {}
    for mo in memory_objects:
        cid = mo["class_id"]
        if cid in removed:
            continue
        m = dict(mo)
        if cid in moved:
            r = moved[cid]
            m["indexes"], dropped_outside[cid] = in_grid(obs_pts[r["index"]])
            m["moved_from"] = np.round(centroids[cid], 3).tolist()
            m["moved_to"] = r["centroid"]
        elif cid not in redetected:
            m["indexes"] = [i for i in mo["indexes"] if i not in changed_idx] or list(mo["indexes"])
        objects.append(m)
    added, added_rows = [], []
    for r in rows:
        if r["decision"] != "added":
            continue
        o = dict(obs_objects[r["index"]])
        o["indexes"], dropped_outside[r["object"]] = in_grid(obs_pts[r["index"]])
        if o["indexes"]:
            added.append(o)
            added_rows.append(r)
    renumber(added, {o["class_id"] for o in memory_objects})
    for o, r in zip(added, added_rows):
        r["added_as"] = o["class_id"]
    objects += added

    vd = UpdateViewDataset(memory_map, {k: {**observations[k], "mask": observations[k]["valid"]} for k in keys})
    vd.indexes_colors_mapping_dict = update_voxel_colors(memory_map.indexes_colors_mapping_dict,
                                                         mem_idx[changed], vd.observed)
    gone = removed + list(moved)
    graph, orphans = update_scene_graph(memory["scene_graph"], gone, objects, vd, cfg)
    timings["update_s"] = round(time.time() - t, 1)

    t = time.time()
    for o in objects:
        o.pop("mask", None)
    embeddings = {o["class_id"]: np.asarray(o["semantic_embedding"], np.float32) for o in objects if "semantic_embedding" in o}
    pkl_path, json_path = save_scene_graph(REPO_ROOT / cfg["outputs"]["scene_graphs_dir"], args.scene_name, graph,
                                           objects, embeddings, view_dataset=vd)
    save_scene_graph_visualization(memory["scene_graph"], dirs["update"] / "scene_graph_before")
    save_scene_graph_visualization(graph, dirs["update"] / "scene_graph_after")
    now_idx = np.fromiter(vd.indexes_colors_mapping_dict.keys(), dtype=np.int64)
    viz.changes_map(dirs["update"] / "changes_map.png", memory_map.index_to_point(now_idx)[::2],
                    np.array([vd.indexes_colors_mapping_dict[i] for i in now_idx.tolist()])[::2],
                    {**{c: centroids[c] for c in removed}, **{f"{c} (moved from)": centroids[c] for c in moved}},
                    {**{o["class_id"]: memory_map.index_to_point(np.asarray(o["indexes"])).mean(0) for o in added},
                     **{f"{c} (moved to)": np.asarray(moved[c]["centroid"]) for c in moved}},
                    f"updated map: {len(removed)} removed, {len(moved)} moved (red -> green), {len(added)} added")
    timings["save_s"] = round(time.time() - t, 1)
    summary["steps"]["5_observation"] = {"scene_graph": str(obs_pkl), "objects": [o["class_id"] for o in obs_objects]}
    summary["steps"]["6_update"] = {
        "removed": removed, "moved": {c: {"from": np.round(centroids[c], 3).tolist(), "to": r["centroid"],
                                          "new_video_object": r["object"], "cosine": r["cosine_to_removed"]}
                                      for c, r in moved.items()},
        "kept_because_redetected": sorted(redetected), "added": [o["class_id"] for o in added],
        "new_video_objects": rows, "voxels_outside_memory_bounds_dropped": dropped_outside,
        "re_placed_after_removal": orphans, "voxels_before": int(len(mem_idx)),
        "voxels_after": int(len(vd.indexes_colors_mapping_dict)), "saved": [str(pkl_path), str(json_path)]}
    logger.info(f"[6 update] removed {removed or 'nothing'}; moved {list(moved) or 'nothing'}; "
                f"added {[o['class_id'] for o in added] or 'nothing'}; kept (re-detected) {sorted(redetected) or 'none'}")
    write_outputs()
    logger.info(f"Outputs: {out}")


def report(s: dict) -> str:
    st, tm = s["steps"], s.get("timings_s", {})
    L = [f"# Scene update (projection method): {s['scene_name']}", "",
         f"Memory `{s['memory']}`, new video `{s['new_video']}`.", ""]
    if "2_lightglue_sim3" in st:
        g = st["2_lightglue_sim3"]
        L += ["## 1-3. Relocalization", "",
              f"LightGlue: {g['pairs_matched']} frame pairs (each new frame vs the {s['settings']['lightglue_candidates']} "
              f"memory frames nearest its ACE pose) -> {g['matches_3d']} 3D matches, {g['inliers']} Sim3 inliers.",
              f"Sim3: scale {g['scale']}, {g['rotation_deg']}°, {g['translation_m']} m; median error {g['median_error_inliers_mm']} mm.", ""]
    if "3_colored_icp" in st:
        c = st["3_colored_icp"]
        L += [f"Colored ICP: {'used' if c['used'] else 'not used'} (correction {c['correction_deg']}° / {c['correction_m']} m), "
              f"fitness@2cm {c['fitness@2cm_before']} -> {c['fitness@2cm_after']}. "
              f"{c['frames_used_for_changes']} frames fit well enough for the change test (median fit {c['frames_fit_median']}).", "",
              "Pictures: `1-3_relocalization/pose_check_*.png` (memory rendered from each pose next to the frame), "
              "`lightglue_*.png`, `3_top_down_sim3_vs_icp.png`, `3_frame_fit.png`.", ""]
    if "4_changes" in st:
        c = st["4_changes"]
        L += ["## 4. Changes (memory voxels projected into the new frames)", "",
              f"The memory's own frames confirm {c['voxels_stable_in_memory']} of its {c['voxels_in_memory']} voxels; only "
              f"those are judged ({c['changed_but_unstable_in_memory_ignored']} voxels the new video would have called "
              "changed are unstable in the memory itself and ignored).", "",
              f"{c['voxels_changed']} of {c['voxels_seen_and_stable']} seen stable voxels changed: {c['voxels_empty_now']} "
              f"empty now, {c['voxels_colour_changed']} colour changed.", "",
              "| memory object | voxels | stable in memory | seen (of stable) | changed of seen | status |",
              "|---|---:|---:|---:|---:|---|"]
        for cid, v in c["objects"].items():
            L.append(f"| {cid} | {v['voxels']} | {v['stable_in_memory']:.0%} | {v['seen_share']:.0%} | "
                     f"{v['changed_share_of_seen']:.0%} | {v['status']} |")
        L += ["", "Changed regions (largest first):", "", "| voxels | centroid | extent (m) | empty now | in |", "|---:|---|---|---:|---|"]
        for r in c["changed_regions"][:15]:
            L.append(f"| {r['voxels']} | {r['centroid']} | {r['extent_m']} | {r['empty_now_share']:.0%} | "
                     + ", ".join(f"{k} {v}" for k, v in r["in"].items()) + " |")
        L += ["", "Pictures: `4_changes/changes.mp4` (every frame: changed voxels on the new frame | memory from the same pose), "
              "`changes_top_down.png`, `frame_*.png`.", ""]
    if "6_update" in st:
        u = st["6_update"]
        L += ["## 5-6. The new video's objects (same pipeline as the memory) and the update", "",
              "| new video object | detections | decision | memory object | cosine | geometric | novelty |",
              "|---|---:|---|---|---:|---:|---:|"]
        for r in u["new_video_objects"]:
            dec = r["decision"] + (f" → {r['added_as']}" if r.get("added_as") else "") + \
                (f" ({r['kind']}, {r['memory_saw_empty']:.0%})" if r.get("kind") else "") + \
                (f" (inside {r['inside']})" if r.get("inside") else "")
            L.append(f"| {r['object']} | {r['detections']} | {dec} | {r.get('memory_id', r['best_memory'])} | "
                     f"{r['cosine']} | {r['geometric']} | {r['novelty']} |")
        moved = ", ".join(f"{c} {v['from']} -> {v['to']}" for c, v in u["moved"].items())
        L += ["", f"- Removed: {', '.join(u['removed']) or 'nothing'}",
              f"- Moved: {moved or 'nothing'}",
              f"- Added: {', '.join(u['added']) or 'nothing'}",
              f"- Kept although the change test removed them (re-detected in place): {', '.join(u['kept_because_redetected']) or 'none'}",
              f"- Voxels: {u['voxels_before']} -> {u['voxels_after']}", "",
              "Pictures: `6_update/changes_map.png`, `scene_graph_before/`, `scene_graph_after/`; the new video's own "
              "scene graph and its pipeline log in `5_observation/`.", ""]
    L += ["## Time", "", "| step | s |", "|---|---:|"] + [f"| {k} | {v} |" for k, v in tm.items()]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
