"""Speed-ups for an update after recording: detection only where it is needed,
frames loaded without the per-frame outlier removal, and a map update that
touches only the changed objects.

Detection on the new video serves one purpose in this method: §8.2, telling
moved / removed / added apart, and confirming an absent object in place. So it
is needed only near
  - objects the evidence did not call present (absent or deferred), and
  - new geometry: new-frame points far from every memory voxel, seen in several
    frames (an added object, or a moved one at its new place).
Around each such place a short window of consecutive frames is detected (the
tracker needs consecutive frames). With nothing absent, deferred or new, the
detector does not run at all.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

logger = logging.getLogger("covis_gate")


# ─── loading ───

def _load_one(d: Path, name: str):
    from scripts.compare_video_poses import edge_free_mask

    rgb = np.asarray(Image.open(d / "rgb" / f"{name}.jpg"), dtype=np.uint8)
    point = np.load(d / "point" / f"{name}.npy").astype(np.float32)
    mask = np.load(d / "mask" / f"{name}.npy").astype(bool)
    return rgb, point, mask, edge_free_mask(point, mask), np.loadtxt(d / "poses" / f"{name}.txt")


def load_frames(new_dir: Path, workers: int) -> dict:
    """The new frames as projection_method.load_new_frames() returns them, but
    without DoVSG's statistical outlier removal (~1 s per frame): the change
    test uses the edge-free mask, which drops the flying pixels at depth edges.
    o["mask"] is the plain valid-depth mask; inlier_masks() refines it for the
    few frames whose points go into the map."""
    from ace_utils import read_K

    names = sorted(p.stem for p in (new_dir / "poses").glob("*.txt"))
    K = read_K(new_dir)
    with ThreadPoolExecutor(workers) as pool:
        loaded = list(pool.map(lambda n: _load_one(new_dir, n), names))
    return {k: {"frame": n, "rgb": rgb.astype(np.float32) / 255.0, "point": point, "mask": mask, "valid": valid,
                "depth": point[..., 2].astype(np.float64), "intrinsic": K, "c2b": c2b}
            for k, (n, (rgb, point, mask, valid, c2b)) in enumerate(zip(names, loaded))}


# ─── where detection is needed ───

def novel_regions(frames: list, mem_tree, bounds, tcfg: dict) -> dict:
    """New geometry: points of the new frames farther than novel_distance_m from
    every memory voxel, on a novel_voxel_m grid, kept where at least
    novel_min_frames frames saw them, then grouped (DBSCAN). Returns the regions
    (largest first) and, per region, the frame seeing most of it."""
    import open3d as o3d

    rng = np.random.default_rng(0)
    vox = tcfg["novel_voxel_m"]
    lo, hi = np.asarray(bounds.lower_bound), np.asarray(bounds.higher_bound)
    per_frame = []
    for o in frames:
        pts = o["point"][o["valid"]]
        if len(pts) > tcfg["novel_points_per_frame"]:
            pts = pts[rng.choice(len(pts), tcfg["novel_points_per_frame"], replace=False)]
        w = pts.astype(np.float64) @ o["pose"][:3, :3].T + o["pose"][:3, 3]
        w = w[np.all((w >= lo) & (w < hi), axis=1)]
        d, _ = mem_tree.query(w, k=1, distance_upper_bound=tcfg["novel_distance_m"])
        per_frame.append(np.unique(np.floor(w[np.isinf(d)] / vox).astype(np.int64), axis=0))
    if not per_frame or not any(len(p) for p in per_frame):
        return {"regions": []}
    allv = np.concatenate([p for p in per_frame if len(p)])
    keys, support = np.unique(allv, axis=0, return_counts=True)
    keys = keys[support >= tcfg["novel_min_frames"]]
    if len(keys) < tcfg["novel_min_voxels"]:
        return {"regions": []}
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector((keys + 0.5) * vox))
    labels = np.asarray(pcd.cluster_dbscan(eps=1.6 * vox, min_points=4))
    regions = []
    for lab in range(labels.max() + 1 if len(labels) else 0):
        mine = keys[labels == lab]
        if len(mine) < tcfg["novel_min_voxels"]:
            continue
        mine_set = {tuple(v) for v in mine.tolist()}
        seen = [sum(tuple(v) in mine_set for v in p.tolist()) for p in per_frame]
        regions.append({"voxels": int(len(mine)), "centroid": np.round((mine + 0.5).mean(0) * vox, 3).tolist(),
                        "best_frame": int(np.argmax(seen)), "frames_seeing": int(np.count_nonzero(seen))})
    regions.sort(key=lambda r: -r["voxels"])
    return {"regions": regions}


def detection_windows(n_frames: int, centres: list, tcfg: dict) -> list:
    """Windows of detect_window consecutive frames around each centre (frame
    index in the whole video), at most detect_max_windows, merged when they
    overlap. Returns the sorted frame indexes."""
    half = tcfg["detect_window"] // 2
    chosen = set()
    for c in centres[:tcfg["detect_max_windows"]]:
        a = max(0, min(c - half, n_frames - tcfg["detect_window"]))
        chosen.update(range(a, min(n_frames, a + tcfg["detect_window"])))
    return sorted(chosen)


def prepare_subset_dataset(new_dir: Path, obs_dir: Path, observations: dict, keys: list, scale: float) -> list:
    """The chosen frames, renumbered 000000, 000001, ... (the static pipeline
    reads them in that order), with their registered poses and scaled depth.
    Returns the original frame names in the new order."""
    if obs_dir.exists():
        shutil.rmtree(obs_dir)
    for sub in ("rgb", "mask", "point", "poses"):
        (obs_dir / sub).mkdir(parents=True)
    names = []
    for i, k in enumerate(keys):
        o = observations[k]
        new = f"{i:06d}"
        names.append(o["frame"])
        os.symlink((new_dir / "rgb" / f"{o['frame']}.jpg").resolve(), obs_dir / "rgb" / f"{new}.jpg")
        os.symlink((new_dir / "mask" / f"{o['frame']}.npy").resolve(), obs_dir / "mask" / f"{new}.npy")
        np.save(obs_dir / "point" / f"{new}.npy", (np.load(new_dir / "point" / f"{o['frame']}.npy") * scale).astype(np.float32))
        np.savetxt(obs_dir / "poses" / f"{new}.txt", o["pose"])
    meta = json.loads((new_dir / "metadata.json").read_text())
    meta["length"] = len(keys)
    (obs_dir / "metadata.json").write_text(json.dumps(meta))
    (obs_dir / "source_frames.json").write_text(json.dumps(names))
    return names


# ─── map update that touches only the changed objects ───

class ChangedMap:
    """The ViewDataset API the graph builder and the saver read (index_to_point,
    resolution, bounds, colours), on the memory's grid: the memory's voxels,
    minus those of removed and moved objects, plus those of added and moved
    objects with the colours the new video gave them. No frame is voxelised."""

    def __init__(self, memory_map, gone_idx, new_colours: dict):
        self.memory = memory_map
        self.resolution, self.bounds, self.voxel_num = memory_map.resolution, memory_map.bounds, memory_map.voxel_num
        gone = set(int(i) for i in gone_idx)
        self.indexes_colors_mapping_dict = {ix: c for ix, c in memory_map.indexes_colors_mapping_dict.items()
                                            if ix not in gone}
        self.indexes_colors_mapping_dict.update(new_colours)
        self.observed = new_colours

    def index_to_point(self, indexes):
        return self.memory.index_to_point(indexes)

    def point_to_index(self, points):
        return self.memory.point_to_index(points)

    def index_to_voxel(self, indexes):
        return self.memory.index_to_voxel(indexes)

    def voxel_to_index(self, voxels):
        return self.memory.voxel_to_index(voxels)


def object_in_memory_grid(memory_map, obs_map, obs_obj) -> tuple[list, dict]:
    """A new-video object's voxels moved onto the memory's grid, with their colours."""
    idx = np.asarray(obs_obj["indexes"])
    pts = obs_map.index_to_point(idx) + obs_map.resolution / 2
    cols = np.array([obs_map.indexes_colors_mapping_dict.get(int(i), (0.5, 0.5, 0.5)) for i in idx.tolist()])
    lo, hi = memory_map.bounds.lower_bound, memory_map.bounds.higher_bound
    ok = np.all((pts >= lo) & (pts < hi), axis=1)
    new_idx = memory_map.point_to_index(pts[ok])
    uniq, inv = np.unique(new_idx, return_inverse=True)
    colour = np.zeros((len(uniq), 3))
    np.add.at(colour, inv, cols[ok])
    colour /= np.bincount(inv, minlength=len(uniq))[:, None]
    return uniq.tolist(), dict(zip(uniq.tolist(), colour))
