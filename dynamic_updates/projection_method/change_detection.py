"""Memory voxels projected into every new frame, judged by depth and colour.

Each frame votes on every memory voxel it can see (in front of the camera,
inside the image, on a pixel with trustworthy depth):

  through   the new depth is more than tau behind the voxel: the space is empty now
  colour    the new depth agrees (within tau) but the colour does not: something
            else is there now
  confirm   depth and colour agree
  (hidden)  the new depth is more than tau in front of it: something is in the way,
            no vote

tau = max(abs_tol, rel_tol x the voxel's depth), because monocular depth is off
by a few percent of the distance. The colour is compared after a per-frame
gain (median ratio of new to memory colour over the depth-confirmed voxels),
so a change of exposure between the videos does not count as a change.

A voxel has changed when at least min_votes frames voted through or colour and
they are more than vote_share of the frames that voted on it.

Monocular depth differs from frame to frame, so the memory's own surfaces are
layers several centimetres thick and their voxel colours are averages; a new
view sees through the front of such a layer, or a colour no single view had,
without anything having changed. So the same test is first run with the
memory's own frames (stable_voxels()): only voxels its own frames confirm are
judged in the new video.
"""
from __future__ import annotations

import numpy as np


def project(points: np.ndarray, pose: np.ndarray, K: np.ndarray, h: int, w: int):
    """World points -> (indexes of the points inside the image, their rows, cols, depths)."""
    inv = np.linalg.inv(pose)
    pc = points @ inv[:3, :3].T + inv[:3, 3]
    z = pc[:, 2]
    ii = np.where(z > 0.05)[0]
    u = np.round(K[0, 0] * pc[ii, 0] / z[ii] + K[0, 2]).astype(np.int64)
    v = np.round(K[1, 1] * pc[ii, 1] / z[ii] + K[1, 2]).astype(np.int64)
    inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    return ii[inside], v[inside], u[inside], z[ii[inside]]


def vote_changes(points: np.ndarray, colours: np.ndarray, frames: list[dict], ccfg: dict) -> dict:
    """frames: dicts with rgb (h, w, 3) in [0, 1], depth (h, w) in m, valid (h, w),
    pose (camera to memory world), K. Returns per-voxel vote counts, the
    changed mask, and per-frame numbers."""
    n = len(points)
    through = np.zeros(n, np.int32)
    colour = np.zeros(n, np.int32)
    confirm = np.zeros(n, np.int32)
    per_frame = []
    for f in frames:
        h, w = f["valid"].shape
        ii, r, c, z = project(points, f["pose"], f["K"], h, w)
        ok = f["valid"][r, c]
        ii, r, c, z = ii[ok], r[ok], c[ok], z[ok]
        diff = f["depth"][r, c] - z
        tau = np.maximum(ccfg["abs_tol_m"], ccfg["rel_tol"] * z)
        is_through = diff > tau
        near = np.abs(diff) <= tau
        new_rgb = f["rgb"][r[near], c[near]]
        mem_rgb = colours[ii[near]]
        gain = np.ones(3)
        if near.sum() >= 500:
            gain = np.clip(np.median(new_rgb, 0) / np.maximum(np.median(mem_rgb, 0), 1e-3), 0.5, 2.0)
        dist = np.linalg.norm(new_rgb - np.clip(mem_rgb * gain, 0, 1), axis=1)
        is_colour = np.zeros(len(ii), bool)
        is_colour[np.where(near)[0][dist > ccfg["colour_thresh"]]] = True
        is_confirm = near & ~is_colour
        through[ii[is_through]] += 1
        colour[ii[is_colour]] += 1
        confirm[ii[is_confirm]] += 1
        per_frame.append({"frame": f["frame"], "voted": int(len(ii)), "through": int(is_through.sum()),
                          "colour": int(is_colour.sum()), "confirm": int(is_confirm.sum()),
                          "gain": np.round(gain, 3).tolist()})
    change = through + colour
    changed = (change >= ccfg["min_votes"]) & (change > ccfg["vote_share"] * (change + confirm))
    return {"through": through, "colour": colour, "confirm": confirm, "changed": changed, "per_frame": per_frame}


def stable_voxels(votes: dict, min_votes: int, confirm_share: float) -> np.ndarray:
    """Voxels the memory's own frames confirm: at least min_votes confirmations,
    and they are at least confirm_share of the frames that voted on it."""
    seen = votes["through"] + votes["colour"] + votes["confirm"]
    return (votes["confirm"] >= min_votes) & (votes["confirm"] >= confirm_share * seen)


def cluster_changes(points: np.ndarray, changed: np.ndarray, eps: float, min_points: int, min_voxels: int) -> list:
    """Changed voxels grouped into regions (DBSCAN); regions smaller than min_voxels are noise."""
    import open3d as o3d

    idx = np.where(changed)[0]
    if len(idx) == 0:
        return []
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points[idx].astype(np.float64)))
    labels = np.asarray(pcd.cluster_dbscan(eps=eps, min_points=min_points))
    regions = []
    for lab in range(labels.max() + 1):
        members = idx[labels == lab]
        if len(members) >= min_voxels:
            regions.append(members)
    return sorted(regions, key=len, reverse=True)


def object_status(objects: list, index_of: dict, votes: dict, judged: np.ndarray, ccfg: dict) -> dict:
    """Per memory object: how much of it the new video saw (of its voxels that
    can be judged, i.e. stable in the memory), and how much of that changed.
    Removed when it was seen enough and most of what was seen changed; an
    object the video did not look at stays."""
    seen = ((votes["through"] + votes["colour"] + votes["confirm"]) > 0) & judged
    out = {}
    for obj in objects:
        rows = np.array([index_of[i] for i in obj["indexes"] if i in index_of], dtype=np.int64)
        n_judged = int(judged[rows].sum())
        n_seen = int(seen[rows].sum())
        n_changed = int(votes["changed"][rows].sum())
        seen_share = n_seen / max(n_judged, 1)
        changed_share = n_changed / max(n_seen, 1)
        through_share = int((votes["changed"][rows] & (votes["through"][rows] >= votes["colour"][rows])).sum()) / max(n_changed, 1)
        if seen_share < ccfg["min_seen_share"]:
            status = "not seen"
        elif changed_share > ccfg["remove_share"]:
            status = "removed"
        else:
            status = "present"
        out[obj["class_id"]] = {"voxels": int(len(rows)), "stable_in_memory": round(n_judged / max(len(rows), 1), 3),
                                "seen_share": round(seen_share, 3),
                                "changed_share_of_seen": round(changed_share, 3),
                                "changed_voxels": n_changed, "of_changes_empty_space": round(through_share, 3),
                                "status": status}
    return out
