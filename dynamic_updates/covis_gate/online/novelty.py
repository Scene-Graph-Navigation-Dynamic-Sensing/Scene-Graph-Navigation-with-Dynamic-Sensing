"""New geometry, found while the video plays (targeted.novel_regions, one keyframe at a time, on the GPU).

A pixel's 3D point is "new" when no memory voxel is within novel_distance_m of it. New points are counted on a
novel_voxel_m grid, one count per keyframe that saw the cell; cells seen by at least novel_min_frames keyframes
are grouped (DBSCAN) and groups of at least novel_min_voxels cells are regions: an added object, or a moved one at
its new place. The keyframe that saw most of a region is where the detector should look.

Batch version: 4000 random points of every 4th keyframe, regions computed once at the end. Here: the same
sampling per keyframe pushed to it, regions recomputed on request (~ms: a few thousand cells).
"""
from __future__ import annotations

from collections import deque

import numpy as np
import torch

import gpu_evidence as G


class NovelGeometry:
    def __init__(self, memory_pts: np.ndarray, lower: np.ndarray, upper: np.ndarray, resolution: float, shape: np.ndarray,
                 tcfg: dict, device):
        self.cfg, self.dev = tcfg, device
        self.occ = G.Occupancy(memory_pts, lower, resolution, shape, tcfg["novel_distance_m"], device)
        self.lower = torch.as_tensor(lower, dtype=torch.float64, device=device)
        self.upper = torch.as_tensor(upper, dtype=torch.float64, device=device)
        self.vox = float(tcfg["novel_voxel_m"])
        self.dims = torch.ceil((self.upper - self.lower) / self.vox).long()
        self.support = torch.zeros(int(self.dims.prod()), dtype=torch.int16, device=device)
        self.frames: deque = deque(maxlen=400)       # (keyframe idx, unique novel cells of that keyframe)
        self.gen = torch.Generator(device=device)
        self.gen.manual_seed(0)

    def update(self, idx: int, depth: torch.Tensor, valid: torch.Tensor, pose: np.ndarray, scale: float, K: np.ndarray):
        ys, xs = valid.nonzero(as_tuple=True)
        if len(ys) == 0:
            return
        n = self.cfg["novel_points_per_frame"]
        if len(ys) > n:
            sel = torch.randperm(len(ys), generator=self.gen, device=self.dev)[:n]
            ys, xs = ys[sel], xs[sel]
        z = depth[ys, xs].double() * scale
        cam = torch.stack([(xs.double() - K[0, 2]) / K[0, 0] * z, (ys.double() - K[1, 2]) / K[1, 1] * z, z], 1)
        R = torch.as_tensor(pose[:3, :3], dtype=torch.float64, device=self.dev)
        t = torch.as_tensor(pose[:3, 3], dtype=torch.float64, device=self.dev)
        w = cam @ R.T + t
        w = w[((w >= self.lower) & (w < self.upper)).all(1)]            # inside the memory's map, as the batch method
        w = w[~self.occ.near(w)]
        if len(w) == 0:
            self.frames.append((idx, torch.zeros(0, dtype=torch.long, device=self.dev)))
            return
        cell = torch.floor(w / self.vox).long() - torch.floor(self.lower / self.vox).long()
        cell = cell.clamp(min=0)
        cell = torch.minimum(cell, self.dims - 1)
        flat = torch.unique((cell[:, 0] * self.dims[1] + cell[:, 1]) * self.dims[2] + cell[:, 2])
        self.support[flat] += 1
        self.frames.append((idx, flat))

    def support_share(self, pts: np.ndarray, tolerance_cells: int = 1) -> float:
        """Share of `pts` (world, memory frame) that lie on geometry the video consistently saw as new: within
        tolerance_cells of a cell seen as new by at least novel_min_frames keyframes."""
        if len(pts) == 0:
            return 0.0
        dims = tuple(int(x) for x in self.dims)
        sup = (self.support >= self.cfg["novel_min_frames"]).float().view(1, 1, *dims)
        k = 2 * tolerance_cells + 1
        sup = torch.nn.functional.max_pool3d(sup, k, 1, tolerance_cells)[0, 0] > 0
        w = torch.as_tensor(pts, dtype=torch.float64, device=self.dev)
        cell = torch.floor(w / self.vox).long() - torch.floor(self.lower / self.vox).long()
        ok = ((cell >= 0) & (cell < self.dims)).all(1)
        cell = torch.minimum(cell.clamp(min=0), self.dims - 1)
        return float((sup[cell[:, 0], cell[:, 1], cell[:, 2]] & ok).float().mean())

    def snapshot(self):
        """What regions_from() needs, taken on the thread that updates the counts (a few ms)."""
        keep = (self.support >= self.cfg["novel_min_frames"]).nonzero().squeeze(1)
        return keep, list(self.frames)

    def regions(self) -> list:
        return self.regions_from(*self.snapshot())

    def regions_from(self, keep: torch.Tensor, frames: list) -> list:
        """Regions of a snapshot, largest first: {"voxels", "centroid", "best_frame", "frames_seeing", "cells"}."""
        import open3d as o3d

        c = self.cfg
        if len(keep) < c["novel_min_voxels"]:
            return []
        d1, d2 = int(self.dims[1]), int(self.dims[2])
        flat = keep.cpu().numpy()
        grid = np.stack([flat // (d1 * d2), (flat // d2) % d1, flat % d2], 1)
        base = np.floor(self.lower.cpu().numpy() / self.vox).astype(np.int64)
        pts = (grid + base + 0.5) * self.vox
        labels = np.asarray(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts)).cluster_dbscan(
            eps=1.6 * self.vox, min_points=4))
        out = []
        for lab in range(labels.max() + 1 if len(labels) else 0):
            m = labels == lab
            if m.sum() < c["novel_min_voxels"]:
                continue
            ids = keep[torch.as_tensor(m, device=self.dev)]
            seen = [(idx, int(torch.isin(cells, ids).sum())) for idx, cells in frames]
            best = max(seen, key=lambda s: s[1])
            out.append({"voxels": int(m.sum()), "centroid": np.round(pts[m].mean(0), 3).tolist(), "best_frame": best[0],
                        "frames_seeing": int(sum(1 for _, k in seen if k > 0)), "cells": flat[m]})
        out.sort(key=lambda r: -r["voxels"])
        return out
