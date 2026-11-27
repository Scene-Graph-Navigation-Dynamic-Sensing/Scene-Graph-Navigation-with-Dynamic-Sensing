"""§6 on the GPU, one keyframe at a time.

evidence.classify_frame / evidence.gather do the same arithmetic in numpy for a
whole video at once (~60 ms per frame: projecting 338k voxels, the colour
median and np.add.at are the three big pieces). Here the memory lives on the
GPU for the whole session and a keyframe costs a few milliseconds.

The arithmetic is the batch code's, step for step (float64 projection and
residual, the same sigma(z), the same exposure gain), so the per-object counts
match the batch counts to within a pixel's rounding; test_online.py checks it.

Also here: the edge-free depth mask (compare_video_poses.edge_free_mask) as
two max-pools, and the "does this frame fit the map" test (frame_fit) as a
lookup in a dilated occupancy volume instead of a KD-tree.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

import theory as T

HIT, THROUGH, OCCLUDED = T.HIT, T.THROUGH, T.OCCLUDED


def edge_free_mask_gpu(depth: torch.Tensor, valid: torch.Tensor, rel: float = 0.03, k: int = 5) -> torch.Tensor:
    """Valid pixels whose depth is locally smooth (no flying pixels at depth edges)."""
    z = torch.where(valid, depth, torch.zeros_like(depth))
    zmax = F.max_pool2d(z[None, None], k, 1, k // 2)[0, 0]
    zmin = -F.max_pool2d(-torch.where(valid, z, torch.full_like(z, 1e6))[None, None], k, 1, k // 2)[0, 0]
    return valid & ((zmax - zmin) < rel * z.clamp_min(1e-6))


class Occupancy:
    """The memory's voxels as a dense boolean volume on the memory's own grid,
    dilated by a sphere: a point is within `radius_m` of a memory voxel when its
    cell is set. 252 x 225 x 116 cells for lap_in_1, so a lookup is one gather."""

    def __init__(self, pts: np.ndarray, lower: np.ndarray, resolution: float, shape: np.ndarray, radius_m: float,
                 device):
        self.dev, self.res = device, float(resolution)
        self.lower = torch.as_tensor(lower, dtype=torch.float64, device=device)
        self.shape = torch.as_tensor(shape, dtype=torch.long, device=device)
        cells = np.floor((pts - lower) / resolution).astype(np.int64)
        ok = np.all((cells >= 0) & (cells < shape), axis=1)
        vol = torch.zeros(tuple(int(s) for s in shape), dtype=torch.float32, device=device)
        c = torch.as_tensor(cells[ok], device=device)
        vol[c[:, 0], c[:, 1], c[:, 2]] = 1.0
        r = int(round(radius_m / resolution))
        ax = torch.arange(-r, r + 1, device=device, dtype=torch.float32)
        gx, gy, gz = torch.meshgrid(ax, ax, ax, indexing="ij")
        kernel = ((gx ** 2 + gy ** 2 + gz ** 2) <= r * r + 1e-6).float()[None, None]
        self.vol = F.conv3d(vol[None, None], kernel, padding=r)[0, 0] > 0.5

    def near(self, world: torch.Tensor) -> torch.Tensor:
        """(M, 3) world points -> bool: within the radius of a memory voxel (False outside the grid)."""
        cell = torch.floor((world - self.lower) / self.res).long()
        inside = ((cell >= 0) & (cell < self.shape)).all(1)
        cell = torch.minimum(torch.clamp(cell, min=0), self.shape - 1)
        return self.vol[cell[:, 0], cell[:, 1], cell[:, 2]] & inside

    def inside(self, world: torch.Tensor) -> torch.Tensor:
        cell = torch.floor((world - self.lower) / self.res).long()
        return ((cell >= 0) & (cell < self.shape)).all(1)


class GpuMemory:
    """Everything the per-frame evidence needs about the memory, on the GPU.

    owner_eff   per voxel: object index, -1 background, -3 not judged (evidence.gather's convention)
    bg_groups   target object -> rows of the background voxels that certify it (evidence.background_groups)
    """

    def __init__(self, pts: np.ndarray, cols: np.ndarray, owner_eff: np.ndarray, n_obj: int, bg_groups: dict,
                 ccfg: dict, device="cuda"):
        self.dev, self.n_obj, self.cfg = torch.device(device), n_obj, ccfg
        self.pts = torch.as_tensor(pts, dtype=torch.float64, device=self.dev)
        self.cols = torch.as_tensor(cols, dtype=torch.float32, device=self.dev)
        self.owner_eff = torch.as_tensor(owner_eff, dtype=torch.long, device=self.dev)
        obj_rows = (self.owner_eff >= 0).nonzero().squeeze(1)
        self.obj_slot = torch.full((len(pts),), -1, dtype=torch.long, device=self.dev)
        self.obj_slot[obj_rows] = torch.arange(len(obj_rows), device=self.dev)
        self.n_slots = int(len(obj_rows))
        self.slot_owner = self.owner_eff[obj_rows]
        self.bg_targets = list(bg_groups)
        rows = [torch.as_tensor(bg_groups[i], dtype=torch.long, device=self.dev) for i in self.bg_targets]
        self.bg_rows = torch.cat(rows) if rows else torch.zeros(0, dtype=torch.long, device=self.dev)
        self.bg_gid = (torch.cat([torch.full((len(r),), g, dtype=torch.long, device=self.dev) for g, r in enumerate(rows)])
                       if rows else torch.zeros(0, dtype=torch.long, device=self.dev))
        self.status = torch.zeros(len(pts), dtype=torch.uint8, device=self.dev)

    def process(self, rgb: torch.Tensor, depth: torch.Tensor, valid: torch.Tensor, pose: np.ndarray, K: np.ndarray,
                scale: float, thr_row: torch.Tensor | None = None) -> dict:
        """One keyframe. rgb (H, W, 3) float [0, 1] (or uint8), depth (H, W) in the new video's units, valid (H, W)
        the edge-free mask, pose camera -> memory world, scale the depth factor of the Sim3.
        Fills thr_row (bool, n_slots) with this frame's through voxels. Returns numpy counts."""
        c = self.cfg
        h, w = valid.shape
        inv = np.linalg.inv(pose)
        R = torch.as_tensor(inv[:3, :3], dtype=torch.float64, device=self.dev)
        t = torch.as_tensor(inv[:3, 3], dtype=torch.float64, device=self.dev)
        pc = self.pts @ R.T + t
        z_all = pc[:, 2]
        ii = (z_all > 0.05).nonzero().squeeze(1)
        z = z_all[ii]
        u = torch.round(K[0, 0] * pc[ii, 0] / z + K[0, 2]).long()
        v = torch.round(K[1, 1] * pc[ii, 1] / z + K[1, 2]).long()
        inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        ii, r, col, z = ii[inside], v[inside], u[inside], z[inside]
        keep = valid[r, col] & (z <= c["max_depth_m"])
        ii, r, col, z = ii[keep], r[keep], col[keep], z[keep]
        res = depth[r, col].double() * scale - z
        sigma = c["sigma0_m"] + c["sigma_k1"] * z + c["sigma_k2"] * z * z
        out = torch.full(res.shape, HIT, dtype=torch.uint8, device=self.dev)
        out[res > 2 * sigma] = THROUGH
        out[res < -2 * sigma] = THROUGH if c["residual"] != "signed" else OCCLUDED
        if c["colour_refine"]:
            hit = (out == HIT).nonzero().squeeze(1)
            img = rgb if rgb.dtype != torch.uint8 else rgb.float() / 255.0
            new_rgb, mem_rgb = img[r[hit], col[hit]], self.cols[ii[hit]]
            gain = torch.ones(3, device=self.dev)
            if len(hit) >= 500:
                gain = (torch.quantile(new_rgb, 0.5, dim=0) / torch.quantile(mem_rgb, 0.5, dim=0).clamp_min(1e-3)).clamp(0.5, 2.0)
            dist = torch.linalg.norm(new_rgb - (mem_rgb * gain).clamp(0, 1), dim=1)
            out[hit[dist > c["colour_thresh"]]] = THROUGH
        own = self.owner_eff[ii]
        obj = own >= 0
        n = self.n_obj
        n_valid = torch.bincount(own[obj], minlength=n)
        n_hit = torch.bincount(own[obj & (out == HIT)], minlength=n)
        thr = obj & (out == THROUGH)
        n_thr = torch.bincount(own[thr], minlength=n)
        if thr_row is not None:
            thr_row.zero_()
            thr_row[self.obj_slot[ii[thr]]] = True
        parts = [n_valid, n_hit, n_thr]
        if len(self.bg_targets):
            self.status.zero_()
            self.status[ii] = out
            st = self.status[self.bg_rows]
            g = len(self.bg_targets)
            parts += [torch.bincount(self.bg_gid[st > 0], minlength=g), torch.bincount(self.bg_gid[st == HIT], minlength=g)]
        flat = torch.cat([p.long() for p in parts]).cpu().numpy()
        res_ = {"n_valid": flat[:n], "n_hit": flat[n:2 * n], "n_thr": flat[2 * n:3 * n]}
        g = len(self.bg_targets)
        res_["bg_valid"] = flat[3 * n:3 * n + g]
        res_["bg_hit"] = flat[3 * n + g:3 * n + 2 * g]
        return res_
