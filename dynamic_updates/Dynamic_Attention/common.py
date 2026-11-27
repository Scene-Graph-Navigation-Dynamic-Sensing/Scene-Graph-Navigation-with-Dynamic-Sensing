"""Paths, the YOLOE patch-feature extractor and projection helpers shared by every Dynamic_Attention script.

Nothing here changes the other methods: covis_gate (and its online/ engine), projection_method and dovsg_method are
only imported.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
GATE = REPO_ROOT / "dynamic_updates" / "covis_gate"
OUT_ROOT = REPO_ROOT / "outputs" / "dynamic_attention"
DATA_DIR = OUT_ROOT / "data"
CKPT_DIR = OUT_ROOT / "checkpoints"
YOLOE_WEIGHTS = REPO_ROOT / "models" / "checkpoints" / "yoloe-26s-seg-pf.pt"
PATCH = 16                       # the presence head's patch grid: YOLOE neck features at stride 16
FEAT_DIM = 384                   # 128 (layer 16, stride 8, pooled to 16) + 256 (layer 19, stride 16)


def setup_paths() -> None:
    for p in (REPO_ROOT / "external" / "Scene-Graph-Generation", REPO_ROOT, REPO_ROOT / "dynamic_updates" / "dovsg_method",
              REPO_ROOT / "dynamic_updates" / "projection_method", GATE, GATE / "online", HERE):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


class YoloeFeatures:
    """Patch features of a whole frame from the YOLOE detector's own neck (frozen), stride 16.

    Only the layers up to 19 run (backbone + neck to the stride-16 output); the segmentation head is skipped, so one
    832x468 frame costs about a third of a full YOLOE pass. Layer 16 (stride 8, 128 ch) is average-pooled to stride 16
    and concatenated with layer 19 (stride 16, 256 ch): 384 numbers per 16x16 patch.
    """

    LAST = 19

    def __init__(self, weights: Path | str = YOLOE_WEIGHTS, device: str = "cuda"):
        import torch
        from ultralytics import YOLOE

        self.torch = torch
        self.dev = torch.device(device)
        self.net = YOLOE(str(weights)).model.to(self.dev).eval()
        self.layers = self.net.model[: self.LAST + 1]

    def __call__(self, rgb_u8: np.ndarray):
        """(H, W, 3) uint8 -> (ceil(H/16), ceil(W/16), 384) float16 torch tensor on the device."""
        torch = self.torch
        h, w = rgb_u8.shape[:2]
        H, W = -(-h // 32) * 32, -(-w // 32) * 32
        x = torch.full((1, 3, H, W), 114 / 255, device=self.dev)
        x[0, :, :h, :w] = torch.from_numpy(rgb_u8).to(self.dev).permute(2, 0, 1).float() / 255
        y = []
        with torch.no_grad():
            for m in self.layers:
                if m.f != -1:
                    x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
                x = m(x)
                y.append(x)
            f8 = torch.nn.functional.avg_pool2d(y[16], 2)
            f = torch.cat([f8, y[19]], 1)[0]                     # (384, H/16, W/16)
        gh, gw = -(-h // PATCH), -(-w // PATCH)
        return f[:, :gh, :gw].permute(1, 2, 0).contiguous().half()


def project(pts: np.ndarray, c2w: np.ndarray, K: np.ndarray):
    """World points -> pixel u, v (float) and camera depth z."""
    w2c = np.linalg.inv(c2w)
    cam = pts @ w2c[:3, :3].T + w2c[:3, 3]
    z = cam[:, 2]
    zs = np.where(np.abs(z) < 1e-6, 1e-6, z)
    return K[0, 0] * cam[:, 0] / zs + K[0, 2], K[1, 1] * cam[:, 1] / zs + K[1, 2], z


def footprint(pts: np.ndarray, c2w: np.ndarray, K: np.ndarray, depth: np.ndarray, valid: np.ndarray | None,
              tol: float = 0.05, min_vox: int = 2):
    """Patches an object's voxels project into, with per-patch shares of visible / occluding / behind voxels.

    depth: (h, w) metres in the map's scale (0 or invalid where unknown). Returns dict or None when out of view:
      patches (P,) flat patch ids on the stride-16 grid, n (P,), vis (P,), occ (P,), behind (P,) shares in [0, 1].
    """
    h, w = depth.shape
    u, v, z = project(pts, c2w, K)
    ok = (z > 0.1) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    if ok.sum() < min_vox:
        return None
    ui, vi, z = u[ok].astype(int), v[ok].astype(int), z[ok]
    d = depth[vi, ui]
    good = d > 0.05 if valid is None else valid[vi, ui] & (d > 0.05)
    gw = -(-w // PATCH)
    pid = (vi // PATCH) * gw + (ui // PATCH)
    vis = good & (np.abs(d - z) < tol)
    occ = good & (d < z - tol)
    beh = good & (d > z + tol)
    uniq, inv, cnt = np.unique(pid, return_inverse=True, return_counts=True)
    keep = cnt >= min_vox
    if not keep.any():
        return None

    def share(flag):
        return np.bincount(inv, weights=flag.astype(np.float64), minlength=len(uniq)) / cnt

    return {"patches": uniq[keep], "n": cnt[keep], "vis": share(vis)[keep], "occ": share(occ)[keep],
            "behind": share(beh)[keep], "in_view": int(ok.sum())}


def descriptor(feats, patches: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Weighted mean feature over the given patches of a (gh, gw, C) feature map -> (C,) float32."""
    import torch

    flat = feats.reshape(-1, feats.shape[-1])
    f = flat[torch.as_tensor(np.asarray(patches), dtype=torch.long, device=flat.device)].float()
    if weights is None:
        out = f.mean(0)
    else:
        wt = f.new_tensor(weights).clamp_min(1e-3)
        out = (f * wt[:, None]).sum(0) / wt.sum()
    return out.cpu().numpy().astype(np.float32)
