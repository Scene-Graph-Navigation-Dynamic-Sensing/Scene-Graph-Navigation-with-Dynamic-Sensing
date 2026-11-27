"""Replay voxel-map construction frame by frame and render it as a video.

`scripts/build_voxel_map.py` shows the *finished* map. This one shows how it
gets there: it walks the prepared DoVSG-format dataset in capture order and,
after every frame, re-renders the voxels accumulated so far next to the RGB
frame that produced them, then muxes the renders into an MP4.

It reimplements DoVSG ViewDataset's voxelisation (point_to_voxel ->
voxel_to_index -> per-index colour average) incrementally instead of in one
batch, so the geometry matches what ViewDataset would produce for the same
dataset and resolution.

  python scripts/animate_voxel_map.py --dataset-dir data/processed/3rscan_c92fb578_gt

Outputs (under --out-dir): frames/*.png, voxel_map_live.mp4, growth.json, run.log
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import yaml
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

logger = logging.getLogger("animate_voxel_map")


# ─── dataset access (mirrors ViewDataset.load_* ) ───────────────────────────
def read_metadata(d: Path) -> dict:
    return json.loads((d / "metadata.json").read_text())


_get_inlier_mask = None


def enable_inlier_mask(scene_graph_generation_root: str):
    """ViewDataset drops per-frame depth outliers before voxelising; opt in to match it."""
    global _get_inlier_mask
    from object_fusion._dovsg_import import ensure_dovsg_importable
    ensure_dovsg_importable(scene_graph_generation_root)
    from dovsg.utils.utils import get_inlier_mask
    _get_inlier_mask = get_inlier_mask


def apply_inlier_mask(point, rgb, mask, nb_neighbors=30, std_ratio=1.5):
    if _get_inlier_mask is None:
        return mask
    return np.logical_and(mask, _get_inlier_mask(point=point, color=rgb / 255, mask=mask,
                                                 nb_neighbors=nb_neighbors, std_ratio=std_ratio))


def load_frame(d: Path, i: int):
    rgb = np.asarray(Image.open(d / "rgb" / f"{i:06d}.jpg"), dtype=np.uint8)
    point = np.load(d / "point" / f"{i:06d}.npy", allow_pickle=True).astype(np.float32)
    mask = np.load(d / "mask" / f"{i:06d}.npy", allow_pickle=True).astype(bool)
    pose = np.loadtxt(d / "poses" / f"{i:06d}.txt").astype(np.float32)
    return rgb, point, mask, pose


def to_world(point: np.ndarray, pose: np.ndarray) -> np.ndarray:
    """Camera-frame points -> world, exactly as ViewDataset.load_data does."""
    return point @ pose[:3, :3].T + pose[:3, 3]


# ─── voxel grid (mirrors ViewDataset.point_to_index / index_to_point) ───────
class VoxelGrid:
    def __init__(self, lower: np.ndarray, upper: np.ndarray, resolution: float):
        self.lower = lower
        self.resolution = resolution
        self.voxel_num = ((upper - lower) / resolution).astype(np.int32)
        # running map, kept as sorted parallel arrays so every update is vectorised
        self.keys = np.zeros(0, dtype=np.int64)
        self.color_sum = np.zeros((0, 3), dtype=np.float64)
        self.hits = np.zeros(0, dtype=np.int64)

    def __len__(self) -> int:
        return int(self.keys.size)

    def point_to_index(self, points: np.ndarray) -> np.ndarray:
        v = np.floor((points - self.lower) / self.resolution).astype(np.int64)
        n = self.voxel_num
        return v[..., 0] * n[1] * n[2] + v[..., 1] * n[2] + v[..., 2]

    def index_to_point(self, indexes: np.ndarray) -> np.ndarray:
        idx = np.asarray(indexes, dtype=np.int64)
        n = self.voxel_num
        v = np.zeros(idx.shape + (3,), dtype=np.int64)
        v[..., 2] = idx % n[2]
        idx = idx // n[2]
        v[..., 1] = idx % n[1]
        v[..., 0] = idx // n[1]
        return v * self.resolution + self.lower

    def integrate(self, gpoint: np.ndarray, color: np.ndarray, mask: np.ndarray) -> int:
        """Fold one frame in. Returns how many voxels were newly occupied."""
        v = np.floor((gpoint - self.lower) / self.resolution)
        in_range = (v >= 0).all(-1) & (v < self.voxel_num).all(-1)
        m = mask & in_range
        if not m.any():
            return 0
        idx = self.point_to_index(gpoint[m])
        col = color[m]

        uniq, inv, cnt = np.unique(idx, return_inverse=True, return_counts=True)
        summed = np.zeros((uniq.size, 3))
        np.add.at(summed, inv, col)
        frame_mean = summed / cnt[:, None]   # this frame's colour per voxel

        pos = np.searchsorted(self.keys, uniq)
        exists = np.zeros(uniq.size, dtype=bool)
        if self.keys.size:
            inb = pos < self.keys.size
            exists[inb] = self.keys[pos[inb]] == uniq[inb]

        # voxels already in the map: fold this frame's colour into the average
        np.add.at(self.color_sum, pos[exists], frame_mean[exists])
        np.add.at(self.hits, pos[exists], 1)

        # newly occupied voxels: append, then re-sort to keep searchsorted valid
        fresh = ~exists
        if fresh.any():
            self.keys = np.concatenate([self.keys, uniq[fresh]])
            self.color_sum = np.concatenate([self.color_sum, frame_mean[fresh]])
            self.hits = np.concatenate([self.hits, np.ones(int(fresh.sum()), dtype=np.int64)])
            order = np.argsort(self.keys)
            self.keys, self.color_sum, self.hits = self.keys[order], self.color_sum[order], self.hits[order]
        return int(fresh.sum())

    def as_pcd(self) -> o3d.geometry.PointCloud:
        cols = self.color_sum / np.maximum(self.hits, 1)[:, None]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.index_to_point(self.keys).astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector(np.clip(cols, 0.0, 1.0))
        return pcd


# ─── passes ────────────────────────────────────────────────────────────────
def compute_bounds(dataset: Path, length: int):
    """ViewDataset fixes the grid from the full sequence before voxelising."""
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    centers = []
    for i in range(length):
        rgb, point, mask, pose = load_frame(dataset, i)
        mask = apply_inlier_mask(point, rgb, mask)
        g = to_world(point, pose)
        centers.append(pose[:3, 3])
        pts = g[mask]
        if len(pts):
            lo = np.minimum(lo, pts.min(axis=0))
            hi = np.maximum(hi, pts.max(axis=0))
    return np.round(lo, 2), np.round(hi, 2), np.array(centers)


def make_renderer(width: int, height: int):
    r = o3d.visualization.rendering.OffscreenRenderer(width, height)
    r.scene.set_background([0.06, 0.07, 0.09, 1.0])
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultUnlit"
    mat.point_size = 3.0
    line = o3d.visualization.rendering.MaterialRecord()
    line.shader = "unlitLine"
    line.line_width = 5.0
    return r, mat, line


def traj_lineset(centers: np.ndarray) -> o3d.geometry.LineSet:
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(centers.astype(np.float64))
    ls.lines = o3d.utility.Vector2iVector([[i, i + 1] for i in range(len(centers) - 1)])
    ls.colors = o3d.utility.Vector3dVector(np.tile([1.0, 0.25, 0.1], (max(len(centers) - 1, 1), 1)))
    return ls


def label(img: np.ndarray, lines: list[str]) -> np.ndarray:
    y = 28
    for t in lines:
        cv2.putText(img, t, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, t, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        y += 26
    return img


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-dir", required=True, type=Path)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "pipeline.yaml")
    ap.add_argument("--resolution", type=float, default=None,
                    help="voxel size in metres (default: object_fusion.voxel_resolution)")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--orbit", type=float, default=0.0,
                    help="degrees of camera orbit per frame (0 = fixed viewpoint)")
    ap.add_argument("--zoom", type=float, default=0.75,
                    help="camera distance as a multiple of the scene diagonal (smaller = closer)")
    ap.add_argument("--view", choices=["iso", "top", "front", "side"], default="iso")
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=6)
    ap.add_argument("--inlier-mask", action="store_true",
                    help="apply ViewDataset's statistical outlier removal (slower, matches build_voxel_map.py)")
    ap.add_argument("--no-video", action="store_true", help="write PNGs only")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    resolution = args.resolution if args.resolution is not None \
        else cfg["object_fusion"]["voxel_resolution"]

    out_dir = args.out_dir or (REPO_ROOT / "outputs" / "voxel_maps" /
                               f"{args.dataset_dir.name}_live_r{resolution}_{time.strftime('%Y%m%d-%H%M%S')}")
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
                        handlers=[logging.FileHandler(out_dir / "run.log"),
                                  logging.StreamHandler(sys.stdout)], force=True)

    if args.inlier_mask:
        enable_inlier_mask(cfg["external"]["scene_graph_generation_root"])
        logger.info("statistical outlier removal enabled")

    meta = read_metadata(args.dataset_dir)
    length = meta["length"] if args.max_frames is None else min(meta["length"], args.max_frames)
    logger.info("dataset %s | %d frames | resolution %.3f m", args.dataset_dir, length, resolution)

    lo, hi, centers = compute_bounds(args.dataset_dir, length)
    grid = VoxelGrid(lo, hi, resolution)
    logger.info("grid dims %s over extent %s m", grid.voxel_num.tolist(), (hi - lo).round(2).tolist())

    center = (lo + hi) / 2
    diag = float(np.linalg.norm(hi - lo))
    offsets = {"iso": np.array([1.0, -1.0, 0.8]), "top": np.array([0.0, 0.0, 1.6]),
               "front": np.array([0.0, -1.7, 0.2]), "side": np.array([1.7, 0.0, 0.2])}
    # normalised so framing holds whatever the room's aspect: distance comes from
    # --zoom * the scene diagonal, not from the raw offset vector's length
    base_off = offsets[args.view] / np.linalg.norm(offsets[args.view])
    up = np.array([0.0, 1.0, 0.0]) if args.view == "top" else np.array([0.0, 0.0, 1.0])

    renderer, mat, line_mat = make_renderer(args.width, args.height)

    growth = []
    for i in range(length):
        rgb, point, mask, pose = load_frame(args.dataset_dir, i)
        mask = apply_inlier_mask(point, rgb, mask)
        gpoint = to_world(point, pose)
        new = grid.integrate(gpoint, (rgb / 255.0).astype(np.float32), mask)

        pcd = grid.as_pcd()
        renderer.scene.clear_geometry()
        renderer.scene.add_geometry("voxels", pcd, mat)
        if i >= 1:
            renderer.scene.add_geometry("traj", traj_lineset(centers[:i + 1]), line_mat)

        ang = np.deg2rad(args.orbit * i)
        off = base_off.copy()
        if args.orbit:
            c, s = np.cos(ang), np.sin(ang)
            off[0], off[1] = base_off[0] * c - base_off[1] * s, base_off[0] * s + base_off[1] * c
        eye = center + off * diag * args.zoom
        renderer.setup_camera(60.0, center.astype(np.float32),
                              eye.astype(np.float32), up.astype(np.float32))
        render = np.asarray(renderer.render_to_image())[:, :, ::-1].copy()  # RGB->BGR

        label(render, [f"frame {i + 1}/{length}",
                       f"occupied voxels: {len(grid):,}",
                       f"+{new:,} new this frame",
                       f"resolution {resolution:.3f} m"])

        # RGB source frame in the left panel, scaled to the render height
        h = args.height
        w = int(rgb.shape[1] * h / rgb.shape[0])
        panel = cv2.resize(rgb[:, :, ::-1], (w, h), interpolation=cv2.INTER_AREA)
        label(panel, [f"rgb/{i:06d}.jpg"])
        canvas = np.hstack([panel, render])
        cv2.imwrite(str(frames_dir / f"{i:06d}.png"), canvas)

        growth.append({"frame": i, "new_voxels": int(new), "total_voxels": len(grid)})
        logger.info("frame %3d: +%6d new -> %8d voxels", i, new, len(grid))

    (out_dir / "growth.json").write_text(json.dumps(
        {"dataset_dir": str(args.dataset_dir), "resolution_m": resolution,
         "inlier_mask": bool(args.inlier_mask),
         "frames": length, "final_voxels": len(grid),
         "grid_dims": grid.voxel_num.tolist(),
         "bounds": {"lower": lo.tolist(), "upper": hi.tolist()},
         "per_frame": growth}, indent=2))

    o3d.io.write_point_cloud(str(out_dir / "voxel_map.ply"), grid.as_pcd())

    if not args.no_video:
        mp4 = out_dir / "voxel_map_live.mp4"
        cmd = ["ffmpeg", "-y", "-framerate", str(args.fps), "-i", str(frames_dir / "%06d.png"),
               "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264",
               "-pix_fmt", "yuv420p", "-crf", "20", str(mp4)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode:
            logger.error("ffmpeg failed:\n%s", r.stderr[-2000:])
        else:
            logger.info("wrote %s", mp4)

    logger.info("=" * 60)
    logger.info("final: %d voxels @ %.3f m over %d frames", len(grid), resolution, length)
    logger.info("results: %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
