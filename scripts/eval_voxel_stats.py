"""Pixel->voxel accounting for a prepared dataset.

Answers, for one depth source: how many pixels were integrated, how many
distinct voxels they produced, and how often a voxel is re-observed across
frames. Renders nothing and writes only a small JSON, so it is safe to run
when disk is tight.

    python scripts/eval_voxel_stats.py data/processed/<ds> [--resolution 0.01]
"""
import argparse, json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.animate_voxel_map import (VoxelGrid, load_frame, to_world, read_metadata,
                                       compute_bounds, enable_inlier_mask, apply_inlier_mask)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset_dir", type=Path)
    ap.add_argument("--resolution", type=float, default=0.01)
    ap.add_argument("--scene-graph-root", default="/home/ravindu/SceneGraphNav/external/Scene-Graph-Generation")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    enable_inlier_mask(args.scene_graph_root)
    n = read_metadata(args.dataset_dir)["length"]
    lo, hi, _ = compute_bounds(args.dataset_dir, n)
    grid = VoxelGrid(lo, hi, args.resolution)

    px_total = px_depth_valid = px_integrated = touched_sum = 0
    per_frame = []
    for i in range(n):
        rgb, point, mask, pose = load_frame(args.dataset_dir, i)
        px_total += mask.size
        px_depth_valid += int(mask.sum())
        mask = apply_inlier_mask(point, rgb, mask)
        gpoint = to_world(point, pose)

        v = np.floor((gpoint - grid.lower) / grid.resolution)
        m = mask & (v >= 0).all(-1) & (v < grid.voxel_num).all(-1)
        touched = int(np.unique(grid.point_to_index(gpoint[m])).size) if m.any() else 0

        new = grid.integrate(gpoint, (rgb / 255.0).astype(np.float32), mask)
        px_integrated += int(m.sum())
        touched_sum += touched
        per_frame.append({"frame": i, "px_integrated": int(m.sum()),
                          "voxels_touched": touched, "new_voxels": int(new)})
        if i % 50 == 0 or i == n - 1:
            print(f"  {i+1}/{n} px={px_integrated:,} voxels={len(grid):,}", flush=True)

    ext = (hi - lo)
    final = len(grid)
    out = {
        "dataset": args.dataset_dir.name, "frames": n, "resolution_m": args.resolution,
        "extent_m": ext.round(2).tolist(), "volume_m3": round(float(np.prod(ext)), 1),
        "final_voxels": final,
        "pixels_total": px_total,
        "pixels_depth_valid": px_depth_valid,
        "pixels_integrated": px_integrated,
        "depth_valid_pct": round(100 * px_depth_valid / px_total, 2),
        "integrated_pct": round(100 * px_integrated / px_total, 2),
        # how many integrated pixel observations collapse into one voxel
        "pixels_per_voxel": round(px_integrated / final, 2),
        # how many times an average voxel is seen across frames (1.0 = never re-seen)
        "observations_per_voxel": round(touched_sum / final, 2),
        # of the voxels a frame touches, the share that were never seen before
        "mean_frac_new_per_frame": round(float(np.mean(
            [f["new_voxels"] / max(f["voxels_touched"], 1) for f in per_frame])), 4),
        "voxels_per_m3": round(final / float(np.prod(ext))),
        "last_frame_new_voxels": per_frame[-1]["new_voxels"],
        "per_frame": per_frame,
    }
    dest = args.out or Path("outputs/voxel_stats") / f"{args.dataset_dir.name}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2))
    print("RESULT " + json.dumps({k: v for k, v in out.items() if k != "per_frame"}))


if __name__ == "__main__":
    main()
