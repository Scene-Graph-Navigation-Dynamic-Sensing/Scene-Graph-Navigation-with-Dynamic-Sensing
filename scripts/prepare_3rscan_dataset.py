"""Convert a 3RScan `sequence/` into the DoVSG dataset layout the pipeline reads.

Writes rgb/{i:06d}.jpg, point/{i:06d}.npy (camera-frame XYZ), mask/{i:06d}.npy
(valid-depth bool), poses/{i:06d}.txt (4x4 camera-to-world) and metadata.json,
matching exactly what DroidSlamPoseEstimator.save_frame_inputs + write_metadata
produce, so object_fusion.view_dataset.ViewDataset consumes it unchanged.

Depth source:
  gt   -- the dataset's own frame-*.depth.pgm (uint16 mm), upsampled to the
          colour resolution. Metric and reliable; the default.
  da3  -- Depth Anything V3. DA3-BASE is affine-invariant, so a per-frame
          scale must come from somewhere or the map will not be metric; see
          --da3-align. A metric checkpoint (DA3METRIC-LARGE) needs no alignment.

Rotation:
  3RScan Tango frames are stored rotated 90 deg (the floor runs along the right
  edge of the 960x540 image). --rotate uprights colour and depth, remaps the
  intrinsics, and post-multiplies the pose rotation by the matching roll about
  the optical axis so the reconstructed cloud is bit-identical to the unrotated
  one. Upright frames matter for the detector, not for the geometry.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

logger = logging.getLogger("prepare_3rscan")

# Rotating the image 90 deg clockwise maps (x, y) -> (H-1-y, x), which is a
# roll of the camera frame about its optical axis: X' = -Y, Y' = X, Z' = Z.
R_ROLL = np.array([[0.0, -1.0, 0.0],
                   [1.0, 0.0, 0.0],
                   [0.0, 0.0, 1.0]], dtype=np.float64)


def parse_info(sequence_dir: Path) -> dict:
    info = {}
    for line in (sequence_dir / "_info.txt").read_text().splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            info[k.strip()] = v.strip()
    return info


def parse_intrinsic(value: str) -> np.ndarray:
    nums = [float(x) for x in value.split()]
    return np.array(nums, dtype=np.float64).reshape(4, 4)[:3, :3]


def rotate_intrinsics(K: np.ndarray, height: int) -> np.ndarray:
    """Intrinsics after a 90 deg clockwise image rotation of a WxH image."""
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    K_new = np.eye(3)
    K_new[0, 0], K_new[1, 1] = fy, fx
    K_new[0, 2], K_new[1, 2] = (height - 1) - cy, cx
    return K_new


def backproject(depth_m: np.ndarray, K: np.ndarray, min_d: float, max_d: float):
    h, w = depth_m.shape
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    valid = (depth_m > min_d) & (depth_m < max_d) & np.isfinite(depth_m)
    z = np.where(valid, depth_m, 0.0)
    points = np.stack([(u - cx) * z / fx, (v - cy) * z / fy, z], axis=-1)
    return points.astype(np.float32), valid


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sequence", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path, help="dataset dir to create")
    ap.add_argument("--depth-source", choices=["gt", "da3"], default="gt")
    ap.add_argument("--model-id", default="depth-anything/DA3-BASE")
    ap.add_argument("--da3-align", choices=["none", "gt_median", "global"], default="gt_median",
                    help="none = trust the model (use with a metric checkpoint); "
                         "gt_median = per-frame median match to GT (upper bound, needs GT); "
                         "global = one median-of-medians scale for the whole sequence")
    ap.add_argument("--rotate", action="store_true",
                    help="upright the 90-deg-rotated 3RScan frames (recommended for detection)")
    ap.add_argument("--rgb-width", type=int, default=0,
                    help="downscale colour to this width (0 = keep native); "
                         "intrinsics are scaled to match")
    ap.add_argument("--min-depth", type=float, default=0.1)
    ap.add_argument("--max-depth", type=float, default=10.0)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        handlers=[logging.FileHandler(args.out / "prepare.log"), logging.StreamHandler(sys.stdout)],
        force=True,
    )
    logger.info("args: %s", vars(args))

    info = parse_info(args.sequence)
    depth_shift = float(info.get("m_depthShift", 1000))
    K_color = parse_intrinsic(info["m_calibrationColorIntrinsic"])
    W, H = int(info["m_colorWidth"]), int(info["m_colorHeight"])
    logger.info("native colour %dx%d, K=%s", W, H, K_color[:2].round(2).tolist())

    color_paths = sorted(args.sequence.glob("frame-*.color.jpg"))[::args.stride]
    if args.limit:
        color_paths = color_paths[:args.limit]
    if not color_paths:
        logger.error("no frames found in %s", args.sequence)
        return 1

    estimator = None
    if args.depth_source == "da3":
        from perception.depth_estimation.depth_anything_estimator import DepthAnythingEstimator
        estimator = DepthAnythingEstimator(
            model_id=args.model_id, device=args.device,
            min_valid_depth_m=args.min_depth, max_valid_depth_m=args.max_depth)
        backend = getattr(estimator, "_backend", "unknown")
        logger.info("depth backend loaded: %s", backend)
        if backend != "v3":
            logger.error("expected the V3 backend, got '%s'", backend)
            return 2

    # a single sequence-wide scale, estimated from a sample of frames
    global_scale = 1.0
    if args.depth_source == "da3" and args.da3_align == "global":
        sample = color_paths[::max(len(color_paths) // 10, 1)]
        scales = []
        for cp in sample:
            stem = re.sub(r"\.color$", "", cp.stem)
            gt = cv2.imread(str(cp.with_name(f"{stem}.depth.pgm")), cv2.IMREAD_UNCHANGED)
            rgb = cv2.cvtColor(cv2.imread(str(cp)), cv2.COLOR_BGR2RGB)
            pred, _ = estimator.estimate(rgb, K_color[0, 0], K_color[1, 1])
            gt_m = gt.astype(np.float32) / depth_shift
            pred_r = cv2.resize(pred, (gt_m.shape[1], gt_m.shape[0]), interpolation=cv2.INTER_NEAREST)
            m = (gt_m > args.min_depth) & (gt_m < args.max_depth) & (pred_r > 0)
            if m.sum() > 100:
                scales.append(float(np.median(gt_m[m]) / np.median(pred_r[m])))
        global_scale = float(np.median(scales))
        logger.info("global DA3 scale from %d frames: %.4f (spread %.3f-%.3f)",
                    len(scales), global_scale, min(scales), max(scales))

    for sub in ("rgb", "point", "mask", "poses", "depth"):
        (args.out / sub).mkdir(parents=True, exist_ok=True)

    K_out = None
    written = 0
    scales_used = []

    for cp in color_paths:
        stem = re.sub(r"\.color$", "", cp.stem)
        gt_path = cp.with_name(f"{stem}.depth.pgm")
        pose_path = cp.with_name(f"{stem}.pose.txt")
        if not pose_path.exists():
            logger.warning("[%s] no pose, skipping", stem)
            continue

        pose = np.loadtxt(pose_path).astype(np.float64)
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            logger.warning("[%s] bad pose, skipping", stem)
            continue

        bgr = cv2.imread(str(cp), cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        gt_raw = cv2.imread(str(gt_path), cv2.IMREAD_UNCHANGED) if gt_path.exists() else None
        gt_m = gt_raw.astype(np.float32) / depth_shift if gt_raw is not None else None

        if args.depth_source == "gt":
            if gt_m is None:
                logger.warning("[%s] no GT depth, skipping", stem)
                continue
            # GT depth shares the colour FOV, so a plain resize is the correct
            # reprojection; nearest keeps depth edges from being interpolated
            # across object boundaries into phantom surfaces.
            depth = cv2.resize(gt_m, (W, H), interpolation=cv2.INTER_NEAREST)
            scale = 1.0
        else:
            # focal at the native colour resolution: a DA3METRIC checkpoint
            # needs it to return metres rather than canonical depth
            pred, _ = estimator.estimate(rgb, K_color[0, 0], K_color[1, 1])
            pred = cv2.resize(pred, (W, H), interpolation=cv2.INTER_NEAREST)
            if args.da3_align == "gt_median":
                if gt_m is None:
                    logger.warning("[%s] no GT depth for alignment, skipping", stem)
                    continue
                pr = cv2.resize(pred, (gt_m.shape[1], gt_m.shape[0]), interpolation=cv2.INTER_NEAREST)
                m = (gt_m > args.min_depth) & (gt_m < args.max_depth) & (pr > 0)
                if m.sum() < 100:
                    logger.warning("[%s] too few valid GT pixels to align, skipping", stem)
                    continue
                scale = float(np.median(gt_m[m]) / np.median(pr[m]))
            elif args.da3_align == "global":
                scale = global_scale
            else:
                scale = 1.0
            depth = pred * scale
        scales_used.append(scale)

        if args.rotate:
            rgb = cv2.rotate(rgb, cv2.ROTATE_90_CLOCKWISE)
            depth = cv2.rotate(depth, cv2.ROTATE_90_CLOCKWISE)
            K_frame = rotate_intrinsics(K_color, H)
            pose = pose.copy()
            pose[:3, :3] = pose[:3, :3] @ R_ROLL.T
        else:
            K_frame = K_color.copy()

        if args.rgb_width and args.rgb_width != rgb.shape[1]:
            s = args.rgb_width / rgb.shape[1]
            new_size = (args.rgb_width, int(round(rgb.shape[0] * s)))
            rgb = cv2.resize(rgb, new_size, interpolation=cv2.INTER_AREA)
            depth = cv2.resize(depth, new_size, interpolation=cv2.INTER_NEAREST)
            K_frame = K_frame.copy()
            K_frame[0, :] *= s
            K_frame[1, :] *= s

        points, valid = backproject(depth, K_frame, args.min_depth, args.max_depth)
        if valid.sum() < 100:
            logger.warning("[%s] only %d valid depth pixels, skipping", stem, int(valid.sum()))
            continue

        idx = written
        cv2.imwrite(str(args.out / "rgb" / f"{idx:06d}.jpg"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        np.save(args.out / "point" / f"{idx:06d}.npy", points)
        np.save(args.out / "mask" / f"{idx:06d}.npy", valid)
        np.savetxt(args.out / "poses" / f"{idx:06d}.txt", pose)
        # Also emit the reference depth as uint16 millimetres, which (with the
        # camera.yaml written below) makes this directory a data/raw dataset as
        # well as a DoVSG one: run_static_scene_graph.py --source <dir> can then
        # use it either with these sensor values (depth.backend=reference) or
        # with estimated depth aligned to them (depth.align=global).
        if gt_m is not None:
            gt_out = cv2.resize(gt_m, (W, H), interpolation=cv2.INTER_NEAREST)
            if args.rotate:
                gt_out = cv2.rotate(gt_out, cv2.ROTATE_90_CLOCKWISE)
            if args.rgb_width and args.rgb_width != gt_out.shape[1]:
                gt_out = cv2.resize(gt_out, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(str(args.out / "depth" / f"{idx:06d}.png"),
                        np.clip(gt_out * 1000.0, 0, 65535).astype(np.uint16))
        K_out = K_frame
        written += 1
        logger.info("[%s] -> %06d  valid=%d (%.1f%%)  scale=%.4f  size=%dx%d",
                    stem, idx, int(valid.sum()), 100.0 * valid.mean(), scale,
                    rgb.shape[1], rgb.shape[0])

    if written < 2:
        logger.error("only %d frame(s) written", written)
        return 1

    metadata = {
        "w": int(rgb.shape[1]),
        "h": int(rgb.shape[0]),
        "K": K_out.flatten().tolist(),
        "dist_coef": [0.0, 0.0, 0.0, 0.0, 0.0],
        "length": written,
    }
    (args.out / "metadata.json").write_text(json.dumps(metadata))

    # data/raw-style camera file, so --source <dir> works on this directory too
    (args.out / "camera.yaml").write_text(
        f"# from {args.sequence}/_info.txt"
        f"{' (rotated 90 deg, rescaled)' if args.rotate or args.rgb_width else ''}\n"
        f"fx: {K_out[0, 0]}\nfy: {K_out[1, 1]}\ncx: {K_out[0, 2]}\ncy: {K_out[1, 2]}\n"
        f"width: {metadata['w']}\nheight: {metadata['h']}\ndepth_scale: 1000\n")

    (args.out / "source.json").write_text(json.dumps({
        "sequence": str(args.sequence),
        "depth_source": args.depth_source,
        "model_id": args.model_id if args.depth_source == "da3" else None,
        "da3_align": args.da3_align if args.depth_source == "da3" else None,
        "rotated": bool(args.rotate),
        "frames": written,
        "scale_mean": float(np.mean(scales_used)) if scales_used else None,
        "scale_std": float(np.std(scales_used)) if scales_used else None,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, indent=2))

    logger.info("wrote %d frames to %s (%dx%d)", written, args.out, metadata["w"], metadata["h"])
    logger.info("metadata K = %s", np.array(metadata["K"]).reshape(3, 3).round(2).tolist())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
