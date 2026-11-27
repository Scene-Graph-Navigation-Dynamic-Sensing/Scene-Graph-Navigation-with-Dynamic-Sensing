"""Evaluate Depth Anything V3 against 3RScan ground-truth depth.

Runs the repo's DepthAnythingEstimator over every frame-XXXXXX.color.jpg in a
3RScan `sequence/` directory and compares the prediction against the paired
frame-XXXXXX.depth.pgm (uint16 millimetres, m_depthShift = 1000).

The 3RScan depth intrinsics are the colour intrinsics scaled by the image
resize (176.594 = 756.832 * 224/960, 240.808 = 756.026 * 172/540), so the two
images share a field of view and the prediction is simply resized to the GT
resolution before comparison -- no reprojection needed.

DA3-BASE is affine-invariant (up to scale), so predictions are aligned to GT
before metrics are computed; --align none is only meaningful for a metric
checkpoint such as DA3METRIC-LARGE.
"""
from __future__ import annotations

import argparse
import csv
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

from perception.depth_estimation.depth_anything_estimator import DepthAnythingEstimator  # noqa: E402

logger = logging.getLogger("eval_depth")


# ─── dataset io ──────────────────────────────────────────────────────────────

def parse_info(sequence_dir: Path) -> dict:
    """Parse _info.txt into a dict (values kept as strings)."""
    info = {}
    info_path = sequence_dir / "_info.txt"
    if info_path.exists():
        for line in info_path.read_text().splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                info[k.strip()] = v.strip()
    return info


def load_gt_depth(path: Path, depth_shift: float) -> np.ndarray:
    """Read a 3RScan .depth.pgm as float32 metres (0 = invalid)."""
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise IOError(f"could not read GT depth {path}")
    return raw.astype(np.float32) / depth_shift


# ─── alignment ───────────────────────────────────────────────────────────────

def align_prediction(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray, mode: str):
    """Return (aligned_pred, scale, shift) for the requested alignment mode."""
    if mode == "none":
        return pred, 1.0, 0.0
    if mode == "median":
        scale = float(np.median(gt[mask]) / np.median(pred[mask]))
        return pred * scale, scale, 0.0
    if mode == "scale_shift":
        # least squares: gt ~= a * pred + b over valid pixels
        a_mat = np.stack([pred[mask], np.ones(mask.sum(), dtype=np.float32)], axis=1)
        sol, *_ = np.linalg.lstsq(a_mat, gt[mask], rcond=None)
        scale, shift = float(sol[0]), float(sol[1])
        return pred * scale + shift, scale, shift
    raise ValueError(f"unknown align mode: {mode}")


# ─── metrics ─────────────────────────────────────────────────────────────────

def depth_metrics(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> dict:
    p = np.clip(pred[mask], 1e-6, None)
    g = gt[mask]
    ratio = np.maximum(p / g, g / p)
    log_diff = np.log(p) - np.log(g)
    return {
        "abs_rel": float(np.mean(np.abs(p - g) / g)),
        "sq_rel": float(np.mean(((p - g) ** 2) / g)),
        "rmse": float(np.sqrt(np.mean((p - g) ** 2))),
        "rmse_log": float(np.sqrt(np.mean(log_diff ** 2))),
        "log10": float(np.mean(np.abs(np.log10(p) - np.log10(g)))),
        "mae": float(np.mean(np.abs(p - g))),
        "silog": float(np.sqrt(np.mean(log_diff ** 2) - np.mean(log_diff) ** 2) * 100.0),
        "delta1": float(np.mean(ratio < 1.25)),
        "delta2": float(np.mean(ratio < 1.25 ** 2)),
        "delta3": float(np.mean(ratio < 1.25 ** 3)),
    }


METRIC_KEYS = ["abs_rel", "sq_rel", "rmse", "rmse_log", "log10", "mae", "silog",
               "delta1", "delta2", "delta3"]


# ─── visualization ───────────────────────────────────────────────────────────

def colorize(depth: np.ndarray, mask: np.ndarray, vmax: float) -> np.ndarray:
    norm = np.clip(depth / max(vmax, 1e-6), 0.0, 1.0)
    img = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    img[~mask] = 0
    return img


def save_comparison(out_path: Path, rgb_bgr, pred, gt, mask, vmax: float) -> None:
    h, w = gt.shape
    rgb_small = cv2.resize(rgb_bgr, (w, h), interpolation=cv2.INTER_AREA)
    pred_img = colorize(pred, mask, vmax)
    gt_img = colorize(gt, mask, vmax)
    err = np.zeros_like(gt)
    err[mask] = np.abs(pred[mask] - gt[mask])
    err_img = cv2.applyColorMap(
        (np.clip(err / 1.0, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    err_img[~mask] = 0
    panel = np.hstack([rgb_small, gt_img, pred_img, err_img])
    panel = cv2.resize(panel, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_NEAREST)
    for i, label in enumerate(["RGB", "GT", "DA3 (aligned)", "|err| 0-1m"]):
        cv2.putText(panel, label, (i * w * 2 + 8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), panel)


# ─── main ────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sequence", required=True, type=Path,
                    help="3RScan sequence/ directory containing frame-*.color.jpg")
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "outputs" / "depth_eval",
                    help="where results, logs and visualizations are written")
    ap.add_argument("--model-id", default="depth-anything/DA3-BASE")
    ap.add_argument("--fallback-model-id",
                    default="depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--align", choices=["median", "scale_shift", "none"], default="median",
                    help="how to align the prediction to GT before scoring "
                         "(DA3-BASE is up-to-scale -> keep median/scale_shift)")
    ap.add_argument("--min-depth", type=float, default=0.1, help="GT valid range, metres")
    ap.add_argument("--max-depth", type=float, default=10.0, help="GT valid range, metres")
    ap.add_argument("--stride", type=int, default=1, help="evaluate every Nth frame")
    ap.add_argument("--limit", type=int, default=0, help="stop after N frames (0 = all)")
    ap.add_argument("--no-vis", action="store_true", help="skip side-by-side images")
    ap.add_argument("--rotate", action="store_true",
                    help="upright the 90-deg-rotated 3RScan frames before inference "
                         "(GT depth is rotated to match, so metrics stay comparable). "
                         "Metric checkpoints infer absolute scale from image cues, so "
                         "orientation matters to them.")
    ap.add_argument("--require-v3", action="store_true",
                    help="abort if the V3 backend did not load (instead of silently using V2)")
    args = ap.parse_args()

    run_id = time.strftime("%Y%m%d-%H%M%S")
    out_dir = args.out_dir / f"{args.model_id.split('/')[-1]}_{args.align}_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)],
        force=True,
    )
    logger.info("run id %s", run_id)
    logger.info("args: %s", vars(args))
    logger.info("writing results to %s", out_dir)

    info = parse_info(args.sequence)
    depth_shift = float(info.get("m_depthShift", 1000))
    logger.info("sequence info: %d entries, m_depthShift=%s, frames=%s",
                len(info), depth_shift, info.get("m_frames.size", "?"))

    color_paths = sorted(args.sequence.glob("frame-*.color.jpg"))
    if not color_paths:
        logger.error("no frame-*.color.jpg found under %s", args.sequence)
        return 1
    color_paths = color_paths[::args.stride]
    if args.limit:
        color_paths = color_paths[:args.limit]
    logger.info("%d frames selected (stride=%d, limit=%d)",
                len(color_paths), args.stride, args.limit)

    estimator = DepthAnythingEstimator(
        model_id=args.model_id,
        fallback_model_id=args.fallback_model_id,
        device=args.device,
        min_valid_depth_m=args.min_depth,
        max_valid_depth_m=args.max_depth,
    )
    backend = getattr(estimator, "_backend", "unknown")
    logger.info("depth backend actually loaded: %s (device=%s)", backend, estimator.device)
    if args.require_v3 and backend != "v3":
        logger.error("--require-v3 set but backend is '%s'. Install the Depth Anything V3 "
                     "package: pip install depth-anything-3", backend)
        return 2

    csv_path = out_dir / "per_frame_metrics.csv"
    rows, per_frame, skipped = [], [], []
    t_total = 0.0

    with csv_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["frame", "valid_px", "valid_frac", "scale", "shift",
                         "infer_ms"] + METRIC_KEYS)

        for i, color_path in enumerate(color_paths):
            stem = re.sub(r"\.color$", "", color_path.stem)
            gt_path = color_path.with_name(f"{stem}.depth.pgm")
            if not gt_path.exists():
                logger.warning("[%s] no GT depth, skipping", stem)
                skipped.append(stem)
                continue

            bgr = cv2.imread(str(color_path), cv2.IMREAD_COLOR)
            if bgr is None:
                logger.warning("[%s] unreadable colour image, skipping", stem)
                skipped.append(stem)
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            gt = load_gt_depth(gt_path, depth_shift)

            if args.rotate:
                bgr = cv2.rotate(bgr, cv2.ROTATE_90_CLOCKWISE)
                rgb = cv2.rotate(rgb, cv2.ROTATE_90_CLOCKWISE)
                gt = cv2.rotate(gt, cv2.ROTATE_90_CLOCKWISE)

            t0 = time.perf_counter()
            pred, _ = estimator.estimate(rgb)
            infer_ms = (time.perf_counter() - t0) * 1000.0
            t_total += infer_ms

            pred_r = cv2.resize(pred, (gt.shape[1], gt.shape[0]),
                                interpolation=cv2.INTER_NEAREST)

            mask = (gt > args.min_depth) & (gt < args.max_depth) & np.isfinite(gt) \
                & np.isfinite(pred_r) & (pred_r > 0)
            valid_px = int(mask.sum())
            if valid_px < 100:
                logger.warning("[%s] only %d valid pixels, skipping", stem, valid_px)
                skipped.append(stem)
                continue

            pred_a, scale, shift = align_prediction(pred_r, gt, mask, args.align)
            m = depth_metrics(pred_a, gt, mask)
            per_frame.append(m)
            rows.append(stem)

            writer.writerow([stem, valid_px, round(valid_px / gt.size, 4),
                             round(scale, 6), round(shift, 6), round(infer_ms, 1)]
                            + [round(m[k], 6) for k in METRIC_KEYS])
            fh.flush()

            logger.info("[%s] %3d/%3d valid=%5d (%.1f%%) scale=%.4f "
                        "AbsRel=%.4f RMSE=%.4f d1=%.4f (%.0f ms)",
                        stem, i + 1, len(color_paths), valid_px,
                        100.0 * valid_px / gt.size, scale,
                        m["abs_rel"], m["rmse"], m["delta1"], infer_ms)

            if not args.no_vis:
                save_comparison(out_dir / "vis" / f"{stem}.jpg",
                                bgr, pred_a, gt, mask, args.max_depth)

    if not per_frame:
        logger.error("no frames evaluated")
        return 1

    summary = {k: float(np.mean([f[k] for f in per_frame])) for k in METRIC_KEYS}
    result = {
        "run_id": run_id,
        "sequence": str(args.sequence),
        "model_id": args.model_id,
        "backend_loaded": backend,
        "device": estimator.device,
        "align": args.align,
        "rotated": bool(args.rotate),
        "depth_range_m": [args.min_depth, args.max_depth],
        "frames_evaluated": len(per_frame),
        "frames_skipped": skipped,
        "mean_infer_ms": t_total / max(len(per_frame), 1),
        "metrics_mean": summary,
    }
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2))

    logger.info("=" * 62)
    logger.info("SUMMARY over %d frames  (model=%s, backend=%s, align=%s)",
                len(per_frame), args.model_id, backend, args.align)
    for k in METRIC_KEYS:
        logger.info("  %-9s %.4f", k, summary[k])
    logger.info("  mean inference %.0f ms/frame", result["mean_infer_ms"])
    logger.info("results: %s", out_dir)
    logger.info("=" * 62)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
