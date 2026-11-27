"""Run every Depth Anything V3 configuration on ONE image; save each result as its own file.

Unlike compare_depth_configs.py (which composes a grid), this writes one file per
output so each result can be opened, cropped or dropped into a slide on its own.

Per configuration it saves:
    <cfg>_depth.png       colorized depth, turbo, 0-max_depth
    <cfg>_depth.npy       the raw depth values in metres
    <cfg>_error.png       |prediction - ground truth|, inferno, 0-1 m
    <cfg>_panel.jpg       the four-up strip: RGB | GT | DA3 (aligned) | |err| 0-1m,
                          the same composite the eval run writes to its vis/ folder
plus the shared inputs (rgb.png, gt_depth.png, gt_depth.npy) and metrics.json.

The image can come from a 3RScan sequence (--sequence + --frame, which brings
ground truth with it) or be any standalone image file (--image), in which case
there is nothing to compare against and only the depth outputs are written.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.eval_depth_3rscan import (  # noqa: E402
    parse_info, load_gt_depth, align_prediction, depth_metrics, save_comparison)

CONFIGS = [
    ("DA3-BASE",        "depth-anything/DA3-BASE",        False, "median"),
    ("DA3-BASE",        "depth-anything/DA3-BASE",        True,  "median"),
    ("DA3METRIC-LARGE", "depth-anything/DA3METRIC-LARGE", False, "none"),
    ("DA3METRIC-LARGE", "depth-anything/DA3METRIC-LARGE", False, "median"),
    ("DA3METRIC-LARGE", "depth-anything/DA3METRIC-LARGE", True,  "none"),
    ("DA3METRIC-LARGE", "depth-anything/DA3METRIC-LARGE", True,  "median"),
]


def colorize_depth(depth, mask, vmax):
    norm = np.clip(depth / max(vmax, 1e-6), 0, 1)
    img = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    if mask is not None:
        img[~mask] = 0
    return img


def colorize_error(err, mask, vmax):
    norm = np.clip(err / max(vmax, 1e-6), 0, 1)
    img = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    img[~mask] = 0
    return img


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--sequence", type=Path, help="3RScan sequence dir (brings ground truth)")
    src.add_argument("--image", type=Path, help="any standalone image file (no ground truth)")
    ap.add_argument("--frame", default="frame-000000", help="with --sequence")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--max-depth", type=float, default=10.0)
    ap.add_argument("--min-depth", type=float, default=0.1)
    ap.add_argument("--err-max", type=float, default=1.0)
    ap.add_argument("--upscale", type=int, default=3,
                    help="enlarge saved maps by this factor (GT is only 224x172)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    from perception.depth_estimation.depth_anything_estimator import DepthAnythingEstimator

    # ── load the one image (+ ground truth if we have it) ──
    if args.sequence:
        cp = args.sequence / f"{args.frame}.color.jpg"
        if not cp.exists():
            print(f"ERROR: no such frame: {cp}")
            return 1
        info = parse_info(args.sequence)
        shift = float(info.get("m_depthShift", 1000))
        gt0 = load_gt_depth(cp.with_name(f"{args.frame}.depth.pgm"), shift)
        name = args.frame
    else:
        cp = args.image
        if not cp.exists():
            print(f"ERROR: no such image: {cp}")
            return 1
        gt0 = None
        name = cp.stem

    bgr0 = cv2.imread(str(cp), cv2.IMREAD_COLOR)
    rgb0 = cv2.cvtColor(bgr0, cv2.COLOR_BGR2RGB)

    out_dir = args.out_dir or (REPO_ROOT / "outputs" / "single_image_depth" / name)
    out_dir.mkdir(parents=True, exist_ok=True)

    def save(path: Path, img):
        if args.upscale > 1:
            img = cv2.resize(img, None, fx=args.upscale, fy=args.upscale,
                             interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(str(path), img)

    cv2.imwrite(str(out_dir / "rgb.png"), bgr0)
    if gt0 is not None:
        gt_mask = (gt0 > args.min_depth) & (gt0 < args.max_depth)
        save(out_dir / "gt_depth.png", colorize_depth(gt0, gt_mask, args.max_depth))
        np.save(out_dir / "gt_depth.npy", gt0)

    models: dict[str, DepthAnythingEstimator] = {}
    results = []

    print(f"image: {cp}")
    print(f"saving to: {out_dir}\n")

    for label, model_id, rot, align in CONFIGS:
        if gt0 is None and align != "none":
            continue   # nothing to align against
        if model_id not in models:
            est = DepthAnythingEstimator(model_id=model_id, device=args.device,
                                         min_valid_depth_m=args.min_depth,
                                         max_valid_depth_m=args.max_depth)
            if getattr(est, "_backend", None) != "v3":
                print(f"ERROR: {model_id} did not load the V3 backend")
                return 2
            models[model_id] = est
        est = models[model_id]

        rgb = cv2.rotate(rgb0, cv2.ROTATE_90_CLOCKWISE) if rot else rgb0
        pred, _ = est.estimate(rgb)

        tag = f"{label}_upright-{'yes' if rot else 'no'}_align-{align}"

        if gt0 is not None:
            gt = cv2.rotate(gt0, cv2.ROTATE_90_CLOCKWISE) if rot else gt0
            pred_r = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
            mask = (gt > args.min_depth) & (gt < args.max_depth) & np.isfinite(pred_r) & (pred_r > 0)
            pred_a, scale, _ = align_prediction(pred_r, gt, mask, align)
            m = depth_metrics(pred_a, gt, mask)

            err = np.zeros_like(gt)
            err[mask] = np.abs(pred_a[mask] - gt[mask])
            save(out_dir / f"{tag}_error.png", colorize_error(err, mask, args.err_max))
            save(out_dir / f"{tag}_depth.png", colorize_depth(pred_a, mask, args.max_depth))
            np.save(out_dir / f"{tag}_depth.npy", pred_a)

            # the same four-up composite the eval run writes to vis/
            bgr = cv2.rotate(bgr0, cv2.ROTATE_90_CLOCKWISE) if rot else bgr0
            save_comparison(out_dir / f"{tag}_panel.jpg", bgr, pred_a, gt, mask, args.max_depth)

            results.append({"config": tag, "model_id": model_id, "upright": rot,
                            "align": align, "scale_applied": round(scale, 4),
                            **{k: round(v, 4) for k, v in m.items()}})
            print(f"{tag:<46} AbsRel {m['abs_rel']:.3f}  RMSE {m['rmse']:.3f}  "
                  f"d1 {m['delta1']:.3f}  scale x{scale:.2f}")
        else:
            save(out_dir / f"{tag}_depth.png", colorize_depth(pred, None, args.max_depth))
            np.save(out_dir / f"{tag}_depth.npy", pred)
            results.append({"config": tag, "model_id": model_id, "upright": rot,
                            "align": align,
                            "depth_min_m": round(float(pred.min()), 3),
                            "depth_max_m": round(float(pred.max()), 3)})
            print(f"{tag:<46} depth range {pred.min():.2f}-{pred.max():.2f} m")

    (out_dir / "metrics.json").write_text(json.dumps({
        "source_image": str(cp),
        "frame": name,
        "has_ground_truth": gt0 is not None,
        "depth_range_m": [args.min_depth, args.max_depth],
        "error_scale_m": args.err_max,
        "results": results,
    }, indent=2))

    n_png = len(list(out_dir.glob("*.png")))
    n_panel = len(list(out_dir.glob("*_panel.jpg")))
    print(f"\nwrote {n_panel} four-up panels + {n_png} images + "
          f"{len(results)} .npy depth maps + metrics.json")
    print(f"folder: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
