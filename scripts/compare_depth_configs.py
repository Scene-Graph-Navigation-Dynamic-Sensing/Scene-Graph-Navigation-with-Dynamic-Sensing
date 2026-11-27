"""Every Depth Anything V3 configuration, side by side, on the same frame(s).

Re-runs inference for each config in the comparison table and lays the results
out as a grid: ground truth, prediction, and absolute error per row. Writes a
PNG for one frame, or a multi-page PDF (one page per frame) for several.

Every row is displayed upright regardless of what the model was fed -- the
"upright" label says what the MODEL saw, which is the thing being compared;
rotating a landscape result for display is cosmetic and changes no number.

The error column uses one shared 0-1 m scale across all rows so rows are
comparable. Unaligned rows saturate it, which is the finding, not a bug.

    python scripts/compare_depth_configs.py --sequence $SEQ --frame frame-000000
    python scripts/compare_depth_configs.py --sequence $SEQ --frame all \
        --out outputs/depth_eval/all_configs.pdf
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.eval_depth_3rscan import (  # noqa: E402
    parse_info, load_gt_depth, align_prediction, depth_metrics)

# (label, model_id, rotate, align) -- the rows of the comparison table
CONFIGS = [
    ("DA3-BASE",        "depth-anything/DA3-BASE",        False, "median"),
    ("DA3METRIC-LARGE", "depth-anything/DA3METRIC-LARGE", False, "none"),
    ("DA3METRIC-LARGE", "depth-anything/DA3METRIC-LARGE", False, "median"),
    ("DA3METRIC-LARGE", "depth-anything/DA3METRIC-LARGE", True,  "none"),
    ("DA3METRIC-LARGE", "depth-anything/DA3METRIC-LARGE", True,  "median"),
]

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8984"
COL_TITLES = ["RGB input", "Ground truth", "DA3 prediction", "|error|  0-1 m"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sequence", required=True, type=Path)
    ap.add_argument("--frame", default="frame-000000",
                    help='a frame stem, a comma-separated list, or "all"')
    ap.add_argument("--out", type=Path,
                    default=REPO_ROOT / "outputs" / "depth_eval" / "config_comparison.png",
                    help=".png for a single frame, .pdf for one page per frame")
    ap.add_argument("--stride", type=int, default=1, help='with --frame all')
    ap.add_argument("--max-depth", type=float, default=10.0)
    ap.add_argument("--min-depth", type=float, default=0.1)
    ap.add_argument("--err-max", type=float, default=1.0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    from perception.depth_estimation.depth_anything_estimator import DepthAnythingEstimator

    info = parse_info(args.sequence)
    shift = float(info.get("m_depthShift", 1000))

    if args.frame == "all":
        frames = [re.sub(r"\.color$", "", p.stem)
                  for p in sorted(args.sequence.glob("frame-*.color.jpg"))][::args.stride]
    else:
        frames = [f.strip() for f in args.frame.split(",") if f.strip()]
    if not frames:
        print("ERROR: no frames selected")
        return 1

    is_pdf = args.out.suffix.lower() == ".pdf"
    if len(frames) > 1 and not is_pdf:
        print("ERROR: multiple frames need a .pdf --out")
        return 1

    models: dict[str, DepthAnythingEstimator] = {}

    def estimator(model_id: str):
        if model_id not in models:
            est = DepthAnythingEstimator(model_id=model_id, device=args.device,
                                         min_valid_depth_m=args.min_depth,
                                         max_valid_depth_m=args.max_depth)
            if getattr(est, "_backend", None) != "v3":
                raise RuntimeError(f"{model_id} did not load the V3 backend")
            models[model_id] = est
        return models[model_id]

    def build_rows(frame: str) -> list[dict]:
        cp = args.sequence / f"{frame}.color.jpg"
        bgr0 = cv2.imread(str(cp), cv2.IMREAD_COLOR)
        gt0 = load_gt_depth(cp.with_name(f"{frame}.depth.pgm"), shift)
        rgb0 = cv2.cvtColor(bgr0, cv2.COLOR_BGR2RGB)

        rows = []
        for label, model_id, rot, align in CONFIGS:
            est = estimator(model_id)
            rgb, gt = rgb0, gt0
            if rot:
                rgb = cv2.rotate(rgb0, cv2.ROTATE_90_CLOCKWISE)
                gt = cv2.rotate(gt0, cv2.ROTATE_90_CLOCKWISE)

            pred, _ = est.estimate(rgb)
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
            mask = (gt > args.min_depth) & (gt < args.max_depth) & np.isfinite(pred) & (pred > 0)
            pred_a, scale, _ = align_prediction(pred, gt, mask, align)
            m = depth_metrics(pred_a, gt, mask)

            err = np.zeros_like(gt)
            err[mask] = np.abs(pred_a[mask] - gt[mask])

            if not rot:   # display upright; cosmetic only
                rot90 = lambda a: cv2.rotate(a, cv2.ROTATE_90_CLOCKWISE)  # noqa: E731
                gt_d, pred_d, err_d = rot90(gt), rot90(pred_a), rot90(err)
                mask_d = rot90(mask.astype(np.uint8)).astype(bool)
                rgb_d = rot90(rgb)
            else:
                gt_d, pred_d, err_d, mask_d, rgb_d = gt, pred_a, err, mask, rgb

            rows.append(dict(label=label, rot=rot, align=align, scale=scale, m=m,
                             rgb=rgb_d, gt=gt_d, pred=pred_d, err=err_d, mask=mask_d))
        return rows

    def render(rows: list[dict], frame: str):
        n = len(rows)
        fig, axes = plt.subplots(n, 4, figsize=(11.5, 2.55 * n))
        fig.patch.set_facecolor("white")

        for r, row in enumerate(rows):
            hide = ~row["mask"]
            masked = lambda a: np.ma.masked_where(hide, a)  # noqa: E731
            panels = [
                (row["rgb"], None, None, None),
                (masked(row["gt"]), "turbo", 0, args.max_depth),
                (masked(row["pred"]), "turbo", 0, args.max_depth),
                (masked(row["err"]), "inferno", 0, args.err_max),
            ]
            for c, (img, cmap, vmin, vmax) in enumerate(panels):
                ax = axes[r, c]
                ax.set_facecolor("black")
                ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
                ax.set_xticks([]); ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_visible(False)
                if r == 0:
                    ax.set_title(COL_TITLES[c], fontsize=10, color=INK, pad=8)

            m = row["m"]
            axes[r, 0].set_ylabel(
                f"{row['label']}\nupright: {'yes' if row['rot'] else 'no'}   align: {row['align']}\n"
                f"AbsRel {m['abs_rel']:.3f}   RMSE {m['rmse']:.3f} m\nδ1 {m['delta1']:.3f}"
                + (f"   scale x{row['scale']:.2f}" if row["align"] != "none" else ""),
                fontsize=8.4, color=INK2, rotation=0, ha="right", va="center", labelpad=14)

        fig.suptitle(f"Depth Anything V3 configurations — {frame}",
                     fontsize=13, color=INK, x=0.5, y=0.995)
        fig.text(0.5, 0.002,
                 "All rows shown upright for comparison; “upright” states what the model was fed. "
                 "Black = no ground-truth depth. Error shares one 0–1 m scale, so unaligned rows saturate.",
                 ha="center", fontsize=8, color=MUTED)
        fig.tight_layout(rect=[0.10, 0.014, 1, 0.985])
        return fig

    args.out.parent.mkdir(parents=True, exist_ok=True)

    if is_pdf:
        from matplotlib.backends.backend_pdf import PdfPages
        with PdfPages(args.out) as pdf:
            for i, frame in enumerate(frames):
                fig = render(build_rows(frame), frame)
                pdf.savefig(fig, facecolor="white")
                plt.close(fig)
                print(f"[{i+1}/{len(frames)}] {frame}")
            pdf.infodict()["Title"] = "Depth Anything V3 - configuration comparison"
    else:
        fig = render(build_rows(frames[0]), frames[0])
        fig.savefig(args.out, dpi=150, facecolor="white")
        plt.close(fig)

    print(f"wrote {args.out}  ({len(frames)} frame(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
