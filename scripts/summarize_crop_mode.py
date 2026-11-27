#!/usr/bin/env python3
"""Tables of outputs/eval/crop_mode/*/results.json (scripts/eval_crop_mode.py),
per video and pooled, one column per crop mode.

    python scripts/summarize_crop_mode.py [--modes bbox,tight,blur,segmented]
    -> prints Markdown, writes outputs/eval/crop_mode/summary.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "outputs" / "eval" / "crop_mode"
VIDEOS = ["initial", "updated_same_angle", "updated_diff_angle", "update_in_same_video", "img6100"]

# (label, getter, pooling: "n" = weighted by detections/tracks/objects, "mean", "sum")
ROWS = [
    ("NOVIC naming, per detection (mapped, lenient)", lambda m: (m["naming"]["detection"]["mapped_lenient"],
                                                                 m["naming"]["detection"]["n"]), "n"),
    ("NOVIC naming, per detection (mapped, strict)", lambda m: (m["naming"]["detection"]["mapped_strict"],
                                                                m["naming"]["detection"]["n"]), "n"),
    ("NOVIC naming, per detection (raw NOVIC label, lenient)", lambda m: (m["naming"]["detection"]["raw_lenient"],
                                                                           m["naming"]["detection"]["n"]), "n"),
    ("NOVIC naming, per track average (mapped, lenient)", lambda m: (m["naming"]["track_average"]["mapped_lenient"],
                                                                     m["naming"]["track_average"]["n"]), "n"),
    ("Same vs different object, per detection (AUC)",
     lambda m: (m["matching"]["detections"]["auc_same_vs_different"], 1), "mean"),
    ("Different-object detection pairs above merge cosine",
     lambda m: (next(v for k, v in m["matching"]["detections"].items() if k.startswith("different_pairs_above")), 1),
     "mean"),
    ("Different-object track averages above merge cosine",
     lambda m: (next(v for k, v in m["matching"]["track_averages"].items() if k.startswith("different_pairs_above")),
                1), "mean"),
    ("Fusion: objects found (recall)", lambda m: (m["fusion"]["recall(object found, purity>=0.5)"],
                                                  m["fusion"]["objects_in_video"]), "n"),
    ("Fusion: fused objects that are one real object (precision)",
     lambda m: (m["fusion"]["precision(fused object is one real object)"], m["fusion"]["fused_objects"]), "n"),
    ("Fusion: objects split in two or more", lambda m: (len(m["fusion"]["split_objects(>1 fused object)"]), 1),
     "sum"),
    ("Fusion: fused objects mixing objects", lambda m: (len(m["fusion"]["merged_objects(purity<0.5 or 2 objects >=20%)"]),
                                                        1), "sum"),
    ("Fusion: naming of found objects (lenient)", lambda m: (m["fusion"]["naming_lenient"],
                                                             m["fusion"]["objects_in_video"]), "n"),
    ("Voxels: purity of found objects", lambda m: (m["fusion"]["voxel_purity"], 1), "mean"),
    ("Voxels: recall of found objects", lambda m: (m["fusion"]["voxel_recall"], 1), "mean"),
]


def fmt(v, pooling):
    if v is None:
        return "–"
    return f"{v:.0f}" if pooling == "sum" else f"{v:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", default="bbox,tight,blur,segmented")
    args = ap.parse_args()
    modes = args.modes.split(",")
    results = {v: json.loads((ROOT / v / "results.json").read_text()) for v in VIDEOS
               if (ROOT / v / "results.json").exists()}
    lines = []

    lines.append("## Pooled over " + ", ".join(results) + "\n")
    lines.append("| Metric | " + " | ".join(modes) + " |")
    lines.append("|---|" + "---|" * len(modes))
    for label, get, pooling in ROWS:
        cells = []
        for mode in modes:
            vals = [get(r["modes"][mode]) for r in results.values() if mode in r["modes"]]
            vals = [(x, n) for x, n in vals if x is not None]
            if not vals:
                cells.append("–")
            elif pooling == "sum":
                cells.append(fmt(sum(x for x, _ in vals), pooling))
            elif pooling == "n":
                cells.append(fmt(sum(x * n for x, n in vals) / max(sum(n for _, n in vals), 1), pooling))
            else:
                cells.append(fmt(sum(x for x, _ in vals) / len(vals), pooling))
        lines.append(f"| {label} | " + " | ".join(cells) + " |")

    lines.append("\n## Per video\n")
    for v, r in results.items():
        vox = r["voxels_per_detection"]
        lines.append(f"### {v} ({r['frames']} frames, {r['labelled_detections']} labelled detections, "
                     f"{len(r['objects'])} objects)\n")
        lines.append("| Metric | " + " | ".join(modes) + " |")
        lines.append("|---|" + "---|" * len(modes))
        for label, get, pooling in ROWS:
            cells = [fmt(get(r["modes"][m])[0], pooling) if m in r["modes"] else "–" for m in modes]
            lines.append(f"| {label} | " + " | ".join(cells) + " |")
        lines.append(f"\nVoxels per detection (same in every mode; fusion lifts the mask): mask "
                     f"{vox['mask']['voxels_mean']:.0f} voxels; lifting the box instead would add "
                     f"{vox['box']['voxels_mean'] - vox['mask']['voxels_mean']:.0f}, of which "
                     f"{vox['box']['other_object']:.0%} of the box's voxels belong to other objects and "
                     f"{vox['box']['background']:.0%} to background.\n")
    text = "\n".join(lines)
    (ROOT / "summary.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
