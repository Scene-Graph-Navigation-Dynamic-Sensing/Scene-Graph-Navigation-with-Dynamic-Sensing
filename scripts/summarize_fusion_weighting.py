#!/usr/bin/env python3
"""Pooled table of outputs/eval/crop_mode/*/fusion_weighting.json
(scripts/eval_fusion_weighting.py); writes fusion_weighting_summary.md."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "outputs" / "eval" / "crop_mode"
VIDEOS = ["initial", "updated_same_angle", "updated_diff_angle", "update_in_same_video", "img6100"]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="", help="result files fusion_weighting<tag>*.json, e.g. _keep")
    args = ap.parse_args()
    data = {}
    for v in VIDEOS:  # fusion_weighting<tag>.json plus any parallel parts fusion_weighting<tag>_p<n>.json
        for f in sorted((ROOT / v).glob(f"fusion_weighting{args.tag}*.json")):
            if not args.tag and "_keep" in f.name:
                continue
            data.setdefault(v, {}).update(json.loads(f.read_text()))
    first = next(iter(data.values()))
    schemes = [s for s in first if all(s in d for d in data.values())]
    rows = []
    for s in schemes:
        rs = [d[s] for d in data.values() if s in d]
        if len(rs) < len(data):
            continue
        n_obj = sum(r["objects_in_video"] for r in rs)
        n_fused = sum(r["fused_objects"] for r in rs)
        good = sum(r["one_real_object"] for r in rs)
        rows.append((s, sum(r["found"] for r in rs) / n_obj, good / max(n_fused, 1),
                     sum(r["split_objects"] for r in rs), sum(r["mixed_objects"] for r in rs),
                     sum(r["voxel_purity"] * r["one_real_object"] for r in rs) / max(good, 1),
                     sum(r["voxel_recall"] * r["one_real_object"] for r in rs) / max(good, 1),
                     sum(r["named_strict"] for r in rs) / n_obj, sum(r["named_lenient"] for r in rs) / n_obj,
                     " / ".join(f"{r['found']}/{r['objects_in_video']}" for r in rs)))
    lines = [f"Pooled over {', '.join(data)} ({sum(d[schemes[0]]['objects_in_video'] for d in data.values())} objects)\n",
             "| Weighting of the running average | objects found | fused objects = one object | splits | "
             "mixed | voxel purity | voxel recall | named (strict) | named (lenient) | found per video |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r[0]} | {r[1]:.3f} | {r[2]:.3f} | {r[3]} | {r[4]} | {r[5]:.3f} | {r[6]:.3f} | {r[7]:.3f} | "
                     f"{r[8]:.3f} | {r[9]} |")
    text = "\n".join(lines)
    (ROOT / f"fusion_weighting{args.tag}_summary.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
