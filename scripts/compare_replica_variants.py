"""Compare tuning variants against the baseline on the Replica dev scenes.

Reads results/replica/<scene>/metrics.json (baseline) and
results/replica_tuning/<variant>/<scene>/metrics.json, and writes
results/replica_tuning/comparison.md: one table per scene and the mean over
the scenes every variant has finished.

    python scripts/compare_replica_variants.py --scenes office_0 room_0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent

COLUMNS = [  # (header, getter, is_percentage)
    ("Nodes", lambda m: m["nodes"]["nodes_total"], False),
    ("Found", lambda m: f'{m["nodes"]["valid_objects"]}/{m["nodes"]["gt_objects_visible"]}', False),
    ("Valid nodes", lambda m: m["nodes"]["valid_node_rate"], True),
    ("Node prec.", lambda m: m["nodes"]["node_precision"], True),
    ("Node recall", lambda m: m["nodes"]["node_recall"], True),
    ("Recall named", lambda m: m["nodes"]["node_recall_named"], True),
    ("N_dup", lambda m: m["nodes"]["n_dup"], False),
    ("Acc@1", lambda m: m["retrieval"].get("acc_at_k", {}).get("1"), True),
    ("Acc@5", lambda m: m["retrieval"].get("acc_at_k", {}).get("5"), True),
    ("AUC", lambda m: m["retrieval"].get("auc"), False),
    ("Obj R@5", lambda m: m["retrieval"].get("object_recall_mapped", {}).get("5"), True),
    ("mIoU ex1", lambda m: m["segmentation"]["1"]["miou"], True),
    ("mAcc ex1", lambda m: m["segmentation"]["1"]["macc"], True),
    ("F-mIoU ex1", lambda m: m["segmentation"]["1"]["fmiou"], True),
    ("mIoU ex4", lambda m: m["segmentation"]["4"]["miou"], True),
    ("F-mIoU ex4", lambda m: m["segmentation"]["4"]["fmiou"], True),
    ("Pipeline min", lambda m: (m["info"].get("pipeline_stage_seconds") or {}).get("total_seconds", 0) / 60, False),
]


def fmt(v, pct):
    if v is None:
        return "–"
    return f"{100 * v:.1f}" if pct else (f"{v:.3f}" if isinstance(v, float) and v < 1.5 else f"{v:.0f}"
                                         if isinstance(v, (int, float)) else str(v))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenes", nargs="+", default=["office_0", "room_0"])
    ap.add_argument("--variants", nargs="*", default=None, help="default: every folder under results/replica_tuning")
    args = ap.parse_args()
    tuning = REPO_ROOT / "results" / "replica_tuning"
    variants = args.variants or sorted(p.name for p in tuning.iterdir() if p.is_dir())
    variants = ["baseline"] + [v for v in variants if v != "baseline"]

    def load(variant, scene):
        p = (REPO_ROOT / "results" / "replica" / scene if variant == "baseline" else tuning / variant / scene)
        p = p / "metrics.json"
        return json.loads(p.read_text()) if p.exists() else None

    head = "| Variant | " + " | ".join(c[0] for c in COLUMNS) + " |"
    sep = "|---" * (len(COLUMNS) + 1) + "|"
    L = ["# Replica tuning: variants vs baseline", "",
         "Dev scenes only (the other six are held out for the final run). Percentages except Nodes, N_dup, AUC "
         "and minutes. Variant settings: `configs/replica_variants/`.", ""]
    data = {(v, s): load(v, s) for v in variants for s in args.scenes}
    for s in args.scenes:
        L += [f"## {s}", "", head, sep]
        for v in variants:
            m = data[(v, s)]
            L.append(f"| {v} | " + (" | ".join(fmt(g(m), p) for _, g, p in COLUMNS) if m else
                                    " | ".join("…" for _ in COLUMNS)) + " |")
        L.append("")
    L += [f"## Mean over {', '.join(args.scenes)}", "", head, sep]
    for v in variants:
        ms = [data[(v, s)] for s in args.scenes]
        if not all(ms):
            L.append(f"| {v} | " + " | ".join("…" for _ in COLUMNS) + " |")
            continue
        vals = []
        for _, g, p in COLUMNS:
            xs = [g(m) for m in ms]
            if any(isinstance(x, str) for x in xs):          # "found/visible": sum over scenes
                found, visible = (sum(int(x.split("/")[i]) for x in xs) for i in (0, 1))
                vals.append(f"{found}/{visible}")
                continue
            xs = [x for x in xs if x is not None]
            vals.append(fmt(float(np.mean(xs)) if xs else None, p))
        L.append(f"| {v} | " + " | ".join(vals) + " |")
    out = tuning / "comparison.md"
    out.write_text("\n".join(L) + "\n")
    print(out.read_text())
    return 0


if __name__ == "__main__":
    sys.exit(main())
