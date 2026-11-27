"""Side-by-side comparison of two Replica result folders on the same scenes
(e.g. the final configuration vs the same plus the mask-box alignment step).

    python scripts/compare_replica_runs.py --a results/replica_final --b results/replica_align \\
        --scenes office_0 room_0 --labels final final+align --out results/replica_align/comparison.md
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def load(rd: Path) -> dict:
    m = json.loads((rd / "metrics.json").read_text())
    log = (rd / "pipeline.log").read_text()
    th = re.search(r"Streaming fusion: (\d+) track commits.*?\{'detection_s': ([\d.]+), 'embedding_gpu0_s': ([\d.]+), "
                   r"'fusion_s': ([\d.]+)", log)
    tr = re.search(r"Tracker: (\{.*?\})\n", log)
    tr = json.loads(tr.group(1)) if tr else {}
    wall = (rd / "pipeline_wall_time_s.txt")
    n, s = m["nodes"], m["status_counts"] if "status_counts" in m else m["nodes"]["status_counts"]
    naming = m.get("naming", {}).get("methods", {})
    seg6 = m["segmentation"]["6"]
    return {
        "Nodes": n["nodes_total"],
        "Objects found (recall)": f"{n['valid_objects']}/{n['gt_objects_visible']} ({100 * n['node_recall']:.1f}%)",
        "Valid nodes": f"{100 * n['valid_node_rate']:.1f}%",
        "**N_dup (duplicates)**": n["n_dup"],
        "**Mixed nodes**": s.get("mixed", 0),
        "Wall / floor / ceiling nodes": s.get("structural", 0),
        "Unlabelled-object nodes": s.get("unlabelled", 0),
        "Node precision": f"{100 * n['node_precision']:.1f}%",
        "**Naming: NOVIC acceptable (WordNet)**": (f"{100 * naming['novic']['acceptable']:.1f}%" if naming else "–"),
        "Naming: pipeline label acceptable": (f"{100 * naming['pipeline']['acceptable']:.1f}%" if naming else "–"),
        "Embedding Acc@1 / Acc@5": f"{100 * m['retrieval']['acc_at_k']['1']:.1f} / {100 * m['retrieval']['acc_at_k']['5']:.1f}",
        "Segmentation mAcc / F-mIoU (exclude-6)": f"{100 * seg6['macc']:.1f} / {100 * seg6['fmiou']:.1f}",
        "Detections (tracker)": tr.get("detections", "–"),
        "Dropped by mask-box filter": tr.get("dropped_by_mask_box", 0),
        "Track commits": int(th.group(1)) if th else "–",
        "**Pipeline total (min)**": f"{float(wall.read_text()) / 60:.1f}" if wall.exists() else "–",
        "Fusion thread (min)": f"{float(th.group(4)) / 60:.1f}" if th else "–",
        "Fusion per commit (s)": f"{float(th.group(4)) / int(th.group(1)):.1f}" if th else "–",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--scenes", nargs="+", required=True)
    ap.add_argument("--labels", nargs=2, default=["A", "B"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    L = []
    for s in args.scenes:
        a, b = load(REPO_ROOT / args.a / s), load(REPO_ROOT / args.b / s)
        L += [f"## {s}", "", f"| Metric | {args.labels[0]} | {args.labels[1]} |", "|---|---|---|"]
        L += [f"| {k} | {a[k]} | {b[k]} |" for k in a]
        L.append("")
    text = "\n".join(L)
    print(text)
    if args.out:
        Path(REPO_ROOT / args.out).write_text(
            f"# {args.labels[0]} vs {args.labels[1]}\n\n`{args.a}` vs `{args.b}`\n\n" + text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
