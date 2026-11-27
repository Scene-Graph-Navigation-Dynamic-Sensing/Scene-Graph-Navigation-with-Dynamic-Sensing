"""Compare detection-validation ablations against the baseline on the same Replica scenes.

Each run folder is <root>/<scene>/ as written by eval_replica_scene.py. Per run it reports, summed / pooled over the
scenes: non-object nodes, object precision, recall, F1, duplicates, the human-like re-scoring (found without wall
fixtures, fake nodes), naming, Acc@5, segmentation, time, what the validation dropped, and the number of
parent/child scene-graph edges whose two nodes carry the same name (the table-with-laptop symptom).

    python scripts/compare_validation_runs.py --scenes office_0 room_0 \\
        --runs final=results/replica_final stuff=results/replica_validation/val_stuff_detect ... \\
        --out results/replica_validation/comparison.md
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def same_name_pairs(graph: dict) -> list[tuple[str, str]]:
    """(parent, child) node ids on a tree edge where both are objects with the same, known name."""
    out = []

    def walk(node):
        for c in node.get("children", []):
            if (node.get("has_embedding") and c.get("has_embedding") and node["node_class"] == c["node_class"]
                    and node["node_class"] != "unknown"):
                out.append((node["node_id"], c["node_id"]))
            walk(c)
    for root in graph if isinstance(graph, list) else [graph]:
        walk(root)
    return out


def all_edges(graph: dict) -> list[tuple[dict, dict]]:
    out = []

    def walk(node):
        for c in node.get("children", []):
            if node.get("has_embedding") and c.get("has_embedding"):
                out.append((node, c))
            walk(c)
    for root in graph if isinstance(graph, list) else [graph]:
        walk(root)
    return out


def classify_pairs(graph: dict, rd: Path) -> dict:
    """Same-name parent/child pairs sorted by ground truth, and the table-with-laptop count over ALL edges.
      duplicate          both nodes on the same GT object (merging them is right)
      same_class         two different real objects of the same GT class (two cushions): correct to keep
      structural_parent  the parent is a wall / floor / ceiling (or multi-object) node named like the object in it
      other              anything else
    named_after_child: parent/child edges (both on real objects) where the parent's name fits the CHILD's GT class
    (WordNet exact / specific) but not its own -- a table named "laptop"."""
    import csv
    from evaluation.naming_judge import grade
    rows = {r["node_id"]: r for r in csv.DictReader(open(rd / "nodes.csv"))}
    real = ("correct", "wrong_name")
    cat = {"duplicate": 0, "same_class": 0, "structural_parent": 0, "other": 0}
    after_child = []
    for p, c in all_edges(graph):
        rp, rc = rows.get(p["node_id"]), rows.get(c["node_id"])
        if not rp or not rc:
            continue
        if p["node_class"] == c["node_class"] and p["node_class"] != "unknown":
            if rp["status"] in real and rc["status"] in real:
                k = ("duplicate" if rp["gt_object_id"] == rc["gt_object_id"] else
                     "same_class" if rp["gt_class"] == rc["gt_class"] else "other")
            elif rp["status"] in ("structural", "mixed"):
                k = "structural_parent"
            else:
                k = "other"
            cat[k] += 1
        if rp["status"] in real and rc["status"] in real and rp["gt_object_id"] != rc["gt_object_id"]:
            ok = ("exact", "specific")
            if grade(p["node_class"], rc["gt_class"]) in ok and grade(p["node_class"], rp["gt_class"]) not in ok:
                after_child.append(f"{p['node_id']}({rp['gt_class']})>{c['node_id']}({rc['gt_class']})")
    return {"pairs": cat, "named_after_child": after_child}


def load_scene(rd: Path) -> dict | None:
    if not (rd / "metrics.json").exists():
        return None
    if not (rd / "reevaluation_grouped.json").exists():
        subprocess.run([sys.executable, str(REPO_ROOT / "scripts" / "reevaluate_grouped.py"), str(rd)],
                       cwd=REPO_ROOT, capture_output=True)
    m = json.loads((rd / "metrics.json").read_text())
    n, sc = m["nodes"], m["nodes"]["status_counts"]
    re_g = json.loads((rd / "reevaluation_grouped.json").read_text()) if (rd / "reevaluation_grouped.json").exists() else {}
    log = (rd / "pipeline.log").read_text() if (rd / "pipeline.log").exists() else ""
    tr = re.search(r"Tracker: (\{.*?\})\n", log)
    tr = json.loads(tr.group(1)) if tr else {}
    nested = re.search(r"Nested-name suppression renamed (\d+)", log)
    graph = json.loads(next((rd / "scene_graph").glob("scene_graph.json")).read_text())
    wall = rd / "pipeline_wall_time_s.txt"
    naming = m.get("naming", {}).get("methods", {})
    seg = m.get("segmentation", {}).get("6")
    return {
        "nodes": n["nodes_total"], "real": sc.get("correct", 0) + sc.get("wrong_name", 0),
        "fake": sc.get("structural", 0) + sc.get("mixed", 0) + sc.get("phantom", 0),
        "structural": sc.get("structural", 0), "mixed": sc.get("mixed", 0),
        "found": n["valid_objects"], "visible": n["gt_objects_visible"], "dup": n["n_dup"],
        "h_found": re_g.get("found_without_fixtures"), "h_visible": re_g.get("visible_without_fixtures"),
        "h_fake": re_g.get("fake_nodes"),
        "novic": naming.get("novic", {}).get("acceptable"), "acc5": m["retrieval"]["acc_at_k"]["5"],
        "macc": seg["macc"] if seg else None, "fmiou": seg["fmiou"] if seg else None,
        "minutes": float(wall.read_text()) / 60 if wall.exists() else None,
        "dropped": {k: v for k, v in tr.get("dropped_by_validation", {}).items() if v},
        "dropped_mask_box": tr.get("dropped_by_mask_box", 0),
        "rejected_tracks": len(re.findall(r"rejected: ", log)),
        "renamed": int(nested.group(1)) if nested else 0,
        "same_name": same_name_pairs(graph),
        "pair_classes": classify_pairs(graph, rd),
    }


def pct(x):
    return "–" if x is None else f"{100 * x:.1f}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenes", nargs="+", required=True)
    ap.add_argument("--runs", nargs="+", required=True, help="label=results_root")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    runs = [r.split("=", 1) for r in args.runs]
    L = ["# Detection validation ablation", "",
         f"Scenes: {', '.join(args.scenes)}. Each run = the final configuration plus ONE check (baseline first). "
         "Counts are summed over the scenes; precision / recall / F1 are pooled (sums, then ratio).", "",
         "- **Object precision** = objects found / (real-object nodes + non-object nodes); duplicates earn nothing.",
         "- **Non-objects** = wall / floor / ceiling nodes + nodes spanning several objects (+ floating).",
         "- **Found (human)** = re-scoring with window panes grouped and wall fixtures (plugs, switches, vents) left out.",
         "- **Same-name pairs** = scene-graph parent/child edges whose two object nodes have the same name "
         "(the table-with-laptop symptom).", "",
         "| Run | Nodes | Non-objects | **Obj. precision** | **Recall (found)** | **F1** | Dup | Found (human) | "
         "Fake (human) | Same-name pairs | NOVIC named | Acc@5 | Seg mAcc / F-mIoU | Time (min) | Dropped / rejected |",
         "|" + "---|" * 15]
    detail = []
    for label, root in runs:
        S = {s: load_scene(REPO_ROOT / root / s) for s in args.scenes}
        if any(v is None for v in S.values()):
            L.append(f"| {label} | " + " | ".join(["(pending)"] + [""] * 13) + " |")
            continue
        V = list(S.values())
        tot = lambda k: sum(v[k] for v in V if v[k] is not None)
        mean = lambda k: (sum(v[k] for v in V if v[k] is not None) / max(sum(v[k] is not None for v in V), 1)
                          if any(v[k] is not None for v in V) else None)
        p = tot("found") / max(tot("real") + tot("fake"), 1)
        r = tot("found") / max(tot("visible"), 1)
        f1 = 2 * p * r / (p + r) if p + r else 0
        dropped = {}
        for v in V:
            for k, c in v["dropped"].items():
                dropped[k] = dropped.get(k, 0) + c
        if tot("dropped_mask_box"):
            dropped["mask_box"] = tot("dropped_mask_box")
        if tot("rejected_tracks"):
            dropped["tracks rejected"] = tot("rejected_tracks")
        if tot("renamed"):
            dropped["renamed"] = tot("renamed")
        same = sum(len(v["same_name"]) for v in V)
        L.append(f"| {label} | {tot('nodes')} | {tot('fake')} | {pct(p)} | {pct(r)} ({tot('found')}/{tot('visible')}) | "
                 f"{f1:.3f} | {tot('dup')} | {tot('h_found')}/{tot('h_visible')} | {tot('h_fake')} | {same} | "
                 f"{pct(mean('novic'))} | {pct(mean('acc5'))} | {pct(mean('macc'))} / {pct(mean('fmiou'))} | "
                 f"{tot('minutes'):.0f} | {', '.join(f'{k} {c}' for k, c in dropped.items()) or '–'} |")
        for s, v in S.items():
            ps = v["found"] / max(v["real"] + v["fake"], 1)
            rs = v["found"] / max(v["visible"], 1)
            detail.append(f"| {label} | {s} | {v['nodes']} | {v['structural']} + {v['mixed']} | {pct(ps)} | {pct(rs)} | "
                          f"{2 * ps * rs / (ps + rs) if ps + rs else 0:.3f} | {v['dup']} | "
                          f"{', '.join(f'{a}>{b}' for a, b in v['same_name']) or '–'} |")
    L += ["", "## Per scene", "", "| Run | Scene | Nodes | Non-objects (structural + mixed) | Obj. precision | Recall | "
          "F1 | Dup | Same-name parent>child |", "|" + "---|" * 9] + detail
    text = "\n".join(L) + "\n"
    print(text)
    if args.out:
        Path(args.out).write_text(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
