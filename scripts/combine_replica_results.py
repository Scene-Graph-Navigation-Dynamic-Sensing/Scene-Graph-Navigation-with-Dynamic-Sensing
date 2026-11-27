"""Combine per-scene Replica results into results/replica/summary.md.

Reads results/replica/<scene>/metrics.json for every evaluated scene and
reports each metric per scene and two ways across scenes:
  * mean over scenes;
  * pooled: node statuses and Acc@k ranks summed over scenes, and, for
    segmentation, ConceptGraphs' "all" row (confusion matrices summed over
    scenes, classes with GT points kept), as eval_replica_semseg.py does.

    python scripts/combine_replica_results.py
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.conceptgraphs_semseg import pooled  # noqa: E402

KS = (1, 3, 5, 10, 25, 50)
EXCLUDES = ("6", "4", "1")   # headline first


def pct(x):
    return "–" if x is None else f"{100 * x:.1f}"


def num(x):
    return "–" if x is None else f"{x:.3f}"


def mean(values):
    v = [x for x in values if x is not None]
    return float(np.mean(v)) if v else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results_root", default="results/replica")
    args = ap.parse_args()
    root = REPO_ROOT / args.results_root
    scenes = sorted(p.parent.name for p in root.glob("*/metrics.json"))
    if not scenes:
        print(f"no metrics.json under {root}")
        return 1
    M = {s: json.loads((root / s / "metrics.json").read_text()) for s in scenes}

    L = ["# Replica: open-vocabulary object quality, all evaluated scenes", "",
         f"Scenes: {', '.join(scenes)} ({len(scenes)} of 8). Per-scene reports: `<scene>/results.md`. "
         f"Protocol: `docs/evaluation_metrics.md` (group A); segmentation follows ConceptGraphs' "
         f"`eval_replica_semseg.py`.", ""]

    # headline: ConceptGraphs reports exclude-6 (README: eval_replica_semseg.py --n_exclude 6)
    seg_scenes = [s for s in scenes if M[s].get("segmentation")]
    seg6 = [M[s]["segmentation"]["6"] for s in seg_scenes]
    p6 = pooled([np.array(g["confusion"]) for g in seg6])
    L += ["## Comparison with ConceptGraphs (background excluded)", "",
          "ConceptGraphs' Table II numbers come from `eval_replica_semseg.py --n_exclude 6` (their README): "
          "other, floor, wall, ceiling, door and window are excluded, as here. Our pipeline does not map "
          "background, so this is the like-for-like setting.", "",
          "| Replica, exclude-6 | mAcc | F-mIoU |", "|---|---|---|",
          "| ConceptGraphs (paper) | 40.63 | 35.95 |",
          "| ConceptGraphs-Detector (paper) | 38.72 | 35.82 |",
          f"| Ours, mean over {len(seg_scenes)} scenes | {pct(mean([g['macc'] for g in seg6]))} | "
          f"{pct(mean([g['fmiou'] for g in seg6]))} |",
          f"| Ours, pooled (ConceptGraphs' \"all\" row) | {pct(p6['macc'])} | {pct(p6['fmiou'])} |", "",
          "Differences: class names are embedded with NOVIC's CLIP model (DFN5B ViT-H-14) instead of LAION "
          "ViT-H-14, since our node embeddings live in that space; frames are every "
          f"{M[scenes[0]]['info'].get('stride', 10)}th of NICE-SLAM's 2,000 (ConceptGraphs: every 5th).", ""]

    # object metrics
    # Object precision counts EVERY node the pipeline outputs: a wall / floor / ceiling piece, a node spanning
    # several objects or a floating node is a non-object and lowers it, and a duplicate earns no credit
    # (objects found / (real-object nodes + non-object nodes)); nodes on objects Replica leaves unlabelled
    # are left out of both sides. With recall it gives an F1 that brute-force proposing cannot raise.
    L += ["## Object metrics", "",
          "Object precision = objects found / (real-object nodes + non-object nodes), duplicates not credited; "
          "non-objects = wall / floor / ceiling pieces, nodes spanning several objects, floating nodes. "
          "F1 combines it with recall, so proposing more nodes only helps if they are real, distinct objects. "
          "Node precision is the ConceptGraphs-style proxy (excludes wall / floor / ceiling nodes, needs the right name).",
          "", "| Scene | Frames | Nodes | Non-objects | **Object precision** | Node recall | **F1** | N_dup | "
          "Valid nodes | Node precision |", "|---|---|---|---|---|---|---|---|---|---|"]
    rows = []
    for s in scenes:
        r, n = M[s]["retrieval"], M[s]["nodes"]
        acc = r.get("acc_at_k", {})
        sc = n["status_counts"]
        real = sc.get("correct", 0) + sc.get("wrong_name", 0)
        fake = sc.get("structural", 0) + sc.get("mixed", 0) + sc.get("phantom", 0)
        oprec = n["valid_objects"] / (real + fake) if real + fake else None
        f1 = 2 * oprec * n["node_recall"] / (oprec + n["node_recall"]) if oprec and n["node_recall"] else None
        row = {"precision": n["node_precision"], "valid": n["valid_node_rate"], "recall": n["node_recall"],
               "n_dup": n["n_dup"], "acc1": acc.get("1"), "acc5": acc.get("5"), "acc10": acc.get("10"),
               "auc": r.get("auc"), "oprec": oprec, "f1": f1, "fake": fake, "real": real,
               "found": n["valid_objects"], "visible": n["gt_objects_visible"]}
        rows.append(row)
        L.append(f"| {s} | {M[s]['info']['frames']} | {n['nodes_total']} | {fake} | {pct(oprec)} | "
                 f"{pct(row['recall'])} ({n['valid_objects']}/{n['gt_objects_visible']}) | {num(f1)} | {row['n_dup']} | "
                 f"{pct(row['valid'])} | {pct(row['precision'])} |")
    L.append("| **mean** | | | " + " | ".join(
        [f"{np.mean([r['fake'] for r in rows]):.1f}", pct(mean([r["oprec"] for r in rows])),
         pct(mean([r["recall"] for r in rows])), num(mean([r["f1"] for r in rows])),
         f"{np.mean([r['n_dup'] for r in rows]):.1f}", pct(mean([r["valid"] for r in rows])),
         pct(mean([r["precision"] for r in rows]))]) + " |")
    fp = sum(r["found"] for r in rows) / max(sum(r["real"] + r["fake"] for r in rows), 1)
    fr = sum(r["found"] for r in rows) / max(sum(r["visible"] for r in rows), 1)
    L.append(f"| **all (pooled)** | | {sum(M[s]['nodes']['nodes_total'] for s in scenes)} | "
             f"{sum(r['fake'] for r in rows)} | {pct(fp)} | {pct(fr)} | {num(2 * fp * fr / (fp + fr))} | "
             f"{sum(r['n_dup'] for r in rows)} | | |")

    status = Counter()
    for s in scenes:
        status.update(M[s]["nodes"]["status_counts"])
    counted = sum(status[k] for k in ("correct", "wrong_name", "mixed", "phantom"))
    found = sum(M[s]["nodes"]["valid_objects"] for s in scenes)
    visible = sum(M[s]["nodes"]["gt_objects_visible"] for s in scenes)
    L += ["", "Pooled over scenes:", "",
          f"- Node precision {pct(status['correct'] / counted if counted else None)}% "
          f"({status['correct']}/{counted} counted nodes); node recall "
          f"{pct(found / visible if visible else None)}% ({found}/{visible} visible objects); "
          f"N_dup {sum(r['n_dup'] for r in rows)}",
          f"- Node status totals: {dict(status)}",
          "- Node precision counts a name as right when it maps to the GT class by text similarity, an automatic "
          "proxy for ConceptGraphs' human-judged precision; the naming section below is the fairer measure.", ""]

    # open-vocabulary naming (scripts/eval_open_vocab_naming.py)
    named = [s for s in scenes if "naming" in M[s]]
    if named:
        L += ["## Open-vocabulary naming (NOVIC vs embedding lookup)", "",
              "Both methods name each node from the same open vocabulary (NOVIC's "
              f"{M[named[0]]['naming']['vocabulary_size']:,} target nouns) and one WordNet judge grades both "
              "(`evaluation/naming_judge.py`). NOVIC: decoded top-1 noun. Embedding lookup: nearest target noun to "
              "the node's image embedding. Acceptable = exact/synonym or more specific; too generic = an ancestor "
              "of the GT class (\"furniture\" for chair). Pipeline label (final name, mapped onto a 245-word "
              "everyday list holding 53 of the 101 Replica classes) is shown for reference, not as open-set.", "",
              "| Scene | Nodes | NOVIC acceptable | Embedding acceptable | NOVIC too generic | "
              "Embedding too generic | Pipeline label acceptable |", "|---|---|---|---|---|---|---|"]
        tot = {m: Counter() for m in ("novic", "embedding", "pipeline")}
        for s in named:
            g = M[s]["naming"]
            meth = g["methods"]
            for m in tot:
                tot[m].update(meth[m]["counts"])
            L.append(f"| {s} | {g['nodes_scored']} | {pct(meth['novic']['acceptable'])} | "
                     f"{pct(meth['embedding']['acceptable'])} | {pct(meth['novic']['share']['generic'])} | "
                     f"{pct(meth['embedding']['share']['generic'])} | {pct(meth['pipeline']['acceptable'])} |")
        n_all = sum(M[s]["naming"]["nodes_scored"] for s in named)

        def share(m, *grades):
            return sum(tot[m][g] for g in grades) / n_all if n_all else None
        L.append(f"| **all (pooled)** | {n_all} | {pct(share('novic', 'exact', 'specific'))} | "
                 f"{pct(share('embedding', 'exact', 'specific'))} | {pct(share('novic', 'generic'))} | "
                 f"{pct(share('embedding', 'generic'))} | {pct(share('pipeline', 'exact', 'specific'))} |")
        L += ["", "Grade totals (exact / more specific / too generic / wrong / no name): " + "; ".join(
            f"{m} {' / '.join(str(tot[m][g]) for g in ('exact', 'specific', 'generic', 'wrong', 'no_name'))}"
            for m in tot), ""]

    # closed-set naming (scripts/eval_closed_set_naming.py)
    closed = [s for s in scenes if "closed_set_naming" in M[s]]
    if closed:
        C = {s: M[s]["closed_set_naming"] for s in closed}
        L += ["## Naming with the class list given (NOVIC vs ConceptGraphs)", "",
              f"Both methods get the scene's Replica class list ({C[closed[0]]['classes']} classes) and pick a name "
              "from it for the same nodes and embeddings. **NOVIC (guided)**: its beam search restricted to the class "
              "names (guided decoding, scores renormalised over the list). **ConceptGraphs rule**: the class whose "
              "text embedding (\"an image of {}\") is nearest to the node's image embedding, as ConceptGraphs assigns "
              "classes. A name counts only if it equals the GT class. This applies ConceptGraphs' class-assignment rule "
              "to our nodes, not a run of the ConceptGraphs system. NOVIC can't encode the class name \"blinds\" and "
              "is given \"blind\" instead; \"logo\" can't be encoded at all (no scored node is a logo).", "",
              "| Scene | Nodes | NOVIC top-1 | ConceptGraphs top-1 | NOVIC top-3 | ConceptGraphs top-3 | "
              "NOVIC top-5 | ConceptGraphs top-5 |", "|---|---|---|---|---|---|---|---|"]
        for s in closed:
            c = C[s]
            nv, cgv = c["novic_guided"], c["conceptgraphs_rule"]
            L.append(f"| {s} | {c['nodes_scored']} | " + " | ".join(
                f"{pct(nv[k])} | {pct(cgv[k])}" for k in ("1", "3", "5")) + " |")
        nr = np.concatenate([np.array(C[s]["novic_ranks"], dtype=int) for s in closed])
        cr = np.concatenate([np.array(C[s]["conceptgraphs_ranks"], dtype=int) for s in closed])
        L.append("| **mean** | | " + " | ".join(
            f"{pct(mean([C[s]['novic_guided'][k] for s in closed]))} | "
            f"{pct(mean([C[s]['conceptgraphs_rule'][k] for s in closed]))}" for k in ("1", "3", "5")) + " |")
        L.append(f"| **all (pooled)** | {len(nr)} | " + " | ".join(
            f"{pct(float(((nr > 0) & (nr <= k)).mean()))} | {pct(float((cr <= k).mean()))}" for k in (1, 3, 5))
            + " |")
        L += ["", "Per node: `<scene>/closed_set_naming.csv`.", ""]

    # closed-set embedding quality
    ranks = np.concatenate([np.array(M[s]["retrieval"].get("ranks", []), dtype=int) for s in scenes])
    V = M[scenes[0]]["retrieval"]["vocabulary_size"]
    L += ["## Embedding quality (closed set)", "",
          f"The node's image embedding ranked against Replica's {V} class names. This measures how well the stored "
          "embedding separates the classes, for comparison with closed-set methods (HOV-SG, ReasoningGraph); it is "
          "not a naming score, since the class list is given in advance.", "",
          "| Scene | Acc@1 | Acc@5 | Acc@10 | AUC |", "|---|---|---|---|---|"]
    for s, row in zip(scenes, rows):
        L.append(f"| {s} | {pct(row['acc1'])} | {pct(row['acc5'])} | {pct(row['acc10'])} | {num(row['auc'])} |")
    L.append("| **mean** | " + " | ".join([pct(mean([r["acc1"] for r in rows])), pct(mean([r["acc5"] for r in rows])),
                                       pct(mean([r["acc10"] for r in rows])), num(mean([r["auc"] for r in rows]))])
             + " |")
    if len(ranks):
        L += ["", "Pooled: " + ", ".join(f"Acc@{k} {pct(float((ranks <= k).mean()))}%" for k in KS if k <= V)
              + f"; AUC {np.mean([(ranks <= k).mean() for k in range(1, V + 1)]):.3f} "
              f"({len(ranks)} nodes, {V} classes)"]

    # segmentation
    for ex in EXCLUDES:
        L += ["", f"## Segmentation, ConceptGraphs exclude-{ex} "
                  f"({', '.join(M[seg_scenes[0]]['segmentation'][ex]['excluded'])} excluded)", "",
              "| Scene | mIoU | mAcc | F-mIoU | mPrecision | mF1 | Classes |", "|---|---|---|---|---|---|---|"]
        seg = [M[s]["segmentation"][ex] for s in seg_scenes]
        for s, g in zip(seg_scenes, seg):
            L.append(f"| {s} | {pct(g['miou'])} | {pct(g['macc'])} | {pct(g['fmiou'])} | {pct(g['mprecision'])} | "
                     f"{pct(g['mf1'])} | {g['num_classes']} |")
        L.append(f"| **mean** | {pct(mean([g['miou'] for g in seg]))} | {pct(mean([g['macc'] for g in seg]))} | "
                 f"{pct(mean([g['fmiou'] for g in seg]))} | {pct(mean([g['mprecision'] for g in seg]))} | "
                 f"{pct(mean([g['mf1'] for g in seg]))} | |")
        p = pooled([np.array(g["confusion"]) for g in seg])
        L.append(f"| **all (pooled)** | {pct(p['miou'])} | {pct(p['macc'])} | {pct(p['fmiou'])} | "
                 f"{pct(p['mprecision'])} | {pct(p['mf1'])} | {p['num_classes']} |")
    L += ["", "Class names are embedded with NOVIC's CLIP model (DFN5B ViT-H-14), not ConceptGraphs' LAION "
              "ViT-H-14, because our node embeddings live in that space."]
    (root / "summary.md").write_text("\n".join(L) + "\n")
    print((root / "summary.md").read_text())
    return 0


if __name__ == "__main__":
    sys.exit(main())
