"""Markdown comparison of pipeline runs against the published Replica baselines (ConceptGraphs, ReasoningGraph).

Each --runs entry is label=results_root with <root>/<scene>/metrics.json from eval_replica_scene.py. For every run:
  * 3D semantic segmentation, ConceptGraphs Table II protocol (exclude-6, pooled = summed confusion matrices)
  * embedding quality, ReasoningGraph Table I protocol (Acc@5/10/25, AUC over the 101 Replica classes, pooled ranks)
  * scene-graph quality next to ConceptGraphs Table I (node precision proxy, duplicates)
  * our object metrics (non-objects, object precision, recall, F1) and open-vocabulary naming (WordNet judge)
Baseline numbers are copied from the papers' tables (see BASELINES); the caveats in the output say where the
protocols differ.

    python scripts/make_baseline_comparison.py --runs "final=results/replica_final" \\
        "panoptic+probe r1=results/frontend_choice/panprobe/run1" --out results/frontend_choice/baseline_comparison.md
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from evaluation.conceptgraphs_semseg import pooled  # noqa: E402

SCENES = ["office_0", "office_1", "office_2", "office_3", "office_4", "room_0", "room_1", "room_2"]
# ConceptGraphs (Gu et al., ICRA 2024) Table II, Replica, exclude-6: (type, mAcc, F-mIoU)
CG_SEG = [("ConceptFusion", "zero-shot", 24.16, 31.31), ("ConceptFusion + SAM", "zero-shot", 31.53, 38.70),
          ("ConceptGraphs", "zero-shot", 40.63, 35.95), ("ConceptGraphs-Detector", "zero-shot", 38.72, 35.82),
          ("OpenSeg", "privileged (fine-tuned)", 41.19, 53.74), ("LSeg", "privileged (fine-tuned)", 33.39, 51.54),
          ("CLIPSeg (rd64-uni)", "privileged (fine-tuned)", 28.21, 39.84)]
# ReasoningGraph Table I, Replica columns: Acc@5, Acc@10, Acc@25, AUC, objects / scene
RG_EMB = [("HOV-SG", 3.66, 6.56, 11.26, 66.82, 72.6), ("ConceptGraphs", 20.70, 30.02, 41.72, 80.50, 40.5),
          ("ReasoningGraph", 53.99, 58.23, 63.16, 87.23, 13.3)]
# ConceptGraphs Table I: node precision (%) and duplicates, ConceptGraphs / ConceptGraphs-Detector (no office_4)
CG_GRAPH = {"office_0": (65, 59, 2, 0), "office_1": (65, 49, 0, 2), "office_2": (75, 67, 3, 3), "office_3": (68, 71, 5, 1),
            "room_0": (78, 56, 3, 4), "room_1": (77, 70, 4, 3), "room_2": (66, 54, 4, 2)}


def load_run(root: Path) -> dict:
    M = {s: json.loads((root / s / "metrics.json").read_text()) for s in SCENES if (root / s / "metrics.json").exists()}
    if not M:
        return {}
    seg = [M[s]["segmentation"]["6"]["confusion"] for s in M if M[s].get("segmentation", {}).get("6", {}).get("confusion")]
    segp = pooled(seg) if seg else None
    V = next(iter(M.values()))["retrieval"]["vocabulary_size"]
    ranks = np.concatenate([np.asarray(M[s]["retrieval"]["ranks"], int) for s in M])
    acc = {k: float((ranks <= k).mean()) for k in (1, 5, 10, 25)}
    auc = float(np.mean([(ranks <= k).mean() for k in range(1, V + 1)]))
    n = {s: M[s]["nodes"] for s in M}
    sc = lambda s, k: n[s]["status_counts"].get(k, 0)  # noqa: E731
    real = sum(sc(s, "correct") + sc(s, "wrong_name") for s in M)
    fake = sum(sc(s, "structural") + sc(s, "mixed") + sc(s, "phantom") for s in M)
    found, vis = sum(n[s]["valid_objects"] for s in M), sum(n[s]["gt_objects_visible"] for s in M)
    p, r = found / max(real + fake, 1), found / max(vis, 1)
    nam = [(M[s]["naming"]["nodes_scored"], M[s]["naming"]["methods"]) for s in M if M[s].get("naming")]
    tot = sum(k for k, _ in nam)
    novic = sum(k * m["novic"]["acceptable"] for k, m in nam) / tot if tot else None
    emb = sum(k * m["embedding"]["acceptable"] for k, m in nam) / tot if tot else None
    cs = [M[s]["closed_set_naming"] for s in M if M[s].get("closed_set_naming")]
    ctot = sum(c["nodes_scored"] for c in cs)
    closed = ({m: sum(c["nodes_scored"] * c[m]["1"] for c in cs) / ctot for m in ("novic_guided", "conceptgraphs_rule")}
              if ctot else None)
    return {"scenes": list(M), "seg": segp, "acc": acc, "auc": auc, "nodes": sum(n[s]["nodes_total"] for s in M),
            "per_scene_prec": {s: n[s]["node_precision"] for s in M}, "per_scene_dup": {s: n[s]["n_dup"] for s in M},
            "node_prec": sum(sc(s, "correct") for s in M) / max(sum(n[s]["nodes_counted"] for s in M), 1),
            "dup": sum(n[s]["n_dup"] for s in M), "fake": fake, "prec": p, "rec": r, "found": found, "vis": vis,
            "f1": 2 * p * r / (p + r) if p + r else 0, "novic": novic, "embed": emb, "closed": closed,
            "obj_per_scene": (real) / len(M)}


def f(x, d=1):
    return "–" if x is None else f"{x:.{d}f}"


def pc(x):
    return "–" if x is None else f"{100 * x:.1f}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True, help="label=results_root")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    runs = [(l, load_run(REPO_ROOT / r)) for l, r in (x.split("=", 1) for x in args.runs)]
    runs = [(l, d) for l, d in runs if d]
    incomplete = [f"{l} ({len(d['scenes'])}/8 scenes)" for l, d in runs if len(d["scenes"]) < 8]

    L = ["# Replica: our front ends vs ConceptGraphs and ReasoningGraph", "",
         f"Generated {time.strftime('%Y-%m-%d %H:%M')} by `scripts/make_baseline_comparison.py`. All 8 Replica scenes "
         "(office_0–4, room_0–2), ground-truth depth and poses, every 10th of 2,000 frames. Baseline numbers are "
         "copied from the papers' tables.", ""]
    if incomplete:
        L += [f"> **Incomplete runs:** {', '.join(incomplete)}. Their pooled numbers cover only the finished scenes.", ""]
    L += ["## Runs compared", "", "| Run | Scenes | Nodes | Non-objects | Object precision | Recall (found) | **F1** | Duplicates |",
          "|---|---|---|---|---|---|---|---|"]
    for l, d in runs:
        L.append(f"| {l} | {len(d['scenes'])} | {d['nodes']} | {d['fake']} | {pc(d['prec'])} | {pc(d['rec'])} "
                 f"({d['found']}/{d['vis']}) | **{d['f1']:.3f}** | {d['dup']} |")
    L += ["", "Object precision = objects found / (real-object nodes + non-object nodes); non-objects = wall / floor / "
          "ceiling pieces and nodes spanning several objects; duplicates earn nothing. The published systems do not "
          "report this metric.", ""]

    L += ["## 1. 3D semantic segmentation (ConceptGraphs Table II, exclude-6, pooled)", "",
          "| Method | Type | mAcc | F-mIoU |", "|---|---|---|---|"]
    L += [f"| {m} | {t} | {a:.2f} | {b:.2f} |" for m, t, a, b in CG_SEG[:4]]
    L += [f"| **Ours: {l}** | zero-shot | **{pc(d['seg']['macc']) if d['seg'] else '–'}** | "
          f"**{pc(d['seg']['fmiou']) if d['seg'] else '–'}** |" for l, d in runs]
    L += [f"| *{m}* | *{t}* | *{a:.2f}* | *{b:.2f}* |" for m, t, a, b in CG_SEG[4:]]
    L += ["", "- Exclude-6 drops other, floor, wall, ceiling, door and window, as ConceptGraphs' `--n_exclude 6`. "
          "Our pipeline does not map background, so this is the like-for-like row.",
          "- Encoder caveat: class names are embedded with DFN5B ViT-H-14 (NOVIC's CLIP); ConceptGraphs used LAION "
          "ViT-H-14. An encoder-matched ConceptGraphs rerun would make this airtight.",
          "- Frame stride works against us (every 10th frame vs every 5th for ConceptGraphs).", ""]

    L += ["## 2. Embedding quality (ReasoningGraph Table I protocol)", "",
          "| Method | Acc@5 | Acc@10 | Acc@25 | AUC | Objects / scene |", "|---|---|---|---|---|---|"]
    L += [f"| {m} | {a:.2f} | {b:.2f} | {c:.2f} | {u:.2f} | {o} |" for m, a, b, c, u, o in RG_EMB]
    L += [f"| **Ours: {l}** | **{pc(d['acc'][5])}** | **{pc(d['acc'][10])}** | **{pc(d['acc'][25])}** | "
          f"**{100 * d['auc']:.2f}** | {d['obj_per_scene']:.0f} |" for l, d in runs]
    L += ["", "- Each node's image embedding is ranked against Replica's 101 class names; not a naming score (the class "
          "list is given).", "- ReasoningGraph ranks over 500+ labels (it reports Acc@500), so our lead at Acc@k is "
          "inflated and the AUCs are not directly comparable; ReasoningGraph also mixes the detector label's text "
          "embedding into each object feature, ours is image-only.", ""]

    L += ["## 3. Scene-graph quality (ConceptGraphs Table I)", "",
          "| Scene | CG prec. | CG-D prec. | CG dup. | CG-D dup. | " + " | ".join(f"{l} prec. / dup." for l, _ in runs) + " |",
          "|" + "---|" * (5 + len(runs))]
    for s in SCENES:
        cg = CG_GRAPH.get(s)
        cells = [f"{pc(d['per_scene_prec'].get(s))} / {d['per_scene_dup'].get(s, '–')}" for _, d in runs]
        L.append(f"| {s} | " + (" | ".join(str(x) for x in cg) if cg else "– | – | – | –") + " | " + " | ".join(cells) + " |")
    L.append("| **All** | 71 (avg) | 61 (avg) | 21 | 15 | " + " | ".join(
        f"{pc(d['node_prec'])} / {d['dup']}" for _, d in runs) + " |")
    L += ["", "- **Not like-for-like.** ConceptGraphs' precision is human-judged (2 of 3 judges accept the caption); ours "
          "counts a node right only if its name maps to the GT class by text similarity, a stricter automatic proxy. "
          "ConceptGraphs does not report office_4 (its duplicates are summed over 7 scenes, ours over 8).", ""]

    L += ["## 4. Naming", "", "| Run | NOVIC acceptable (open vocabulary) | Embedding lookup acceptable | "
          "NOVIC top-1 (class list given) | ConceptGraphs rule top-1 |", "|---|---|---|---|---|"]
    for l, d in runs:
        c = d["closed"] or {}
        L.append(f"| {l} | {pc(d['novic'])} | {pc(d['embed'])} | {pc(c.get('novic_guided'))} | {pc(c.get('conceptgraphs_rule'))} |")
    L += ["", "- Open vocabulary: both methods name from NOVIC's 42,919 nouns; a WordNet judge grades exact / more "
          "specific as acceptable. Class list given: both pick from the scene's Replica classes; '–' = not computed "
          "for that run (scripts/eval_closed_set_naming.py).", ""]

    L += ["## Reading this", "",
          "- Pick the front end by **F1 and non-objects** (first table), pooled over 8 scenes and averaged over the two runs; "
          "segmentation and Acc@k say whether the cleaner map costs semantic coverage.",
          "- Differences between the two runs of one configuration are the noise floor; a gap smaller than that is a tie.", ""]
    text = "\n".join(L) + "\n"
    if args.out:
        Path(args.out).write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
