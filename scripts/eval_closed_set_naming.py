"""Closed-set naming: both methods get the Replica class list.

  NOVIC (guided)     NOVIC's beam search restricted to the scene's Replica class
                     names (guided decoding, `guide_targets`), scores renormalised
                     over the list (gencfg ..._gr_...)
  ConceptGraphs rule each node gets the class whose text embedding ("an image of
                     {}") is nearest to its image embedding, as ConceptGraphs
                     assigns classes (the embedding ranking behind Acc@k)

Same nodes (matched cleanly to a countable object), same embeddings, same class
list, and a name is right only when it equals the GT class, so no judge is
needed. This applies ConceptGraphs' class-assignment rule to our nodes; it is
not a run of the ConceptGraphs system.

Reads finished evaluation folders; writes <scene>/closed_set_naming.csv and
metrics.json["closed_set_naming"]. scripts/combine_replica_results.py turns them
into the summary section.

    HF_HUB_OFFLINE=1 python scripts/eval_closed_set_naming.py results/replica_final/*/
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.replica_gt import IGNORED_CLASSES, normalize  # noqa: E402

GT_ROOT = Path("/home/ravindu/SceneGraphNav/datasets/replica/extracted")
GENCFG = "beam_k10_vnone_gr_t1_a0"      # guided, renormalised over the list
KS = (1, 3, 5)
# class names NOVIC's tokenizer can't encode, given to it in a form it can and mapped back
# ("logo" has no encodable form; no scored node is a logo)
NOVIC_ALIASES = {"blinds": "blind"}
BACK = {v: k for k, v in NOVIC_ALIASES.items()}


def class_list(scene: str) -> list[str]:
    """The scene's Replica classes, exactly as the Acc@k ranking uses them (evaluation/replica_gt.py)."""
    info = json.loads((GT_ROOT / scene / "habitat" / "info_semantic.json").read_text())
    return sorted({normalize(c["name"]) for c in info["classes"]} - IGNORED_CLASSES)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("result_dir", nargs="+")
    ap.add_argument("--gpu", default="0")
    args = ap.parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.gpu)

    cfg = yaml.safe_load(open(REPO_ROOT / "configs" / "pipeline.yaml"))
    from object_fusion._dovsg_import import ensure_dovsg_importable
    ensure_dovsg_importable(cfg["external"]["scene_graph_generation_root"])
    from scene_graph.storage.serializer import load_scene_graph
    from semantic_decoding.novic.novic_decoder import NOVICDecoder

    novic_cfg = cfg["semantic_decoding"]
    decoders = {}                                   # one decoder per distinct class list
    for rd in map(Path, args.result_dir):
        metrics = json.loads((rd / "metrics.json").read_text())
        scene = metrics["info"]["scene"]
        classes = class_list(scene)
        key = tuple(classes)
        if key not in decoders:
            for d in decoders.values():
                d.close()
            decoders = {key: NOVICDecoder(novic_dir=novic_cfg["novic_dir"],
                                          checkpoint_path=novic_cfg["novic_checkpoint"],
                                          gencfg=GENCFG, guide_targets=[NOVIC_ALIASES.get(c, c) for c in classes],
                                          device="cuda")}
            if decoders[key].is_stub:
                raise SystemExit("NOVIC failed to load; run with HF_HUB_OFFLINE=1")
        decoder = decoders[key]

        emb = load_scene_graph(next((rd / "scene_graph").glob("*.pkl")))["node_embeddings"]
        rows = []
        with open(rd / "nodes.csv") as f:
            for r in csv.DictReader(f):
                if r["status"] in ("correct", "wrong_name") and int(r["gt_rank_in_embedding"] or 0) > 0:
                    rows.append(r)
        tags = decoder.decode_embeddings(
            np.stack([np.asarray(emb[r["node_id"]], np.float32) for r in rows])) if rows else []

        out_rows, novic_rank, cg_rank = [], [], []
        for r, t in zip(rows, tags):
            preds = [BACK.get(p, p) for p, _ in t.top_k]
            nr = preds.index(r["gt_class"]) + 1 if r["gt_class"] in preds else 0     # 0 = not in NOVIC's top 10
            novic_rank.append(nr)
            cg_rank.append(int(r["gt_rank_in_embedding"]))
            out_rows.append({"node_id": r["node_id"], "gt_class": r["gt_class"],
                             "novic_guided": BACK.get(t.label, t.label), "novic_score": round(t.score, 3), "novic_gt_rank": nr or "",
                             "conceptgraphs_rule": r["embedding_class"], "conceptgraphs_gt_rank": cg_rank[-1]})
        nr, cr = np.array(novic_rank), np.array(cg_rank)
        res = {"classes": len(classes), "nodes_scored": len(rows), "gencfg": GENCFG,
               "novic_guided": {k: float(((nr > 0) & (nr <= k)).mean()) if len(nr) else None for k in KS},
               "conceptgraphs_rule": {k: float((cr <= k).mean()) if len(cr) else None for k in KS},
               "novic_ranks": nr.tolist(), "conceptgraphs_ranks": cr.tolist()}
        with open(rd / "closed_set_naming.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()) if out_rows else ["node_id"])
            w.writeheader()
            w.writerows(out_rows)
        metrics["closed_set_naming"] = res
        (rd / "metrics.json").write_text(json.dumps(metrics, indent=1))
        print(f"{scene}: {len(rows)} nodes, {len(classes)} classes | top-1 NOVIC guided "
              f"{res['novic_guided'][1]:.1%} vs ConceptGraphs rule {res['conceptgraphs_rule'][1]:.1%}")
    for d in decoders.values():
        d.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
