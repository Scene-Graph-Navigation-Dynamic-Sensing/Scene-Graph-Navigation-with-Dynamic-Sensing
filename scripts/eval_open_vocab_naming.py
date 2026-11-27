"""Open-vocabulary naming, NOVIC vs nearest-noun embedding lookup, on an equal
footing (evaluation/naming_judge.py explains the protocol).

Reads finished evaluation folders (results/.../<scene>/ with nodes.csv and
scene_graph/*.pkl), so no pipeline or segmentation re-run is needed. Writes:

  <scene>/naming.csv            one row per scored node: GT class, the three
                                names and their grades
  <scene>/metrics.json          gains a "naming" section
  <scene>/results.md            gains / refreshes the "Open-vocabulary naming"
                                section

    HF_HUB_OFFLINE=1 python scripts/eval_open_vocab_naming.py results/replica_final/*/
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation import naming_judge as nj  # noqa: E402

PROMPT = "an image of {}"           # same prompt as the closed-set Acc@k (ConceptGraphs')
CACHE = REPO_ROOT / "data" / "cache" / "novic_target_text_embeddings.npz"
SECTION = "## Open-vocabulary naming (NOVIC vs embedding lookup)"
GRADE_LABEL = {"exact": "exact / synonym", "specific": "more specific", "generic": "too generic",
               "wrong": "wrong", "no_name": "no name"}


def scored_rows(result_dir: Path, embeddings: dict) -> list[dict]:
    rows = []
    with open(result_dir / "nodes.csv") as f:
        for r in csv.DictReader(f):
            if r["status"] not in ("correct", "wrong_name") or r["node_id"] not in embeddings:
                continue
            rows.append({"node_id": r["node_id"], "gt_class": r["gt_class"],
                         "novic": r["novic_top5"].split(" | ")[0], "pipeline": r["novic_name"]})
    return rows


def section_md(m: dict, vocab_size: int) -> str:
    n = m["nodes_scored"]
    L = [SECTION, "",
         f"Both methods name each of the {n} nodes matched cleanly to a countable object from the same open "
         f"vocabulary, NOVIC's {vocab_size:,} target nouns, and one WordNet judge grades every name "
         "(`evaluation/naming_judge.py`). **NOVIC**: its decoded top-1 noun. **Embedding lookup**: the target noun "
         "whose text embedding is nearest to the node's image embedding. **Pipeline label**: the node's final name, "
         "for reference only; it is mapped onto a 245-word everyday list that contains 53 of Replica's 101 classes, "
         "so it is not open-set. Acceptable = exact/synonym + more specific.", "",
         "| Grade | NOVIC | Embedding lookup | Pipeline label |", "|---|---|---|---|"]
    meth = m["methods"]
    L.append(f"| **Acceptable** | **{meth['novic']['acceptable']:.1%}** | **{meth['embedding']['acceptable']:.1%}** | "
             f"{meth['pipeline']['acceptable']:.1%} |")
    for g in nj.GRADES:
        L.append(f"| {GRADE_LABEL[g]} | " + " | ".join(
            f"{meth[x]['share'][g]:.1%} ({meth[x]['counts'][g]})" for x in nj.METHODS) + " |")
    L += ["", "Per node: `naming.csv`. The closed-set Acc@k above ranks the embedding against Replica's own "
          "class list, so it measures embedding quality, not naming.", ""]
    return "\n".join(L)


def write_section(results_md: Path, text: str) -> None:
    md = results_md.read_text()
    if SECTION in md:                                   # replace up to the next "## " heading
        md = re.sub(re.escape(SECTION) + r".*?(?=\n## )", text.rstrip("\n") + "\n", md, flags=re.S)
    else:                                               # insert before "## Nodes"
        md = md.replace("\n## Nodes\n", "\n" + text + "\n## Nodes\n", 1)
    results_md.write_text(md)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("result_dir", nargs="+")
    ap.add_argument("--gpu", default="0")
    args = ap.parse_args()

    import os
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.gpu)
    cfg = yaml.safe_load(open(REPO_ROOT / "configs" / "pipeline.yaml"))
    from object_fusion._dovsg_import import ensure_dovsg_importable
    ensure_dovsg_importable(cfg["external"]["scene_graph_generation_root"])
    from scene_graph.storage.serializer import load_scene_graph
    from semantic_decoding.novic.novic_decoder import NOVICDecoder

    novic_cfg = cfg["semantic_decoding"]
    decoder = NOVICDecoder(novic_dir=novic_cfg["novic_dir"], checkpoint_path=novic_cfg["novic_checkpoint"],
                           gencfg=novic_cfg["gencfg"], device="cuda")
    if decoder.is_stub:
        raise SystemExit("NOVIC failed to load; run with HF_HUB_OFFLINE=1")
    nouns, noun_emb = load_vocabulary(decoder)
    decoder.close()

    for rd in map(Path, args.result_dir):
        payload = load_scene_graph(next((rd / "scene_graph").glob("*.pkl")))
        m = evaluate_dir(rd, payload["node_embeddings"], nouns, noun_emb)
        print(f"{rd}: {m['nodes_scored']} nodes, acceptable NOVIC {m['methods']['novic']['acceptable']:.1%} "
              f"vs embedding {m['methods']['embedding']['acceptable']:.1%} "
              f"(pipeline {m['methods']['pipeline']['acceptable']:.1%})")
    return 0


def load_vocabulary(decoder) -> tuple[list[str], np.ndarray]:
    nouns = decoder.target_nouns()
    return nouns, nj.vocabulary_text_embeddings(decoder, nouns, PROMPT, CACHE)


def evaluate_dir(rd: Path, emb: dict, nouns: list[str], noun_emb: np.ndarray) -> dict:
    """Score one finished evaluation folder; writes naming.csv, metrics.json["naming"], results.md section."""
    rows = scored_rows(rd, emb)
    looked_up = nj.nearest_nouns(np.stack([np.asarray(emb[r["node_id"]], np.float32) for r in rows]),
                                 nouns, noun_emb) if rows else []
    for r, w in zip(rows, looked_up):
        r["embedding"] = w
    m = nj.naming_metrics(rows)
    m.update(vocabulary_size=len(nouns), prompt=PROMPT)

    with open(rd / "naming.csv", "w", newline="") as f:
        cols = ["node_id", "gt_class"] + [x for k in nj.METHODS for x in (k, f"grade_{k}")]
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    metrics = json.loads((rd / "metrics.json").read_text())
    metrics["naming"] = m
    (rd / "metrics.json").write_text(json.dumps(metrics, indent=1))
    write_section(rd / "results.md", section_md(m, len(nouns)))
    return m


if __name__ == "__main__":
    sys.exit(main())
