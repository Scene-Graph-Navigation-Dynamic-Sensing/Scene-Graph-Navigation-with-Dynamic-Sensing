"""Open-vocabulary nearest-noun lookup with YOLOE, graded like the CLIP lookup.

Does YOLOE's object embedding stay aligned with text over a full open
vocabulary, or only over common detection classes? Each node's averaged YOLOE
embedding (yoloe_node_embeddings.npz, from scripts/eval_yoloe_embedding_acc.py)
is named by its nearest noun among NOVIC's 42,919 target nouns, embedded with
YOLOE's text encoder (MobileCLIP2 + RepRTA, yoloe-26x-seg.pt). The name is
graded by the same WordNet judge (evaluation/naming_judge.py) as the CLIP
lookup in naming.csv (scripts/eval_open_vocab_naming.py), on the same nodes.

Writes <scene>/yoloe_lookup.csv and prints, per scene and pooled, the share of
acceptable names (exact/synonym or more specific) for YOLOE and for CLIP on
the same nodes.

    python scripts/eval_yoloe_open_vocab_lookup.py results/replica_final/*/
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation import naming_judge as nj  # noqa: E402

NOUNS_CACHE = REPO_ROOT / "data" / "cache" / "novic_target_text_embeddings.npz"   # the CLIP test's noun list
WEIGHTS = REPO_ROOT / "external" / "Scene-Graph-Generation" / "yoloe-26x-seg.pt"
PROMPTS = {"an image of": "an image of {}", "plain": "{}"}   # the CLIP test used "an image of {}"


def yoloe_noun_embeddings(nouns: list[str], template: str, tag: str) -> np.ndarray:
    cache = REPO_ROOT / "data" / "cache" / f"yoloe26x_target_text_embeddings_{tag}.npz"
    if cache.exists():
        d = np.load(cache)
        if len(d["emb"]) == len(nouns):
            return d["emb"].astype(np.float32)
    from ultralytics import YOLOE
    model = YOLOE(str(WEIGHTS))
    out = []
    for i in range(0, len(nouns), 2000):
        pe = model.get_text_pe([template.format(n) for n in nouns[i:i + 2000]])[0].float().cpu().numpy()
        out.append(pe / np.linalg.norm(pe, axis=1, keepdims=True))
    emb = np.concatenate(out)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, emb=emb.astype(np.float16))
    return emb


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("result_dir", nargs="+")
    ap.add_argument("--gpu", default="0")
    args = ap.parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.gpu)

    nouns = [str(n) for n in np.load(NOUNS_CACHE)["nouns"]]
    text = {k: yoloe_noun_embeddings(nouns, t, k.replace(" ", "_")) for k, t in PROMPTS.items()}
    print(f"{len(nouns)} nouns embedded with YOLOE's text encoder")

    pooled = {k: Counter() for k in ["clip", *[f"yoloe ({k})" for k in PROMPTS]]}
    n_all = 0
    print(f"\n{'scene':9s} {'nodes':>5s}  {'CLIP lookup':>11s}  " + "  ".join(f"{'YOLOE (' + k + ')':>20s}" for k in PROMPTS))
    for rd in map(Path, args.result_dir):
        y = np.load(rd / "yoloe_node_embeddings.npz")
        yemb = dict(zip([str(i) for i in y["node_ids"]], y["emb"]))
        rows = [r for r in csv.DictReader(open(rd / "naming.csv")) if r["node_id"] in yemb]
        missing = sum(1 for _ in csv.DictReader(open(rd / "naming.csv"))) - len(rows)
        E = np.stack([yemb[r["node_id"]] for r in rows])
        out = []
        for r in rows:
            out.append({"node_id": r["node_id"], "gt_class": r["gt_class"],
                        "clip_lookup": r["embedding"], "grade_clip": r["grade_embedding"]})
        for k, T in text.items():
            idx = np.argmax(E @ T.T, axis=1)
            for o, i in zip(out, idx):
                o[f"yoloe_{k}"] = nouns[i]
                o[f"grade_yoloe_{k}"] = nj.grade(nouns[i], o["gt_class"])
        with open(rd / "yoloe_lookup.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
            w.writeheader(); w.writerows(out)
        n = len(out)
        n_all += n
        shares = {}
        for key, col in [("clip", "grade_clip"), *[(f"yoloe ({k})", f"grade_yoloe_{k}") for k in PROMPTS]]:
            c = Counter(o[col] for o in out)
            pooled[key].update(c)
            shares[key] = (c["exact"] + c["specific"]) / n
        print(f"{rd.name:9s} {n:5d}  {shares['clip']:11.1%}  "
              + "  ".join(f"{shares[f'yoloe ({k})']:20.1%}" for k in PROMPTS)
              + (f"   ({missing} nodes without YOLOE embedding skipped)" if missing else ""))

    print(f"\n{'pooled':9s} {n_all:5d}  " + "  ".join(
        f"{(pooled[k]['exact'] + pooled[k]['specific']) / n_all:{11 if k == 'clip' else 20}.1%}" for k in pooled))
    print("\ngrade totals (exact / more specific / too generic / wrong / no name):")
    for k, c in pooled.items():
        print(f"  {k:20s} " + " / ".join(str(c[g]) for g in nj.GRADES))
    return 0


if __name__ == "__main__":
    sys.exit(main())
