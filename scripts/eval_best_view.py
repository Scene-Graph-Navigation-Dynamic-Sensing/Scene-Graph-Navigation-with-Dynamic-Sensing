#!/usr/bin/env python3
"""Name each object from its single best view instead of an averaged embedding.

Fusion names an object by decoding the running average of its views' CLIP
embeddings. Here NOVIC instead decodes ONE view per track / object, chosen by:

  largest view          biggest mask area among views not cut by the image border
  quality score         sqrt(area) x not-cut x sharpness
  YOLOE confidence      most confident detection
  most typical view     the view nearest the median embedding (the medoid)
  NOVIC's own pick      decode every view, keep the most confident name

and, for reference, against the current average, the outlier-rejected average
(scripts/eval_view_weighting.py) and a vote (most common name over all
views' decodes). Scored per track and per object (all of an object's tracks
pooled), with WordNet strict / lenient correctness after the everyday-names
mapping, on outputs/eval/crop_mode/<video>/{cache.pkl, labels.json}.

    python scripts/eval_best_view.py
"""
from __future__ import annotations

import json
import logging
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.eval_view_weighting import ROOT, VIDEOS, track_average, view_features  # noqa: E402

logger = logging.getLogger("eval_best_view")

SINGLE = ["largest view", "quality score", "YOLOE confidence", "most typical view", "NOVIC's own pick"]
REFERENCE = ["average (current)", "average, outlier rejection", "vote over all views"]
METHODS = REFERENCE[:2] + SINGLE + REFERENCE[2:]


def pick(views: list[dict], method: str) -> int:
    a = np.array([v["area"] for v in views])
    full = ~np.array([v["truncated"] for v in views])
    sharp = np.array([v["sharp"] for v in views])
    if method == "largest view":
        return int(np.argmax(np.where(full, a, a * 1e-3)))  # an uncut view if there is one
    if method == "quality score":
        return int(np.argmax(np.sqrt(a) * np.where(full, 1.0, 0.2) * np.sqrt(sharp / max(np.median(sharp), 1e-9))))
    if method == "YOLOE confidence":
        return int(np.argmax([v["conf"] for v in views]))
    if method == "most typical view":
        E = np.stack([v["emb"] for v in views])
        E = E / np.linalg.norm(E, axis=1, keepdims=True)
        med = np.median(E, axis=0)
        return int(np.argmax(E @ (med / np.linalg.norm(med))))
    raise ValueError(method)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    from object_fusion.instance_tagger import to_everyday_names
    from scripts.eval_crop_mode import Judge, load_models
    from scripts.run_static_scene_graph import load_config, load_everyday_names

    cfg = load_config(REPO_ROOT / "configs" / "pipeline_fyp_videos.yaml")
    _, novic = load_models(cfg, tracker=False)
    judge, everyday = Judge(), load_everyday_names(cfg)
    weight = cfg["label_mapper"].get("everyday_name_weight", 0.2)

    def decode(E):
        E = np.asarray(E, np.float32)
        rs = [r for k in range(0, len(E), 64) for r in novic.decode_embeddings(E[k:k + 64])]
        raw = [r.label for r in rs]
        mapped = to_everyday_names(raw, E, everyday, novic.embed_text, weight) if everyday else raw
        return mapped, [float(r.score) for r in rs]

    pooled = {lvl: {m: [] for m in METHODS} for lvl in ("track", "object")}
    detail = {}
    for video in VIDEOS:
        if not (ROOT / video / "labels.json").exists():
            continue
        cache = pickle.loads((ROOT / video / "cache.pkl").read_bytes())
        labels = {int(k): v for k, v in json.loads((ROOT / video / "labels.json").read_text()).items()}
        views = [v for v in view_features(cache) if labels.get(v["tid"], "ignore") != "ignore"]
        # every view decoded once: NOVIC's own pick and the vote use them
        names, scores = decode([v["emb"] for v in views])
        for v, n, s in zip(views, names, scores):
            v["name"], v["score"] = n, s
        groups = {"track": {}, "object": {}}
        for v in views:
            groups["track"].setdefault(v["tid"], []).append(v)
            groups["object"].setdefault(labels[v["tid"]], []).append(v)
        groups["track"] = {k: g for k, g in groups["track"].items() if len(g) >= 3}
        detail[video] = {}
        for lvl, gs in groups.items():
            keys = list(gs)
            cls = [(labels[k] if lvl == "track" else k).split("|")[0] for k in keys]
            out = {}
            avg = [track_average(gs[k], "uniform (current)") for k in keys]
            rob = [track_average(gs[k], "outlier rejection") for k in keys]
            out["average (current)"] = decode(avg)[0]
            out["average, outlier rejection"] = decode(rob)[0]
            for m in SINGLE[:-1]:
                out[m] = decode([gs[k][pick(gs[k], m)]["emb"] for k in keys])[0]
            out["NOVIC's own pick"] = [max(gs[k], key=lambda v: v["score"])["name"] for k in keys]
            out["vote over all views"] = [Counter(v["name"] for v in gs[k]).most_common(1)[0][0] for k in keys]
            for m in METHODS:
                pooled[lvl][m] += [judge(p, c) for p, c in zip(out[m], cls)]
            detail[video][lvl] = {f"{k} ({c})": {m: out[m][i] for m in METHODS} for i, (k, c) in enumerate(zip(keys, cls))}
        logger.info(f"{video}: {len(groups['track'])} tracks, {len(groups['object'])} objects, {len(views)} views")
    novic.close()

    summary = {lvl: {m: {"n": len(v), "strict": round(float(np.mean([a for a, _ in v])), 3),
                         "lenient": round(float(np.mean([b for _, b in v])), 3)} for m, v in d.items()}
               for lvl, d in pooled.items()}
    (ROOT / "best_view.json").write_text(json.dumps({"pooled": summary, "per_video": detail}, indent=2, default=str))
    n_t, n_o = summary["track"][METHODS[0]]["n"], summary["object"][METHODS[0]]["n"]
    print(f"\n| How the name is decoded | per track ({n_t}) strict | lenient | per object ({n_o}) strict | lenient |")
    print("|---|---|---|---|---|")
    for m in METHODS:
        t, o = summary["track"][m], summary["object"][m]
        print(f"| {m} | {t['strict']:.3f} | {t['lenient']:.3f} | {o['strict']:.3f} | {o['lenient']:.3f} |")


if __name__ == "__main__":
    main()
