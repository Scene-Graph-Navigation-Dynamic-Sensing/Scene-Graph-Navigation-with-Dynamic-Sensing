#!/usr/bin/env python3
"""Which views should count more in an object's average CLIP embedding?

Fusion keeps each object's embedding as a running average over its
detections, every view weighted equally (DoVSG merge_obj_to_obj). Views are
not equal: some show part of the object (cut by the image border), some show
it far away or motion-blurred. This scores per-view weightings of each
BoT-SORT track's average by what the average is used for:

  naming    NOVIC's decode of the weighted track average against the track's
            labelled class (WordNet strict / lenient, as in eval_crop_mode.py)
  matching  how well weighted averages tell objects apart: ROC AUC of the
            cosine of two tracks of the same object vs of different objects

It reuses outputs/eval/crop_mode/<video>/cache.pkl (detections, masks, box-crop
embeddings) and labels.json from scripts/eval_crop_mode.py; no detection is
re-run.

    python scripts/eval_view_weighting.py
"""
from __future__ import annotations

import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logger = logging.getLogger("eval_view_weighting")
ROOT = REPO_ROOT / "outputs" / "eval" / "crop_mode"
VIDEOS = ["initial", "updated_same_angle", "updated_diff_angle", "update_in_same_video", "img6100"]


def view_features(cache) -> list[dict]:
    """Per detection: track, mask area share, border truncation, median depth,
    YOLOE confidence, sharpness of the box crop, embedding."""
    import cv2
    from PIL import Image

    ds = Path(cache["dataset_dir"])
    out = []
    for f, fr in enumerate(cache["frames"]):
        if not len(fr["boxes"]):
            continue
        h, w = fr["shape"]
        masks = np.unpackbits(fr["masks"], axis=-1, count=w).astype(bool)
        gray = cv2.cvtColor(np.asarray(Image.open(ds / "rgb" / f"{f:06d}.jpg")), cv2.COLOR_RGB2GRAY)
        z = np.load(ds / "point" / f"{f:06d}.npy")[..., 2]
        valid = np.load(ds / "mask" / f"{f:06d}.npy").astype(bool)
        for i, t in enumerate(fr["tids"]):
            m = masks[i]
            ys, xs = np.nonzero(m)
            if len(xs) == 0:
                continue
            border = 2
            truncated = xs.min() <= border or ys.min() <= border or xs.max() >= w - 1 - border or ys.max() >= h - 1 - border
            x1, y1, x2, y2 = fr["boxes"][i].astype(int)
            patch = gray[max(0, y1):y2, max(0, x1):x2]
            sharp = float(cv2.Laplacian(patch, cv2.CV_64F).var()) if patch.size > 16 else 0.0
            zz = z[m & valid]
            out.append({"f": f, "i": i, "tid": int(t), "area": float(m.sum() / (h * w)), "truncated": bool(truncated),
                        "depth": float(np.median(zz)) if len(zz) else np.nan, "conf": float(fr["confs"][i]),
                        "sharp": sharp, "emb": np.asarray(fr["emb"]["bbox"][i], np.float32)})
    return out


def weights(views: list[dict], scheme: str) -> np.ndarray:
    a = np.array([v["area"] for v in views])
    trunc = np.array([v["truncated"] for v in views])
    d = np.array([v["depth"] for v in views])
    d = np.where(np.isfinite(d), d, np.nanmedian(d) if np.isfinite(d).any() else 1.0)
    conf = np.array([v["conf"] for v in views])
    sharp = np.array([v["sharp"] for v in views])
    full = np.where(trunc, 0.2, 1.0)
    rel = lambda x: x / max(np.median(x), 1e-9)  # noqa: E731 -- scale-free within a track
    if scheme == "uniform (current)":
        return np.ones(len(views))
    if scheme == "mask area":
        return a
    if scheme == "sqrt mask area":
        return np.sqrt(a)
    if scheme == "not truncated":
        return full
    if scheme == "closer (1/depth^2)":
        return 1.0 / np.maximum(d, 0.1) ** 2
    if scheme == "YOLOE confidence":
        return conf
    if scheme == "sharpness":
        return np.sqrt(rel(sharp))
    if scheme == "area x not-truncated x sharpness":
        return np.sqrt(a) * full * np.sqrt(rel(sharp))
    if scheme == "best 30% views (area, not truncated)":
        score = np.sqrt(a) * full
        k = max(1, int(np.ceil(0.3 * len(views))))
        w = np.zeros(len(views))
        w[np.argsort(-score)[:k]] = 1.0
        return w
    raise ValueError(scheme)


SCHEMES = ["uniform (current)", "mask area", "sqrt mask area", "not truncated", "closer (1/depth^2)",
           "YOLOE confidence", "sharpness", "area x not-truncated x sharpness",
           "best 30% views (area, not truncated)", "outlier rejection", "outlier rejection + area x not-truncated"]


def track_average(views: list[dict], scheme: str) -> np.ndarray:
    E = np.stack([v["emb"] for v in views])
    En = E / np.linalg.norm(E, axis=1, keepdims=True)
    if scheme.startswith("outlier rejection"):
        # drop views far from the track's own consensus (occlusion, a track
        # briefly on another object): cosine to the median direction below
        # median - 2.5 MAD of the track's cosines
        med = np.median(En, axis=0)
        cos = En @ (med / np.linalg.norm(med))
        mad = np.median(np.abs(cos - np.median(cos))) + 1e-6
        keep = cos >= np.median(cos) - 2.5 * mad
        base = np.ones(len(views)) if scheme == "outlier rejection" else weights(views, "area x not-truncated x sharpness")
        w = base * keep
    else:
        w = weights(views, scheme)
    if w.sum() <= 0:
        w = np.ones(len(views))
    avg = (E * w[:, None]).sum(0) / w.sum()
    return (avg / np.linalg.norm(avg)).astype(np.float32)


def roc_auc(pos, neg):
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    ranks = np.concatenate([pos, neg]).argsort().argsort() + 1
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    from object_fusion.instance_tagger import to_everyday_names
    from scripts.eval_crop_mode import Judge, load_models
    from scripts.run_static_scene_graph import load_config, load_everyday_names

    cfg = load_config(REPO_ROOT / "configs" / "pipeline_fyp_videos.yaml")
    _, novic = load_models(cfg, tracker=False)
    judge, everyday = Judge(), load_everyday_names(cfg)
    weight = cfg["label_mapper"].get("everyday_name_weight", 0.2)

    per_video, pooled = {}, {s: {"strict": [], "lenient": [], "auc": []} for s in SCHEMES}
    for video in VIDEOS:
        if not (ROOT / video / "labels.json").exists():
            continue
        cache = pickle.loads((ROOT / video / "cache.pkl").read_bytes())
        labels = {int(k): v for k, v in json.loads((ROOT / video / "labels.json").read_text()).items()}
        views = [v for v in view_features(cache) if labels.get(v["tid"], "ignore") != "ignore"]
        tracks = sorted({v["tid"] for v in views if sum(1 for u in views if u["tid"] == v["tid"]) >= 3})
        by_track = {t: [v for v in views if v["tid"] == t] for t in tracks}
        cls = [labels[t].split("|")[0] for t in tracks]
        inst = np.array([labels[t] for t in tracks])
        stats = {"tracks": len(tracks),
                 "truncated_share": round(float(np.mean([v["truncated"] for v in views])), 3)}
        for s in SCHEMES:
            A = np.stack([track_average(by_track[t], s) for t in tracks])
            raw = [r.label for k in range(0, len(A), 64) for r in novic.decode_embeddings(A[k:k + 64])]
            mapped = to_everyday_names(raw, A, everyday, novic.embed_text, weight) if everyday else raw
            j = [judge(p, c) for p, c in zip(mapped, cls)]
            cos = A @ A.T
            iu = np.triu_indices(len(A), 1)
            same = (inst[:, None] == inst[None, :])[iu]
            auc = roc_auc(cos[iu][same], cos[iu][~same])
            stats[s] = {"strict": round(float(np.mean([a for a, _ in j])), 3),
                        "lenient": round(float(np.mean([b for _, b in j])), 3),
                        "auc_same_vs_different": round(auc, 3),
                        "names": dict(zip([f"t{t} {labels[t]}" for t in tracks], mapped))}
            pooled[s]["strict"] += [a for a, _ in j]
            pooled[s]["lenient"] += [b for _, b in j]
            if np.isfinite(auc):
                pooled[s]["auc"].append(auc)
        per_video[video] = stats
        logger.info(f"{video}: {len(tracks)} tracks; " + ", ".join(
            f"{s}: {stats[s]['lenient']}" for s in SCHEMES))
    novic.close()

    summary = {s: {"tracks": len(pooled[s]["lenient"]),
                   "naming_strict": round(float(np.mean(pooled[s]["strict"])), 3),
                   "naming_lenient": round(float(np.mean(pooled[s]["lenient"])), 3),
                   "auc_same_vs_different (mean over videos)": round(float(np.mean(pooled[s]["auc"])), 3)}
               for s in SCHEMES}
    (ROOT / "view_weighting.json").write_text(json.dumps({"pooled": summary, "per_video": per_video}, indent=2,
                                                         default=str))
    print("\n| Weighting of views in a track's average | tracks | naming strict | naming lenient | "
          "same-vs-different AUC |\n|---|---|---|---|---|")
    for s in SCHEMES:
        r = summary[s]
        print(f"| {s} | {r['tracks']} | {r['naming_strict']:.3f} | {r['naming_lenient']:.3f} | "
              f"{r['auc_same_vs_different (mean over videos)']:.3f} |")


if __name__ == "__main__":
    main()
