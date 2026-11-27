#!/usr/bin/env python3
"""Stage 2 of the relation comparison (relateanything env, PYTHONNOUSERSITE=1): RelateAnything on every prepared
frame, with the ground-truth object masks. Saves, per scan, every scored (subject, object, predicate) of every
frame -- not just the top-k -- so stage 3 can pool over views.

    PYTHONNOUSERSITE=1 ~/miniconda3/envs/relateanything/bin/python evaluation/relations/ra_predict.py \\
        --root ../results/relations_3dssg [--model maelic/relsgg-vits16plus]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from relsgg import RelateAnything

# 3DSSG's relation words, phrased as the model expects, plus a few layout words as distractors.
VOCAB = ["standing on", "lying on", "supported by", "on", "inside", "lying in", "standing in",
         "hanging on", "hanging in", "attached to", "leaning against", "part of", "built into", "connected to",
         "covering", "next to", "in front of", "behind", "to the left of", "to the right of"]
MIN_AREA_FRAC = 0.002          # regions smaller than this share of the image are skipped


@torch.no_grad()
def score_all(ra: RelateAnything, image_bgr, boxes, masks):
    """RelateAnything.predict() up to the score matrix: every kept pair, every predicate."""
    N = len(boxes)
    img_t, W, H = ra._to_chw(image_bgr, ra.img_size)
    b = boxes.astype(np.float32).copy()
    b[:, [0, 2]] /= W
    b[:, [1, 3]] /= H
    cxcywh = np.stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2, b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]], -1)
    boxes_t = torch.from_numpy(cxcywh.astype(np.float32)).unsqueeze(0).to(ra.device)
    cov, fill = ra._rasterize(masks, boxes, W, H)
    out = ra.model(img_t.to(ra.device), boxes_t, box_counts=torch.tensor([N], device=ra.device), targets=None,
                   cov=cov.to(ra.device), fill=fill.to(ra.device))
    scores = ra.contract.scores(out["logits"][0].float(), out["pair_logits"][0].float()).cpu().numpy()
    s, o = out["sub_idx"][0].cpu().numpy(), out["obj_idx"][0].cpu().numpy()
    keep = out["valid_mask"][0].cpu().numpy().astype(bool) & (s < N) & (o < N) & (s != o)
    return s[keep], o[keep], scores[keep]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--model", default="maelic/relsgg-vits16plus")
    ap.add_argument("--max_boxes", type=int, default=60)
    args = ap.parse_args()

    ra = RelateAnything.from_pretrained(args.model, device="cuda")
    ra.set_vocabulary(VOCAB)
    print("model", args.model, "| vocabulary", len(VOCAB), "| contract", ra.contract.describe())
    for sdir in sorted(p for p in args.root.iterdir() if (p / "meta.json").exists()):
        dst = sdir / f"ra_{args.model.split('/')[-1]}.npz"
        if dst.exists():
            continue
        meta = json.loads((sdir / "meta.json").read_text())
        S, O, F, SC = [], [], [], []
        t0, nfr = time.time(), 0
        for fi, fr in enumerate(meta["frames"]):
            img = cv2.imread(str(sdir / "frames" / f"{fr['name']}.jpg"))
            ids = cv2.imread(str(sdir / "frames" / f"{fr['name']}.ids.png"), cv2.IMREAD_UNCHANGED)
            H, W = ids.shape
            oids, counts = np.unique(ids[ids > 0], return_counts=True)
            big = counts >= MIN_AREA_FRAC * H * W
            oids, counts = oids[big], counts[big]
            if len(oids) < 2:
                continue
            oids = oids[np.argsort(-counts)][:args.max_boxes]                 # largest regions first
            masks = np.stack([ids == k for k in oids])
            boxes = []
            for m in masks:
                ys, xs = np.nonzero(m)
                boxes.append([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1])
            s, o, sc = score_all(ra, img, np.asarray(boxes, np.float32), masks)
            S.append(oids[s]); O.append(oids[o]); F.append(np.full(len(s), fi)); SC.append(sc.astype(np.float16))
            nfr += 1
        np.savez_compressed(dst, sub=np.concatenate(S) if S else np.zeros(0, int),
                            obj=np.concatenate(O) if O else np.zeros(0, int),
                            frame=np.concatenate(F) if F else np.zeros(0, int),
                            scores=np.concatenate(SC) if SC else np.zeros((0, len(VOCAB)), np.float16),
                            vocab=np.array(VOCAB))
        print(f"{sdir.name}: {nfr} frames, {sum(len(x) for x in S)} scored pairs, "
              f"{(time.time() - t0) / max(nfr, 1) * 1000:.0f} ms/frame")


if __name__ == "__main__":
    main()
