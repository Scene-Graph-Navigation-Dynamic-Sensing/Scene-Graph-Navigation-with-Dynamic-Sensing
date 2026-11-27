#!/usr/bin/env python3
"""
DTD pipeline: YOLOE-26 (prompt-free) + BoT-SORT -> batched CLIP ViT-H -> batched NOVIC.

Two tagging policies:
  every   : every live track is cropped + CLIP-encoded + NOVIC-decoded every frame (majority-vote label)
  confirm : a track is tagged until its name is confirmed, then the name is locked and the
            track is only followed by BoT-SORT (no more CLIP/NOVIC for it)

Reuses ../yolo_clip_novic (copied unchanged from dtd_pipeline). Run with the scene_graph env.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
YCN_DIR = os.path.join(os.path.dirname(HERE), "yolo_clip_novic")
sys.path.insert(0, YCN_DIR)

from common.bench import Bench  # noqa: E402
from pipeline import YoloClipNovicPipeline  # noqa: E402

logging.basicConfig(level=logging.WARNING)

YOLO_WEIGHTS = "/home/ravindu/SceneGraphNav/models/checkpoints/yoloe-26s-seg-pf.pt"
NOVIC_CKPT = "/home/ravindu/SceneGraphNav/external/novic/outputs/ovod_20240628_142131/ovod_chunk0433_20240630_235415.train"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", choices=["every", "confirm"], required=True)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--confirm-votes", type=int, default=3, help="min votes before a name can be locked")
    ap.add_argument("--confirm-frac", type=float, default=0.6, help="top-noun share needed to lock")
    ap.add_argument("--max-votes", type=int, default=10, help="lock the majority noun after this many votes regardless")
    ap.add_argument("--frames", type=int, default=0)
    ap.add_argument("--source", default=None, help="video file or folder of images (default: IMG_6100.mp4)")
    ap.add_argument("--out-root", default=None, help="output root (default: pipeline_benchmark/outputs)")
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    with open(os.path.join(YCN_DIR, "config.yaml")) as f:
        cfg = yaml.safe_load(f)
    cfg.update(yolo_weights=YOLO_WEIGHTS, novic_checkpoint=NOVIC_CKPT, fp16=args.fp16, device="cuda",
               crops_dir="", display=False)

    name = args.name or ("p1_dtd_every_frame" if args.policy == "every" else "p4_dtd_confirm_then_track")
    bench_cfg = {"policy": args.policy, "yolo": os.path.basename(YOLO_WEIGHTS), "tracker": cfg["tracker_type"],
                 "clip": f"{cfg['clip_model']}/{cfg['clip_pretrained']}", "fp16": args.fp16,
                 "novic": os.path.basename(NOVIC_CKPT), "conf": cfg["conf_thresh"], "imgsz": cfg["img_size"]}
    if args.policy == "confirm":
        bench_cfg.update(confirm_votes=args.confirm_votes, confirm_frac=args.confirm_frac, max_votes=args.max_votes)

    bench = Bench(name, bench_cfg, max_frames=args.frames, video=args.source, out_root=args.out_root)
    p = YoloClipNovicPipeline(cfg)
    logging.getLogger("trajectory.trajectory_manager").setLevel(logging.WARNING)
    bench.models_loaded()

    locked: dict[int, str] = {}

    for idx, frame in bench.frames():
        t = bench.timer()
        with t("detect_track"):
            dets = p.detector.update(frame)
            lc = p.trajectory_manager.step(dets)

        live = lc["pending_tagging"] + lc["active"] + lc.get("degraded", [])
        to_tag = live if args.policy == "every" else [tr for tr in live if tr.track_id not in locked]

        if to_tag:
            with t("clip"):
                emb = p.clip_encoder.encode_batch(frame, [tr.bbox for tr in to_tag], pad_frac=p.pad_frac)
            with t("novic"):
                res = p.novic_decoder.decode_batch(emb)
            for tr, (label, score) in zip(to_tag, res):
                p.trajectory_manager.register_label(tr.track_id, label, score)
                if args.policy == "confirm":
                    votes = sum(tr.label_frequencies.values())
                    top, n = max(tr.label_frequencies.items(), key=lambda kv: kv[1])
                    if (votes >= args.confirm_votes and n / votes >= args.confirm_frac) or votes >= args.max_votes:
                        locked[tr.track_id] = top

        objects = []
        for tr in p.trajectory_manager.tracks.values():
            if tr.frames_since_seen != 0:
                continue
            label = locked.get(tr.track_id)
            if label is None and tr.label_frequencies:
                label = max(tr.label_frequencies.items(), key=lambda kv: kv[1])[0]
            objects.append({"id": tr.track_id, "label": label, "score": tr.label_logit_score, "box": tr.bbox})
        bench.record(idx, frame, objects, t, extra={"n_tagged": len(to_tag)})

    s = bench.finish()
    print(f"locked tracks: {len(locked)}")


if __name__ == "__main__":
    main()
