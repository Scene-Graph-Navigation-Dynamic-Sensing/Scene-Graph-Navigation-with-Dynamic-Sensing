#!/usr/bin/env python3
"""
RAM++ -> YOLOE-26 (text-prompted with RAM tags) -> BoT-SORT.

Follows dtd_pipeline copy/comparison/run_ram_yoloe.py: RAM++ tags every frame, the YOLOE vocabulary
grows when new tags appear (append-only so class indices stay stable for the tracker), YOLOE tracks
at conf 0.10 and a track is only reported once it has reached conf 0.35. Run with locate-env.
"""

from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from common.bench import Bench  # noqa: E402
from common_ram import RamTagger  # noqa: E402

YOLOE_WEIGHTS = "/home/ravindu/SceneGraphNav/models/checkpoints/yoloe-26s-seg.pt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=0)
    ap.add_argument("--source", default=None, help="video file or folder of images (default: IMG_6100.mp4)")
    ap.add_argument("--out-root", default=None, help="output root (default: pipeline_benchmark/outputs)")
    ap.add_argument("--ram-every", type=int, default=1, help="run RAM++ every N frames")
    ap.add_argument("--name", default="p2_ram_yoloe_botsort")
    args = ap.parse_args()

    # YOLOE's MobileCLIP text encoder is resolved relative to the cwd
    os.chdir(os.path.dirname(HERE))
    from ultralytics import YOLO

    cfg = {"tagger": "RAM++ swin_l 384", "detector": os.path.basename(YOLOE_WEIGHTS), "tracker": "botsort.yaml",
           "track_conf": 0.10, "confirm_conf": 0.35, "iou": 0.5, "imgsz": 640, "ram_every": args.ram_every}
    bench = Bench(args.name, cfg, max_frames=args.frames, video=args.source, out_root=args.out_root)
    ram = RamTagger()
    yolo = YOLO(YOLOE_WEIGHTS)
    bench.models_loaded()

    vocab: list[str] = []
    confirmed: set[int] = set()

    for idx, frame in bench.frames():
        t = bench.timer()
        if idx % args.ram_every == 0:
            with t("ram"):
                tags = ram.tags(frame)
            new = [c for c in tags if c not in vocab]
            if new:
                with t("set_classes"):
                    vocab.extend(new)
                    yolo.set_classes(list(vocab))
        with t("yoloe_track"):
            r = yolo.track(frame, persist=True, conf=0.10, iou=0.5, agnostic_nms=True,
                           tracker="botsort.yaml", verbose=False)[0]

        objects = []
        if r.boxes is not None and r.boxes.id is not None:
            names = r.names
            for box, tid, conf, cls in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.id.cpu().numpy(),
                                           r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy()):
                tid = int(tid)
                if conf >= 0.35:
                    confirmed.add(tid)
                if tid in confirmed:
                    objects.append({"id": tid, "label": names[int(cls)], "score": float(conf), "box": box})
        bench.record(idx, frame, objects, t, extra={"vocab_size": len(vocab)})

    bench.finish()
    with open(os.path.join(bench.out_dir, "ram_vocab.txt"), "w") as f:
        f.write("\n".join(vocab))


if __name__ == "__main__":
    main()
