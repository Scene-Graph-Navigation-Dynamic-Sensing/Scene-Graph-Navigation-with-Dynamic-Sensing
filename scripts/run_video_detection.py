#!/usr/bin/env python3
"""
run_video_detection.py — Detect + tag objects in a video, write an annotated copy
====================================================================================
Runs the same YOLOE-26 → crop → NOVIC pipeline as main.py, but over the frames
of a video file instead of a folder of images, and writes one annotated output
video with bounding boxes + name-tag labels burned in (no live GUI window,
since this runs on a headless/remote machine with no display attached).

Usage:
    python scripts/run_video_detection.py data/IMG_6100.MOV

    # common options
    python scripts/run_video_detection.py data/IMG_6100.MOV --frame_stride 2
    python scripts/run_video_detection.py data/IMG_6100.MOV --output outputs/IMG_6100_annotated.mp4
    python scripts/run_video_detection.py data/IMG_6100.MOV --max_frames 300
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from main import (
    NOVIC_CHECKPOINT_DEFAULT,
    NOVIC_DIR_DEFAULT,
    YOLOE_WEIGHTS_DEFAULT,
    _draw_boxes,
    run_pipeline,
)
from perception.detection.yoloe_seg_detector import YOLOESegDetector
from semantic_decoding.novic.novic_decoder import NOVICDecoder

logger = logging.getLogger("run_video_detection")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("video", help="Path to input video file (e.g. .MOV, .mp4).")
    p.add_argument("--output", default=None,
                   help="Output video path. Default: outputs/<video_stem>_annotated.mp4")
    p.add_argument("--frame_stride", type=int, default=1,
                   help="Process every Nth frame (skipped frames reuse the last "
                        "annotation). Higher = faster, less accurate tracking of motion.")
    p.add_argument("--max_frames", type=int, default=None,
                   help="Stop after processing this many frames (for a quick preview run).")

    p.add_argument("--yoloe_weights", default=YOLOE_WEIGHTS_DEFAULT)
    p.add_argument("--conf_thresh", type=float, default=0.25)
    p.add_argument("--iou_thresh", type=float, default=0.50)
    p.add_argument("--img_size", type=int, default=640)

    p.add_argument("--novic_dir", default=NOVIC_DIR_DEFAULT)
    p.add_argument("--novic_checkpoint", default=NOVIC_CHECKPOINT_DEFAULT)
    p.add_argument("--gencfg", default="beam_k10_vnone_gp_t1_a0")

    p.add_argument("--crop_mode", default="bbox", choices=["segmented", "bbox"])
    p.add_argument("--crop_padding_px", type=int, default=15)
    p.add_argument("--device", default="cuda")
    return p


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    args = build_arg_parser().parse_args()

    video_path = Path(args.video)
    if not video_path.is_file():
        logger.error(f"Video not found: {video_path}")
        sys.exit(1)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error(f"Could not open video: {video_path}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info(f"Input video: {video_path}  ({width}x{height} @ {fps:.1f}fps, {total_frames} frames)")

    out_path = Path(args.output) if args.output else REPO_ROOT / "outputs" / f"{video_path.stem}_annotated.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        logger.error(f"Could not open VideoWriter for: {out_path}")
        sys.exit(1)

    logger.info(f"Loading YOLOE-26-seg-pf  : {args.yoloe_weights}")
    detector = YOLOESegDetector(
        weights_path=args.yoloe_weights,
        conf_thresh=args.conf_thresh,
        iou_thresh=args.iou_thresh,
        img_size=args.img_size,
        device=args.device,
    )

    logger.info(f"Loading NOVIC decoder    : {args.novic_checkpoint}")
    tagger = NOVICDecoder(
        novic_dir=args.novic_dir,
        checkpoint_path=args.novic_checkpoint,
        gencfg=args.gencfg,
        device=args.device,
    )
    if tagger.is_stub:
        logger.warning("NOVIC checkpoint not loaded → stub-decoder mode (placeholder tags).")

    frame_idx = 0
    processed = 0
    last_annotated_bgr = None
    t_start = time.time()

    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            if args.max_frames is not None and processed >= args.max_frames:
                break

            if frame_idx % args.frame_stride == 0:
                image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                _, _, _, results = run_pipeline(
                    image_rgb=image_rgb,
                    detector=detector,
                    tagger=tagger,
                    crop_mode=args.crop_mode,
                    crop_padding_px=args.crop_padding_px,
                )

                if results:
                    boxes = np.stack([r.bbox for r in results])
                    confs = np.array([r.conf for r in results])
                    labels = [f"{r.tag}  {r.score:.2f}" for r in results]
                    annotated_rgb = _draw_boxes(image_rgb, boxes, confs, labels=labels)
                else:
                    annotated_rgb = image_rgb

                last_annotated_bgr = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)
                processed += 1
                if processed % 10 == 0:
                    elapsed = time.time() - t_start
                    logger.info(f"Processed {processed} frame(s) (video frame {frame_idx}) — {elapsed:.1f}s elapsed")

            writer.write(last_annotated_bgr if last_annotated_bgr is not None else frame_bgr)
            frame_idx += 1
    finally:
        cap.release()
        writer.release()
        tagger.close()

    elapsed = time.time() - t_start
    logger.info(f"Done. Read {frame_idx} frame(s), ran detection on {processed} of them in {elapsed:.1f}s.")
    logger.info(f"Annotated video saved → {out_path}")


if __name__ == "__main__":
    main()
