#!/usr/bin/env python3
"""
Run the yolo_clip_novic pipeline (old approach) on a video source.

Usage:
    python yolo_clip_novic/run.py --source ../data/living_room.mp4
    python yolo_clip_novic/run.py --source 0 --no-display

Prints a per-track "most common noun" report on exit (q / Ctrl-C).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import cv2
import yaml

# Make this folder self-contained: it resolves its own detectors/tagging/
# trajectory subpackages and pipeline.py relative to itself, not the repo root.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline import YoloClipNovicPipeline  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s",
                     datefmt="%H:%M:%S")
logger = logging.getLogger("run")
# register_label() logs every confirmation at INFO - thousands of lines on a full
# video, which buries the progress and the final report.
logging.getLogger("trajectory.trajectory_manager").setLevel(logging.WARNING)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "config.yaml"))
    p.add_argument("--source", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--novic-checkpoint", default=None)
    p.add_argument("--fp16", action="store_true", help="fp16 CLIP encode (same nouns on the 60-frame check, ~3x faster)")
    p.add_argument("--retag-min-votes", type=int, default=None,
                   help="tag each track every frame until it has N votes, then every --retag-interval frames (0=always)")
    p.add_argument("--retag-interval", type=int, default=None)
    p.add_argument("--no-display", action="store_true")
    p.add_argument("--output", default=None, help="write annotated video here (mp4)")
    p.add_argument("--crops-dir", default=None, help="save track crops here (default: off)")
    p.add_argument("--frames", type=int, default=0, help="stop after N frames (0 = full video)")
    return p.parse_args()


def _finalize_video(raw_path: str, output: str) -> None:
    """Re-encode OpenCV's mp4v output to H.264 so it plays in browsers/VS Code."""
    import shutil
    import subprocess
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"raw video missing: {raw_path}")
    if shutil.which("ffmpeg"):
        r = subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", raw_path,
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", output],
        )
        if r.returncode == 0:
            os.remove(raw_path)
            logger.info(f"wrote {output}")
            return
    os.replace(raw_path, output)
    logger.warning(f"ffmpeg unavailable/failed; wrote mp4v video to {output} (may not play everywhere)")


def main():
    args = parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.source is not None:
        config["source"] = args.source
    if args.device is not None:
        config["device"] = args.device
    if args.novic_checkpoint is not None:
        config["novic_checkpoint"] = args.novic_checkpoint
    elif not config.get("novic_checkpoint"):
        config["novic_checkpoint"] = os.getenv("NOVIC_CHECKPOINT", "")
    if not config.get("yolo_weights") or config["yolo_weights"] == "yolo11n.pt":
        config["yolo_weights"] = os.getenv("YOLO_WEIGHTS", config["yolo_weights"])

    if args.retag_min_votes is not None:
        config["retag_min_votes"] = args.retag_min_votes
    if args.retag_interval is not None:
        config["retag_interval"] = args.retag_interval
    if args.fp16:
        config["fp16"] = True
    if args.crops_dir is not None:
        config["crops_dir"] = args.crops_dir
    output = args.output or config.get("output_path") or ""

    display = config.get("display", True) and not args.no_display

    pipeline = YoloClipNovicPipeline(config)

    source = config["source"]
    cap = cv2.VideoCapture(int(source) if str(source).isdigit() else source)
    if not cap.isOpened():
        raise SystemExit(f"cannot open source: {source}")

    # Per-process temp name: two runs pointed at the same --output must not share
    # (and then delete) each other's raw file.
    raw_path = f"{output}.{os.getpid()}.raw.mp4" if output else ""
    writer = None
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1

            result = pipeline.process_frame(frame)

            if output:
                if writer is None:
                    h, w = frame.shape[:2]
                    writer = cv2.VideoWriter(raw_path, cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (w, h))
                writer.write(pipeline.draw(frame, result))

            if frame_idx % 25 == 0:
                logger.info(
                    f"frame {frame_idx}/{total_frames or '?'} | {result['fps']:.2f} FPS | "
                    f"{len(pipeline.trajectory_manager.tracks)} live tracks"
                )

            if display:
                annotated = pipeline.draw(frame, result)
                cv2.imshow("yolo_clip_novic", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if args.frames and frame_idx >= args.frames:
                break
    except KeyboardInterrupt:
        logger.info("interrupted")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if display:
            cv2.destroyAllWindows()
        pipeline.shutdown()

        # Report BEFORE touching the video: a failed encode must never cost the results.
        summary = pipeline.get_track_summary()
        logger.info("=" * 60)
        logger.info("Most common noun per track:")
        for row in summary:
            logger.info(
                f"  track {row['track_id']:>4}: {row['most_common_label']:<24} "
                f"({row['vote_fraction']:.0%} of {row['total_observations']} observations)"
            )
        logger.info("=" * 60)

        if output:
            import json
            try:
                with open(output + ".nouns.json", "w") as f:
                    json.dump(summary, f, indent=2)
                logger.info(f"wrote {output}.nouns.json")
            except OSError as e:
                logger.error(f"could not write noun report: {e}")

        if writer is not None:
            try:
                _finalize_video(raw_path, output)
            except Exception as e:  # keep the raw video rather than lose it
                logger.error(f"video finalize failed ({e}); raw video kept at {raw_path}")

if __name__ == "__main__":
    main()
