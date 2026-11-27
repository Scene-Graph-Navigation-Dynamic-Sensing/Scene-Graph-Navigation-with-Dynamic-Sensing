"""
Old approach: YOLO -> BoT-SORT -> batched CLIP -> batched NOVIC.

    frame -> YOLO+BoT-SORT (class-agnostic boxes, persistent IDs)
          -> batch pad-crop + CLIP encode ALL pending tracks in one forward
          -> batch NOVIC decode ALL embeddings in one forward
          -> per-track majority-vote noun ("monitoring")

This differs from the very first version of this pipeline in one way:
tagging is BATCHED. The original looped `clip_encoder.encode()` then
`novic_decoder.decode()` once per track. Batching collects every pending crop
for the frame and runs one CLIP forward and one NOVIC forward.

Measured on an RTX 5000 (fp32 CLIP, as this pipeline runs it), looped -> batched:

    N tracks   CLIP encode    NOVIC decode
        4       1.06x          2.31x
        8       1.04x          2.66x
       16       1.13x          1.95x

So batching is a real win for NOVIC but barely helps CLIP: ViT-H at 378px is
already compute-bound at batch size 1, leaving nothing for batching to recover.
CLIP cost stays ~linear in the number of tracks (~250 ms per crop in fp32),
which is the bottleneck the unified single-backbone model (../unified_pipeline/)
removes by encoding the whole frame once instead of once per object.

Monitoring: TrajectoryManager.register_label() already does majority-vote
label tracking per track (Track.label_frequencies), so "what is the most
common noun this track has been called" falls out for free. This module
adds a convenience report over that state plus per-label crop archiving.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Optional

import cv2
import numpy as np
import torch

from detectors.yolo_botsort_tracker import YoloBotsortTracker
from tagging.clip_encoder import CLIPEncoder
from tagging.novic_decoder import NOVICDecoder
from trajectory.track import Track, TrackStatus
from trajectory.trajectory_manager import TrajectoryManager

logger = logging.getLogger(__name__)


class YoloClipNovicPipeline:
    def __init__(self, config: dict):
        self.config = config
        self.device = config.get("device", "cuda")

        self.detector = YoloBotsortTracker(
            weights_path=config["yolo_weights"],
            conf_thresh=config.get("conf_thresh", 0.35),
            iou_thresh=config.get("iou_thresh", 0.5),
            device=self.device,
            tracker_type=config.get("tracker_type", "botsort.yaml"),
            img_size=config.get("img_size", 640),
        )

        self.clip_encoder = CLIPEncoder(
            model_name=config.get("clip_model", "ViT-H-14-378-quickgelu"),
            pretrained=config.get("clip_pretrained", "dfn5b"),
            device=self.device,
            fp16=config.get("fp16", False),
        )

        self.novic_decoder = NOVICDecoder(
            checkpoint_path=config.get("novic_checkpoint", ""),
            beam_k=config.get("beam_k", 10),
            vocab_free_temp=config.get("vocab_free_temp", 2.0),
            genre_penalty=config.get("genre_penalty", True),
            temperature=config.get("temperature", 1.0),
            alpha=config.get("alpha", 0.0),
            device=self.device,
        )

        self.trajectory_manager = TrajectoryManager(
            max_missed_frames=config.get("max_missed_frames", 25),
            degradation_threshold=0.0,  # not used; every returned track is re-tagged
        )

        self.pad_frac = config.get("pad_frac", 0.15)
        self.retag_min_votes = config.get("retag_min_votes", 0)
        self.retag_interval = max(1, config.get("retag_interval", 10))

        self._frame_count = 0
        self._fps_buffer: list[float] = []

        # Final reports for tracks the trajectory manager has already evicted, so
        # the end-of-run summary covers objects that left the scene too.
        self._evicted_summaries: dict[int, dict] = {}

        # noun-voting crop archive: track_id -> label -> {count, max_score, sum_score, best_crop}
        self.crops_dir = config.get("crops_dir", "")
        self.track_crop_history: dict[int, dict[str, dict]] = {}
        if self.crops_dir:
            os.makedirs(self.crops_dir, exist_ok=True)
            logger.info(f"Crop archiving enabled: {self.crops_dir}")

        logger.info(f"YoloClipNovicPipeline initialized (device={self.device})")

    # ─────────────────────────────── main loop ────────────────────────────────

    def process_frame(self, frame: np.ndarray) -> dict:
        t_start = time.time()
        self._frame_count += 1

        # ─── Stage 1: detect + track ───
        detections = self.detector.update(frame)

        # ─── Stage 2: lifecycle ───
        lifecycle = self.trajectory_manager.step(detections)
        for gone in lifecycle["removed"]:
            row = self._summarize(gone)
            if row is not None:
                self._evicted_summaries[gone.track_id] = row

        # ─── Stage 3: batch-tag every returned track this frame ───
        to_tag = lifecycle["pending_tagging"] + lifecycle["active"] + lifecycle.get("degraded", [])
        self._batch_tag(frame, self._select_for_tagging(to_tag))

        frame_time = time.time() - t_start
        self._fps_buffer.append(frame_time)
        if len(self._fps_buffer) > 30:
            self._fps_buffer.pop(0)
        avg = sum(self._fps_buffer) / len(self._fps_buffer)

        return {
            "lifecycle": lifecycle,
            "tracks": self.trajectory_manager.tracks,
            "frame_time_ms": frame_time * 1000,
            "fps": 1.0 / (avg + 1e-8),
        }

    def _select_for_tagging(self, tracks: list[Track]) -> list[Track]:
        """
        Optional re-tag budget. Default (retag_min_votes=0) re-tags every track every frame.

        With retag_min_votes=N, a track is tagged every frame until it has N votes, then only
        every retag_interval frames (staggered by track id so the work spreads across frames).
        Labels stay crop-based, so accuracy is preserved to the extent the majority vote has
        already settled; CLIP work drops from tracks x frames to roughly tracks x N.
        """
        if self.retag_min_votes <= 0:
            return tracks
        keep = []
        for t in tracks:
            votes = sum(t.label_frequencies.values())
            if votes < self.retag_min_votes or (self._frame_count + t.track_id) % self.retag_interval == 0:
                keep.append(t)
        return keep

    def _batch_tag(self, frame: np.ndarray, tracks: list[Track]) -> None:
        """One CLIP forward + one NOVIC forward for every track this frame."""
        if not tracks:
            return

        bboxes = [t.bbox for t in tracks]
        embeddings = self.clip_encoder.encode_batch(frame, bboxes, pad_frac=self.pad_frac)
        results = self.novic_decoder.decode_batch(embeddings)

        for track, (label, score) in zip(tracks, results):
            self.trajectory_manager.register_label(track.track_id, label, score)
            self._record_crop(frame, track, label, score)

    def _record_crop(self, frame: np.ndarray, track: Track, label: str, score: float) -> None:
        """Archive the best crop seen so far for each (track, label) pair."""
        if not self.crops_dir:
            return

        history = self.track_crop_history.setdefault(track.track_id, {})
        entry = history.setdefault(
            label, {"count": 0, "max_score": -1.0, "sum_score": 0.0, "best_crop": None}
        )
        entry["count"] += 1
        entry["sum_score"] += float(score)
        if score > entry["max_score"]:
            entry["max_score"] = float(score)
            entry["best_crop"] = CLIPEncoder.pad_crop(frame, track.bbox, self.pad_frac)

    # ────────────────────────────── monitoring ────────────────────────────────

    def get_track_summary(self) -> list[dict]:
        """
        Per-track monitoring report: the majority-vote noun and how confident
        that vote is. This is the "most common noun for that object" the
        trajectory manager has been accumulating in Track.label_frequencies
        every frame via register_label().
        """
        rows = dict(self._evicted_summaries)
        for track in self.trajectory_manager.tracks.values():
            row = self._summarize(track)
            if row is not None:
                rows[track.track_id] = row
        return [rows[tid] for tid in sorted(rows)]

    @staticmethod
    def _summarize(track: Track) -> Optional[dict]:
        if not track.label_frequencies:
            return None
        total = sum(track.label_frequencies.values())
        top_label, top_count = max(track.label_frequencies.items(), key=lambda kv: kv[1])
        return {
            "track_id": track.track_id,
            "most_common_label": top_label,
            "vote_count": top_count,
            "vote_fraction": top_count / total,
            "total_observations": total,
            "all_labels": dict(track.label_frequencies),
            "status": track.status.value,
        }

    # ─────────────────────────────── drawing ───────────────────────────────────

    def draw(self, frame: np.ndarray, result: Optional[dict] = None, confirmed_only: bool = False) -> np.ndarray:
        """
        Draw every track tagged with its most common noun so far.

        The tag is the majority vote over all NOVIC labels the track has received
        (Track.label_frequencies), not the latest frame's label, so it does not
        flicker when a single frame decodes differently.
        """
        annotated = frame.copy()
        h = frame.shape[0]
        scale = max(0.6, h / 1080 * 0.9)          # keep text legible at 1080p+
        thick = max(2, int(round(scale * 2)))

        for track in self.trajectory_manager.tracks.values():
            if confirmed_only and track.status != TrackStatus.CONFIRMED:
                continue
            if not track.label_frequencies:
                continue                              # not tagged yet: draw nothing

            noun, votes = max(track.label_frequencies.items(), key=lambda kv: kv[1])
            total = sum(track.label_frequencies.values())
            x1, y1, x2, y2 = track.bbox.astype(int)
            color = (0, 255, 0) if votes / total >= 0.6 else (0, 200, 255)  # amber = unstable vote

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thick)
            text = f"{noun} #{track.track_id}"
            (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
            top = max(0, y1 - th - base - 4)
            cv2.rectangle(annotated, (x1, top), (x1 + tw, top + th + base + 4), color, -1)
            cv2.putText(annotated, text, (x1, top + th + 1),
                        cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick)

        if result:
            cv2.putText(annotated, f"FPS: {result['fps']:.1f}", (10, int(40 * scale) + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 255, 255), thick)
        return annotated

    # ─────────────────────────────── shutdown ──────────────────────────────────

    def shutdown(self) -> None:
        """Save the best archived crop per (track, most-common-label) and log the report."""
        if self.crops_dir and self.track_crop_history:
            logger.info(f"Saving summary crops to {self.crops_dir}...")
            for track_id, labels in self.track_crop_history.items():
                if not labels:
                    continue
                top_label = max(labels.items(), key=lambda kv: kv[1]["count"])[0]
                data = labels[top_label]
                safe = top_label.replace(" ", "_").replace("/", "_")
                avg = data["sum_score"] / max(1, data["count"])
                fname = f"track_{track_id}_{safe}_freq_{data['count']}_avgconf_{avg:.3f}.jpg"
                if data["best_crop"] is not None:
                    cv2.imwrite(os.path.join(self.crops_dir, fname), data["best_crop"])
            logger.info("Crop archiving complete.")

        for row in self.get_track_summary():
            logger.info(
                f"Track {row['track_id']}: '{row['most_common_label']}' "
                f"({row['vote_count']}/{row['total_observations']} votes, "
                f"{row['vote_fraction']:.0%})"
            )

    def get_stats(self) -> dict:
        return {
            "frame_count": self._frame_count,
            "trajectory": self.trajectory_manager.get_stats(),
            "avg_fps": 1.0 / (sum(self._fps_buffer) / max(len(self._fps_buffer), 1) + 1e-8),
        }

    def reset(self) -> None:
        self.detector.reset()
        self.trajectory_manager.reset()
        self._frame_count = 0
        self._fps_buffer.clear()
        self._evicted_summaries.clear()
