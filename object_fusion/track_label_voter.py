"""Per-track noun voting — the dtd_pipeline labelling scheme, applied to 3D.

Ported from /home/ravindu/rashmika/dtd_pipeline: BoT-SORT gives every object a
persistent track id, NOVIC decodes a noun from each crop, and every decode is a
*vote* counted in `label_frequencies` for that id
(trajectory/trajectory_manager.py::register_label). The name of an object is
then the most frequent noun it ever received, not the last one
(pipeline.py::_summarize), which is what stops the label flickering when one
blurred or half-occluded frame decodes differently.

Two things differ here because the target is a 3D map rather than a video
overlay:

  * Votes are re-aggregated per fused 3D instance. One physical object usually
    collects several track ids over a scan, since the tracker drops ids through
    occlusions and viewpoint changes, so the instance's name is the argmax over
    the summed votes of every track that fed it (`label_for_tracks`).
  * Decoding is throttled. dtd_pipeline decodes every track every frame; here a
    track is re-decoded only until it has `min_votes`, then every
    `revote_interval` frames, since a static scan sees the same object for
    hundreds of frames and NOVIC's beam search is the expensive step.

Best crops are archived exactly as dtd_pipeline does, as
track_<id>_<noun>_freq_<count>_avgconf_<score>.jpg.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class TrackLabelVoter:
    def __init__(self, min_votes: int = 3, revote_interval: int = 15, keep_crops: bool = True):
        self.min_votes = min_votes
        self.revote_interval = max(1, revote_interval)
        self.keep_crops = keep_crops
        # track_id -> {noun: {"count", "sum_score", "max_score", "best_crop"}}
        self.tracks: dict[int, dict[str, dict]] = defaultdict(dict)
        self.decodes = 0
        self.detections_seen = 0

    def needs_vote(self, track_id: int, frame_idx: int) -> bool:
        """Untracked detections (id < 0) always decode: with no id there is
        nothing to accumulate votes against."""
        self.detections_seen += 1
        if track_id < 0:
            return True
        votes = self.vote_count(track_id)
        if votes < self.min_votes:
            return True
        return (frame_idx + track_id) % self.revote_interval == 0

    def register(self, track_id: int, label: str, score: float, crop: np.ndarray | None = None) -> None:
        entry = self.tracks[track_id].setdefault(
            label, {"count": 0, "sum_score": 0.0, "max_score": -1.0, "best_crop": None})
        entry["count"] += 1
        entry["sum_score"] += float(score)
        self.decodes += 1
        if score > entry["max_score"]:
            entry["max_score"] = float(score)
            if self.keep_crops and crop is not None and crop.size > 3:
                entry["best_crop"] = crop.copy()

    def vote_count(self, track_id: int) -> int:
        return sum(e["count"] for e in self.tracks.get(track_id, {}).values())

    def label_for_tracks(self, track_ids) -> tuple[str | None, dict]:
        """Majority noun over the pooled votes of several track ids, with the
        stats behind it (how dominant the winner was, and its mean score)."""
        pooled: dict[str, dict] = {}
        for tid in set(int(t) for t in track_ids):
            for label, e in self.tracks.get(tid, {}).items():
                agg = pooled.setdefault(label, {"count": 0, "sum_score": 0.0})
                agg["count"] += e["count"]
                agg["sum_score"] += e["sum_score"]
        if not pooled:
            return None, {}

        label, best = max(pooled.items(), key=lambda kv: kv[1]["count"])
        total = sum(v["count"] for v in pooled.values())
        return label, {
            "vote_count": best["count"],
            "total_votes": total,
            "vote_fraction": best["count"] / total,
            "mean_score": best["sum_score"] / best["count"],
            "all_labels": {k: v["count"] for k, v in sorted(
                pooled.items(), key=lambda kv: -kv[1]["count"])},
            "track_ids": sorted(int(t) for t in set(track_ids)),
        }

    def save_crops(self, out_dir: Path) -> int:
        """One image per (track, noun), named as dtd_pipeline names them."""
        if not self.keep_crops:
            return 0
        import cv2

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        written = 0
        for tid, labels in self.tracks.items():
            for label, e in labels.items():
                if e["best_crop"] is None:
                    continue
                avg = e["sum_score"] / e["count"]
                safe = "".join(c if c.isalnum() else "_" for c in label)
                name = f"track_{tid}_{safe}_freq_{e['count']}_avgconf_{avg:.3f}.jpg"
                cv2.imwrite(str(out_dir / name), cv2.cvtColor(e["best_crop"], cv2.COLOR_RGB2BGR))
                written += 1
        logger.info(f"Saved {written} best track crops to {out_dir}")
        return written

    def summary(self) -> dict:
        per_track = []
        for tid, labels in sorted(self.tracks.items()):
            label, stats = self.label_for_tracks([tid])
            if label is not None:
                per_track.append({"track_id": tid, "most_common_label": label, **stats})
        return {
            "tracks_voted": len(per_track),
            "novic_decodes": self.decodes,
            "detections_seen": self.detections_seen,
            "decode_reduction": (round(1.0 - self.decodes / self.detections_seen, 4)
                                 if self.detections_seen else 0.0),
            "per_track": per_track,
        }
