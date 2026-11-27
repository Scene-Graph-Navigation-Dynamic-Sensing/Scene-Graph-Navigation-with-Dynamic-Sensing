"""
Step 3: Trajectory Manager — Track Lifecycle Orchestration

This is the central hub that decides:
  - Which tracks are NEW (→ tag them via NOVIC)
  - Which tracks are ALIVE (→ propagate labels for free, zero NOVIC cost)
  - Which tracks are DEAD (→ evict from memory)

It implements Object Addition (brand-new tracks) and Object Removal
(tracks missing for too many frames), and provides hooks for label
registration from the tagging subsystem.
"""

from trajectory.track import Track, TrackStatus
import numpy as np
import time
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class TrajectoryManager:
    """
    Manages the lifecycle of all tracked objects in the DTD pipeline.

    Responsibilities:
        1. Object Addition: Create Track instances for newly detected objects.
        2. Label Propagation: Confirmed tracks carry their label for free (no re-tagging).
        3. Object Removal: Evict tracks that have been missing for too long.
        4. Degradation Detection: Flag tracks whose confidence has dropped.

    Args:
        max_missed_frames: Frames without a match before a track is removed.
        degradation_threshold: Label confidence below which a track is marked DEGRADED.
    """

    def __init__(
        self,
        max_missed_frames: int = 25,
        degradation_threshold: float = 0.3,
    ):
        self.tracks: dict[int, Track] = {}
        self.max_missed_frames = max_missed_frames
        self.degradation_threshold = degradation_threshold

        # Statistics
        self._total_tracks_created = 0
        self._total_tracks_removed = 0
        self._total_labels_registered = 0

    def step(self, detections: list[dict]) -> dict:
        """
        Process one frame's worth of detections and update track lifecycle.

        Args:
            detections: Output of YOLO26Tracker.update() or PFRPNTracker.update().
                        Each dict has: {"track_id": int, "bbox": np.ndarray, "conf": float}

        Returns:
            Dict with three lists of Track objects:
                - "pending_tagging": Tracks that need NOVIC labeling (new or accumulating).
                - "active": Tracks with confirmed labels (label propagates for free).
                - "removed": Tracks that were evicted this frame.
                - "degraded": Tracks flagged for quality reconstruction.
        """
        seen_ids = set()
        pending, active, degraded = [], [], []

        for det in detections:
            tid = det["track_id"]
            bbox = det["bbox"]
            conf = det["conf"]
            seen_ids.add(tid)

            if tid not in self.tracks:
                # Object Addition: brand-new track
                new_track = Track(
                    track_id=tid,
                    bbox=bbox,
                    status=TrackStatus.PENDING_TAGGING,
                )
                new_track.current_yolo_label = det.get("label")
                new_track.current_yolo_conf = det.get("conf", 0.0)
                self.tracks[tid] = new_track
                pending.append(new_track)
                self._total_tracks_created += 1
                logger.debug(f"New track created: ID={tid}")

            else:
                track = self.tracks[tid]
                track.mark_seen(bbox)
                track.current_yolo_label = det.get("label")
                track.current_yolo_conf = det.get("conf", 0.0)

                if track.status == TrackStatus.PENDING_TAGGING:
                    # Still accumulating frames for fusion (Step 6)
                    pending.append(track)

                elif track.status == TrackStatus.CONFIRMED:
                    # Check for confidence degradation
                    if track.label_logit_score < self.degradation_threshold:
                        track.status = TrackStatus.DEGRADED
                        degraded.append(track)
                        logger.debug(
                            f"Track {tid} degraded: score={track.label_logit_score:.3f}"
                        )
                    else:
                        # Label propagates automatically — zero NOVIC cost
                        active.append(track)

                elif track.status == TrackStatus.DEGRADED:
                    degraded.append(track)

        # Object Removal: anything not matched this frame
        removed = []
        for tid in list(self.tracks.keys()):
            if tid not in seen_ids:
                track = self.tracks[tid]
                track.mark_missed()
                if track.frames_since_seen > self.max_missed_frames:
                    removed.append(self.tracks.pop(tid))
                    self._total_tracks_removed += 1
                    logger.debug(
                        f"Track removed: ID={tid}, lived for {track.frames_tracked} frames"
                    )

        return {
            "pending_tagging": pending,
            "active": active,
            "removed": removed,
            "degraded": degraded,
        }

    def register_label(
        self,
        track_id: int,
        label: str,
        score: float,
        provisional: bool = False,
    ):
        """
        Assign a label to a track.

        Args:
            track_id: The track to label.
            label: Open-vocabulary label string from NOVIC.
            score: Confidence score from the NOVIC decode.
            provisional: If True, set as provisional (fast-path) label only.
        """
        if track_id not in self.tracks:
            logger.warning(f"Cannot register label: track {track_id} not found.")
            return

        track = self.tracks[track_id]

        if provisional:
            track.label_frequencies[label] = track.label_frequencies.get(label, 0) + 1
            most_freq_label = max(track.label_frequencies.items(), key=lambda x: x[1])[0]
            track.provisional_label = most_freq_label
            track.provisional_score = score
            logger.debug(f"Track {track_id}: provisional label = '{label}' ({score:.3f}) -> most freq = '{most_freq_label}'")
        else:
            track.label_frequencies[label] = track.label_frequencies.get(label, 0) + 1
            most_freq_label = max(track.label_frequencies.items(), key=lambda x: x[1])[0]
            track.label = most_freq_label
            track.label_logit_score = score
            track.status = TrackStatus.CONFIRMED
            self._total_labels_registered += 1
            logger.info(
                f"Track {track_id}: confirmed label = '{label}' ({score:.3f}) -> most freq = '{most_freq_label}'"
            )

    def get_track(self, track_id: int) -> Optional[Track]:
        """Retrieve a track by ID, or None if not found."""
        return self.tracks.get(track_id)

    def get_all_active_tracks(self) -> list[Track]:
        """Return all tracks that currently have confirmed labels."""
        return [
            t for t in self.tracks.values()
            if t.status == TrackStatus.CONFIRMED
        ]

    def get_stats(self) -> dict:
        """Return pipeline statistics for monitoring."""
        status_counts = {}
        for track in self.tracks.values():
            status_counts[track.status.value] = status_counts.get(track.status.value, 0) + 1

        return {
            "active_tracks": len(self.tracks),
            "total_created": self._total_tracks_created,
            "total_removed": self._total_tracks_removed,
            "total_labeled": self._total_labels_registered,
            "status_breakdown": status_counts,
        }

    def reset(self):
        """Clear all tracks and reset statistics."""
        self.tracks.clear()
        self._total_tracks_created = 0
        self._total_tracks_removed = 0
        self._total_labels_registered = 0
        logger.info("TrajectoryManager reset.")
