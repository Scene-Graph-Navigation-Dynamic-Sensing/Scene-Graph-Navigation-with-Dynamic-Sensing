"""
Step 1: Core Data Structure — Track

Every downstream module reads/writes this object.
The Track dataclass represents a single tracked object across frames,
carrying its identity, spatial state, label, and multi-cue fusion buffers.
"""

from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import time
from typing import Optional


class TrackStatus(Enum):
    """Lifecycle states for a tracked object."""
    PENDING_TAGGING = "pending_tagging"   # new track, no label yet
    CONFIRMED = "confirmed"               # label registered and stable
    DEGRADED = "degraded"                 # confidence dropped, needs reconstruction
    LOST = "lost"                         # no match for N frames, pending removal


@dataclass
class Track:
    """
    Unified track representation used throughout the DTD pipeline.

    Attributes:
        track_id: Unique integer identifier assigned by the tracker backend.
        bbox: Current bounding box [x1, y1, x2, y2] in pixel coordinates.
        status: Lifecycle state (PENDING_TAGGING → CONFIRMED → DEGRADED → LOST).
        label: Open-vocabulary label produced by NOVIC decoder, or None if not yet tagged.
        label_logit_score: Confidence score from the NOVIC decode (beam-search log-prob).
        provisional_label: Fast-path label emitted after frame 1 (unfused, low-confidence)
                           for immediate downstream use (e.g., grasp planning).
        provisional_score: Confidence for the provisional label.
        embedding_buffer: List of CLIP/SigLIP embeddings accumulated over first N frames
                          for multi-cue fusion (Step 6).
        intra_conf_buffer: Per-frame appearance/motion/semantic confidence scores.
        inter_conf_buffer: Frame-to-frame cycle-consistency scores.
        position_3d: Optional 3D position [x, y, z] if depth sensor is available (Step 9).
        prev_bbox: Previous frame's bounding box, used for motion feature computation.
        status_pending_reconstruction: Flag for SAM2MOT-style quality reconstruction (Step 7).
        frames_since_seen: Number of consecutive frames where the track was not matched.
        frames_tracked: Total number of frames this track has been actively matched.
        last_updated: Timestamp of the last successful match.
        creation_time: Timestamp when the track was first created.
    """
    track_id: int
    bbox: np.ndarray                      # [x1, y1, x2, y2], current frame
    status: TrackStatus = TrackStatus.PENDING_TAGGING
    label: Optional[str] = None
    label_logit_score: float = 0.0

    # Fast-path provisional label (Step 9 — robotics adaptation)
    provisional_label: Optional[str] = None
    provisional_score: float = 0.0

    # Buffers for multi-cue fusion (Step 6)
    embedding_buffer: list = field(default_factory=list)     # CLIP embeddings, first N frames
    intra_conf_buffer: list = field(default_factory=list)    # appearance/motion/semantic scores
    inter_conf_buffer: list = field(default_factory=list)    # frame-to-frame cycle consistency

    # Track label frequencies
    label_frequencies: dict = field(default_factory=dict)
    
    # YOLO fast-path cascade properties
    current_yolo_label: Optional[str] = None
    current_yolo_conf: float = 0.0

    # Optional 3D position (Step 9 — robotics adaptation)
    position_3d: Optional[np.ndarray] = None                 # [x, y, z] from depth sensor

    # Previous bounding box for motion features
    prev_bbox: Optional[np.ndarray] = None

    # SAM2MOT-style reconstruction flag (Step 7)
    status_pending_reconstruction: bool = False

    # Temporal bookkeeping
    frames_since_seen: int = 0
    frames_tracked: int = 0
    last_updated: float = field(default_factory=time.time)
    creation_time: float = field(default_factory=time.time)

    def mark_seen(self, bbox: np.ndarray):
        """Update track with a new matched detection."""
        self.prev_bbox = self.bbox.copy() if self.bbox is not None else None
        self.bbox = bbox
        self.frames_since_seen = 0
        self.frames_tracked += 1
        self.last_updated = time.time()

    def mark_missed(self):
        """Increment the missed-frame counter (no detection matched this frame)."""
        self.frames_since_seen += 1

    @property
    def age_seconds(self) -> float:
        """Time elapsed since track creation."""
        return time.time() - self.creation_time

    @property
    def time_to_first_label(self) -> Optional[float]:
        """Time from creation to first confirmed label (None if not yet labeled)."""
        if self.label is not None:
            return self.last_updated - self.creation_time
        return None

    @property
    def is_labeled(self) -> bool:
        """Whether this track has received a confirmed label."""
        return self.label is not None and self.status == TrackStatus.CONFIRMED

    @property
    def display_label(self) -> str:
        """Best available label for visualization: confirmed > provisional > pending."""
        if self.label is not None:
            return self.label
        if self.provisional_label is not None:
            return f"~{self.provisional_label}"
        return "…"

    def compute_motion_speed(self) -> float:
        """
        Compute displacement speed between current and previous bounding box centers.
        Returns 0.0 if no previous bbox is available.
        Used as a motion feature for the Self-Gated Fusion Network (Step 6).
        """
        if self.prev_bbox is None:
            return 0.0
        cx_curr = (self.bbox[0] + self.bbox[2]) / 2.0
        cy_curr = (self.bbox[1] + self.bbox[3]) / 2.0
        cx_prev = (self.prev_bbox[0] + self.prev_bbox[2]) / 2.0
        cy_prev = (self.prev_bbox[1] + self.prev_bbox[3]) / 2.0
        return float(np.sqrt((cx_curr - cx_prev) ** 2 + (cy_curr - cy_prev) ** 2))

    def __repr__(self) -> str:
        return (
            f"Track(id={self.track_id}, status={self.status.value}, "
            f"label={self.display_label}, score={self.label_logit_score:.3f}, "
            f"frames={self.frames_tracked}, missed={self.frames_since_seen})"
        )
