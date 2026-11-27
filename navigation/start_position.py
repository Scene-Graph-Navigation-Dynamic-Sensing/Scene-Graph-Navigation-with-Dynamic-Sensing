"""Resolves "where is the robot right now" for a navigation plan.

No such concept exists anywhere else in this repo (confirmed: no
start_point/current_position/robot_pose/odometry tracking of any kind), so
this module makes the gap explicit instead of a caller silently guessing.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal, Sequence

import numpy as np

logger = logging.getLogger(__name__)

StartFallback = Literal["last_pose", "world_origin"]


def _last_pose_translation(recorder_dir: Path) -> list[float]:
    """Reads the last captured frame's camera pose (poses/{idx:06}.txt, a
    4x4 world<-camera transform -- same file layout ViewDataset.load_pose()
    reads) and returns its translation, in the same gravity-aligned world
    frame the voxel map / scene graph / task_planning navigation_target use.
    This assumes the robot hasn't moved since that frame was captured -- a
    placeholder, not real localization."""
    poses_dir = Path(recorder_dir) / "poses"
    pose_files = sorted(poses_dir.glob("*.txt"))
    if not pose_files:
        raise FileNotFoundError(f"No pose files found under {poses_dir}")
    pose = np.loadtxt(pose_files[-1]).astype(np.float32)
    return pose[:3, 3].tolist()


def resolve_start_xyz(
    explicit_start_xyz: Sequence[float] | None,
    recorder_dir: str | Path | None,
    fallback: StartFallback = "last_pose",
) -> tuple[list[float], str]:
    """Resolution order, highest priority first:
      1. `explicit_start_xyz` if given -- always preferred; this is the only
         path that reflects an actually-known robot position.
      2. If `recorder_dir` is given and `fallback == "last_pose"`: the last
         captured frame's camera-pose translation (placeholder -- assumes no
         motion since capture, not real localization).
      3. Otherwise: world origin [0, 0, 0], with a loud warning -- only
         useful for smoke-testing the pipeline end-to-end.

    Returns (start_xyz, source) where `source` records which path was used,
    so a NavigationPlan can carry that information forward for debugging.
    """
    if explicit_start_xyz is not None:
        return list(explicit_start_xyz), "explicit"

    if recorder_dir is not None and fallback == "last_pose":
        start_xyz = _last_pose_translation(Path(recorder_dir))
        logger.warning(
            "start_xyz not given explicitly -- using the last captured frame's pose "
            f"from {recorder_dir} as a placeholder for the robot's current position. "
            "This assumes the robot hasn't moved since that frame was captured."
        )
        return start_xyz, "last_pose"

    logger.warning(
        "start_xyz not given and no recorder_dir to fall back on -- using world origin "
        "[0, 0, 0], which is very likely wrong for a real scene. Pass --start-xyz explicitly."
    )
    return [0.0, 0.0, 0.0], "world_origin"
