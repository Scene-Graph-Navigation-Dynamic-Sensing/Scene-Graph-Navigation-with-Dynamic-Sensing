"""Re-express a dataset's poses in a gravity-aligned world (z up, lowest surface at z=0).

DoVSG's relation builder assumes that frame: it treats [0, 0, 1] as the floor
normal, the floor as z=0, and decides "on" by comparing z heights. DROID-SLAM's
world is instead the first camera's frame (x right, y down, z forward), so
without this step "on" is judged along the camera's viewing direction and
nearly everything ends up "on floor".

DoVSG itself levels the world from floor masks found by a text-prompted
GroundingDINO+SAM2 detector (dovsg/controller.py::transform_pose_with_floor).
This pipeline is prompt-free and many captures never show the floor, so the
estimate here needs no detector:

  1. prior  -- a hand-held camera is roughly upright, so the mean of each
               frame's camera-up axis (-y) in world coordinates approximates up;
  2. refine -- RANSAC-extract the largest planes from the reconstructed cloud and
               take the biggest one within max_tilt_deg of the prior (floor or
               tabletop — both are horizontal). The prior alone is not enough:
               a camera looking down at a table tilts it by the pitch angle;
  3. level  -- rotate that normal onto +z, then shift so a low percentile of the
               cloud's height sits at z=0.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import open3d as o3d

logger = logging.getLogger(__name__)


def _rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Smallest rotation taking unit vector a onto unit vector b."""
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if np.linalg.norm(v) < 1e-9:
        if c > 0:
            return np.eye(3)
        # 180 degrees about any axis perpendicular to a
        axis = np.cross(a, [1.0, 0.0, 0.0] if abs(a[0]) < 0.9 else [0.0, 1.0, 0.0])
        axis /= np.linalg.norm(axis)
        return 2.0 * np.outer(axis, axis) - np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))


def estimate_gravity_alignment(
    points_world: np.ndarray,
    camera_rotations: np.ndarray,
    # The prior is off by however far the camera pitches down on average —
    # 33 deg on TUM freiburg1_desk, which looks down at the desk throughout.
    # 45 still separates horizontal surfaces from walls (~90 deg away).
    max_tilt_deg: float = 45.0,
    num_planes: int = 5,
    floor_percentile: float = 1.0,
    voxel_size: float = 0.03,
) -> tuple[np.ndarray, dict]:
    """points_world: (N,3) reconstructed cloud; camera_rotations: (F,3,3) camera-to-world.
    Returns (4x4 world->aligned transform, diagnostics)."""
    up_prior = (camera_rotations @ np.array([0.0, -1.0, 0.0])).mean(axis=0)
    up_prior /= np.linalg.norm(up_prior)

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points_world.astype(np.float64)))
    pcd = pcd.voxel_down_sample(voxel_size)
    remaining = pcd
    cos_tol = np.cos(np.radians(max_tilt_deg))
    best_normal, best_inliers = None, 0
    for _ in range(num_planes):
        if len(remaining.points) < 100:
            break
        (a, b, c, _d), inliers = remaining.segment_plane(distance_threshold=voxel_size,
                                                         ransac_n=3, num_iterations=1000)
        normal = np.array([a, b, c]) / np.linalg.norm([a, b, c])
        if np.dot(normal, up_prior) < 0:
            normal = -normal
        if np.dot(normal, up_prior) >= cos_tol and len(inliers) > best_inliers:
            best_normal, best_inliers = normal, len(inliers)
        remaining = remaining.select_by_index(inliers, invert=True)

    up = best_normal if best_normal is not None else up_prior
    R = _rotation_between(up, np.array([0.0, 0.0, 1.0]))
    z = (points_world @ R.T)[:, 2]
    floor_z = float(np.percentile(z, floor_percentile))

    T = np.eye(4)
    T[:3, :3] = R
    T[2, 3] = -floor_z
    info = {
        "up_source": "plane" if best_normal is not None else "camera_prior",
        "plane_inliers": best_inliers,
        "prior_to_up_deg": float(np.degrees(np.arccos(np.clip(np.dot(up_prior, up), -1, 1)))),
        "floor_z_before": floor_z,
    }
    return T, info


def gravity_align_dataset(dataset_dir: Path, frame_stride: int = 5, **kwargs) -> dict:
    """Rewrites dataset_dir/poses/*.txt in a z-up world. The unaligned poses are
    kept in poses_unaligned/ so the step can be inspected or undone."""
    dataset_dir = Path(dataset_dir)
    backup = dataset_dir / "poses_unaligned"
    pose_files = sorted((dataset_dir / "poses").glob("*.txt"))
    # always align from the original poses, so running this twice is harmless
    poses = np.stack([np.loadtxt(backup / p.name if (backup / p.name).exists() else p) for p in pose_files])

    clouds = []
    for p, T in list(zip(pose_files, poses))[::frame_stride]:
        pts = np.load(dataset_dir / "point" / f"{p.stem}.npy").reshape(-1, 3)
        valid = np.load(dataset_dir / "mask" / f"{p.stem}.npy").reshape(-1)
        pts = pts[valid][::8]
        clouds.append(pts @ T[:3, :3].T + T[:3, 3])

    T_align, info = estimate_gravity_alignment(np.concatenate(clouds), poses[:, :3, :3], **kwargs)

    backup.mkdir(exist_ok=True)
    for p, T in zip(pose_files, poses):
        if not (backup / p.name).exists():
            np.savetxt(backup / p.name, T)
        np.savetxt(p, T_align @ T)

    info["transform"] = T_align.tolist()
    logger.info(f"Gravity-aligned {len(pose_files)} poses: up from {info['up_source']} "
                f"({info['prior_to_up_deg']:.1f} deg from camera prior), lowest surface moved to z=0")
    return info
