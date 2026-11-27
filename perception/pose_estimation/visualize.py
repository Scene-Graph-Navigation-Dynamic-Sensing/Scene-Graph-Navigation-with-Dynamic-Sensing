"""Camera trajectory visualization (outputs/visualizations/pose/trajectory.png)."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def save_trajectory_visualization(out_dir: Path, poses: dict[int, np.ndarray]) -> Path | None:
    """Plots the camera-position path (top-down and side views) from a set of 4x4 poses."""
    if not poses:
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    indices = sorted(poses.keys())
    positions = np.array([poses[i][:3, 3] for i in indices])

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(10, 5))

    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    ax3d.plot(positions[:, 0], positions[:, 1], positions[:, 2], "-o", markersize=2, linewidth=1)
    ax3d.scatter(*positions[0], c="green", s=60, label="start")
    ax3d.scatter(*positions[-1], c="red", s=60, label="end")
    ax3d.set_xlabel("X")
    ax3d.set_ylabel("Y")
    ax3d.set_zlabel("Z")
    ax3d.set_title("Camera trajectory (3D)")
    ax3d.legend()

    ax2d = fig.add_subplot(1, 2, 2)
    ax2d.plot(positions[:, 0], positions[:, 1], "-o", markersize=2, linewidth=1)
    ax2d.scatter(positions[0, 0], positions[0, 1], c="green", s=60, label="start")
    ax2d.scatter(positions[-1, 0], positions[-1, 1], c="red", s=60, label="end")
    ax2d.set_xlabel("X")
    ax2d.set_ylabel("Y")
    ax2d.set_title("Top-down (X-Y)")
    ax2d.axis("equal")
    ax2d.legend()

    out_path = out_dir / "trajectory.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def save_trajectory_comparison(out_dir: Path, aligned: dict, title: str = "") -> Path:
    """Estimated vs ground-truth trajectory (`aligned` comes from
    scripts.eval_pose.align_trajectories, so the estimate is already in the
    ground-truth frame), plus the per-frame error curves underneath."""
    out_dir.mkdir(parents=True, exist_ok=True)
    gt, est = aligned["gt_positions"], aligned["est_positions"]
    frames = aligned["frames"]
    terr, rerr = aligned["translation_error_m"], aligned["rotation_error_deg"]

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(15, 9))
    gt_kw = dict(color="#2c7fb8", linewidth=2, label="ground truth")
    est_kw = dict(color="#d95f02", linewidth=1.5, linestyle="--", label="DROID-SLAM estimate")

    ax = fig.add_subplot(2, 3, 1, projection="3d")
    ax.plot(gt[:, 0], gt[:, 1], gt[:, 2], **gt_kw)
    ax.plot(est[:, 0], est[:, 1], est[:, 2], **est_kw)
    ax.scatter(*gt[0], c="green", s=50)
    ax.set_title("Trajectory (3D)")
    ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]"); ax.set_zlabel("Z [m]")
    ax.legend(fontsize=8)

    for pos, (a, b, name) in enumerate(((0, 1, "X-Y (top-down)"), (0, 2, "X-Z (side)")), start=2):
        ax = fig.add_subplot(2, 3, pos)
        ax.plot(gt[:, a], gt[:, b], **gt_kw)
        ax.plot(est[:, a], est[:, b], **est_kw)
        ax.scatter(gt[0, a], gt[0, b], c="green", s=50, zorder=5, label="start")
        ax.set_title(name)
        ax.set_xlabel("XYZ"[a] + " [m]"); ax.set_ylabel("XYZ"[b] + " [m]")
        ax.axis("equal"); ax.grid(alpha=0.3); ax.legend(fontsize=8)

    # per-frame error, and how the two trajectories separate over time
    ax = fig.add_subplot(2, 3, 4)
    ax.plot(frames, terr * 100, color="#d95f02")
    ax.axhline(terr.mean() * 100, color="gray", linestyle=":", label=f"mean {terr.mean()*100:.1f} cm")
    ax.set_title("Translation error per frame")
    ax.set_xlabel("frame"); ax.set_ylabel("error [cm]"); ax.grid(alpha=0.3); ax.legend(fontsize=8)

    ax = fig.add_subplot(2, 3, 5)
    ax.plot(frames, rerr, color="#7570b3")
    ax.axhline(rerr.mean(), color="gray", linestyle=":", label=f"mean {rerr.mean():.2f}°")
    ax.set_title("Rotation error per frame")
    ax.set_xlabel("frame"); ax.set_ylabel("error [deg]"); ax.grid(alpha=0.3); ax.legend(fontsize=8)

    ax = fig.add_subplot(2, 3, 6)
    for axis, colour in zip(range(3), ("#1b9e77", "#d95f02", "#7570b3")):
        ax.plot(frames, (est[:, axis] - gt[:, axis]) * 100, color=colour, label=f"{'XYZ'[axis]} error")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Per-axis position error")
    ax.set_xlabel("frame"); ax.set_ylabel("estimate − ground truth [cm]")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)

    fig.suptitle(title or "DROID-SLAM vs ground truth", fontsize=13)
    out_path = out_dir / "trajectory_vs_groundtruth.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
