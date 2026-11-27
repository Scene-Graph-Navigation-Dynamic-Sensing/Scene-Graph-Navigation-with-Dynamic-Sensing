"""Camera pose estimation via DROID-SLAM, with a CPU colored-ICP fallback.

This does not re-implement DROID-SLAM or ICP registration — both already
exist and work in Scene-Graph-Generation (dovsg/scripts/pose_estimation.py
and dovsg/scripts/cpu_pose_estimation.py, driven the same way by
main_live.py::run_pose_estimation()). This module is a thin wrapper that
prepares the on-disk dataset layout those scripts expect, shells out to
whichever one the config selects, and reads the resulting poses back in.

DROID-SLAM needs its compiled CUDA extensions (droid_backends, lietorch,
pytorch_scatter) built under Scene-Graph-Generation/{DROID-SLAM,lietorch}.
If they aren't available, set pose.backend: "cpu_icp" in the config to use
the geometry-only Open3D colored-ICP fallback instead (no CUDA required).
"""
from __future__ import annotations

import logging
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class DroidSlamPoseEstimator:
    def __init__(
        self,
        backend: str = "droid_slam",
        scene_graph_generation_root: str = "",
        dovsg_pose_estimation_script: str = "dovsg/scripts/pose_estimation.py",
        dovsg_cpu_pose_estimation_script: str = "dovsg/scripts/cpu_pose_estimation.py",
        stride: int = 1,
    ):
        self.backend = backend
        self.root = Path(scene_graph_generation_root)
        self.pose_script = dovsg_pose_estimation_script
        self.cpu_icp_script = dovsg_cpu_pose_estimation_script
        self.stride = stride

        if not self.root.exists():
            raise FileNotFoundError(
                f"external.scene_graph_generation_root does not exist: {self.root}. "
                "Set it in configs/pipeline.yaml to your Scene-Graph-Generation checkout."
            )

    # ─── Per-frame input staging ───

    def save_frame_inputs(
        self,
        dataset_dir: Path,
        frame_idx: int,
        depth_meters: np.ndarray,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
    ) -> None:
        """Writes point/*.npy + mask/*.npy (camera-frame XYZ back-projection +
        valid-depth mask) unconditionally — object_fusion's ViewDataset needs
        these in this exact format regardless of which pose backend is used
        (it world-transforms them itself once poses are known: gpoint = point
        @ pose[:3,:3].T + pose[:3,3], see dovsg/memory/view_dataset.py).
        Additionally writes depth/*.npy (millimeters) + calib.txt when the
        DROID-SLAM backend is selected, since dovsg/scripts/pose_estimation.py
        reads those instead. The RGB frame itself is expected to already be
        at dataset_dir/rgb/{frame_idx:06d}.jpg, written by the capture loop.
        """
        h, w = depth_meters.shape
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        valid = (depth_meters > 0.1) & (depth_meters < 10.0) & np.isfinite(depth_meters)
        safe_depth = np.where(valid, depth_meters, 0.0)
        x = (u - cx) * safe_depth / fx
        y = (v - cy) * safe_depth / fy
        z = safe_depth
        points = np.stack([x, y, z], axis=-1).astype(np.float32)

        point_dir = dataset_dir / "point"
        mask_dir = dataset_dir / "mask"
        point_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)
        np.save(point_dir / f"{frame_idx:06d}.npy", points)
        np.save(mask_dir / f"{frame_idx:06d}.npy", valid)

        if self.backend == "droid_slam":
            depth_dir = dataset_dir / "depth"
            depth_dir.mkdir(parents=True, exist_ok=True)
            calib_path = dataset_dir / "calib.txt"
            with open(calib_path, "w") as f:
                f.write(f"{fx} {fy} {cx} {cy}")
            depth_mm = np.clip(depth_meters * 1000.0, 0.0, 65535.0).astype(np.float32)
            np.save(depth_dir / f"{frame_idx:06d}.npy", depth_mm)

    # ─── Trajectory estimation ───

    def estimate_trajectory(self, dataset_dir: Path) -> None:
        """Run the selected pose-estimation script; writes dataset_dir/poses/*.txt."""
        dataset_dir = Path(dataset_dir)

        if self.backend == "droid_slam":
            self._run_droid_slam(dataset_dir)
        elif self.backend == "cpu_icp":
            self._run_cpu_icp(dataset_dir)
        else:
            raise ValueError(f"Unknown pose backend: {self.backend}")

    def _run_droid_slam(self, dataset_dir: Path) -> None:
        # Mirrors dovsg/scripts/pose_estimation.py's own sys.path setup
        # (droid_path = "./DROID-SLAM/droid_slam", relative to cwd=self.root)
        # — without this, "import droid" fails to even find the module and
        # this check reports DROID-SLAM as unavailable regardless of whether
        # its compiled extensions are actually built.
        check = subprocess.run(
            # torch must be imported before droid_backends/lietorch so their
            # libc10.so dependency is already loaded into the process.
            [sys.executable, "-c", "import sys, torch; sys.path.append('./DROID-SLAM/droid_slam'); import droid"],
            cwd=self.root,
            capture_output=True,
            text=True,
        )
        if check.returncode != 0:
            raise RuntimeError(
                "DROID-SLAM is not importable in this environment (compiled droid_backends/"
                "lietorch extensions missing or no CUDA). Set pose.backend: 'cpu_icp' in "
                f"configs/pipeline.yaml to use the fallback instead. Underlying error:\n{check.stderr}"
            )

        logger.info("Estimating camera poses with DROID-SLAM (RGB-D)...")
        subprocess.run(
            [
                sys.executable,
                self.pose_script,
                "--datadir", str(dataset_dir),
                "--calib", str(dataset_dir / "calib.txt"),
                "--pose_path", "poses_droidslam",
                "--stride", str(self.stride),
            ],
            cwd=self.root,
            check=True,
        )

        poses_src = dataset_dir / "poses_droidslam"
        poses_dst = dataset_dir / "poses"
        poses_dst.mkdir(parents=True, exist_ok=True)
        for pose_file in sorted(poses_src.glob("*.txt")):
            shutil.copy2(pose_file, poses_dst / pose_file.name)
        logger.info(f"DROID-SLAM poses copied into {poses_dst}")

    def _run_cpu_icp(self, dataset_dir: Path) -> None:
        logger.info("Estimating camera poses with CPU colored ICP (fallback)...")
        subprocess.run(
            [sys.executable, self.cpu_icp_script, "--datadir", str(dataset_dir)],
            cwd=self.root,
            check=True,
        )

    # ─── Reading results back ───

    def load_poses(self, dataset_dir: Path) -> dict[int, np.ndarray]:
        """Returns {frame_idx: 4x4 pose matrix} for every pose file written."""
        poses_dir = Path(dataset_dir) / "poses"
        poses: dict[int, np.ndarray] = {}
        for pose_file in sorted(poses_dir.glob("*.txt")):
            frame_idx = int(pose_file.stem)
            poses[frame_idx] = np.loadtxt(pose_file)
        return poses
