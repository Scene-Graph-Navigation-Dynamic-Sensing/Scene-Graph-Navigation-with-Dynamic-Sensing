"""Convert a ZED2i/Isaac-Sim synthetic capture into this pipeline's plain
`data/raw/<name>/` dataset layout (see data/raw/README.md): rgb/, camera.yaml,
depth/, poses/, matching stems. Unlike prepare_3rscan_dataset.py this writes
no point/mask/DoVSG-metadata files -- run_static_scene_graph.py's own
capture_frames() builds those later from whatever depth/pose backend the
config picks.

Source layout expected (as produced by the ROS/Isaac-Sim recorder):
  rgb/<stem>.png        colour frames
  depth/<stem>.npy      float32 metres, ground-truth (Isaac Sim depth_registered)
  frames.csv            frame_idx,stamp_sec,stamp_nanosec,rgb_path,depth_path
                         -- the authoritative frame list; iterated here instead
                         of listing rgb/ directly, since this recorder's rgb/
                         and depth/ folders can have one more file than
                         frames.csv lists.
  odom.csv              stamp_sec,stamp_nanosec,x,y,z,qx,qy,qz,qw,... -- ground
                         truth pose of the robot base (/<robot>/odom), *not*
                         the camera.
  imu.csv               stamp_sec,stamp_nanosec,wx,wy,wz,ax,ay,az -- raw IMU,
                         copied through unchanged; no pipeline stage consumes
                         it yet (kept for future dynamic-sensing work).
  metadata.json          camera intrinsics (a ROS CameraInfo dump) + topic names.

Camera pose caveat (read before trusting the trajectory):
  odom.csv is the robot base pose, and this dataset ships no extrinsics/TF
  relating the base to the camera. This script assumes the camera is
  coincident with the base origin (zero translation offset) and applies only
  the fixed ROS body(FLU)->camera-optical(RDF) axis-convention rotation
  below. If the real camera mount has a nonzero offset or tilt, the written
  trajectory will be systematically off by that same fixed amount -- this is
  a documented approximation, not a bug.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import shutil
from pathlib import Path

import numpy as np

logger = logging.getLogger("prepare_synthetic_dataset")

# Fixed rotation from ROS body frame (FLU: x-forward, y-left, z-up) to camera
# optical frame (RDF: x-right, y-down, z-forward), assuming the camera looks
# along the body's forward axis with no roll/tilt and no translation offset
# (see module docstring). Derivation: for a vector v with body-frame
# coordinates v_b and optical-frame coordinates v_o, v_o = R @ v_b where
#   R = [[0, -1, 0], [0, 0, -1], [1, 0, 0]]
# (forward -> +z_opt, left -> -x_opt, up -> -y_opt). The camera axes
# expressed in body coordinates are then the columns of R.T, so if R_wb is
# the body-to-world rotation (from the odom quaternion), the camera-to-world
# rotation is R_wc = R_wb @ R.T:
BODY_TO_OPTICAL_T = np.array([
    [0.0, 0.0, 1.0],
    [-1.0, 0.0, 0.0],
    [0.0, -1.0, 0.0],
], dtype=np.float64)


def quat_to_matrix(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Hamilton quaternion (x, y, z, w) -> 3x3 rotation matrix."""
    n = (qx * qx + qy * qy + qz * qz + qw * qw) ** 0.5
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ], dtype=np.float64)


def odom_pose_to_camera_to_world(x: float, y: float, z: float,
                                  qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Robot-base pose (odom.csv row) -> 4x4 camera-to-world matrix, under
    the zero-offset assumption documented at the top of this file."""
    r_wb = quat_to_matrix(qx, qy, qz, qw)
    r_wc = r_wb @ BODY_TO_OPTICAL_T
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = r_wc
    T[:3, 3] = [x, y, z]
    return T


def load_odom(path: Path) -> np.ndarray:
    """Returns an (N, 8) array of [t, x, y, z, qx, qy, qz, qw], sorted by t."""
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            t = float(row["stamp_sec"]) + float(row["stamp_nanosec"]) * 1e-9
            rows.append((t, float(row["x"]), float(row["y"]), float(row["z"]),
                         float(row["qx"]), float(row["qy"]), float(row["qz"]), float(row["qw"])))
    odom = np.array(rows, dtype=np.float64)
    return odom[np.argsort(odom[:, 0])]


def nearest_odom_row(odom: np.ndarray, t: float) -> np.ndarray:
    idx = int(np.searchsorted(odom[:, 0], t))
    candidates = [i for i in (idx - 1, idx) if 0 <= i < len(odom)]
    best = min(candidates, key=lambda i: abs(odom[i, 0] - t))
    return odom[best]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sequence", required=True, type=Path, help="synthetic_dataset_01-style capture dir")
    ap.add_argument("--out", required=True, type=Path, help="data/raw/<name> dir to create")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    metadata = json.loads((args.sequence / "metadata.json").read_text())
    K = metadata["camera"]["K"]
    fx, fy, cx, cy = K[0], K[4], K[2], K[5]
    width, height = metadata["camera"]["width"], metadata["camera"]["height"]

    odom = load_odom(args.sequence / "odom.csv")

    rgb_out = args.out / "rgb"
    depth_out = args.out / "depth"
    poses_out = args.out / "poses"
    for d in (rgb_out, depth_out, poses_out):
        d.mkdir(parents=True, exist_ok=True)

    written, skipped = 0, 0
    with open(args.sequence / "frames.csv") as f:
        for row in csv.DictReader(f):
            t = float(row["stamp_sec"]) + float(row["stamp_nanosec"]) * 1e-9
            rgb_path = args.sequence / row["rgb_path"]
            depth_path = args.sequence / row["depth_path"]
            if not rgb_path.exists() or not depth_path.exists():
                logger.warning("frame %s: missing %s, skipping", row["frame_idx"], row["rgb_path"])
                skipped += 1
                continue

            stem = rgb_path.stem
            shutil.copy2(rgb_path, rgb_out / rgb_path.name)
            shutil.copy2(depth_path, depth_out / depth_path.name)

            odom_row = nearest_odom_row(odom, t)
            dt = abs(odom_row[0] - t)
            if dt > 0.1:
                logger.warning("frame %s: nearest odom sample is %.3fs away", stem, dt)
            T = odom_pose_to_camera_to_world(*odom_row[1:])
            np.savetxt(poses_out / f"{stem}.txt", T)
            written += 1

    imu_src = args.sequence / "imu.csv"
    if imu_src.exists():
        shutil.copy2(imu_src, args.out / "imu.csv")

    (args.out / "camera.yaml").write_text(
        f"# converted from {args.sequence}/metadata.json (camera-to-world poses "
        f"derived from odom.csv under a zero-offset body->camera assumption -- see script docstring)\n"
        f"fx: {fx}\nfy: {fy}\ncx: {cx}\ncy: {cy}\nwidth: {width}\nheight: {height}\n")

    logger.info("wrote %d frames to %s (%d skipped)", written, args.out, skipped)
    if written < 2:
        logger.error("only %d frame(s) written", written)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
