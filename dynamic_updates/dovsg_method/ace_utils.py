"""DoVSG's ACE relocalizer (external/DovSG/ace) on this project's captures.

ACE scales the intrinsics by 384 / image height but resizes the image's
shorter side to 384 (TF.resize with an int). For a landscape image these
agree; for a portrait one (the fyp phone videos are 478 x 850) the camera
model it trains and tests with is off by ~1.8x. So portrait frames go into
ACE rotated 90 degrees clockwise, with intrinsics and poses rotated to match,
and the pose ACE returns is rotated back. Landscape frames pass unchanged.
"""
from __future__ import annotations

import contextlib
import json
import os
import sys
from pathlib import Path

import numpy as np

# camera axes of the clockwise-rotated image: x' = -y, y' = x, z' = z
R_ROT = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])


def is_portrait(h: int, w: int) -> bool:
    return h > w


def rotate_image(img: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.rot90(img, k=-1))


def rotate_intrinsics(K: np.ndarray, h: int) -> np.ndarray:
    """K of the clockwise-rotated image: u' = h - 1 - v, v' = u."""
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    return np.array([[fy, 0.0, h - 1 - cy], [0.0, fx, cx], [0.0, 0.0, 1.0]])


def rotate_pose(c2w: np.ndarray) -> np.ndarray:
    """Camera-to-world pose of the rotated camera."""
    out = c2w.copy()
    out[:3, :3] = c2w[:3, :3] @ R_ROT.T
    return out


def unrotate_pose(c2w_rot: np.ndarray) -> np.ndarray:
    out = c2w_rot.copy()
    out[:3, :3] = c2w_rot[:3, :3] @ R_ROT
    return out


def read_K(capture_dir: Path) -> np.ndarray:
    k = json.loads((capture_dir / "metadata.json").read_text())["K"]
    return np.array([[k[0], 0, k[2]], [0, k[4], k[5]], [0, 0, 1]], dtype=np.float64)


@contextlib.contextmanager
def working_dir(path: Path):
    """DoVSG's ACE code reads ace/configs/*.yml and the encoder relative to its repo root."""
    old = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def make_ace_importable(dovsg_root: Path) -> None:
    """`ace` from DoVSG, appended (not prepended) so `dovsg` still resolves to
    Scene-Graph-Generation's package. dsacstar (ACE's RANSAC PnP) is built by
    build_dsacstar.sh into DovSG/third_party/dsacstar_ext."""
    for p in (dovsg_root / "third_party" / "dsacstar_ext", dovsg_root):
        if str(p) not in sys.path:
            sys.path.append(str(p))


def write_ace_dataset(capture_dir: Path, out_dir: Path) -> int:
    """ACE's training layout (rgb/, poses/, calibration/, one file per frame)
    from a processed capture, rotated to landscape when the frames are portrait.
    Poses are the capture's poses/: the frame the voxel map is built in."""
    from PIL import Image

    K = read_K(capture_dir)
    names = sorted(p.stem for p in (capture_dir / "rgb").glob("*.jpg"))
    for sub in ("rgb", "poses", "calibration"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)
    for name in names:
        rgb = np.asarray(Image.open(capture_dir / "rgb" / f"{name}.jpg"))
        pose = np.loadtxt(capture_dir / "poses" / f"{name}.txt")
        h, w = rgb.shape[:2]
        K_i = K
        if is_portrait(h, w):
            rgb, K_i, pose = rotate_image(rgb), rotate_intrinsics(K, h), rotate_pose(pose)
        Image.fromarray(rgb).save(out_dir / "rgb" / f"{name}.jpg", quality=95)
        np.savetxt(out_dir / "poses" / f"{name}.txt", pose)
        np.savetxt(out_dir / "calibration" / f"{name}.txt", K_i)
    return len(names)


def ace_poses(dovsg_root: Path, ace_model: Path, observations: dict) -> list[tuple[np.ndarray, int]]:
    """controller.py test_ace(): per frame, ACE scene coordinates + RANSAC PnP.
    Returns (camera-to-world pose in the memory's coordinates, inlier count)."""
    make_ace_importable(dovsg_root)
    from ace.test_ace import test_ace

    inputs, rotated = {}, {}
    for k, o in observations.items():
        rgb = (o["rgb"] * 255).astype(np.uint8)
        h, w = rgb.shape[:2]
        rotated[k] = is_portrait(h, w)
        if rotated[k]:
            inputs[k] = {"rgb": rotate_image(rgb), "depth": rotate_image(o["depth"]), "mask": rotate_image(o["mask"]),
                         "intrinsic": rotate_intrinsics(o["intrinsic"], h)}
        else:
            inputs[k] = {"rgb": rgb, "depth": o["depth"], "mask": o["mask"], "intrinsic": o["intrinsic"]}
    with working_dir(dovsg_root):
        out = test_ace(str(Path(ace_model).resolve()), inputs)
    poses = []
    for k, (pose, inliers) in zip(observations, out):
        pose = np.asarray(pose, dtype=np.float64)
        poses.append((unrotate_pose(pose) if rotated[k] else pose, int(inliers)))
    return poses
