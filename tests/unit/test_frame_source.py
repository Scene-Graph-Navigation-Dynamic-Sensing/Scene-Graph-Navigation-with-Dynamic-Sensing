import tempfile
from pathlib import Path

import cv2
import numpy as np

from perception.frame_source import CameraIntrinsics, FrameSource


def _write_folder_dataset(root: Path, n: int = 5, with_camera: bool = True) -> None:
    for sub in ("rgb", "depth", "poses"):
        (root / sub).mkdir(parents=True)
    for i in range(n):
        img = np.full((48, 64, 3), i * 40, dtype=np.uint8)
        cv2.imwrite(str(root / "rgb" / f"{i:06d}.png"), img)
        cv2.imwrite(str(root / "depth" / f"{i:06d}.png"), np.full((48, 64), 5000 * (i + 1), dtype=np.uint16))
        T = np.eye(4)
        T[0, 3] = i
        np.savetxt(root / "poses" / f"{i:06d}.txt", T)
    if with_camera:
        (root / "camera.yaml").write_text("fx: 50\nfy: 60\ncx: 32\ncy: 24\ndepth_scale: 5000\n")


def test_intrinsics_scale_with_resolution():
    K = CameraIntrinsics(fx=500, fy=400, cx=320, cy=240, width=640, height=480)
    half = K.scaled_to(320, 240)
    assert (half.fx, half.fy, half.cx, half.cy, half.width, half.height) == (250, 200, 160, 120, 320, 240)


def test_folder_reads_frames_in_order_with_stride_and_limit():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        _write_folder_dataset(root, n=5)
        names = [f.name for f in FrameSource(root, stride=2, max_frames=2)]
        assert names == ["000000", "000002"]


def test_folder_loads_intrinsics_depth_scale_and_reference_data():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        _write_folder_dataset(root, n=3)
        src = FrameSource(root)
        assert src.kind == "folder"
        # width/height default to the native rgb resolution when camera.yaml omits them
        assert src.intrinsics == CameraIntrinsics(fx=50, fy=60, cx=32, cy=24, width=64, height=48)
        frames = list(src)
        # depth_scale from camera.yaml: raw 10000 / 5000 = 2 m
        assert np.allclose(frames[1].ref_depth_m, 2.0)
        poses = src.reference_poses()
        assert sorted(poses) == ["000000", "000001", "000002"]
        assert poses["000002"][0, 3] == 2


def test_folder_without_camera_yaml_has_no_intrinsics():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        _write_folder_dataset(root, n=2, with_camera=False)
        assert FrameSource(root).intrinsics is None


def test_video_file_is_sampled_by_frame_stride():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "clip.avi"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48))
        for i in range(7):
            writer.write(np.full((48, 64, 3), i * 30, dtype=np.uint8))
        writer.release()

        src = FrameSource(path, stride=3)
        assert src.kind == "video"
        assert [f.name for f in src] == ["000000", "000003", "000006"]
