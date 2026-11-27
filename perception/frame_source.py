"""One input interface for the pipeline: a live stream, a video file, or a dataset folder.

Stream  -- DroidCam URL ("http://<phone-ip>:4747/video") or a webcam index ("0").
           Sampled by wall-clock time (capture_interval_s), since the source
           keeps producing frames whether or not we read them.
Video   -- any file cv2 can decode. Sampled by frame count (stride); sampling
           a file by wall-clock time would make the selected frames depend on
           how fast this machine happens to decode.
Folder  -- a dataset directory (see data/raw/README.md):
               rgb/          frames, any sortable names (.jpg/.png)
               camera.yaml   fx fy cx cy [width height] [depth_scale]
               depth/        optional reference depth, same stems as rgb/
               poses/        optional ground-truth 4x4 camera-to-world, same stems
           A folder holding a single video file (and no rgb/) is read as a video,
           so camera.yaml can sit beside it.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
import yaml

logger = logging.getLogger(__name__)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}


@dataclass(frozen=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    def scaled_to(self, width: int, height: int) -> "CameraIntrinsics":
        sx, sy = width / self.width, height / self.height
        return replace(self, fx=self.fx * sx, fy=self.fy * sy, cx=self.cx * sx, cy=self.cy * sy,
                       width=width, height=height)


@dataclass
class Frame:
    rgb: np.ndarray                     # HxWx3 uint8 RGB at native resolution
    name: str                           # stable id: file stem, or source frame number
    ref_depth_m: np.ndarray | None = None


def _read_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"cannot read image {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


class FrameSource:
    def __init__(
        self,
        uri: str | int,
        capture_interval_s: float = 0.3,
        stride: int = 1,
        max_frames: int = 0,
        depth_scale: float = 1000.0,
    ):
        self.uri = str(uri)
        self.capture_interval_s = capture_interval_s
        self.stride = max(1, int(stride))
        self.max_frames = int(max_frames)
        self.depth_scale = depth_scale
        self.intrinsics: CameraIntrinsics | None = None
        self._rgb_paths: list[Path] = []
        self._video_path: Path | None = None

        path = Path(self.uri)
        if path.is_dir():
            self._init_folder(path)
        elif path.is_file() and path.suffix.lower() in VIDEO_EXTS:
            self.kind = "video"
            self._video_path = path
        else:
            self.kind = "stream"

        logger.info("frame source: %s (%s)", self.uri, self.kind)

    def _init_folder(self, root: Path) -> None:
        self.root = root
        camera_yaml = root / "camera.yaml"
        cam = yaml.safe_load(camera_yaml.read_text()) if camera_yaml.exists() else None
        if cam and "depth_scale" in cam:
            self.depth_scale = float(cam["depth_scale"])

        rgb_dir = root / "rgb"
        if rgb_dir.is_dir():
            self.kind = "folder"
            self._rgb_paths = sorted(p for p in rgb_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
            if not self._rgb_paths:
                raise FileNotFoundError(f"no images in {rgb_dir}")
            sample_hw = _read_rgb(self._rgb_paths[0]).shape[:2]
        else:
            videos = sorted(p for p in root.iterdir() if p.suffix.lower() in VIDEO_EXTS)
            if len(videos) != 1:
                raise FileNotFoundError(f"{root}: expected an rgb/ folder or exactly one video file, "
                                        f"found {len(videos)} videos")
            self.kind = "video"
            self._video_path = videos[0]
            cap = cv2.VideoCapture(str(self._video_path))
            sample_hw = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
            cap.release()

        if cam:
            self.intrinsics = CameraIntrinsics(
                fx=float(cam["fx"]), fy=float(cam["fy"]), cx=float(cam["cx"]), cy=float(cam["cy"]),
                width=int(cam.get("width", sample_hw[1])), height=int(cam.get("height", sample_hw[0])),
            )

    # ─── optional ground truth (folder datasets only) ───

    def _companion(self, subdir: str, stem: str, exts: tuple[str, ...]) -> Path | None:
        if self.kind != "folder":
            return None
        for ext in exts:
            p = self.root / subdir / f"{stem}{ext}"
            if p.exists():
                return p
        return None

    def has_reference_depth(self) -> bool:
        return self.kind == "folder" and (self.root / "depth").is_dir()

    def load_reference_depth(self, name: str) -> np.ndarray | None:
        p = self._companion("depth", name, (".png", ".npy", ".pgm", ".tiff", ".tif"))
        if p is None:
            return None
        if p.suffix == ".npy":
            return np.load(p).astype(np.float32)
        raw = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        return raw.astype(np.float32) / self.depth_scale if raw is not None else None

    def reference_poses(self) -> dict[str, np.ndarray]:
        """{frame name: 4x4 camera-to-world}, empty when the dataset ships none."""
        if self.kind != "folder" or not (self.root / "poses").is_dir():
            return {}
        poses = {}
        for p in sorted((self.root / "poses").glob("*.txt")):
            T = np.loadtxt(p)
            if T.shape == (4, 4) and np.isfinite(T).all():
                poses[p.stem] = T
        return poses

    def sample(self, n: int) -> list[Frame]:
        """Up to n frames spread evenly over a folder dataset, with reference depth."""
        if self.kind != "folder":
            raise RuntimeError("sampling needs random access — only available for folder datasets")
        paths = self._selected_paths()
        picks = paths[:: max(len(paths) // n, 1)][:n]
        return [Frame(_read_rgb(p), p.stem, self.load_reference_depth(p.stem)) for p in picks]

    # ─── iteration ───

    def _selected_paths(self) -> list[Path]:
        paths = self._rgb_paths[:: self.stride]
        return paths[: self.max_frames] if self.max_frames else paths

    def __iter__(self) -> Iterator[Frame]:
        if self.kind == "folder":
            for p in self._selected_paths():
                yield Frame(_read_rgb(p), p.stem, self.load_reference_depth(p.stem))
        elif self.kind == "video":
            yield from self._iter_video()
        else:
            yield from self._iter_stream()

    def _iter_video(self) -> Iterator[Frame]:
        cap = cv2.VideoCapture(str(self._video_path))
        if not cap.isOpened():
            raise RuntimeError(f"cannot open video {self._video_path}")
        emitted, idx = 0, -1
        try:
            while not self.max_frames or emitted < self.max_frames:
                ok, bgr = cap.read()
                if not ok:
                    break
                idx += 1
                if idx % self.stride:
                    continue
                emitted += 1
                yield Frame(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), f"{idx:06d}")
        finally:
            cap.release()

    def _iter_stream(self) -> Iterator[Frame]:
        cap = cv2.VideoCapture(int(self.uri)) if self.uri.isdigit() else cv2.VideoCapture(self.uri)
        if not cap.isOpened():
            raise RuntimeError(f"cannot connect to video source {self.uri}")
        for _ in range(5):  # let exposure settle; DroidCam's first frames are often dark
            cap.read()
        emitted, idx, last = 0, -1, 0.0
        try:
            while not self.max_frames or emitted < self.max_frames:
                ok, bgr = cap.read()
                if not ok:
                    break
                idx += 1
                now = time.time()
                if now - last < self.capture_interval_s:
                    continue
                last = now
                emitted += 1
                yield Frame(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), f"{idx:06d}")
        finally:
            cap.release()
