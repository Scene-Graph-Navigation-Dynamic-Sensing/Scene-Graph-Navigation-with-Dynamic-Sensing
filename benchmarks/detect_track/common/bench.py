"""Shared harness: frame loop, per-stage CUDA-synced timing, prediction log, annotated video."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from contextlib import contextmanager

import cv2
import numpy as np
import torch

VIDEO = "/home/ravindu/SceneGraphNav/datasets/IMG_6100.mp4"
OUT_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
VIS_SIZE = (1280, 720)


def read_frames(source: str):
    """Yield (index, BGR frame) from a video file or a folder of images (sorted by name)."""
    if os.path.isdir(source):
        files = sorted(f for f in os.listdir(source) if f.lower().endswith((".png", ".jpg", ".jpeg")))
        for i, f in enumerate(files):
            yield i, cv2.imread(os.path.join(source, f), cv2.IMREAD_COLOR)
        return
    cap = cv2.VideoCapture(source)
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        yield i, frame
        i += 1
    cap.release()


class StageTimer:
    def __init__(self):
        self.stages: dict[str, float] = {}

    @contextmanager
    def __call__(self, name: str):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        yield
        torch.cuda.synchronize()
        self.stages[name] = self.stages.get(name, 0.0) + (time.perf_counter() - t0) * 1000


def _color(tid: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(tid * 7919 + 17)
    return tuple(int(c) for c in rng.integers(60, 255, 3))


def draw(frame: np.ndarray, objects: list[dict], masks: dict | None = None, title: str = "") -> np.ndarray:
    """Draw at VIS_SIZE. masks: {track_id: bool HxW at original resolution}."""
    h, w = frame.shape[:2]
    sx, sy = VIS_SIZE[0] / w, VIS_SIZE[1] / h
    img = cv2.resize(frame, VIS_SIZE, interpolation=cv2.INTER_AREA)
    if masks:
        overlay = img.copy()
        for tid, m in masks.items():
            small = cv2.resize(m.astype(np.uint8), VIS_SIZE, interpolation=cv2.INTER_NEAREST).astype(bool)
            overlay[small] = _color(tid)
        img = cv2.addWeighted(overlay, 0.45, img, 0.55, 0)
    for o in objects:
        x1, y1, x2, y2 = o["box"]
        p1, p2 = (int(x1 * sx), int(y1 * sy)), (int(x2 * sx), int(y2 * sy))
        c = _color(o["id"])
        cv2.rectangle(img, p1, p2, c, 2)
        text = f"{o['label'] or '...'} #{o['id']}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ty = max(th + 4, p1[1])
        cv2.rectangle(img, (p1[0], ty - th - 4), (p1[0] + tw + 4, ty), c, -1)
        cv2.putText(img, text, (p1[0] + 2, ty - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    if title:
        cv2.putText(img, title, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    return img


class Bench:
    """
    Usage:
        bench = Bench("p1_dtd_batch", config={...})
        ... load models ...
        bench.models_loaded()
        for idx, frame in bench.frames():
            t = bench.timer()
            with t("stage"): ...
            bench.record(idx, frame, objects, t, masks=None)
        bench.finish()

    objects: list of {"id": int, "label": str, "score": float, "box": [x1,y1,x2,y2]} in original pixels.
    """

    def __init__(self, name: str, config: dict, video: str | None = None, max_frames: int = 0, save_video: bool = True,
                 out_root: str | None = None):
        self.name = name
        self.video = video or VIDEO
        self.config = dict(config, source=self.video)
        self.max_frames = max_frames
        self.out_dir = os.path.join(out_root or OUT_ROOT, name)
        os.makedirs(self.out_dir, exist_ok=True)
        self._t_start = time.perf_counter()
        self._load_s = None
        self._pred_f = open(os.path.join(self.out_dir, "predictions.jsonl"), "w")
        self._rows: list[dict] = []
        self._writer = None
        self._raw_video = os.path.join(self.out_dir, "annotated.raw.mp4") if save_video else None
        self._frame_t0 = None
        torch.cuda.reset_peak_memory_stats()

    def models_loaded(self):
        torch.cuda.synchronize()
        self._load_s = time.perf_counter() - self._t_start

    def _read(self):
        """Yield BGR frames from a video file or a folder of images (sorted by name)."""
        if os.path.isdir(self.video):
            self.fps_in = 30.0
            self.n_total = sum(f.lower().endswith((".png", ".jpg", ".jpeg")) for f in os.listdir(self.video))
        else:
            cap = cv2.VideoCapture(self.video)
            self.fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
            self.n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
        for _, frame in read_frames(self.video):
            yield frame

    def frames(self):
        self.resolution = None
        idx = 0
        for frame in self._read():
            if self.resolution is None:
                self.resolution = [frame.shape[1], frame.shape[0]]
            torch.cuda.synchronize()
            self._frame_t0 = time.perf_counter()
            yield idx, frame
            idx += 1
            if self.max_frames and idx >= self.max_frames:
                break

    @staticmethod
    def timer() -> StageTimer:
        return StageTimer()

    def record(self, idx: int, frame: np.ndarray, objects: list[dict], timer: StageTimer,
               masks: dict | None = None, extra: dict | None = None):
        torch.cuda.synchronize()
        total_ms = (time.perf_counter() - self._frame_t0) * 1000   # excludes drawing/writing below
        objs = [{"id": int(o["id"]), "label": o.get("label") or "", "score": round(float(o.get("score", 0.0)), 4),
                 "box": [round(float(v), 1) for v in o["box"]]} for o in objects]
        if masks:
            for o in objs:
                m = masks.get(o["id"])
                if m is not None:
                    o["mask_area"] = int(m.sum())
        row = {"frame": idx, "total_ms": round(total_ms, 2),
               "stages_ms": {k: round(v, 2) for k, v in timer.stages.items()}, "objects": objs}
        if extra:
            row.update(extra)
        self._pred_f.write(json.dumps(row) + "\n")
        self._rows.append(row)

        if self._raw_video:
            vis = draw(frame, objs, masks, f"{self.name}  {total_ms:.0f} ms/frame")
            if self._writer is None:
                self._writer = cv2.VideoWriter(self._raw_video, cv2.VideoWriter_fourcc(*"mp4v"), self.fps_in, VIS_SIZE)
            self._writer.write(vis)
        if idx % 20 == 0:
            print(f"[{self.name}] frame {idx}/{self.n_total} {total_ms:.0f} ms, {len(objs)} objects", flush=True)

    def finish(self):
        self._pred_f.close()
        rows = self._rows[1:] if len(self._rows) > 1 else self._rows   # frame 0 = warm-up
        tot = np.array([r["total_ms"] for r in rows])
        stage_names = sorted({k for r in rows for k in r["stages_ms"]})
        stages = {k: round(float(np.mean([r["stages_ms"].get(k, 0.0) for r in rows])), 2) for k in stage_names}
        ids = {o["id"] for r in self._rows for o in r["objects"]}
        summary = {
            "pipeline": self.name,
            "config": self.config,
            "frames": len(self._rows),
            "input_resolution": self.resolution,
            "model_load_s": round(self._load_s or 0.0, 1),
            "mean_ms": round(float(tot.mean()), 1),
            "median_ms": round(float(np.median(tot)), 1),
            "p95_ms": round(float(np.percentile(tot, 95)), 1),
            "max_ms": round(float(tot.max()), 1),
            "fps_mean": round(1000.0 / float(tot.mean()), 2),
            "realtime_factor_vs_30fps": round((1000.0 / float(tot.mean())) / 30.0, 3),
            "stages_mean_ms": stages,
            "peak_gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
            "unique_track_ids": len(ids),
            "mean_objects_per_frame": round(float(np.mean([len(r["objects"]) for r in self._rows])), 2),
            "gpu": torch.cuda.get_device_name(0),
            "note": "timings exclude model loading, frame decoding and visualisation; frame 0 excluded as warm-up",
        }
        with open(os.path.join(self.out_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        print(json.dumps(summary, indent=2))

        if self._writer is not None:
            self._writer.release()
            final = os.path.join(self.out_dir, "annotated.mp4")
            if shutil.which("ffmpeg"):
                r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", self._raw_video, "-c:v", "libx264",
                                    "-pix_fmt", "yuv420p", "-crf", "26", final])
                if r.returncode == 0:
                    os.remove(self._raw_video)
                    return summary
            os.replace(self._raw_video, final)
        return summary
