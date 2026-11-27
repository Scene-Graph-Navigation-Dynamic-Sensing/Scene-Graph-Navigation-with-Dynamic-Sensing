#!/usr/bin/env python3
"""Tile the five annotated videos into outputs/comparison_grid.mp4 (3x2 grid, 640x360 per tile)."""

import argparse
import os
import shutil
import subprocess

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_ap = argparse.ArgumentParser()
_ap.add_argument("--out-root", default=os.path.join(HERE, "outputs"))
OUT = os.path.abspath(_ap.parse_args().out_root)
NAMES = ["p1_dtd_every_frame", "p4_dtd_confirm_then_track", "p2_ram_yoloe_botsort",
         "p3a_ram_gdino_sam2_botsort", "p3b_ram_gdino_sam2_videotrack"]
TILE = (640, 360)

caps = [cv2.VideoCapture(os.path.join(OUT, n, "annotated.mp4")) for n in NAMES]
raw = os.path.join(OUT, "comparison_grid.raw.mp4")
w = cv2.VideoWriter(raw, cv2.VideoWriter_fourcc(*"mp4v"), 30, (TILE[0] * 3, TILE[1] * 2))
while True:
    tiles, alive = [], False
    for c, n in zip(caps, NAMES):
        ok, f = c.read()
        alive |= ok
        f = cv2.resize(f, TILE) if ok else np.zeros((TILE[1], TILE[0], 3), np.uint8)
        tiles.append(f)
    if not alive:
        break
    legend = np.zeros((TILE[1], TILE[0], 3), np.uint8)
    for i, n in enumerate(NAMES):
        cv2.putText(legend, n, (15, 50 + 50 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    tiles.append(legend)
    w.write(np.vstack([np.hstack(tiles[:3]), np.hstack(tiles[3:])]))
w.release()
final = os.path.join(OUT, "comparison_grid.mp4")
if shutil.which("ffmpeg") and subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", raw, "-c:v", "libx264",
                                              "-pix_fmt", "yuv420p", "-crf", "26", final]).returncode == 0:
    os.remove(raw)
else:
    os.replace(raw, final)
print(final)
