"""Pictures of each step of the DoVSG update."""
from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402


def render_voxels(points: np.ndarray, colors: np.ndarray, pose: np.ndarray, K: np.ndarray, h: int, w: int) -> np.ndarray:
    """Voxels seen from camera `pose`, z-buffered, pinholes between 1 cm voxels filled."""
    inv = np.linalg.inv(pose)
    pc = points @ inv[:3, :3].T + inv[:3, 3]
    front = pc[:, 2] > 0.05
    pc, cols = pc[front], colors[front]
    u = np.round(K[0, 0] * pc[:, 0] / pc[:, 2] + K[0, 2]).astype(int)
    v = np.round(K[1, 1] * pc[:, 1] / pc[:, 2] + K[1, 2]).astype(int)
    m = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    u, v, z, cols = u[m], v[m], pc[m, 2], cols[m]
    order = np.argsort(-z)
    img = np.zeros((h, w, 3), np.float32)
    img[v[order], u[order]] = cols[order]
    filled = cv2.dilate(img, np.ones((3, 3), np.uint8))
    hole = img.sum(2) == 0
    img[hole] = filled[hole]
    return img


def pose_check(path: Path, rgb: np.ndarray, renders: list, titles: list, frame: str) -> None:
    """The new frame next to the memory rendered from each candidate pose."""
    n = 1 + len(renders)
    fig, ax = plt.subplots(1, n, figsize=(3.0 * n, 5.0))
    ax[0].imshow(rgb)
    ax[0].set_title(f"new frame {frame}", fontsize=9)
    for a, r, t in zip(ax[1:], renders, titles):
        a.imshow(np.clip(0.55 * r + 0.45 * rgb * (r.sum(2, keepdims=True) > 0), 0, 1))
        a.set_title(t, fontsize=9)
    for a in ax:
        a.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def match_pair(path: Path, img_new: np.ndarray, img_mem: np.ndarray, kp_new: np.ndarray, kp_mem: np.ndarray,
               title: str) -> None:
    h, w = img_new.shape[:2]
    sx, sy = w / img_mem.shape[1], h / img_mem.shape[0]
    canvas = np.concatenate([img_new, cv2.resize(img_mem, (w, h))], axis=1)
    fig, ax = plt.subplots(figsize=(7, 6.5))
    ax.imshow(canvas)
    step = max(1, len(kp_new) // 150)
    for a, b in zip(kp_new[::step], kp_mem[::step]):
        ax.plot([a[0], b[0] * sx + w], [a[1], b[1] * sy], lw=0.6, color="lime", alpha=0.8)
    ax.set_title(f"{title}: {len(kp_new)} matches", fontsize=10)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def bar(path: Path, labels: list, values: list, highlight: int | None, title: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(max(6, 0.12 * len(values) + 3), 3.4))
    colors = ["tab:orange" if i == highlight else "tab:blue" for i in range(len(values))]
    ax.bar(range(len(values)), values, color=colors, width=0.85)
    step = max(1, len(labels) // 25)
    ax.set_xticks(range(0, len(labels), step))
    ax.set_xticklabels(labels[::step], rotation=90, fontsize=7)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def top_down_clouds(path: Path, memory_pts: np.ndarray, before: np.ndarray, after: np.ndarray, t_before: str,
                    t_after: str) -> None:
    """Memory (grey) and new observations (red) from above; z is up (gravity-aligned map)."""
    fig, ax = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
    for a, pts, t in ((ax[0], before, t_before), (ax[1], after, t_after)):
        a.scatter(memory_pts[:, 0], memory_pts[:, 1], s=0.3, c="0.7", linewidths=0, label="memory")
        a.scatter(pts[:, 0], pts[:, 1], s=0.3, c="tab:red", alpha=0.45, linewidths=0, label="new frames")
        a.set_title(t, fontsize=10)
        a.set_aspect("equal")
        a.set_xlabel("x (m)")
    ax[0].set_ylabel("y (m)")
    ax[0].legend(loc="upper right", markerscale=15, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def seen_through_map(path: Path, points: np.ndarray, colors: np.ndarray, deleted: np.ndarray, objects: dict,
                     title: str) -> None:
    """Memory from above, deleted voxels red, each object labelled with its deleted share."""
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(points[~deleted, 0], points[~deleted, 1], s=0.5, c=colors[~deleted], linewidths=0)
    ax.scatter(points[deleted, 0], points[deleted, 1], s=1.0, c="red", linewidths=0,
               label=f"seen through ({int(deleted.sum())} voxels)")
    for cid, v in objects.items():
        ax.annotate(f"{cid}\n{v['deleted_share']:.0%}", v["centroid"][:2], fontsize=7, ha="center",
                    color="red" if v["removed"] else "black")
    ax.set_aspect("equal")
    ax.legend(loc="upper right", markerscale=8)
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def frame_overlay(path: Path, rgb: np.ndarray, hit: np.ndarray) -> None:
    out = rgb.copy()
    out[hit] = 0.4 * out[hit] + 0.6 * np.array([1.0, 0.0, 0.0])
    Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8)).save(path)


def changes_map(path: Path, points: np.ndarray, colors: np.ndarray, removed: dict, added: dict, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(points[:, 0], points[:, 1], s=0.5, c=colors, linewidths=0)
    for cid, c in removed.items():
        ax.scatter(*c[:2], s=180, facecolors="none", edgecolors="red", lw=2)
        ax.annotate(f"removed {cid}", c[:2], color="red", fontsize=8)
    for cid, c in added.items():
        ax.scatter(*c[:2], s=180, facecolors="none", edgecolors="lime", lw=2)
        ax.annotate(f"added {cid}", c[:2], color="green", fontsize=8)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
