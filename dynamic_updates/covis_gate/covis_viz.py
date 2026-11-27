"""Pictures of each step of the co-visibility gated update."""
from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import theory as T  # noqa: E402

STATUS_COLOUR = {"present": "tab:green", "deferred": "tab:orange", "absent": "tab:red", "removed": "tab:red",
                 "moved": "tab:purple", "added": "tab:blue"}


def retrieval_matrix(path: Path, counts: np.ndarray, new_names: list, mem_names: list) -> None:
    fig, ax = plt.subplots(figsize=(max(6, 0.12 * len(mem_names) + 2), max(3, 0.18 * len(new_names) + 1.5)))
    im = ax.imshow(counts, aspect="auto", cmap="viridis")
    ax.set_yticks(range(len(new_names)))
    ax.set_yticklabels(new_names, fontsize=6)
    step = max(1, len(mem_names) // 30)
    ax.set_xticks(range(0, len(mem_names), step))
    ax.set_xticklabels(mem_names[::step], rotation=90, fontsize=6)
    ax.set_xlabel("memory frame")
    ax.set_ylabel("new frame")
    ax.set_title("retrieval: LightGlue matches (no ACE)", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def anchor_match_pair(path: Path, img_new, img_mem, kp_new, kp_mem, is_anchor, title: str) -> None:
    h, w = img_new.shape[:2]
    sx, sy = w / img_mem.shape[1], h / img_mem.shape[0]
    canvas = np.concatenate([img_new, cv2.resize(img_mem, (w, h))], axis=1)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.imshow(canvas)
    step = max(1, len(kp_new) // 200)
    for a, b, anc in zip(kp_new[::step], kp_mem[::step], is_anchor[::step]):
        ax.plot([a[0], b[0] * sx + w], [a[1], b[1] * sy], lw=0.6, color="lime" if anc else "red", alpha=0.8)
    ax.set_title(f"{title}: {int(is_anchor.sum())} matches on anchors (green, used), "
                 f"{int((~is_anchor).sum())} elsewhere (red, not used)", fontsize=9)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def phi_bars(path: Path, phi: dict, tau: float) -> None:
    names = list(phi)
    vals = [phi[n]["phi"] for n in names]
    fig, ax = plt.subplots(figsize=(max(5, 0.5 * len(names) + 2), 3.4))
    ax.bar(range(len(names)), vals, color=["tab:red" if v < tau else "tab:green" for v in vals])
    ax.axhline(tau, color="k", ls="--", lw=1, label=f"tau_phi = {tau}")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([f"{n}\n({phi[n]['matches']})" for n in names], rotation=60, fontsize=7, ha="right")
    ax.set_ylabel("inlier share phi_j")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    ax.set_title("§5.5: anchors' agreement with the consensus pose (matches in brackets)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def survival_curves(path: Path, recs: dict, dt_now: dict) -> None:
    """§3.3: each object's S(dt), its dt this visit marked."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    horizon = max(max(dt_now.values(), default=1.0) * 2, 1.0)
    x = np.linspace(0, horizon, 200)
    for cid, r in recs.items():
        line, = ax.plot(x, [(r["b"] / (r["b"] + t)) ** r["a"] for t in x], lw=1, label=f"{cid} ({r['a'] / r['b']:.3g}/d)")
        ax.plot(dt_now[cid], T.survival(r["a"], r["b"], dt_now[cid]), "o", color=line.get_color(), ms=4)
    ax.set_xlabel("days since last confirmed")
    ax.set_ylabel("S(dt): still in place")
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=6, ncol=2, loc="lower left")
    ax.set_title("§3 dynamism prior: Lomax survival per object (dot: this visit)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def belief_trajectories(path: Path, names: list, traj: np.ndarray, prior_L: np.ndarray, labels: list,
                        thresholds: tuple, frame_names: list) -> None:
    n = len(names)
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 2.4 * rows), squeeze=False, sharex=True, sharey=True)
    x = np.arange(len(frame_names) + 1)
    for k, ax in enumerate(axes.ravel()):
        if k >= n:
            ax.axis("off")
            continue
        P = T.sigmoid(np.concatenate([[prior_L[k]], traj[:, k]]))
        ax.plot(x, P, color=STATUS_COLOUR.get(labels[k], "k"), lw=1.3)
        ax.axhline(thresholds[0], color="tab:red", ls=":", lw=0.8)
        ax.axhline(thresholds[1], color="tab:green", ls=":", lw=0.8)
        ax.set_title(f"{names[k]}: {labels[k]} (P {P[-1]:.2f})", fontsize=8)
        ax.set_ylim(-0.02, 1.02)
    for ax in axes[-1]:
        ax.set_xlabel("frames (0 = prior)", fontsize=8)
    fig.suptitle("§4 belief of presence after each frame; dotted: Chow thresholds", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def gate_over_frames(path: Path, names: list, rho: np.ndarray, is_anchor: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    for k, nm in enumerate(names):
        if not is_anchor[k]:
            ax.plot(rho[:, k], lw=1, label=nm)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("frame")
    ax.set_ylabel("rho (co-visibility trust)")
    ax.set_title("§7 gate of each target per frame (anchors: rho = 1)", fontsize=10)
    ax.legend(fontsize=6, ncol=3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def decision_bars(path: Path, names: list, P: np.ndarray, coverage: np.ndarray, labels: list, thresholds: tuple,
                  v_min: float) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(max(8, 0.6 * len(names) + 4), 4), sharex=True)
    xs = range(len(names))
    c = [STATUS_COLOUR.get(l, "k") for l in labels]
    ax[0].bar(xs, P, color=c)
    ax[0].axhline(thresholds[0], color="tab:red", ls="--", lw=1)
    ax[0].axhline(thresholds[1], color="tab:green", ls="--", lw=1)
    ax[0].set_ylabel("P(present)")
    ax[1].bar(xs, coverage, color=c)
    ax[1].axhline(v_min, color="k", ls="--", lw=1)
    ax[1].set_ylabel("gated through coverage")
    for a in ax:
        a.set_xticks(list(xs))
        a.set_xticklabels([f"{n}\n{l}" for n, l in zip(names, labels)], rotation=70, fontsize=7, ha="right")
    fig.suptitle("§8 decision: Chow thresholds (left), coverage condition for absent (right)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def outcome_top_down(path: Path, pts: np.ndarray, cols: np.ndarray, totals: np.ndarray, centroids: dict,
                     labels: dict, P: dict, title: str) -> None:
    """Memory from above; a voxel's colour is its majority outcome over the visit."""
    fig, ax = plt.subplots(figsize=(8, 8))
    seen = totals.sum(1) > 0
    ax.scatter(pts[~seen, 0], pts[~seen, 1], s=0.3, c=cols[~seen], alpha=0.4, linewidths=0)
    major = np.argmax(totals, 1)
    for k, colour in enumerate(((0.1, 0.7, 0.1), (0.9, 0.1, 0.1), (0.4, 0.4, 0.9))):
        m = seen & (major == k)
        ax.scatter(pts[m, 0], pts[m, 1], s=0.6, color=colour, linewidths=0,
                   label=f"{['hit', 'through', 'occluded'][k]} ({int(m.sum())})")
    for cid, c in centroids.items():
        ax.annotate(f"{cid}\n{labels[cid]} {P[cid]:.2f}", c[:2], fontsize=7, ha="center",
                    color=STATUS_COLOUR.get(labels[cid], "k"), weight="bold")
    ax.set_aspect("equal")
    ax.legend(loc="upper right", markerscale=8, fontsize=8)
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def frame_overlay(rgb: np.ndarray, r, c, out, owner_rows, label_pos: dict) -> np.ndarray:
    """New frame with each projected object voxel's outcome: hit green,
    through red, occluded blue; background voxels are not drawn."""
    img = rgb.copy()
    h, w = img.shape[:2]
    for code, colour in ((T.HIT, (0.1, 0.85, 0.1)), (T.THROUGH, (0.95, 0.1, 0.1)), (T.OCCLUDED, (0.3, 0.4, 1.0))):
        m = (out == code) & owner_rows
        layer = np.zeros((h, w), np.uint8)
        layer[r[m], c[m]] = 1
        layer = cv2.dilate(layer, np.ones((3, 3), np.uint8)).astype(bool)
        img[layer] = 0.45 * img[layer] + 0.55 * np.array(colour)
    img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    for text, (x, y, colour) in label_pos.items():
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
        x = int(np.clip(x - tw // 2, 0, max(w - tw, 0)))
        y = int(np.clip(y, th + 2, h - 2))
        cv2.rectangle(img, (x - 1, y - th - 2), (x + tw + 1, y + 2), (0, 0, 0), -1)
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, colour, 1, cv2.LINE_AA)
    return img
