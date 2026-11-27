"""Renders the before / after voxel maps and scene graphs of a scene update.

    python scripts/render_update_check.py --before fyp_initial --after fyp_same_video_update_final \
        --out results/fyp_update_check

Writes, for each of before / after:
  voxel_map_<tag>_rgb.png      top-down and oblique views of the voxel map in its own colours
  voxel_map_<tag>_objects.png  the same, each object node in its own colour and labelled
  scene_graph_<tag>.png        the scene graph (nodes and on / inside / belong edges)
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def load(name: str):
    from navigation.voxel_map import PersistedVoxelMap
    from scene_graph.storage.serializer import load_scene_graph
    p = load_scene_graph(REPO_ROOT / "outputs" / "scene_graphs" / f"{name}.pkl")
    vm = PersistedVoxelMap.from_dict(p["voxel_map"])
    d = vm.indexes_colors_mapping_dict
    idx = np.fromiter(d.keys(), dtype=np.int64, count=len(d))
    rgb = np.array(list(d.values()), dtype=np.float64)
    rgb = rgb / 255 if rgb.max() > 1 else rgb
    pts = vm.index_to_point(idx)
    objs = [(o["class_id"], vm.index_to_point(np.asarray(o["indexes"])))
            for o in p["instance_objects"] if len(o.get("indexes", [])) > 0]
    return pts, rgb, objs


def views(ax_top, ax_side, pts, colours, s=0.3):
    sub = np.random.default_rng(0).permutation(len(pts))[:400000]       # plot at most 400k voxels
    p, c = pts[sub], colours[sub]
    o = np.argsort(p[:, 2])                                              # top-down: highest drawn last
    ax_top.scatter(p[o, 0], p[o, 1], c=c[o], s=s, linewidths=0)
    ax_top.set_aspect("equal"); ax_top.set_title("top-down"); ax_top.set_xlabel("x (m)"); ax_top.set_ylabel("y (m)")
    a = np.deg2rad(35)                                                   # oblique: rotate about z, look from the side
    x = p[:, 0] * np.cos(a) - p[:, 1] * np.sin(a)
    depth = p[:, 0] * np.sin(a) + p[:, 1] * np.cos(a)
    o = np.argsort(-depth)
    ax_side.scatter(x[o], p[o, 2] + 0.35 * depth[o], c=c[o], s=s, linewidths=0)
    ax_side.set_aspect("equal"); ax_side.set_title("oblique"); ax_side.set_xticks([]); ax_side.set_yticks([])


def render_maps(tag: str, title: str, pts, rgb, objs, out: Path):
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    views(ax[0], ax[1], pts, rgb)
    fig.suptitle(f"{title}: voxel map ({len(pts):,} voxels, 1 cm)")
    fig.tight_layout(); fig.savefig(out / f"voxel_map_{tag}_rgb.png", dpi=110); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    grey = np.full((len(pts), 3), 0.85)
    views(ax[0], ax[1], pts, grey, s=0.2)
    cmap = plt.get_cmap("tab20")
    for i, (name, p) in enumerate(objs):
        c = np.tile(cmap(i % 20)[:3], (len(p), 1))
        ax[0].scatter(p[:, 0], p[:, 1], c=c, s=0.6, linewidths=0)
        ax[0].annotate(name, p.mean(0)[:2], fontsize=8, weight="bold", ha="center",
                       bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.7, lw=0))
    ax[1].set_title("oblique (map only)")
    fig.suptitle(f"{title}: object nodes ({len(objs)})")
    fig.tight_layout(); fig.savefig(out / f"voxel_map_{tag}_objects.png", dpi=110); plt.close(fig)


def graph_dot_from_json(name: str) -> str:
    """The saved JSON is a tree (each node with its children and relation to its parent)."""
    root = json.loads((REPO_ROOT / "outputs" / "scene_graphs" / f"{name}.json").read_text())
    lines = ["digraph {", "  rankdir=TB; node [shape=egg style=filled color=lightblue2];"]

    def walk(n, parent=None):
        lines.append(f'  "{n["node_id"]}";')
        if parent:
            lines.append(f'  "{parent}" -> "{n["node_id"]}" [label="{n.get("relation") or ""}"];')
        for c in n.get("children") or []:
            walk(c, n["node_id"])
    walk(root)
    return "\n".join(lines + ["}"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--before", required=True)
    ap.add_argument("--after", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default="configs/pipeline_fyp_videos_final.yaml")
    args = ap.parse_args()
    out = REPO_ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(open(REPO_ROOT / args.config))
    from object_fusion._dovsg_import import ensure_dovsg_importable
    ensure_dovsg_importable(cfg["external"]["scene_graph_generation_root"])

    for tag, name in (("before", args.before), ("after", args.after)):
        render_maps(tag, f"{tag} ({name})", *load(name), out)
        dot = graph_dot_from_json(name)
        (out / f"scene_graph_{tag}.dot").write_text(dot)
        subprocess.run(["dot", "-Tpng", "-Gdpi=110", "-o", str(out / f"scene_graph_{tag}.png")],
                       input=dot.encode(), check=True)
        print(f"{tag}: wrote voxel maps and scene graph for {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
