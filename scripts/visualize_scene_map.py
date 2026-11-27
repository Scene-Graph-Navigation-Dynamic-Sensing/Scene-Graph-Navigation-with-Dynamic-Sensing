#!/usr/bin/env python3
"""
visualize_scene_map.py — Plot a saved scene graph's centroids against the
real reconstructed point cloud
=============================================================================
Sanity check for scene_graph/storage/serializer.py's persisted centroids:
reloads the original per-instance points from the raw capture (indexes ->
view_dataset.index_to_point(), same as fusion/scene-graph building used),
plots them colored per instance, and overlays each node's stored centroid as
a labeled star -- so you can see by eye whether "forklift_0"'s centroid
actually sits on the forklift-shaped point cluster, not just trust the
number in the JSON.

Points are decoded with the voxel-map snapshot saved in the .pkl. Older
pickles without one need the run's raw capture directory (data/processed/
<scene>/) still on disk, or --dataset_dir to force that path.

Usage:
    python scripts/visualize_scene_map.py --scene_graph outputs/scene_graphs/synth_stride2_test.pkl

    # if the capture dir isn't data/processed/<pkl stem> for some reason:
    python scripts/visualize_scene_map.py --scene_graph outputs/scene_graphs/synth_stride2_test.pkl \\
        --dataset_dir data/processed/synth_stride2_test
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from object_fusion._dovsg_import import ensure_dovsg_importable
from object_fusion.view_dataset import load_view_dataset
from scene_graph.storage.serializer import load_scene_graph

logger = logging.getLogger("visualize_scene_map")

_MAX_POINTS_PER_OBJECT = 3000  # subsample dense objects so plotting stays fast


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scene_graph", required=True, help="Path to a saved outputs/scene_graphs/<scene>.pkl")
    p.add_argument("--dataset_dir", default=None, help="Default: data/processed/<scene graph's stem>")
    p.add_argument("--config", default="configs/pipeline.yaml", help="For external.scene_graph_generation_root")
    p.add_argument("--output", default=None, help="Default: outputs/visualizations/scene_map/<scene>.png")
    return p


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    args = build_arg_parser().parse_args()

    scene_graph_path = Path(args.scene_graph)
    scene_name = scene_graph_path.stem
    dataset_dir = Path(args.dataset_dir) if args.dataset_dir else REPO_ROOT / "data" / "processed" / scene_name

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    cfg = yaml.safe_load(config_path.read_text())
    scene_graph_generation_root = cfg["external"]["scene_graph_generation_root"]

    ensure_dovsg_importable(scene_graph_generation_root)
    data = load_scene_graph(scene_graph_path)
    scene_graph = data["scene_graph"]
    instance_objects = data["instance_objects"]
    node_centroids = data.get("node_centroids", {})

    if not node_centroids:
        logger.warning(
            "This scene graph has no stored centroids (saved before centroid persistence was added, "
            "or every node lacked a matching instance) -- only the point cloud will be plotted."
        )

    # Voxel indexes only decode to the correct real-world point at the same
    # resolution they were encoded with, or every index_to_point() call below
    # silently returns points at the wrong scale/offset (centroids, computed
    # correctly at save time, would then appear disconnected from the cloud).
    # Pickles with a voxel-map snapshot record the run's own resolution; older
    # ones fall back to --config, which must then be the run's config.
    #
    # A pickle with a voxel-map snapshot decodes its own indexes, with no raw
    # capture needed. That is also the only correct way for a graph from
    # scripts/update_scene_graph.py: its indexes are in the memory run's grid,
    # not in the grid of the new video's capture.
    voxel_map = data.get("voxel_map")
    if voxel_map is not None and not args.dataset_dir:
        from navigation.voxel_map import PersistedVoxelMap

        logger.info(f"Decoding points with the scene graph's own voxel map (resolution={voxel_map['resolution']}).")
        view_dataset = PersistedVoxelMap.from_dict(voxel_map)
    else:
        if not dataset_dir.is_dir():
            raise FileNotFoundError(
                f"{dataset_dir} not found -- a scene graph without a voxel-map snapshot needs the run's raw "
                "capture directory (pass --dataset_dir if it's not at data/processed/<scene name>)."
            )
        if voxel_map is not None:
            voxel_resolution = voxel_map["resolution"]
        else:
            voxel_resolution = cfg["object_fusion"]["voxel_resolution"]
            logger.warning(f"No voxel-map snapshot in this scene graph; using {config_path.name}'s "
                           f"voxel_resolution={voxel_resolution}. Pass the run's own --config if that is wrong.")
        logger.info(f"Reloading point cloud from {dataset_dir} (resolution={voxel_resolution}) ...")
        view_dataset = load_view_dataset(
            recorder_dir=str(dataset_dir),
            scene_graph_generation_root=scene_graph_generation_root,
            resolution=voxel_resolution,
        )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    cmap = matplotlib.colormaps.get_cmap("turbo")
    n = max(len(instance_objects), 1)

    fig = plt.figure(figsize=(16, 8))
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    ax2d = fig.add_subplot(1, 2, 2)

    rng = np.random.default_rng(0)
    for i, obj in enumerate(instance_objects):
        indexes = obj.get("indexes")
        if indexes is None or len(indexes) == 0:
            continue
        pts = np.asarray(view_dataset.index_to_point(indexes))
        if len(pts) > _MAX_POINTS_PER_OBJECT:
            pts = pts[rng.choice(len(pts), _MAX_POINTS_PER_OBJECT, replace=False)]
        color = cmap(i / max(n - 1, 1))
        ax3d.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=1, color=color, alpha=0.35)
        ax2d.scatter(pts[:, 0], pts[:, 1], s=1, color=color, alpha=0.35)

    labeled = 0
    for node_id, node in scene_graph.object_nodes.items():
        centroid = node_centroids.get(node_id)
        if centroid is None:
            continue
        x, y, z = centroid
        ax3d.scatter([x], [y], [z], marker="*", s=250, color="black", edgecolor="white", linewidth=0.5, zorder=5)
        ax3d.text(x, y, z, f"  {node.node_class}", fontsize=8, weight="bold")
        ax2d.scatter([x], [y], marker="*", s=250, color="black", edgecolor="white", linewidth=0.5, zorder=5)
        ax2d.annotate(node.node_class, (x, y), fontsize=8, weight="bold", xytext=(4, 4), textcoords="offset points")
        labeled += 1

    ax3d.set_xlabel("X"); ax3d.set_ylabel("Y"); ax3d.set_zlabel("Z")
    ax3d.set_title(f"{scene_name} — 3D")
    ax2d.set_xlabel("X"); ax2d.set_ylabel("Y")
    ax2d.set_title(f"{scene_name} — top-down (X-Y)")
    ax2d.set_aspect("equal", adjustable="datalim")
    fig.suptitle(f"Scene map: {len(instance_objects)} instances, {labeled} labeled centroids")
    fig.tight_layout()

    out_path = Path(args.output) if args.output else REPO_ROOT / "outputs" / "visualizations" / "scene_map" / f"{scene_name}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    logger.info(f"Saved scene map: {out_path}")


if __name__ == "__main__":
    main()
