"""Ported from Scene-Graph-Generation/dovsg/navigation/pathplanning.py
(DoVSG). Deltas applied during this port:
  - takes a `voxel_map: PersistedVoxelMap` (navigation/voxel_map.py) instead
    of a live ViewDataset, and `output_dir` instead of `memory_dir`.
  - dropped an unused `import torch`.
  - `pointcloud_visualization` now defaults to False, and `open3d` is
    imported lazily inside that branch only -- a headless entc run
    shouldn't need an interactive 3D viewer, or open3d, to plan a path.
  - narrowed the original's bare `except:` around A* planning to the
    specific exceptions AStar/AStarPlanner actually raise on failure
    (ValueError), so unrelated bugs aren't silently swallowed.
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
import matplotlib.colors as mcolors
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from navigation.core.astar import AStarPlanner
from navigation.core.occupancy import build_occupancy_map

logger = logging.getLogger(__name__)


class PathPlanner:
    def __init__(
        self,
        voxel_map,
        output_dir: Path,
        resolution: float = 0.05,
        occ_avoid_radius: float = 0.4,
        min_height: float = 0.2,
        max_height: float = 1.4,
        conservative: bool = False,
        occ_threshold: int = 100,
        heuristic: str = "euclidean",
        pointcloud_visualization: bool = False,
    ):
        self.output_dir = Path(output_dir)
        self.resolution = resolution
        self.occ_avoid_radius = occ_avoid_radius
        self.min_height = min_height
        self.max_height = max_height
        self.conservative = conservative
        self.occ_threshold = occ_threshold
        self.pointcloud_visualization = pointcloud_visualization

        self.voxel_map = voxel_map

        self.occ_map = build_occupancy_map(
            voxel_map=voxel_map,
            min_height=self.min_height,
            max_height=self.max_height,
            resolution=self.resolution,
            occ_threshold=self.occ_threshold,
            conservative=self.conservative,
            occ_avoid=int(np.ceil(self.occ_avoid_radius / self.resolution)),
        )

        self.astarplanner = AStarPlanner(
            occ_map=self.occ_map,
            resolution=self.resolution,
            occ_avoid_radius=self.occ_avoid_radius,
            conservative=self.conservative,
            heuristic=heuristic,
        )

        original_point = self.occ_map.origin
        ycells, xcells = self.occ_map.grid.shape
        self.min_x, self.min_y = original_point
        self.max_x, self.max_y = self.min_x + xcells * self.resolution, self.min_y + ycells * self.resolution

    def generate_paths(self, start_xyz, end_xyz, visualize: bool = False):
        """Returns a list of (x, y, theta) waypoints from start_xyz to
        end_xyz, or None if no path could be found (e.g. start/goal on an
        occupied cell with no free neighbor, or goal unreachable)."""
        end_xy = end_xyz[:2]

        try:
            paths = self.astarplanner.plan(
                start_xy=start_xyz[:2], end_xy=end_xy, remove_line_of_sight_points=True
            )
        except ValueError as exc:
            logger.warning(f"A* planning failed: {exc}")
            paths = None

        if visualize:
            self.visualization(start_xyz, paths, end_xyz)

        return paths

    def visualization(self, start_xy, paths, end_xyz):
        if self.pointcloud_visualization:
            import open3d  # noqa: F401  (only needed for this optional 3D view)
            from navigation.core.visualizations import visualize

            indexes = list(self.voxel_map.indexes_colors_mapping_dict.keys())
            pcd = self.voxel_map.index_to_pcd(indexes)
            visualize(pcd, paths, end_xyz, self.min_height)

        w, h = self.occ_map.grid[::-1].shape
        size = 16
        fig, ax = plt.subplots(figsize=(size, int(size * h / w)), facecolor="black")

        cmap = mcolors.ListedColormap(["green", "orange"])
        bounds = [0, 1]
        norm = mcolors.BoundaryNorm(bounds, cmap.N)
        cbar_img = ax.imshow(
            self.occ_map.grid[::-1],
            extent=(self.min_x, self.max_x, self.min_y, self.max_y),
            cmap=cmap,
            norm=norm,
        )
        cbar = plt.colorbar(cbar_img, ax=ax, ticks=[0, 1])
        cbar.ax.set_yticklabels(["False", "True"])

        ax.scatter(start_xy[0], start_xy[1], s=100, c="r")
        if paths:
            xs, ys, thetas = zip(*paths)
            ax.scatter(xs, ys, c="cyan", s=20)
        ax.scatter(end_xyz[0], end_xyz[1], s=100, c="b")

        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis("off")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.output_dir / "navigation_vis.jpg"
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()
        return output_path
