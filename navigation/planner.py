"""Orchestrates a single navigation-planning query: load a saved scene
graph's persisted voxel map, build a 2D occupancy grid, and run A* from a
start point to a goal point (typically task_planning's navigation_target).

Mirrors task_planning/planner.py's TaskPlan/plan() shape: a NavigationPlan
dataclass with .to_dict(), and a plan_navigation() function that does the
loading + planning + failure handling.
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from navigation.core.path_facade import PathPlanner
from navigation.voxel_map import PersistedVoxelMap
from scene_graph.storage.serializer import load_scene_graph

logger = logging.getLogger(__name__)


@dataclass
class NavigationPlan:
    room: str
    receptacle_id: str | None
    start_xyz: list[float]
    goal_xyz: list[float]
    waypoints: list[list[float]] | None
    success: bool
    message: str
    resolution: float
    occ_avoid_radius: float
    min_height: float
    max_height: float
    visualization_path: str | None

    def to_dict(self) -> dict:
        return asdict(self)


def plan_navigation(
    scene_graph_pkl_path: str | Path,
    goal_xyz: Sequence[float],
    start_xyz: Sequence[float],
    navigation_cfg: dict,
    room: str | None = None,
    receptacle_id: str | None = None,
    visualize: bool = True,
    output_dir: str | Path | None = None,
) -> NavigationPlan:
    """navigation_cfg: the pipeline config's `navigation:` section (see
    configs/pipeline.yaml). Raises VoxelMapNotPersistedError (from
    navigation/voxel_map.py) if the given .pkl has no persisted voxel map --
    i.e. it predates navigation support, or was saved without view_dataset.
    """
    scene_graph_pkl_path = Path(scene_graph_pkl_path)
    room = room or scene_graph_pkl_path.stem

    data = load_scene_graph(scene_graph_pkl_path)
    voxel_map = PersistedVoxelMap.from_dict(data.get("voxel_map"))

    out_dir = Path(output_dir) if output_dir else Path(navigation_cfg.get("output_dir", "outputs/navigation_plans"))

    planner = PathPlanner(
        voxel_map=voxel_map,
        output_dir=out_dir,
        resolution=navigation_cfg.get("resolution", 0.05),
        occ_avoid_radius=navigation_cfg.get("occ_avoid_radius", 0.4),
        min_height=navigation_cfg.get("min_height", 0.2),
        max_height=navigation_cfg.get("max_height", 1.4),
        conservative=navigation_cfg.get("conservative", False),
        occ_threshold=navigation_cfg.get("occ_threshold", 100),
        heuristic=navigation_cfg.get("heuristic", "euclidean"),
        pointcloud_visualization=navigation_cfg.get("pointcloud_visualization", False),
    )

    waypoints = planner.generate_paths(start_xyz=list(start_xyz), end_xyz=list(goal_xyz), visualize=visualize)

    visualization_path = None
    if visualize:
        visualization_path = str(out_dir / "navigation_vis.jpg")

    if waypoints is None:
        return NavigationPlan(
            room=room,
            receptacle_id=receptacle_id,
            start_xyz=list(start_xyz),
            goal_xyz=list(goal_xyz),
            waypoints=None,
            success=False,
            message="A* planning failed -- start or goal likely sits on an occupied cell "
                    "with no reachable free neighbor, or the goal is unreachable from start. "
                    "See logged warning for details.",
            resolution=planner.resolution,
            occ_avoid_radius=planner.occ_avoid_radius,
            min_height=planner.min_height,
            max_height=planner.max_height,
            visualization_path=visualization_path,
        )

    return NavigationPlan(
        room=room,
        receptacle_id=receptacle_id,
        start_xyz=list(start_xyz),
        goal_xyz=list(goal_xyz),
        waypoints=[list(wp) for wp in waypoints],
        success=True,
        message="",
        resolution=planner.resolution,
        occ_avoid_radius=planner.occ_avoid_radius,
        min_height=planner.min_height,
        max_height=planner.max_height,
        visualization_path=visualization_path,
    )
