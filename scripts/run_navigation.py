#!/usr/bin/env python3
"""
run_navigation.py — Plan a path across a saved scene graph's voxel map
===========================================================================
Loads one already-built scene graph (outputs/scene_graphs/<scene>.pkl,
saved with its voxel map persisted) and a task plan
(outputs/task_plans/<room>_<timestamp>.json, from run_task_planning.py),
builds a 2D occupancy grid from the voxel map, runs A* from a start point to
the task plan's navigation_target, and prints/saves the resulting waypoints
plus a visualization image.

This stops at producing a path plan -- no robot/ROS/motor-control execution,
same way run_task_planning.py stops at producing a navigation_target.

Usage:
    python scripts/run_navigation.py \\
        --scene_graph outputs/scene_graphs/synth_test.pkl \\
        --task_plan outputs/task_plans/synth_test_20260924_120000.json \\
        --start_xyz 0 0 0

    # fall back to the last captured frame's pose as the start point:
    python scripts/run_navigation.py \\
        --scene_graph outputs/scene_graphs/synth_test.pkl \\
        --task_plan outputs/task_plans/synth_test_20260924_120000.json \\
        --recorder_dir data/raw/synth_test
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from navigation.planner import plan_navigation
from navigation.start_position import resolve_start_xyz
from navigation.voxel_map import VoxelMapNotPersistedError

logger = logging.getLogger("run_navigation")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scene_graph", required=True, help="Path to a saved outputs/scene_graphs/<scene>.pkl")
    p.add_argument("--task_plan", default=None, help="Path to a saved outputs/task_plans/<room>_<ts>.json")
    p.add_argument("--goal_xyz", type=float, nargs=3, default=None,
                   help="Explicit goal point, overrides --task_plan's navigation_target if both given")
    p.add_argument("--start_xyz", type=float, nargs=3, default=None, help="Explicit robot start point")
    p.add_argument("--recorder_dir", default=None,
                   help="Fallback start_xyz source: uses the last captured frame's pose "
                        "if --start_xyz isn't given (see navigation/start_position.py)")
    p.add_argument("--config", default="configs/pipeline.yaml")
    p.add_argument("--output_dir", default=None, help="Default: outputs/navigation_plans/")
    p.add_argument("--no_visualize", action="store_true", help="Skip writing navigation_vis.jpg")
    return p


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    args = build_arg_parser().parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    cfg = yaml.safe_load(config_path.read_text())
    nav_cfg = cfg.get("navigation", {})

    room = None
    receptacle_id = None
    goal_xyz = args.goal_xyz
    if args.task_plan:
        task_plan = json.loads(Path(args.task_plan).read_text())
        room = task_plan.get("room")
        receptacle_id = task_plan.get("receptacle_id")
        if goal_xyz is None:
            goal_xyz = task_plan.get("navigation_target")

    if goal_xyz is None:
        logger.error("No goal point: pass --goal_xyz, or --task_plan pointing at a plan with a "
                     "non-null navigation_target.")
        return 1

    start_xyz, start_source = resolve_start_xyz(
        explicit_start_xyz=args.start_xyz,
        recorder_dir=args.recorder_dir,
        fallback=nav_cfg.get("start_xyz_fallback", "last_pose"),
    )

    out_dir = Path(args.output_dir) if args.output_dir else REPO_ROOT / "outputs" / "navigation_plans"

    try:
        nav_plan = plan_navigation(
            scene_graph_pkl_path=args.scene_graph,
            goal_xyz=goal_xyz,
            start_xyz=start_xyz,
            navigation_cfg=nav_cfg,
            room=room,
            receptacle_id=receptacle_id,
            visualize=not args.no_visualize,
            output_dir=out_dir,
        )
    except VoxelMapNotPersistedError as exc:
        logger.error(str(exc))
        return 1

    print(f"\nRoom         : {nav_plan.room}")
    print(f"Start ({start_source}): {nav_plan.start_xyz}")
    print(f"Goal         : {nav_plan.goal_xyz}")
    print(f"Success      : {nav_plan.success}")
    if not nav_plan.success:
        print(f"Message      : {nav_plan.message}")
    else:
        print(f"Waypoints    : {len(nav_plan.waypoints)}")
    if nav_plan.visualization_path:
        print(f"Visualization: {nav_plan.visualization_path}\n")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{nav_plan.room}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out_path.write_text(json.dumps(nav_plan.to_dict(), indent=2))
    logger.info(f"Saved navigation plan: {out_path}")

    return 0 if nav_plan.success else 2


if __name__ == "__main__":
    sys.exit(main())
