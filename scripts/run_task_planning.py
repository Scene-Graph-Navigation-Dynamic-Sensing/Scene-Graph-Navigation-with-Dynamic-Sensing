#!/usr/bin/env python3
"""
run_task_planning.py — Ground a human command against a saved scene graph
===========================================================================
Loads one already-built scene graph (outputs/scene_graphs/<scene>.pkl) and a
natural-language command, asks Gemini which receptacle the command targets
(task_planning/prompts.py), and prints/saves a navigation target resolved
from the centroids scene_graph/storage/serializer.py persists at save time.

"Room" here is just the scene graph's own name -- this pipeline has no
multi-room concept, so picking which scene graph to query is on the caller,
not this script (see task_planning/planner.py's module docstring / the
approved plan for the full rationale).

Usage:
    # --command given up front:
    python scripts/run_task_planning.py \\
        --scene_graph outputs/scene_graphs/synth_test.pkl \\
        --command "bring me the red mug"

    # --command omitted: prompts for it interactively
    python scripts/run_task_planning.py --scene_graph outputs/scene_graphs/synth_test.pkl
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

from object_fusion._dovsg_import import ensure_dovsg_importable
from task_planning.gemini_client import GeminiClient
from task_planning.planner import plan

logger = logging.getLogger("run_task_planning")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scene_graph", required=True, help="Path to a saved outputs/scene_graphs/<scene>.pkl")
    p.add_argument("--command", default=None,
                   help="Natural-language human command, e.g. 'bring me the red mug'. "
                        "If omitted, you'll be prompted for it interactively.")
    p.add_argument("--config", default="configs/pipeline.yaml")
    p.add_argument("--output_dir", default=None, help="Default: outputs/task_plans/")
    return p


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    args = build_arg_parser().parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    cfg = yaml.safe_load(config_path.read_text())

    # the pickled scene_graph contains dovsg.memory.scene_graph.graph.ObjectNode
    # instances -- dovsg must be importable before unpickling it.
    ensure_dovsg_importable(cfg["external"]["scene_graph_generation_root"])

    command = args.command
    if not command:
        command = input("What should the robot bring / where should it go? ").strip()
        if not command:
            logger.error("No command given. Exiting.")
            return

    tp_cfg = cfg.get("task_planning", {})
    gemini_client = GeminiClient(api_key=tp_cfg.get("gemini_api_key", ""), model=tp_cfg.get("model", "gemini-2.5-flash"))
    if gemini_client.is_stub:
        logger.warning("Gemini client in stub mode (no/invalid API key) -- returning a placeholder plan.")

    task_plan = plan(Path(args.scene_graph), command, gemini_client, semantic_decoding_cfg=cfg.get("semantic_decoding"))

    print(f"\nCommand   : {task_plan.command}")
    print(f"Room      : {task_plan.room}")
    print(f"Thought   : {task_plan.thought}")
    print(f"Target    : {task_plan.receptacle_id} ({task_plan.receptacle_class})")
    print(f"Navigate to: {task_plan.navigation_target}\n")

    out_dir = Path(args.output_dir) if args.output_dir else REPO_ROOT / "outputs" / "task_plans"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{task_plan.room}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out_path.write_text(json.dumps(task_plan.to_dict(), indent=2))
    logger.info(f"Saved task plan: {out_path}")


if __name__ == "__main__":
    main()
