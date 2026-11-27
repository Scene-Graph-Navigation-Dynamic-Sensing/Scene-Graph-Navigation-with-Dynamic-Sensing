"""Single place that makes Scene-Graph-Generation's dovsg package importable.

Reused directly (not copied) per the design plan: dovsg/memory/view_dataset.py
and dovsg/memory/instances/{instance_process,instance_utils}.py are proven,
working 3D-fusion code with no coupling to any particular detector — only
the detection-ingestion entry point (InstanceProcess.gsam2_obs_to_detection_list)
is GroundingDINO+SAM2-specific, and that's what instance_fusion.py replaces.
"""
from __future__ import annotations

import sys


def ensure_dovsg_importable(scene_graph_generation_root: str) -> None:
    if scene_graph_generation_root not in sys.path:
        sys.path.insert(0, scene_graph_generation_root)
