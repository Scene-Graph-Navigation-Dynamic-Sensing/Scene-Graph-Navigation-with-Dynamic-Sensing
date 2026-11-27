"""Scene graph PDF render — thin wrapper around SceneGraph.visualize()
(dovsg/memory/scene_graph/graph.py), which requires the graphviz package
and the `dot` binary. Writes outputs/visualizations/scene_graph/scene_graph.pdf.
"""
from __future__ import annotations

from pathlib import Path


def save_scene_graph_visualization(scene_graph, out_dir: Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scene_graph.visualize(save_dir=str(out_dir))
