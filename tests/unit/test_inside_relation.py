"""Flat things lying on a desk are "on" it; things deep inside a box are "inside" it."""
from pathlib import Path

import numpy as np
import yaml

from scene_graph.builder.dovsg_relations import build_scene_graph_dovsg

REPO_ROOT = Path(__file__).resolve().parents[2]
SGG_ROOT = yaml.safe_load((REPO_ROOT / "configs/pipeline.yaml").read_text())["external"][
    "scene_graph_generation_root"]


class _FakeViewDataset:
    def __init__(self, points):
        self.points = np.asarray(points, dtype=np.float32)

    def index_to_point(self, indexes):
        return self.points[np.asarray(indexes)]


def _box(lo, hi, n, rng):
    return rng.uniform(lo, hi, size=(n, 3))


def _relations(parts):
    points, objects, start = [], [], 0
    for name, pts in parts:
        points.append(pts)
        objects.append({"class_id": f"{name}_0", "class_name": name,
                        "indexes": np.arange(start, start + len(pts))})
        start += len(pts)
    graph = build_scene_graph_dovsg(_FakeViewDataset(np.vstack(points)), objects, SGG_ROOT)
    return {cid: (node.parent.node_id, node.parent_relation)
            for cid, node in graph.object_nodes.items() if cid != "floor_0"}


def test_wallet_lying_on_a_desk_is_on_it_not_inside():
    rng = np.random.default_rng(0)
    desk = np.vstack([_box([0, 0, 0.84], [1.6, 1.0, 0.90], 4000, rng),     # the top
                      _box([0, 0, 0.0], [0.05, 0.05, 0.84], 150, rng),     # a leg
                      _box([1.5, 0, 0.90], [1.6, 1.0, 1.30], 150, rng)])   # cables rising at the back
    wallet = _box([0.6, 0.4, 0.86], [0.8, 0.6, 0.92], 400, rng)
    rel = _relations([("desk", desk), ("wallet", wallet)])
    assert rel["wallet_0"] == ("desk_0", "on")


def test_cup_standing_in_a_deep_box_is_inside_it():
    # the box sits on a shelf 0.8 m up: anything within 15 cm of the floor is
    # put on the floor before any support test (DoVSG's stand_floor_threshold)
    rng = np.random.default_rng(1)
    walls = [_box([0, 0, 0], [0.5, 0.02, 0.4], 500, rng), _box([0, 0.48, 0], [0.5, 0.5, 0.4], 500, rng),
             _box([0, 0, 0], [0.02, 0.5, 0.4], 500, rng), _box([0.48, 0, 0], [0.5, 0.5, 0.4], 500, rng),
             _box([0, 0, 0], [0.5, 0.5, 0.02], 1000, rng)]
    cup = _box([0.2, 0.2, 0.02], [0.3, 0.3, 0.15], 400, rng)
    shelf = np.array([0, 0, 0.8])
    rel = _relations([("box", np.vstack(walls) + shelf), ("cup", cup + shelf)])
    assert rel["cup_0"] == ("box_0", "inside")
