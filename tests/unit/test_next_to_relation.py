"""Objects side by side on the same support are "next to" each other; stacked,
far apart, at different heights, or on different supports are not."""
from pathlib import Path

import numpy as np
import yaml

from scene_graph.builder.dovsg_relations import build_scene_graph_dovsg
from scene_graph.storage.serializer import scene_graph_to_next_to, scene_graph_to_tree

REPO_ROOT = Path(__file__).resolve().parents[2]
SGG_ROOT = yaml.safe_load((REPO_ROOT / "configs/pipeline.yaml").read_text())["external"][
    "scene_graph_generation_root"]


class _FakeViewDataset:
    def __init__(self, points):
        self.points = np.asarray(points, dtype=np.float32)

    def index_to_point(self, indexes):
        return self.points[np.asarray(indexes)]


def _box(lo, hi, rng, n=400):
    return rng.uniform(lo, hi, size=(n, 3))


def _graph(parts):
    points, objects, start = [], [], 0
    for name, pts in parts:
        points.append(pts)
        objects.append({"class_id": f"{name}_0", "class_name": name,
                        "indexes": np.arange(start, start + len(pts))})
        start += len(pts)
    return build_scene_graph_dovsg(_FakeViewDataset(np.vstack(points)), objects, SGG_ROOT)


def _pairs(graph):
    return {(e["a"], e["b"]) for e in scene_graph_to_next_to(graph)}


def test_chair_beside_a_desk_is_next_to_it():
    rng = np.random.default_rng(0)
    desk = _box([0, 0, 0], [1.2, 0.6, 0.75], rng, 2000)
    chair = _box([1.3, 0.1, 0], [1.7, 0.5, 0.9], rng, 1000)        # 10 cm gap
    assert _pairs(_graph([("desk", desk), ("chair", chair)])) == {("chair_0", "desk_0")}


def test_far_apart_objects_are_not_next_to_each_other():
    rng = np.random.default_rng(1)
    desk = _box([0, 0, 0], [1.2, 0.6, 0.75], rng, 2000)
    chair = _box([3.0, 0.1, 0], [3.4, 0.5, 0.9], rng, 1000)
    assert _pairs(_graph([("desk", desk), ("chair", chair)])) == set()


def test_thing_on_a_desk_is_not_next_to_the_desk_or_a_chair_beside_it():
    rng = np.random.default_rng(2)
    desk = _box([0, 0, 0], [1.2, 0.6, 0.75], rng, 2000)
    cup = _box([0.5, 0.2, 0.75], [0.6, 0.3, 0.85], rng)             # on the desk (parent-child)
    chair = _box([1.3, 0.1, 0], [1.7, 0.5, 0.9], rng, 1000)
    pairs = _pairs(_graph([("desk", desk), ("cup", cup), ("chair", chair)]))
    assert pairs == {("chair_0", "desk_0")}                          # cup has a different parent


def test_two_things_on_the_same_desk_are_next_to_each_other_and_show_in_json():
    rng = np.random.default_rng(3)
    desk = _box([0, 0, 0], [1.2, 0.6, 0.75], rng, 3000)
    cup = _box([0.5, 0.2, 0.75], [0.6, 0.3, 0.85], rng)
    mouse = _box([0.7, 0.2, 0.75], [0.78, 0.3, 0.79], rng)          # 10 cm from the cup
    graph = _graph([("desk", desk), ("cup", cup), ("mouse", mouse)])
    assert ("cup_0", "mouse_0") in _pairs(graph)
    tree = scene_graph_to_tree(graph, {})
    cup_json = next(c for c in tree["children"][0]["children"] if c["node_id"] == "cup_0")
    assert [n["node_id"] for n in cup_json["next_to"]] == ["mouse_0"]


def test_objects_at_different_heights_are_not_next_to_each_other():
    rng = np.random.default_rng(4)
    table = _box([0, 0, 0], [1.0, 0.6, 0.5], rng, 2000)
    shelf_item = _box([1.05, 0.2, 0.0], [1.2, 0.4, 0.1], rng)       # on the floor, beside table: next to
    high_item = _box([1.05, 0.2, 1.5], [1.2, 0.4, 1.6], rng)        # up a wall: no vertical overlap
    pairs = _pairs(_graph([("table", table), ("item", shelf_item), ("hi", high_item)]))
    assert ("item_0", "table_0") in pairs
    assert not any("hi_0" in p for p in pairs)
