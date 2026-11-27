"""An object that is gone from the new video leaves the scene graph; whatever
it carried is placed again, and a still-visible object is never removed."""
from pathlib import Path

import numpy as np
import yaml

from scene_graph.builder.dovsg_relations import build_scene_graph_dovsg, remove_nodes
from scripts.update_scene_graph import (DEFAULT_UPDATE_CFG, find_need_to_delete_indexes, keep_redetected,
                                        removed_objects)

REPO_ROOT = Path(__file__).resolve().parents[2]
SGG_ROOT = yaml.safe_load((REPO_ROOT / "configs/pipeline.yaml").read_text())["external"][
    "scene_graph_generation_root"]
UCFG = dict(DEFAULT_UPDATE_CFG)


class _FakeView:
    def __init__(self, points):
        self.points = np.asarray(points, dtype=np.float32)

    def index_to_point(self, indexes):
        return self.points[np.asarray(indexes)]


def _box(lo, hi, rng, n):
    return rng.uniform(lo, hi, size=(n, 3))


def _graph():
    rng = np.random.default_rng(0)
    parts = [("table", _box([0, 0, 0], [1.2, 0.6, 0.75], rng, 3000)),
             ("cup", _box([0.5, 0.2, 0.75], [0.6, 0.3, 0.85], rng, 300)),
             ("mouse", _box([0.8, 0.2, 0.75], [0.88, 0.3, 0.79], rng, 300))]
    points, objects, start = [], [], 0
    for name, pts in parts:
        points.append(pts)
        objects.append({"class_id": f"{name}_0", "class_name": name, "indexes": np.arange(start, start + len(pts))})
        start += len(pts)
    return build_scene_graph_dovsg(_FakeView(np.vstack(points)), objects, SGG_ROOT)


def test_removing_a_leaf_object_removes_only_that_node():
    graph = _graph()
    assert graph.object_nodes["cup_0"].parent.node_id == "table_0"
    assert remove_nodes(graph, ["cup_0"]) == []
    assert "cup_0" not in graph.object_nodes
    assert {"table_0", "mouse_0"} <= set(graph.object_nodes)
    assert "cup_0" not in graph.object_nodes["table_0"].children


def test_removing_a_support_returns_what_it_carried_as_orphans():
    graph = _graph()
    assert sorted(remove_nodes(graph, ["table_0"])) == ["cup_0", "mouse_0"]
    assert not {"table_0", "cup_0", "mouse_0"} & set(graph.object_nodes)


def test_removing_support_and_child_together_does_not_orphan_the_child():
    graph = _graph()
    assert remove_nodes(graph, ["table_0", "cup_0"]) == ["mouse_0"]


def test_removing_an_unknown_id_changes_nothing():
    graph = _graph()
    before = set(graph.object_nodes)
    assert remove_nodes(graph, ["ghost_9"]) == []
    assert set(graph.object_nodes) == before


def test_object_is_removed_only_above_the_delete_rate():
    objs = [{"class_id": "a_0", "indexes": np.arange(0, 10)},
            {"class_id": "b_0", "indexes": np.arange(10, 20)},
            {"class_id": "c_0", "indexes": np.arange(20, 30)}]
    deleted = np.array([*range(0, 8), *range(10, 15), 25])  # a: 80%, b: 50% (not "more than"), c: 10%
    removed, share = removed_objects(objs, deleted, UCFG)
    assert removed == ["a_0"]
    assert share == {"a_0": 0.8, "b_0": 0.5, "c_0": 0.1}


class _FakeMemoryMap:
    def __init__(self, points):
        self.points = points
        self.indexes_colors_mapping_dict = {i: 0 for i in range(len(points))}

    def index_to_point(self, idx):
        return self.points[np.asarray(idx)]


class _FakeObs:
    """A camera at the origin looking down +z at a 40x40 image."""
    K = np.array([[40.0, 0, 20], [0, 40.0, 20], [0, 0, 1]])

    def __init__(self, depth_value, n_frames=6):
        self.poses = [np.eye(4)] * n_frames
        self.depths = [np.full((40, 40), depth_value)] * n_frames
        self.masks = [np.ones((40, 40), bool)] * n_frames

    def wait_all(self):
        return self


def _block():
    rng = np.random.default_rng(1)
    return _box([-0.1, -0.1, 1.0], [0.1, 0.1, 1.02], rng, 200)  # a thin object 1 m in front of the camera


def test_voxels_are_deleted_when_the_new_video_sees_a_wall_behind_them():
    mem = _FakeMemoryMap(_block())
    gone = find_need_to_delete_indexes(mem, _FakeObs(depth_value=2.0), {**UCFG, "delete_frame_stride": 1})
    assert len(gone) == len(mem.points)


def test_voxels_stay_when_the_new_video_still_sees_their_surface():
    mem = _FakeMemoryMap(_block())
    gone = find_need_to_delete_indexes(mem, _FakeObs(depth_value=1.0), {**UCFG, "delete_frame_stride": 1})
    assert len(gone) == 0


def test_voxels_outside_the_view_are_not_deleted():
    mem = _FakeMemoryMap(_block() + np.array([0.0, 0.0, -3.0]))  # behind the camera
    gone = find_need_to_delete_indexes(mem, _FakeObs(depth_value=2.0), {**UCFG, "delete_frame_stride": 1})
    assert len(gone) == 0


def test_removal_is_cancelled_when_the_detector_sees_the_object_again():
    memory = [{"class_id": "cup_0", "num_detections": 10}, {"class_id": "mouse_0", "num_detections": 8}]
    fused = [{"memory_ids": ["cup_0"], "num_detections": 15}]   # 5 new detections of the cup
    assert keep_redetected(["cup_0", "mouse_0"], fused, memory, UCFG) == ["mouse_0"]


def test_one_stray_match_does_not_cancel_a_removal():
    memory = [{"class_id": "cup_0", "num_detections": 10}]
    fused = [{"memory_ids": ["cup_0"], "num_detections": 11}]
    assert keep_redetected(["cup_0"], fused, memory, UCFG) == ["cup_0"]
