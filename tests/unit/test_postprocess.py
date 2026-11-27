import numpy as np

from object_fusion.postprocess import (drop_structural_instances, find_support_surfaces,
                                       merge_fragmented_instances)


class _FakeViewDataset:
    """indexes are row ids into a points array."""

    def __init__(self, points):
        self.points = np.asarray(points, dtype=np.float32)

    def index_to_point(self, indexes):
        return self.points[np.asarray(indexes)]


def _box(lo, hi, n=400, rng=None):
    rng = rng or np.random.default_rng(0)
    return rng.uniform(lo, hi, size=(n, 3))


def _obj(name, start, count, embedding):
    return {"class_id": name, "indexes": np.arange(start, start + count),
            "clip_ft": np.asarray(embedding, dtype=np.float32), "num_detections": 1,
            "track_ids": [hash(name) % 100]}


def test_drops_large_flat_floor_height_instances():
    rng = np.random.default_rng(0)
    floor = _box([-1, -1, 0.0], [1, 1, 0.05], 400, rng)     # 2x2 m, 5 cm thick, at z=0
    table = _box([0, 0, 0.70], [0.8, 0.8, 0.78], 400, rng)  # same footprint idea, but up high
    vd = _FakeViewDataset(np.vstack([floor, table]))
    objs = [_obj("floor_like", 0, 400, [1, 0]), _obj("table", 400, 400, [0, 1])]

    kept, dropped = drop_structural_instances(objs, vd)
    assert [o["class_id"] for o in dropped] == ["floor_like"]
    assert [o["class_id"] for o in kept] == ["table"]


def test_small_object_on_the_floor_is_not_dropped():
    rng = np.random.default_rng(1)
    mat = _box([0, 0, 0.0], [0.3, 0.3, 0.04], 400, rng)     # flat and low, but tiny
    vd = _FakeViewDataset(mat)
    kept, dropped = drop_structural_instances([_obj("mat", 0, 400, [1, 0])], vd)
    assert dropped == [] and len(kept) == 1


def _named(obj, label, score, top_k=()):
    obj["track_names"] = [{"label": label, "score": score, "top_k": list(top_k), "num_detections": 1}]
    return obj


def _fragment_pair(rng):
    """A sofa surface and a fragment re-observing part of that same surface."""
    sofa = _box([0, 0, 0], [2.0, 1.0, 0.9], 600, rng)
    piece = sofa[:300] + rng.normal(0, 0.005, size=(300, 3))
    return _FakeViewDataset(np.vstack([sofa, piece]))


def test_merges_a_fragment_that_sits_inside_a_lookalike_object():
    rng = np.random.default_rng(2)
    vd = _fragment_pair(rng)
    objs = [_obj("sofa", 0, 600, [1, 0]), _obj("fragment", 600, 300, [0.97, 0.24])]

    merged = merge_fragmented_instances(objs, vd)
    assert len(merged) == 1
    assert merged[0]["class_id"] == "sofa"
    assert len(merged[0]["indexes"]) == 900          # union of both point sets
    assert len(merged[0]["track_ids"]) == 2          # tracks from both are kept
    expected = np.array([1, 0]) + np.array([0.97, 0.24])   # one detection each
    assert np.allclose(merged[0]["clip_ft"], expected / np.linalg.norm(expected))
    assert merged[0]["num_detections"] == 2


def test_object_inside_a_larger_objects_box_but_not_on_its_points_stays_separate():
    # a desk: top surface at 0.75 m plus four legs, so its box spans the floor
    # to the top; a bag stands under it, inside that box, touching none of it
    rng = np.random.default_rng(5)
    top = _box([0, 0, 0.73], [2.0, 1.0, 0.75], 600, rng)
    legs = np.vstack([_box([x, y, 0], [x + 0.05, y + 0.05, 0.73], 100, rng)
                      for x, y in [(0, 0), (1.95, 0), (0, 0.95), (1.95, 0.95)]])
    bag = _box([0.8, 0.3, 0.0], [1.2, 0.6, 0.5], 300, rng)
    vd = _FakeViewDataset(np.vstack([top, legs, bag]))
    objs = [_obj("desk", 0, 1000, [1, 0]), _obj("bag", 1000, 300, [1, 0])]   # identical embeddings

    assert len(merge_fragmented_instances(objs, vd)) == 2


def test_confident_different_names_block_a_merge():
    vd = _fragment_pair(np.random.default_rng(6))
    objs = [_named(_obj("laptop", 0, 600, [1, 0]), "laptop", 0.6),
            _named(_obj("bottle", 600, 300, [0.97, 0.24]), "bottled water", 0.5)]
    assert len(merge_fragmented_instances(objs, vd)) == 2


def test_names_that_list_each_other_do_not_block_a_merge():
    vd = _fragment_pair(np.random.default_rng(7))
    objs = [_named(_obj("a", 0, 600, [1, 0]), "laptop", 0.6, [("laptop", 0.6), ("laptop computer", 0.2)]),
            _named(_obj("b", 600, 300, [0.97, 0.24]), "laptop computer", 0.5, [("laptop computer", 0.5)])]
    assert len(merge_fragmented_instances(objs, vd)) == 1


def test_a_low_confidence_name_does_not_block_a_merge():
    vd = _fragment_pair(np.random.default_rng(8))
    objs = [_named(_obj("a", 0, 600, [1, 0]), "sofa", 0.6),
            _named(_obj("b", 600, 300, [0.97, 0.24]), "eyebrow", 0.02)]
    assert len(merge_fragmented_instances(objs, vd)) == 1


def test_contained_but_visually_different_objects_stay_separate():
    # a cup inside a shelf's volume is contained, but does not look like it
    rng = np.random.default_rng(3)
    shelf = _box([0, 0, 0], [2.0, 1.0, 0.9], 600, rng)
    cup = _box([0.5, 0.2, 0.2], [0.7, 0.4, 0.4], 300, rng)
    vd = _FakeViewDataset(np.vstack([shelf, cup]))
    objs = [_obj("shelf", 0, 600, [1, 0]), _obj("cup", 600, 300, [0, 1])]

    assert len(merge_fragmented_instances(objs, vd)) == 2


def test_disjoint_objects_stay_separate():
    rng = np.random.default_rng(4)
    a = _box([0, 0, 0], [0.4, 0.4, 0.4], 300, rng)
    b = _box([3, 3, 0], [3.4, 3.4, 0.4], 300, rng)
    vd = _FakeViewDataset(np.vstack([a, b]))
    objs = [_obj("a", 0, 300, [1, 0]), _obj("b", 300, 300, [1, 0])]   # identical embeddings
    assert len(merge_fragmented_instances(objs, vd)) == 2


def test_finds_a_desk_top_but_not_objects_on_it_or_the_floor():
    rng = np.random.default_rng(9)
    # a slightly tilted top (0.84-0.92 m across its width) plus one leg
    top = _box([0, 0, 0], [1.7, 1.5, 0.02], 3000, rng)
    top[:, 2] += 0.84 + 0.06 * top[:, 0] / 1.7
    desk = np.vstack([top, _box([0, 0, 0.0], [0.05, 0.05, 0.84], 200, rng)])
    laptop = _box([0.5, 0.5, 0.9], [0.9, 0.8, 1.15], 800, rng)
    rug = _box([-2, -2, 0.0], [0, 0, 0.03], 2000, rng)
    vd = _FakeViewDataset(np.vstack([desk, laptop, rug]))
    objs = [_obj("desk", 0, 3200, [1, 0]), _obj("laptop", 3200, 800, [0, 1]), _obj("rug", 4000, 2000, [1, 0])]

    assert find_support_surfaces(objs, vd) == {0}
