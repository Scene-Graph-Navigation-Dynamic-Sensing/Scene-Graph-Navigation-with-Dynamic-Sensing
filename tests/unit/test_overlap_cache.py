"""The voxel-lookup overlap matrix must equal upstream DoVSG's FAISS search.

InstanceFusion overrides compute_overlap_matrix_2set to look neighbours up by
voxel index instead of searching every point against every point. All points
sit on the voxel grid, so the two must agree wherever the float search is not
on a knife edge: with a radius of 1.5 voxels, face (1.0) and edge (1.41)
neighbours are clearly inside and corner (1.73) neighbours clearly outside.
Cached voxel sets must also be invalidated when an object changes.
"""
import numpy as np
import torch
import yaml
from pathlib import Path

from object_fusion.instance_fusion import build_instance_fusion_class

REPO_ROOT = Path(__file__).resolve().parents[2]
SGG_ROOT = yaml.safe_load((REPO_ROOT / "configs/pipeline.yaml").read_text())["external"][
    "scene_graph_generation_root"]

RES = 0.01


class _GridViewDataset:
    """A voxel grid with ViewDataset's index <-> voxel <-> point encoding."""

    def __init__(self, voxel_num=(20, 20, 20)):
        self.resolution = RES
        self.voxel_num = np.array(voxel_num, dtype=np.int32)
        self.lower = np.array([-0.1, 0.2, 0.0])

    def voxel_to_index(self, v):
        v = np.asarray(v)
        return v[..., 0] * self.voxel_num[1] * self.voxel_num[2] + v[..., 1] * self.voxel_num[2] + v[..., 2]

    def index_to_voxel(self, ix):
        ix = np.asarray(ix)
        v = np.zeros(ix.shape + (3,), dtype=np.int32)
        v[..., 2] = ix % self.voxel_num[2]
        ix = ix // self.voxel_num[2]
        v[..., 1] = ix % self.voxel_num[1]
        v[..., 0] = ix // self.voxel_num[1]
        return v

    def index_to_point(self, ix):
        return self.index_to_voxel(ix) * self.resolution + self.lower


def _obj(indexes):
    return {"indexes": np.asarray(indexes), "bbox": None}


class _Bag(list):
    def get_values(self, key):
        return [o[key] for o in self]


def _boxes(n):
    # unit cubes all at the origin, so every pair passes the IoU gate
    corners = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
                        [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]], dtype=np.float32)
    return torch.from_numpy(np.stack([corners] * n))


def _blob(vd, rng, centre, n):
    """n voxels around a centre voxel (clipped to the grid), as indexes."""
    v = np.clip(np.asarray(centre) + rng.integers(-4, 5, size=(n, 3)), 0, vd.voxel_num - 1)
    return np.unique(vd.voxel_to_index(v))


def _upstream(fusion, map_objs, new_objs):
    from dovsg.memory.instances.instance_process import InstanceProcess

    return InstanceProcess.compute_overlap_matrix_2set(
        fusion, _Bag(map_objs), _Bag(new_objs), _boxes(len(map_objs)), _boxes(len(new_objs)))


def _ours(fusion, map_objs, new_objs):
    return fusion.compute_overlap_matrix_2set(
        _Bag(map_objs), _Bag(new_objs), _boxes(len(map_objs)), _boxes(len(new_objs)))


def test_voxel_lookup_matches_upstream_and_invalidates_on_change():
    InstanceFusion = build_instance_fusion_class(SGG_ROOT)
    vd = _GridViewDataset()
    rng = np.random.default_rng(0)
    fusion = InstanceFusion(downsample_voxel_size=1.5 * RES)
    fusion.view_dataset = vd
    map_objs = [_obj(_blob(vd, rng, (6, 6, 6), 150)), _obj(_blob(vd, rng, (12, 10, 9), 150))]
    new_objs = [_obj(_blob(vd, rng, (8, 7, 7), 120)), _obj(_blob(vd, rng, (14, 12, 12), 120)),
                _obj(_blob(vd, rng, (19, 19, 19), 60))]  # touches the grid's upper edge

    expected = _upstream(fusion, map_objs, new_objs)
    got = _ours(fusion, map_objs, new_objs)
    assert np.allclose(got, expected), f"{got} != {expected}"
    assert 0 < expected.max() < 1, "the case should have partial overlaps"
    assert fusion._overlap_index_misses == 2 and fusion._overlap_index_hits == 0

    # same objects again: served from cache, same numbers
    assert np.allclose(_ours(fusion, map_objs, new_objs), expected)
    assert fusion._overlap_index_hits == 2

    # a merge grows one object's voxel set -> its cached set must not be reused
    map_objs[0] = _obj(np.union1d(map_objs[0]["indexes"], _blob(vd, rng, (14, 12, 12), 200)))
    expected2 = _upstream(fusion, map_objs, new_objs)
    got2 = _ours(fusion, map_objs, new_objs)
    assert np.allclose(got2, expected2), f"{got2} != {expected2}"
    assert not np.allclose(got2[0], got[0]), "changed object should change its overlaps"


def test_one_voxel_radius_counts_face_neighbours_only():
    InstanceFusion = build_instance_fusion_class(SGG_ROOT)
    vd = _GridViewDataset()
    fusion = InstanceFusion(downsample_voxel_size=RES)
    fusion.view_dataset = vd
    centre = vd.voxel_to_index(np.array([10, 10, 10]))
    queries = [[10, 10, 10], [11, 10, 10], [10, 9, 10], [10, 10, 11],  # itself + face neighbours: in
               [11, 11, 10], [11, 11, 11], [12, 10, 10]]                # edge, corner, two away: out
    new_objs = [_obj([vd.voxel_to_index(np.array(q))]) for q in queries]
    got = _ours(fusion, [_obj([centre])], new_objs)[0]
    assert got.tolist() == [1, 1, 1, 1, 0, 0, 0]
