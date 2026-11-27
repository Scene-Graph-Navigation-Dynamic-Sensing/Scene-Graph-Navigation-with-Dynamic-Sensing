import numpy as np
import pytest

from navigation.core.bounds import Bounds
from navigation.voxel_map import PersistedVoxelMap, VoxelMapNotPersistedError


def test_index_to_point_matches_hand_computed_voxels():
    bounds = Bounds(xmin=0.0, xmax=2.0, ymin=0.0, ymax=2.0, zmin=0.0, zmax=2.0)
    resolution = 0.5
    vm = PersistedVoxelMap(resolution=resolution, bounds=bounds, indexes_colors_mapping_dict={})

    # Same voxel_num formula as ViewDataset: (higher - lower) / resolution.
    voxel_num = ((bounds.higher_bound - bounds.lower_bound) / resolution).astype(np.int32)
    assert tuple(vm.voxel_num) == tuple(voxel_num)

    voxels = np.array([[0, 0, 0], [1, 2, 3], [3, 3, 3]])
    hand_indexes = (
        voxels[:, 0] * voxel_num[1] * voxel_num[2] + voxels[:, 1] * voxel_num[2] + voxels[:, 2]
    )
    hand_points = voxels * resolution + bounds.lower_bound

    assert np.array_equal(vm.voxel_to_index(voxels), hand_indexes)
    decoded_points = vm.index_to_point(hand_indexes)
    assert np.allclose(decoded_points, hand_points)


def test_point_to_index_roundtrips_through_voxel_center():
    bounds = Bounds(xmin=0.0, xmax=5.0, ymin=0.0, ymax=5.0, zmin=0.0, zmax=5.0)
    vm = PersistedVoxelMap(resolution=0.1, bounds=bounds, indexes_colors_mapping_dict={})
    point = np.array([1.23, 2.34, 0.56])
    idx = vm.point_to_index(point)
    decoded = vm.index_to_point(idx)
    # decoded lands on the voxel's lower corner, within one resolution of the original point
    assert np.all(np.abs(decoded - point) < 0.1)


def test_from_dict_reconstructs_bounds_and_mapping():
    d = {
        "resolution": 0.05,
        "bounds": (0.0, 3.0, 0.0, 3.0, 0.0, 2.0),
        "indexes_colors_mapping_dict": {1: (0.1, 0.2, 0.3), 2: (0.4, 0.5, 0.6)},
    }
    vm = PersistedVoxelMap.from_dict(d)
    assert vm.resolution == 0.05
    assert vm.bounds.xmax == 3.0
    assert vm.indexes_colors_mapping_dict == d["indexes_colors_mapping_dict"]


def test_from_dict_none_raises_not_persisted_error():
    with pytest.raises(VoxelMapNotPersistedError):
        PersistedVoxelMap.from_dict(None)
