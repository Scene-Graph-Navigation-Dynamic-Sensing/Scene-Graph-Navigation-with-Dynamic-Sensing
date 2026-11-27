import numpy as np

from navigation.core.bounds import Bounds
from navigation.core.occupancy import build_occupancy_map
from navigation.voxel_map import PersistedVoxelMap


def _voxel_map(points, voxel_resolution=0.01):
    """Builds a PersistedVoxelMap whose indexes_colors_mapping_dict has one
    entry per distinct voxel among `points` (mirrors how ViewDataset's own
    fusion already dedups raw points into one averaged-color entry per
    voxel -- so a dense obstacle needs many *distinct* fine-resolution
    voxels, not the same point repeated, to build up a count)."""
    bounds = Bounds(xmin=0.0, xmax=5.0, ymin=0.0, ymax=5.0, zmin=0.0, zmax=3.0)
    vm = PersistedVoxelMap(resolution=voxel_resolution, bounds=bounds, indexes_colors_mapping_dict={})
    indexes = vm.point_to_index(np.array(points))
    vm.indexes_colors_mapping_dict = {int(idx): (1.0, 1.0, 1.0) for idx in np.atleast_1d(indexes)}
    return vm


def _dense_column(x, y, z0, n, dz=0.01):
    """n distinct 3D points stacked in z at (x, y), one per voxel."""
    return [(x, y, z0 + i * dz) for i in range(n)]


def test_obstacle_cell_marked_occupied_above_threshold():
    pts = _dense_column(2.0, 2.0, z0=1.0, n=150)  # 150 distinct voxels within the height slab
    vm = _voxel_map(pts)
    occ = build_occupancy_map(vm, min_height=0.2, max_height=3.0, resolution=0.5, occ_threshold=100, occ_avoid=0)
    x, y = occ.to_pt((2.0, 2.0))
    assert occ.grid[y, x]


def test_below_threshold_cell_stays_free():
    pts = _dense_column(2.0, 2.0, z0=1.0, n=50)  # fewer voxels than occ_threshold
    vm = _voxel_map(pts)
    occ = build_occupancy_map(vm, min_height=0.2, max_height=3.0, resolution=0.5, occ_threshold=100, occ_avoid=0)
    x, y = occ.to_pt((2.0, 2.0))
    assert not occ.grid[y, x]


def test_points_outside_height_range_are_ignored():
    pts = _dense_column(2.0, 2.0, z0=1.5, n=150)  # entirely above max_height
    vm = _voxel_map(pts)
    occ = build_occupancy_map(vm, min_height=0.2, max_height=1.0, resolution=0.5, occ_threshold=100, occ_avoid=0)
    x, y = occ.to_pt((2.0, 2.0))
    assert not occ.grid[y, x]


def test_dilation_expands_occupied_block():
    pts = _dense_column(2.0, 2.0, z0=1.0, n=150)
    vm = _voxel_map(pts)
    occ = build_occupancy_map(vm, min_height=0.2, max_height=3.0, resolution=0.5, occ_threshold=100, occ_avoid=2)
    x, y = occ.to_pt((2.0, 2.0))
    assert occ.grid[y, x + 1]
    assert occ.grid[y + 1, x]


def test_conservative_marks_unobserved_cells_occupied():
    pts = _dense_column(2.0, 2.0, z0=1.0, n=150)
    vm = _voxel_map(pts)
    occ_default = build_occupancy_map(vm, min_height=0.2, max_height=3.0, resolution=0.5, occ_threshold=100, occ_avoid=0, conservative=False)
    occ_conservative = build_occupancy_map(vm, min_height=0.2, max_height=3.0, resolution=0.5, occ_threshold=100, occ_avoid=0, conservative=True)
    x, y = occ_default.to_pt((4.5, 4.5))  # far corner, never observed
    assert not occ_default.grid[y, x]
    assert occ_conservative.grid[y, x]
