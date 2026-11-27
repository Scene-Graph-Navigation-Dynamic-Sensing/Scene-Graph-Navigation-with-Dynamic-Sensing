"""Ported from Scene-Graph-Generation/dovsg/navigation/occupancy_map.py
(DoVSG). Algorithm unchanged; two deltas from the original:
  - the `view_dataset` parameter is retyped to accept navigation/voxel_map.py's
    PersistedVoxelMap instead of a live dovsg.memory.view_dataset.ViewDataset
    -- the function body only ever touches `.bounds`, `.indexes_colors_mapping_dict`
    and `.index_to_point(...)`, all of which PersistedVoxelMap also provides,
    so no other change was needed (duck-typed in the original too).
  - commented-out matplotlib debug blocks were dropped.
"""
from __future__ import annotations

import numpy as np

from navigation.core.grid_map import Map


def build_occupancy_map(
    voxel_map,
    min_height: float,
    max_height: float,
    resolution: float = 0.01,
    occ_threshold: int = 100,
    conservative: bool = False,
    occ_avoid: int = 2,
) -> Map:
    """Bins `voxel_map`'s occupied 3D points into a 2D (x, y) occupancy grid.

    A cell is occupied if the count of points whose z falls within
    [min_height, max_height] meets `occ_threshold`; occupied cells are then
    dilated by `occ_avoid` cells (robot-radius inflation). If `conservative`,
    cells with zero observed points at all (never scanned) are also marked
    occupied -- safer near unmapped regions, at the cost of blocking
    potentially-valid free space.
    """
    bounds = voxel_map.bounds
    points = voxel_map.index_to_point(list(voxel_map.indexes_colors_mapping_dict.keys()))
    origin = (bounds.xmin, bounds.ymin)

    xbins, ybins = int(bounds.xdiff / resolution) + 2, int(bounds.ydiff / resolution) + 2
    counts = np.zeros((ybins, xbins), dtype=np.int32).flatten()
    any_counts = np.zeros((ybins, xbins), dtype=np.int32).flatten()

    xs = np.floor(((points[:, 0] - origin[0] + resolution / 2) / resolution)).astype(np.int32)
    ys = np.floor(((points[:, 1] - origin[1] + resolution / 2) / resolution)).astype(np.int32)

    # Counts the number of occupying points in each cell.
    occ_xys = np.logical_and(points[:, 2] >= min_height, points[:, 2] <= max_height)
    occ_inds = ys[occ_xys] * xbins + xs[occ_xys]
    np.add.at(counts, occ_inds, 1)

    # Keeps track of the cells that have any points from anywhere.
    inds = ys * xbins + xs
    inds = np.clip(inds, 0, any_counts.size - 1)
    np.add.at(any_counts, inds, 1)

    assert counts is not None, "No points in the dataset"
    counts = counts.reshape((ybins, xbins))
    any_counts = any_counts.reshape((ybins, xbins))

    occ_map = counts >= occ_threshold
    occ_map_copy = occ_map.copy()

    for i in range(occ_map.shape[0]):
        for j in range(occ_map.shape[1]):
            if occ_map_copy[i, j]:
                occ_map[
                    max(0, i - occ_avoid): min(occ_map.shape[0] - 1, i + occ_avoid),
                    max(0, j - occ_avoid): min(occ_map.shape[1] - 1, j + occ_avoid),
                ] = True

    if conservative:
        no_counts = any_counts == 0
        occ_map = np.logical_or(occ_map, no_counts)

    return Map(occ_map, resolution, origin)
