"""PersistedVoxelMap -- closes the gap that made standalone navigation
planning impossible: scene_graph/storage/serializer.py never persisted the
fusion pipeline's ViewDataset (bounds + occupied-voxel index/color map), only
the scene graph and node centroids, because the voxel map only exists
in-memory during a single pipeline run.

This is a deliberately minimal reimplementation of just the read-only
voxel-index API that navigation/core/occupancy.py and
navigation/core/path_facade.py need (`bounds`, `indexes_colors_mapping_dict`,
`index_to_point`, `index_to_pcd`) -- NOT a general ViewDataset replacement.
The encode/decode formulas below are copied from
Scene-Graph-Generation/dovsg/memory/view_dataset.py's
point_to_voxel/voxel_to_point/voxel_to_index/index_to_voxel, so a
PersistedVoxelMap decodes indexes produced by the real ViewDataset
identically.

Kept in sync by contract with scene_graph/storage/serializer.py's
`voxel_map` pickle block (see `_build_voxel_map_snapshot` there) -- if either
side's encoding changes, the other must change with it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from navigation.core.bounds import Bounds


class VoxelMapNotPersistedError(Exception):
    """Raised when a saved scene graph .pkl predates voxel_map persistence
    (or otherwise has no voxel_map block), so navigation can't run against
    it -- the scene must be re-saved via a fresh fusion + save_scene_graph
    run first."""


@dataclass
class PersistedVoxelMap:
    resolution: float
    bounds: Bounds
    indexes_colors_mapping_dict: dict[int, tuple[float, float, float]]
    voxel_num: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.voxel_num = (
            (self.bounds.higher_bound - self.bounds.lower_bound) / self.resolution
        ).astype(np.int32)

    def point_to_voxel(self, points) -> np.ndarray:
        points = np.asarray(points)
        return np.floor((points - self.bounds.lower_bound) / self.resolution).astype(np.int32)

    def voxel_to_point(self, voxels) -> np.ndarray:
        voxels = np.asarray(voxels)
        return voxels * self.resolution + self.bounds.lower_bound

    def voxel_to_index(self, voxels) -> np.ndarray:
        voxels = np.asarray(voxels)
        return (
            voxels[..., 0] * self.voxel_num[1] * self.voxel_num[2]
            + voxels[..., 1] * self.voxel_num[2]
            + voxels[..., 2]
        )

    def index_to_voxel(self, indexes) -> np.ndarray:
        indexes = np.asarray(indexes)
        voxels = np.zeros(indexes.shape + (3,), dtype=np.int32)
        voxels[..., 2] = indexes % self.voxel_num[2]
        indexes = indexes // self.voxel_num[2]
        voxels[..., 1] = indexes % self.voxel_num[1]
        voxels[..., 0] = indexes // self.voxel_num[1]
        return voxels

    def point_to_index(self, points) -> np.ndarray:
        return self.voxel_to_index(self.point_to_voxel(points))

    def index_to_point(self, indexes) -> np.ndarray:
        return self.voxel_to_point(self.index_to_voxel(indexes))

    def index_to_pcd(self, indexes):
        """Only used by the optional 3D Open3D visualization path -- lazy
        import so open3d isn't a hard dependency of navigation/ otherwise."""
        import open3d as o3d

        points = self.index_to_point(indexes)
        colors = np.array([self.indexes_colors_mapping_dict[idx] for idx in indexes])
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        return pcd

    @classmethod
    def from_dict(cls, d: dict | None) -> "PersistedVoxelMap":
        if d is None:
            raise VoxelMapNotPersistedError(
                "This scene graph .pkl has no persisted voxel_map -- it was saved before "
                "navigation support was added, or by a caller that didn't pass view_dataset "
                "to save_scene_graph(). Re-run the fusion pipeline and save the scene again."
            )
        xmin, xmax, ymin, ymax, zmin, zmax = d["bounds"]
        return cls(
            resolution=d["resolution"],
            bounds=Bounds(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, zmin=zmin, zmax=zmax),
            indexes_colors_mapping_dict=d["indexes_colors_mapping_dict"],
        )
