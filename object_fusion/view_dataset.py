"""Global voxel-map builder — thin re-export of DoVSG's ViewDataset.

ViewDataset (dovsg/memory/view_dataset.py) is detector-agnostic: it only
needs a per-scan directory of rgb/point/mask/poses + metadata.json to
back-project every frame into one global voxel-indexed point cloud
("global location of the object" -> Voxel Map, in the target design).
Reused as-is; no adaptation needed. The dataset directory it expects is
populated by scripts/run_static_scene_graph.py during capture, using
the same layout Scene-Graph-Generation/main_live.py already writes
(rgb/{idx:06d}.jpg, point/{idx:06d}.npy, mask/{idx:06d}.npy,
poses/{idx:06d}.txt, metadata.json with w/h/K/dist_coef/length).
"""
from __future__ import annotations

import json
from pathlib import Path

from object_fusion._dovsg_import import ensure_dovsg_importable


def load_view_dataset(
    recorder_dir: str,
    scene_graph_generation_root: str,
    interval: int = 1,
    resolution: float = 0.01,
    use_inlier_mask: bool = True,
):
    ensure_dovsg_importable(scene_graph_generation_root)
    from dovsg.memory.view_dataset import ViewDataset

    return ViewDataset(
        recorder_dir=recorder_dir,
        interval=interval,
        use_inlier_mask=use_inlier_mask,
        resolution=resolution,
    )


def _frame_inliers(recorder_dir: str, cnt: int, nb_neighbors: int, std_ratio: float,
                   scene_graph_generation_root: str):
    """Pass 1 of build_view_dataset_parallel(), one frame: ViewDataset.load_data()'s
    statistical-outlier mask and the frame's world extent."""
    import numpy as np
    from PIL import Image

    ensure_dovsg_importable(scene_graph_generation_root)
    from dovsg.utils.utils import get_inlier_mask

    d = Path(recorder_dir)
    image = np.asarray(Image.open(d / f"rgb/{cnt:06}.jpg"), dtype=np.uint8)
    mask = np.load(d / f"mask/{cnt:06}.npy", allow_pickle=True).astype(np.bool_)
    point = np.load(d / f"point/{cnt:06}.npy", allow_pickle=True).astype(np.float32)
    pose = np.loadtxt(d / f"poses/{cnt:06}.txt").astype(np.float32)
    mask = np.logical_and(mask, get_inlier_mask(point=point, color=image / 255, mask=mask,
                                                nb_neighbors=nb_neighbors, std_ratio=std_ratio))
    world = (point @ pose[:3, :3].T + pose[:3, 3])[mask]
    if len(world) == 0:
        return mask, None, None
    return mask, np.amin(world, axis=0), np.amax(world, axis=0)


def _frame_voxels(recorder_dir: str, cnt: int, mask, lower, higher, resolution: float, voxel_num):
    """Pass 2, one frame: ViewDataset.voxelize() -- the pixel -> voxel index map
    and each voxel's mean colour in this frame."""
    import numpy as np
    from PIL import Image

    d = Path(recorder_dir)
    image = np.asarray(Image.open(d / f"rgb/{cnt:06}.jpg"), dtype=np.uint8)
    point = np.load(d / f"point/{cnt:06}.npy", allow_pickle=True).astype(np.float32)
    pose = np.loadtxt(d / f"poses/{cnt:06}.txt").astype(np.float32)
    gpoint = point @ pose[:3, :3].T + pose[:3, 3]
    color = (image / 255).astype(np.float32)
    range_mask = ((gpoint - lower) >= 0).all(-1) * ((higher - gpoint) >= 0).all(-1)
    index_mask = np.logical_and(mask, range_mask)
    mapping = -np.ones(gpoint.shape[:2], dtype=np.int32)
    voxels = np.floor((gpoint[index_mask] - lower) / resolution).astype(np.int32)
    mapping[index_mask] = (voxels[..., 0] * voxel_num[1] * voxel_num[2] + voxels[..., 1] * voxel_num[2]
                           + voxels[..., 2])
    unique, inverse, counts = np.unique(mapping[index_mask], return_inverse=True, return_counts=True)
    summed = np.zeros((unique.size, 3))
    np.add.at(summed, inverse, color[index_mask])
    return mapping, index_mask, summed / counts[:, np.newaxis], unique


def build_view_dataset_parallel(recorder_dir: str, scene_graph_generation_root: str, resolution: float = 0.01,
                                workers: int = 8, nb_neighbors: int = 30, std_ratio: float = 1.5):
    """DoVSG's ViewDataset(recorder_dir, interval=1, use_inlier_mask=True), with
    its per-frame work spread over `workers` processes. The outlier filter and
    the voxelization of one frame do not depend on any other frame; only the
    map bounds need all of them, so: pass 1 (masks + extents, parallel) ->
    bounds, rounded as ViewDataset does -> pass 2 (voxel indexes + colours,
    parallel) -> the global colour map, averaged as
    calculate_all_global_voxel_indexes_and_colors() does. Same arithmetic in
    the same order, so the map is the one ViewDataset builds. Per-frame images
    and world points are not kept (nothing reads them after the map)."""
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    import numpy as np

    ensure_dovsg_importable(scene_graph_generation_root)
    from dovsg.memory.view_dataset import Bounds, ViewDataset

    vd = ViewDataset.__new__(ViewDataset)
    vd.recorder_dir = Path(recorder_dir)
    vd.interval, vd.use_inlier_mask, vd.resolution = 1, True, resolution
    vd.nb_neighbors, vd.std_ratio = nb_neighbors, std_ratio
    vd.metadata = vd.read_metadata()
    frames = list(range(vd.length))

    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        first = list(pool.map(_frame_inliers, [recorder_dir] * len(frames), frames, [nb_neighbors] * len(frames),
                              [std_ratio] * len(frames), [scene_graph_generation_root] * len(frames)))
        masks = [m for m, _, _ in first]
        lows = [lo for _, lo, _ in first if lo is not None]
        highs = [hi for _, _, hi in first if hi is not None]
        min_bounds = np.minimum.reduce([np.array([np.inf] * 3), *lows])
        max_bounds = np.maximum.reduce([np.array([-np.inf] * 3), *highs])
        vd.bounds = Bounds.from_arr(np.round(np.stack([min_bounds, max_bounds], axis=1), 2))
        vd.voxel_num = ((vd.bounds.higher_bound - vd.bounds.lower_bound) / resolution).astype(np.int32)
        second = list(pool.map(_frame_voxels, [recorder_dir] * len(frames), frames, masks,
                               [vd.bounds.lower_bound] * len(frames), [vd.bounds.higher_bound] * len(frames),
                               [resolution] * len(frames), [vd.voxel_num] * len(frames)))

    vd.images, vd.global_points = [], []
    vd.masks = masks
    vd.names = [f"{cnt:06}" for cnt in frames]
    vd.append_length_log = [len(frames)]
    vd.pixel_index_mappings = [m for m, _, _, _ in second]
    vd.pixel_index_masks = [k for _, k, _, _ in second]
    all_colours = np.vstack([c for _, _, c, _ in second])
    all_indexes = np.hstack([u for _, _, _, u in second])
    unique, inverse, counts = np.unique(all_indexes, return_inverse=True, return_counts=True)
    summed = np.zeros((unique.size, all_colours.shape[1]))
    np.add.at(summed, inverse, all_colours)
    vd.indexes_colors_mapping_dict = dict(zip(unique, summed / counts[:, np.newaxis]))
    return vd


def load_view_dataset_for_transfer(workers: int = 1, **kwargs):
    """load_view_dataset() for a separate process: the voxel map is CPU work
    that holds Python's GIL, so on a thread it slows detection down instead
    of overlapping with it. Drops what nothing reads after the map is built
    (per-frame images, masks and world points: ~400 MB) and packs the voxel
    colour map into two arrays, so the result pickles back quickly. Undo with
    restore_view_dataset()."""
    import numpy as np

    if workers > 1:
        vd = build_view_dataset_parallel(workers=workers, **kwargs)
    else:
        vd = load_view_dataset(**kwargs)
    vd.images, vd.masks, vd.global_points = [], [], []
    for attr in ("_scene_colors", "_scene_indexes"):
        if hasattr(vd, attr):
            setattr(vd, attr, [])
    colours = vd.indexes_colors_mapping_dict
    vd._packed_colours = (np.fromiter(colours.keys(), dtype=np.int64, count=len(colours)),
                          np.asarray(list(colours.values()), dtype=np.float32))
    vd.indexes_colors_mapping_dict = {}
    return vd


def restore_view_dataset(vd):
    keys, values = vd.__dict__.pop("_packed_colours")
    vd.indexes_colors_mapping_dict = dict(zip(keys.tolist(), values))
    return vd


def voxelize_frame_in_grid(dataset_dir: str, i: int, sim3: tuple, lower: tuple, upper: tuple, resolution: float,
                           voxel_num: tuple, scene_graph_generation_root: str, nb_neighbors: int = 30,
                           std_ratio: float = 1.5) -> dict:
    """One frame of another run placed in a saved map's voxel grid: the frame's
    points are scaled and moved by the Sim3 (scale, R, t) into the saved world
    and indexed with the saved map's bounds -- DoVSG ViewDataset.voxelize()
    with foreign bounds. Module-level so a process pool can run it."""
    import numpy as np
    from PIL import Image

    ensure_dovsg_importable(scene_graph_generation_root)
    from dovsg.utils.utils import get_inlier_mask

    scale, R, t = sim3
    d = Path(dataset_dir)
    image = np.asarray(Image.open(d / "rgb" / f"{i:06d}.jpg"), dtype=np.uint8)
    point = np.load(d / "point" / f"{i:06d}.npy").astype(np.float32) * scale
    mask = np.load(d / "mask" / f"{i:06d}.npy").astype(bool)
    pose = np.loadtxt(d / "poses" / f"{i:06d}.txt")
    pose_m = np.eye(4)  # camera-to-saved-world: rotate by R, move the centre by the full Sim3
    pose_m[:3, :3] = R @ pose[:3, :3]
    pose_m[:3, 3] = scale * R @ pose[:3, 3] + t
    mask &= get_inlier_mask(point=point, color=image / 255, mask=mask, nb_neighbors=nb_neighbors,
                            std_ratio=std_ratio)
    gpoint = point @ pose_m[:3, :3].T + pose_m[:3, 3]
    lo, hi = np.asarray(lower), np.asarray(upper)
    valid = mask & np.all((gpoint >= lo) & (gpoint < hi), axis=-1)
    mapping = -np.ones(point.shape[:2], dtype=np.int64)
    # same encoding as ViewDataset.point_to_index / navigation.voxel_map.PersistedVoxelMap
    v = np.floor((gpoint[valid] - lo) / resolution).astype(np.int64)
    mapping[valid] = v[:, 0] * voxel_num[1] * voxel_num[2] + v[:, 1] * voxel_num[2] + v[:, 2]
    return {"image": image, "mask": mask, "pose": pose_m, "depth": np.where(mask, point[..., 2], 0.0).astype(np.float32),
            "mapping": mapping, "valid": valid}


def write_metadata(dataset_dir: Path, width: int, height: int, K: list, length: int, dist_coef: list | None = None) -> None:
    """Writes the metadata.json ViewDataset.read_metadata() expects."""
    metadata = {
        "w": width,
        "h": height,
        "K": K,  # flattened 3x3 intrinsic matrix, row-major
        "dist_coef": dist_coef or [0.0, 0.0, 0.0, 0.0, 0.0],
        "length": length,
    }
    with open(Path(dataset_dir) / "metadata.json", "w") as f:
        json.dump(metadata, f)
