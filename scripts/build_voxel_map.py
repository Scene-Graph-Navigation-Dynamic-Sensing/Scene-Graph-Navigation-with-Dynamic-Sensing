"""Build and visualize the global voxel map from a prepared DoVSG-format dataset.

Loads the dataset through object_fusion.view_dataset.ViewDataset -- the same
voxel map the scene-graph pipeline consumes -- and writes it out as a coloured
point cloud plus rendered views, so the map can be inspected without running
the rest of the pipeline.

Outputs (under --out-dir):
  voxel_map.ply        coloured voxel centres, one point per occupied voxel
  trajectory.ply       camera centres as a polyline
  views/*.png          offscreen renders from several angles
  voxel_map_stats.json occupancy, extent, per-frame contribution
  run.log

--interactive opens an Open3D window instead of rendering offscreen (needs a
display). --compare-mesh overlays the 3RScan reference mesh to eyeball drift.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import open3d as o3d
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from object_fusion.view_dataset import load_view_dataset  # noqa: E402

logger = logging.getLogger("build_voxel_map")


def voxel_map_to_pcd(view_dataset) -> o3d.geometry.PointCloud:
    """The occupied voxels ViewDataset accumulated, as an Open3D cloud."""
    indexes = np.array(list(view_dataset.indexes_colors_mapping_dict.keys()))
    colors = np.array(list(view_dataset.indexes_colors_mapping_dict.values()))
    points = view_dataset.index_to_point(indexes)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    pcd.colors = o3d.utility.Vector3dVector(np.clip(colors, 0.0, 1.0))
    return pcd


def load_trajectory(dataset_dir: Path, length: int) -> np.ndarray:
    centers = []
    for i in range(length):
        pose = np.loadtxt(dataset_dir / "poses" / f"{i:06d}.txt")
        centers.append(pose[:3, 3])
    return np.array(centers)


def trajectory_lineset(centers: np.ndarray) -> o3d.geometry.LineSet:
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(centers)
    ls.lines = o3d.utility.Vector2iVector([[i, i + 1] for i in range(len(centers) - 1)])
    ls.colors = o3d.utility.Vector3dVector(np.tile([1.0, 0.0, 0.0], (max(len(centers) - 1, 1), 1)))
    return ls


def render_views(geometries, out_dir: Path, width=1600, height=1200) -> list[Path]:
    """Offscreen renders from a few fixed angles around the scene."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pts = np.concatenate([np.asarray(g.points) for g in geometries
                          if isinstance(g, o3d.geometry.PointCloud)], axis=0)
    center = pts.mean(axis=0)
    diag = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))

    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
    renderer.scene.set_background([1.0, 1.0, 1.0, 1.0])
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultUnlit"
    mat.point_size = 3.0
    line_mat = o3d.visualization.rendering.MaterialRecord()
    line_mat.shader = "unlitLine"
    line_mat.line_width = 4.0

    for i, g in enumerate(geometries):
        renderer.scene.add_geometry(
            f"g{i}", g, line_mat if isinstance(g, o3d.geometry.LineSet) else mat)

    # 3RScan poses are not gravity-aligned, so "top" here means along +Z of the
    # scan frame rather than true up; the named angles are just viewpoints.
    offsets = {
        "iso":   np.array([1.0, -1.0, 0.8]),
        "top":   np.array([0.0, 0.0, 1.6]),
        "front": np.array([0.0, -1.7, 0.2]),
        "side":  np.array([1.7, 0.0, 0.2]),
    }
    written = []
    for name, off in offsets.items():
        # normalise: the offsets above encode direction, not distance (|iso| is
        # 1.63), so without this the camera sits 1.6x further back than the 0.7
        # diagonal intends and a large scan renders as a speck
        eye = center + off / np.linalg.norm(off) * diag * 0.7
        up = np.array([0.0, 1.0, 0.0]) if name == "top" else np.array([0.0, 0.0, 1.0])
        renderer.setup_camera(60.0, center.astype(np.float32),
                              eye.astype(np.float32), up.astype(np.float32))
        img = renderer.render_to_image()
        p = out_dir / f"{name}.png"
        o3d.io.write_image(str(p), img)
        written.append(p)
        logger.info("rendered %s", p)
    return written


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-dir", required=True, type=Path,
                    help="output of scripts/prepare_3rscan_dataset.py")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "pipeline.yaml")
    ap.add_argument("--resolution", type=float, default=None,
                    help="voxel size in metres (default: object_fusion.voxel_resolution)")
    ap.add_argument("--no-inlier-mask", action="store_true",
                    help="skip the statistical outlier removal ViewDataset applies per frame")
    ap.add_argument("--interactive", action="store_true", help="open an Open3D window")
    ap.add_argument("--compare-mesh", type=Path, default=None,
                    help="3RScan labels.instances.annotated.v2.ply to overlay")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    resolution = args.resolution if args.resolution is not None \
        else cfg["object_fusion"]["voxel_resolution"]

    out_dir = args.out_dir or (REPO_ROOT / "outputs" / "voxel_maps" /
                               f"{args.dataset_dir.name}_r{resolution}_{time.strftime('%Y%m%d-%H%M%S')}")
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        handlers=[logging.FileHandler(out_dir / "run.log"), logging.StreamHandler(sys.stdout)],
        force=True,
    )
    logger.info("args: %s", vars(args))
    logger.info("voxel resolution: %.4f m", resolution)

    src_info = {}
    src_path = args.dataset_dir / "source.json"
    if src_path.exists():
        src_info = json.loads(src_path.read_text())
        logger.info("dataset source: %s", src_info)

    t0 = time.perf_counter()
    view_dataset = load_view_dataset(
        recorder_dir=str(args.dataset_dir),
        scene_graph_generation_root=cfg["external"]["scene_graph_generation_root"],
        resolution=resolution,
        use_inlier_mask=not args.no_inlier_mask,
    )
    build_s = time.perf_counter() - t0
    logger.info("ViewDataset built in %.1f s", build_s)

    pcd = voxel_map_to_pcd(view_dataset)
    n_vox = len(pcd.points)
    bounds = view_dataset.bounds
    extent = np.array([bounds.xdiff, bounds.ydiff, bounds.zdiff])
    logger.info("occupied voxels: %d", n_vox)
    logger.info("scene extent (m): %.2f x %.2f x %.2f", *extent)
    logger.info("voxel grid dims: %s", view_dataset.voxel_num.tolist())

    ply_path = out_dir / "voxel_map.ply"
    o3d.io.write_point_cloud(str(ply_path), pcd)
    logger.info("wrote %s", ply_path)

    centers = load_trajectory(args.dataset_dir, view_dataset.length)
    traj = trajectory_lineset(centers)
    traj_pcd = o3d.geometry.PointCloud()
    traj_pcd.points = o3d.utility.Vector3dVector(centers)
    traj_pcd.paint_uniform_color([1.0, 0.0, 0.0])
    o3d.io.write_point_cloud(str(out_dir / "trajectory.ply"), traj_pcd)

    geoms = [pcd, traj]
    if args.compare_mesh and args.compare_mesh.exists():
        mesh = o3d.io.read_triangle_mesh(str(args.compare_mesh))
        mesh.compute_vertex_normals()
        geoms.append(mesh)
        logger.info("overlaying reference mesh: %s (%d verts)",
                    args.compare_mesh, len(mesh.vertices))

    stats = {
        "dataset_dir": str(args.dataset_dir),
        "source": src_info,
        "resolution_m": resolution,
        "frames": int(view_dataset.length),
        "occupied_voxels": int(n_vox),
        "voxel_grid_dims": view_dataset.voxel_num.tolist(),
        "extent_m": extent.round(3).tolist(),
        "bounds": {"lower": bounds.lower_bound.round(3).tolist(),
                   "upper": bounds.higher_bound.round(3).tolist()},
        "trajectory_length_m": float(np.linalg.norm(np.diff(centers, axis=0), axis=1).sum()),
        "build_seconds": round(build_s, 1),
        "inlier_mask": not args.no_inlier_mask,
    }
    (out_dir / "voxel_map_stats.json").write_text(json.dumps(stats, indent=2))

    if args.interactive:
        logger.info("opening interactive viewer (close the window to finish)")
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
        o3d.visualization.draw_geometries(geoms + [frame], window_name="Voxel map")
    else:
        render_views(geoms, out_dir / "views")

    logger.info("=" * 60)
    logger.info("voxel map: %d voxels @ %.3f m over %d frames", n_vox, resolution, view_dataset.length)
    logger.info("results: %s", out_dir)
    logger.info("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
