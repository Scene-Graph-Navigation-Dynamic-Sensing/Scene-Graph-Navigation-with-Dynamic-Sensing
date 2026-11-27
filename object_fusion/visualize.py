"""3D instance visualization for the object-fusion stage.

open_interactive_viewer() reuses DoVSG's vis_instances() as-is (it already
has color-by-instance/class/clip-similarity toggles wired to keyboard
shortcuts) for interactive inspection. save_instances_screenshot() is a
small standalone offscreen renderer (Open3D only, no DoVSG dependency) that
produces the non-interactive artifact the per-stage visualization
requirement asks for, since vis_instances() itself is a blocking
interactive window rather than something that hands back a saved image.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import open3d as o3d

from object_fusion._dovsg_import import ensure_dovsg_importable


def open_interactive_viewer(instance_objects, view_dataset, scene_graph_generation_root: str, device: str = "cuda"):
    ensure_dovsg_importable(scene_graph_generation_root)
    from dovsg.memory.instances.visualize_instances import vis_instances

    cmap = matplotlib.colormaps.get_cmap("turbo")
    classes = sorted({obj["class_name"] for obj in instance_objects})
    class_colors = {c: cmap(i / max(len(classes) - 1, 1))[:3] for i, c in enumerate(classes)}

    vis_instances(
        instance_objects=instance_objects,
        class_colors=class_colors,
        view_dataset=view_dataset,
        show_background=True,
        device=device,
    )


def save_instances_screenshot(instance_objects, view_dataset, out_path: Path, voxel_size: float = 0.01) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cmap = matplotlib.colormaps.get_cmap("turbo")
    n = max(len(instance_objects), 1)

    geometries = []
    for i, obj in enumerate(instance_objects):
        pcd = view_dataset.index_to_pcd(indexes=obj["indexes"])
        pcd = pcd.voxel_down_sample(voxel_size)
        color = np.array(cmap(i / max(n - 1, 1))[:3])
        pcd.paint_uniform_color(color)
        geometries.append(pcd)

    renderer = o3d.visualization.rendering.OffscreenRenderer(1280, 960)
    renderer.scene.set_background([1.0, 1.0, 1.0, 1.0])
    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = "defaultUnlit"

    all_points = []
    for i, pcd in enumerate(geometries):
        renderer.scene.add_geometry(f"obj_{i}", pcd, material)
        all_points.append(np.asarray(pcd.points))

    if all_points:
        combined = np.concatenate(all_points, axis=0)
        center = combined.mean(axis=0)
        diag = float(np.linalg.norm(combined.max(axis=0) - combined.min(axis=0)))
        # setup_camera(fov, center, eye, up) — an isometric-ish offset from
        # the scene center, far enough back (per the bbox diagonal) to frame
        # everything; Z is up, matching DoVSG's convention elsewhere.
        eye = center + np.array([diag, -diag, diag * 0.8], dtype=np.float32)
        renderer.setup_camera(
            60.0,
            center.astype(np.float32),
            eye.astype(np.float32),
            np.array([0.0, 0.0, 1.0], dtype=np.float32),
        )

    image = renderer.render_to_image()
    o3d.io.write_image(str(out_path), image)
    return out_path
