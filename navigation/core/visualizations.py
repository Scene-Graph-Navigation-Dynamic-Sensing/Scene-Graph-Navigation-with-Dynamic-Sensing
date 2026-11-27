"""Ported from Scene-Graph-Generation/dovsg/navigation/visualizations.py
(DoVSG), unchanged. Optional 3D Open3D viewer for a planned path -- opens a
blocking interactive window (`visualizer.run()`), so it only makes sense on
a machine with a display; navigation/core/path_facade.py only imports this
module when `pointcloud_visualization=True` is explicitly requested, never
by default. Not needed for headless entc runs, which use the 2D matplotlib
navigation_vis.jpg output instead.
"""
from __future__ import annotations

import math

import numpy as np
import open3d as o3d


def generate_colors(num_points):
    colors = []
    for i in range(num_points):
        t = i / (num_points - 1)
        color = [1 - t, 0, t]
        colors.append(color)
    return colors


def create_dashed_cylinder_line(points, radius=0.03, dash_length=0.07, gap_length=0.03):
    colors = generate_colors(len(points))
    geometries = []
    for i in range(len(points) - 1):
        start_point = np.array(points[i])
        end_point = np.array(points[i + 1])
        start_color = colors[i]
        end_color = colors[i + 1]
        vec = end_point - start_point
        seg_length = np.linalg.norm(vec)
        vec_normalized = vec / seg_length
        n_dashes = math.ceil(seg_length / (dash_length + gap_length))

        for j in range(n_dashes):
            new_dash_length = min(dash_length, seg_length - j * (dash_length + gap_length))
            dash_start = start_point + vec_normalized * j * (new_dash_length + gap_length)
            dash_end = dash_start + vec_normalized * new_dash_length
            cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=new_dash_length)
            cylinder.translate((dash_start + dash_end) / 2)

            z_axis = np.array([0, 0, 1])
            rotation_axis = np.cross(z_axis, vec)
            rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
            rotation_angle = np.arccos(np.dot(z_axis, vec_normalized))
            rotation_matrix = cylinder.get_rotation_matrix_from_axis_angle(rotation_axis * rotation_angle)
            cylinder.rotate(rotation_matrix, center=dash_start)

            t = j / n_dashes
            interpolated_color = (1 - t) * np.array(start_color) + t * np.array(end_color)
            cylinder.paint_uniform_color(interpolated_color)
            geometries.append(cylinder)

    return geometries


def visualize(pcd, path, end_xyz, min_height):
    if path is not None:
        path = np.array(np.array(path).tolist())
        start_point = path[0, :]
    end_point = np.array(end_xyz)

    if path is not None:
        path[:, 2] = min_height
        lines = create_dashed_cylinder_line(path)

    if path is not None:
        start_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
    end_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)

    if path is not None:
        start_sphere.translate(start_point)
    end_sphere.translate(end_point)

    if path is not None:
        start_sphere.paint_uniform_color([0, 1, 0])  # Green start
    end_sphere.paint_uniform_color([1, 0, 0])  # Red end

    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window(visible=True)
    if path is not None:
        geometries = [pcd, *lines, start_sphere, end_sphere]
    else:
        geometries = [pcd, end_sphere]
    visualizer.poll_events()
    visualizer.update_renderer()
    for geometry in geometries:
        visualizer.add_geometry(geometry)
    visualizer.run()
    visualizer.destroy_window()
