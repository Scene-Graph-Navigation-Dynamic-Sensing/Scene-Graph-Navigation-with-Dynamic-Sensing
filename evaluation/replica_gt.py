"""Replica ground truth for open-vocabulary evaluation.

Each scene's habitat/mesh_semantic.ply stores an object_id on every face, and
habitat/info_semantic.json maps object ids to class names. The ground truth is
a set of points sampled uniformly over the mesh surface (about one per square
centimetre, plus every face's centroid), each carrying its face's object id.
Sampling by area rather than using the mesh vertices matters: flat objects
such as a TV screen are a few large faces with almost no vertices, while the
wall around them is densely tessellated, so nearest-vertex matching would hand
the screen to the wall.

Class handling, shared by every metric:
  * ignored      -- no usable label: "undefined", the anonymised text/picture
                    placeholders, and objects whose class is not in the class
                    list (class_id -1, e.g. "other-leaf"). Left out everywhere.
  * structural   -- wall, floor, ceiling. Kept for segmentation; left out of
                    the object metrics (a node on a wall is neither a hit nor
                    a false positive).
  * countable    -- everything else: the objects the object metrics count.

Visibility: an object the walkthrough never shows cannot be found, so the
object metrics only count objects with at least `min_visible_vertices`
surface samples visible (in front of the camera, inside the image, and within
`depth_tol_m` of the frame's depth, i.e. not occluded) in some frame.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

IGNORED_CLASSES = {"undefined", "anonymize text", "anonymize picture"}
STRUCTURAL_CLASSES = {"wall", "floor", "ceiling"}


def normalize(name: str) -> str:
    return name.replace("-", " ").replace("_", " ").strip().lower()


@dataclass
class ReplicaGT:
    scene: str
    xyz: np.ndarray                    # (N, 3) surface samples, mesh frame
    object_id: np.ndarray              # (N,) per sample, 0 = none
    objects: dict[int, dict]           # id -> {"class", "kind": countable|structural|ignored}
    vocabulary: list[str]              # normalised class names usable as labels
    visible_count: np.ndarray | None = None   # (N,) frames each sample is visible in
    visible_objects: set[int] = field(default_factory=set)

    def kind(self, oid: int) -> str:
        return self.objects.get(int(oid), {"kind": "ignored"})["kind"]

    def label(self, oid: int) -> str:
        return self.objects.get(int(oid), {"class": "none"})["class"]

    def countable_visible(self) -> list[int]:
        return sorted(o for o in self.visible_objects if self.kind(o) == "countable")


def sample_surface(verts: np.ndarray, faces: list, face_obj: np.ndarray, per_m2: float = 1e4,
                   seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Area-weighted points on the mesh (polygons fanned into triangles),
    plus each triangle's centroid so no face is left without a sample."""
    tris, tri_obj = [], []
    for k in range(3, max(len(f) for f in faces) + 1):
        sel = [i for i, f in enumerate(faces) if len(f) == k]
        if not sel:
            continue
        F = np.stack([faces[i] for i in sel]).astype(np.int64)
        for j in range(1, k - 1):
            tris.append(F[:, [0, j, j + 1]])
            tri_obj.append(face_obj[sel])
    T = np.concatenate(tris)
    obj = np.concatenate(tri_obj)
    a, b, c = verts[T[:, 0]], verts[T[:, 1]], verts[T[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    rng = np.random.default_rng(seed)
    n = rng.poisson(area * per_m2)
    rep = np.repeat(np.arange(len(T)), n)
    r1, r2 = rng.random(len(rep)), rng.random(len(rep))
    flip = r1 + r2 > 1
    r1[flip], r2[flip] = 1 - r1[flip], 1 - r2[flip]
    pts = a[rep] + r1[:, None] * (b[rep] - a[rep]) + r2[:, None] * (c[rep] - a[rep])
    cent = (a + b + c) / 3
    return np.concatenate([cent, pts]), np.concatenate([obj, obj[rep]])


def load_replica_gt(scene_dir: Path) -> ReplicaGT:
    from plyfile import PlyData

    ply = PlyData.read(str(scene_dir / "habitat" / "mesh_semantic.ply"))
    v = ply["vertex"]
    verts = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)
    faces = ply["face"]
    xyz, object_id = sample_surface(verts, list(faces["vertex_indices"]),
                                    np.asarray(faces["object_id"], dtype=np.int64))

    info = json.loads((scene_dir / "habitat" / "info_semantic.json").read_text())
    class_names = {c["id"]: normalize(c["name"]) for c in info["classes"]}
    objects = {}
    for o in info["objects"]:
        cls = normalize(o["class_name"])
        if o.get("class_id", -1) < 0 or cls in IGNORED_CLASSES or o["class_id"] not in class_names:
            kind = "ignored"
        elif cls in STRUCTURAL_CLASSES:
            kind = "structural"
        else:
            kind = "countable"
        objects[int(o["id"])] = {"class": cls, "kind": kind}
    vocabulary = sorted({n for n in class_names.values() if n not in IGNORED_CLASSES})
    return ReplicaGT(scene=scene_dir.name, xyz=xyz, object_id=object_id, objects=objects, vocabulary=vocabulary)


def compute_visibility(gt: ReplicaGT, frames, T_mesh_to_map: np.ndarray, depth_tol_m: float = 0.05,
                       min_visible_vertices: int = 50) -> None:
    """frames: iterable of (K 3x3, cam_to_map 4x4, depth HxW in metres, 0 = invalid).
    An object counts as visible with at least min_visible_vertices visible
    surface samples (about that many cm^2)."""
    pts = gt.xyz @ T_mesh_to_map[:3, :3].T + T_mesh_to_map[:3, 3]
    count = np.zeros(len(pts), dtype=np.int32)
    for K, c2w, depth in frames:
        z_img = project_depth_test(pts, K, c2w, depth, depth_tol_m)
        count += z_img
    gt.visible_count = count
    ids, n = np.unique(gt.object_id[count > 0], return_counts=True)
    gt.visible_objects = {int(i) for i, c in zip(ids, n) if i != 0 and c >= min_visible_vertices}


def project_depth_test(pts_map: np.ndarray, K: np.ndarray, c2w: np.ndarray, depth: np.ndarray,
                       tol: float) -> np.ndarray:
    """Boolean per point: in front of the camera, inside the image, and within
    tol of the frame's depth at its pixel (so not hidden behind something)."""
    w2c = np.linalg.inv(c2w)
    cam = pts_map @ w2c[:3, :3].T + w2c[:3, 3]
    z = cam[:, 2]
    ok = z > 0.05
    u = np.full(len(pts_map), -1, dtype=np.int64)
    v = np.full(len(pts_map), -1, dtype=np.int64)
    u[ok] = np.round(K[0, 0] * cam[ok, 0] / z[ok] + K[0, 2]).astype(np.int64)
    v[ok] = np.round(K[1, 1] * cam[ok, 1] / z[ok] + K[1, 2]).astype(np.int64)
    h, w = depth.shape
    ok &= (u >= 0) & (u < w) & (v >= 0) & (v < h)
    d = np.zeros(len(pts_map))
    d[ok] = depth[v[ok], u[ok]]
    ok &= (d > 0) & (np.abs(z - d) <= tol)
    return ok
