#!/usr/bin/env python3
"""Stage 1 of the relation comparison (scene_graph env): per 3RScan scan, with GROUND-TRUTH objects,
  - an object-id image per frame: the instance mesh ray-cast from the frame's pose (exact occlusion), rotated
    upright (3RScan's Tango frames are stored sideways; the rotation puts world-down at the image bottom),
    plus the rotated RGB,
  - the voxel-geometry relations: scene_graph/builder/dovsg_relations.py on each GT object's mesh points
    (floor shifted to z = 0, as the pipeline's gravity alignment does),
  - the 3DSSG ground-truth relations.

    python evaluation/relations/prepare_3rscan.py --scans 30 --out ../results/relations_3dssg
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import yaml
from plyfile import PlyData

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
DATA = Path("/home/ravindu/SceneGraphNav/datasets")
SCANS = DATA / "3rscan" / "3rscan_data"
SSG = DATA / "3dssg"
STRUCT = {"floor", "wall", "ceiling"}
log = logging.getLogger("prepare")


def read_info(seq: Path) -> dict:
    info = {}
    for line in (seq / "_info.txt").read_text().splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            info[k.strip()] = v.strip()
    K = np.array(info["m_calibrationColorIntrinsic"].split(), float).reshape(4, 4)[:3, :3]
    return {"K": K, "W": int(info["m_colorWidth"]), "H": int(info["m_colorHeight"])}


def upright_k(c2w: np.ndarray) -> int:
    """Quarter turns (np.rot90 k, counter-clockwise) that put world-down (-z) at the image bottom."""
    d = c2w[:3, :3].T @ np.array([0.0, 0.0, -1.0])        # world-down in camera coords (x right, y down)
    v, best, best_k = np.array([d[0], d[1]]), -9.0, 0
    for k in range(4):
        if v[1] > best:
            best, best_k = v[1], k
        v = np.array([v[1], -v[0]])                          # effect of one CCW quarter turn on a vector
    return best_k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scans", type=int, default=30)
    ap.add_argument("--frame_stride", type=int, default=2)
    ap.add_argument("--out", type=Path, default=Path("/home/ravindu/SceneGraphNav/results/relations_3dssg"))
    ap.add_argument("--config", default=str(REPO / "configs" / "pipeline.yaml"))
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    for name in ("dovsg_relations", "scene_graph.builder.dovsg_relations"):
        logging.getLogger(name).setLevel(logging.WARNING)

    cfg = yaml.safe_load(open(args.config))
    sg = cfg["scene_graph"]
    from scene_graph.builder.dovsg_relations import build_scene_graph_dovsg

    rel = {s["scan"]: s for s in json.load(open(SSG / "relationships.json"))["scans"]}
    objs_all = {s["scan"]: {int(o["id"]): o["label"] for o in s["objects"]}
                for s in json.load(open(SSG / "objects.json"))["scans"]}
    # deterministic choice: scans with labels, frames (extracted or zipped) and a mesh, sorted by id
    cand = sorted(s for s in rel if s in objs_all and (SCANS / s / "labels.instances.annotated.v2.ply").exists()
                  and ((SCANS / s / "sequence" / "_info.txt").exists() or (SCANS / s / "sequence.zip").exists()))
    chosen = cand[:args.scans]
    log.info("%d candidate scans, using %d", len(cand), len(chosen))
    args.out.mkdir(parents=True, exist_ok=True)

    for si, scan in enumerate(chosen):
        out = args.out / scan
        if (out / "meta.json").exists():
            log.info("[%d/%d] %s: done already", si + 1, len(chosen), scan)
            continue
        sdir = SCANS / scan
        seq = sdir / "sequence"
        if not (seq / "_info.txt").exists():
            with zipfile.ZipFile(sdir / "sequence.zip") as z:
                z.extractall(seq)
        info = read_info(seq)

        ply = PlyData.read(str(sdir / "labels.instances.annotated.v2.ply"))
        v = ply["vertex"]
        xyz = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
        vid = np.asarray(v["objectId"], np.int64)
        faces = np.vstack(ply["face"]["vertex_indices"]).astype(np.int64)
        labels = objs_all[scan]
        face_obj = vid[faces[:, 0]]

        mesh = o3d.t.geometry.TriangleMesh()
        mesh.vertex.positions = o3d.core.Tensor(xyz.astype(np.float32))
        mesh.triangle.indices = o3d.core.Tensor(faces.astype(np.int32))
        rc = o3d.t.geometry.RaycastingScene()
        rc.add_triangles(mesh)

        (out / "frames").mkdir(parents=True, exist_ok=True)
        frames = []
        names = sorted(p.name.split(".")[0] for p in seq.glob("frame-*.pose.txt"))[::args.frame_stride]
        for name in names:
            c2w = np.loadtxt(seq / f"{name}.pose.txt")
            if not np.isfinite(c2w).all():
                continue
            rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
                o3d.core.Tensor(info["K"]), o3d.core.Tensor(np.linalg.inv(c2w)), info["W"], info["H"])
            hit = rc.cast_rays(rays)
            prim = hit["primitive_ids"].numpy().astype(np.int64)
            ids = np.zeros(prim.shape, np.uint16)
            ok = prim != o3d.t.geometry.RaycastingScene.INVALID_ID
            ids[ok] = face_obj[prim[ok]]
            k = upright_k(c2w)
            ids = np.ascontiguousarray(np.rot90(ids, k))
            rgb = cv2.imread(str(seq / f"{name}.color.jpg"))
            rgb = np.ascontiguousarray(np.rot90(rgb, k))
            cv2.imwrite(str(out / "frames" / f"{name}.jpg"), rgb, [cv2.IMWRITE_JPEG_QUALITY, 92])
            cv2.imwrite(str(out / "frames" / f"{name}.ids.png"), ids)
            frames.append({"name": name, "rot_k": k})

        # ── voxel-geometry relations on the GT objects ──
        floor_pts = xyz[np.isin(vid, [i for i, l in labels.items() if l == "floor"])]
        z0 = np.percentile(floor_pts[:, 2], 2) if len(floor_pts) > 100 else np.percentile(xyz[:, 2], 1)
        pts = xyz - np.array([0.0, 0.0, z0])
        inst = []
        for oid, lab in labels.items():
            if lab in STRUCT:
                continue
            idx = np.nonzero(vid == oid)[0]
            if len(idx) >= 20:
                inst.append({"class_id": str(oid), "class_name": lab, "indexes": idx})

        class _VD:                                            # view_dataset stand-in: indexes -> points
            @staticmethod
            def index_to_point(i):
                return pts[np.asarray(i)]

        graph = build_scene_graph_dovsg(
            _VD(), inst, cfg["external"]["scene_graph_generation_root"], resolution=sg["resolution"],
            stand_floor_threshold=sg["stand_floor_threshold"], alphashape_alpha=sg["alphashape_alpha"],
            alpha_shape_overleaf_rate_threshold=sg["alpha_shape_overleaf_rate_threshold"],
            inside_threshold=sg["inside_threshold"], part_level_classes=sg.get("part_level_classes"),
            support_max_gap_m=sg["support_max_gap_m"], support_max_penetration_m=sg["support_max_penetration_m"],
            next_to_max_gap_m=sg["next_to_max_gap_m"], next_to_min_vertical_overlap=sg["next_to_min_vertical_overlap"])
        vox = []
        for cid, node in graph.object_nodes.items():
            if node.parent is None:
                continue
            parent = "floor" if node.parent.node_id == "floor_0" else node.parent.node_id
            vox.append([cid, parent, node.parent_relation])
        vox_next = [[e["a"], e["b"], "next to"] for e in getattr(graph, "next_to", [])]

        gt = [[str(s), str(o), p] for s, o, _, p in rel[scan]["relationships"]]
        meta = {"scan": scan, "K": info["K"].tolist(), "W": info["W"], "H": info["H"],
                "objects": {str(k): l for k, l in labels.items()}, "frames": frames,
                "voxel_tree": vox, "voxel_next_to": vox_next, "gt": gt}
        (out / "meta.json").write_text(json.dumps(meta))
        log.info("[%d/%d] %s: %d frames, %d objects (%d non-structural), %d voxel edges, %d GT relations",
                 si + 1, len(chosen), scan, len(frames), len(labels), len(inst), len(vox), len(gt))


if __name__ == "__main__":
    main()
