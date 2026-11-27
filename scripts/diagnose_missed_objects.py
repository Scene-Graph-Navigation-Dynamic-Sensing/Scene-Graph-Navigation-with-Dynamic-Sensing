"""Why did the scene graph miss these Replica objects?

For every visible, countable GT object that no node matched, measure how much
of its visible surface lies within --dist of *any* node's voxels:

  absorbed   >= 50% covered: the object is in the map, inside another node
             (over-merging, or a mask that spilled over it)
  partial    10-50% covered: a fragment of it is in some node
  absent     < 10% covered: it never reached the map (not detected, too few
             detections, dropped by a filter or the denoiser)

Reads a finished evaluation folder (results/.../<scene>/, whose scene_graph/
holds the .pkl) plus the prepared frames in data/raw/ for visibility, so it
works on runs whose per-pixel arrays were already trimmed.

    python scripts/diagnose_missed_objects.py results/replica_tuning/fixes_x_bf/office_0
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.replica_gt import compute_visibility, load_replica_gt  # noqa: E402

GT_ROOT = Path("/home/ravindu/SceneGraphNav/datasets/replica/extracted")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("result_dir", nargs="+")
    ap.add_argument("--dist", type=float, default=0.03)
    ap.add_argument("--frame_step", type=int, default=2, help="use every Nth prepared frame for visibility")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(REPO_ROOT / "configs" / "pipeline.yaml"))
    from object_fusion._dovsg_import import ensure_dovsg_importable
    ensure_dovsg_importable(cfg["external"]["scene_graph_generation_root"])
    from scipy.spatial import cKDTree
    from navigation.voxel_map import PersistedVoxelMap
    from scene_graph.storage.serializer import load_scene_graph

    for rd in map(Path, args.result_dir):
        m = json.loads((rd / "metrics.json").read_text())
        scene = m["info"]["scene"]
        stride = m["info"].get("stride") or 10
        raw = REPO_ROOT / "data" / "raw" / (f"replica_{scene}" + ("" if stride == 10 else f"_s{stride}"))
        pkl = next((rd / "scene_graph").glob("*.pkl"))
        tag = pkl.stem.replace(f"replica_{scene}", "").lstrip("_")
        processed = REPO_ROOT / "data" / "processed" / pkl.stem
        T_align = np.array(json.loads((processed / "gravity_alignment.json").read_text())["transform"])
        T_inv = np.linalg.inv(T_align)

        payload = load_scene_graph(pkl)
        vmap = PersistedVoxelMap.from_dict(payload["voxel_map"])
        node_pts, node_ids = [], []
        for obj in payload["instance_objects"]:
            if obj["class_id"] in payload["node_embeddings"] and len(obj.get("indexes", [])) > 0:
                p = vmap.index_to_point(np.asarray(obj["indexes"])) + vmap.resolution / 2
                node_pts.append(p @ T_inv[:3, :3].T + T_inv[:3, 3])          # into the mesh frame
                node_ids.append(obj["class_id"])
        lab = np.concatenate([np.full(len(p), i) for i, p in enumerate(node_pts)])
        tree = cKDTree(np.concatenate(node_pts))

        gt = load_replica_gt(GT_ROOT / scene)
        c = yaml.safe_load(open(raw / "camera.yaml"))
        K = np.array([[c["fx"], 0, c["cx"]], [0, c["fy"], c["cy"]], [0, 0, 1]])
        names = sorted(p.stem for p in (raw / "rgb").glob("*.jpg"))[:: args.frame_step]
        frames = [(K, np.loadtxt(raw / "poses" / f"{n}.txt"),
                   cv2.imread(str(raw / "depth" / f"{n}.png"), cv2.IMREAD_UNCHANGED) / 1000.0) for n in names]
        compute_visibility(gt, frames, np.eye(4))

        found = set()
        with open(rd / "gt_objects.csv") as f:
            for r in csv.DictReader(f):
                if r["found"] == "True":
                    found.add(int(r["gt_object_id"]))
        rows = []
        for oid in gt.countable_visible():
            if oid in found:
                continue
            sel = (gt.object_id == oid) & (gt.visible_count > 0)
            d, nn = tree.query(gt.xyz[sel])
            near = d <= args.dist
            cover = float(near.mean())
            top = Counter(lab[nn[near]].tolist()).most_common(1)
            kind = "absorbed" if cover >= 0.5 else "partial" if cover >= 0.1 else "absent"
            rows.append({"gt_object_id": oid, "class": gt.label(oid), "visible_samples": int(sel.sum()),
                         "coverage": round(cover, 3), "kind": kind,
                         "covering_node": node_ids[top[0][0]] if top else ""})
        out = rd / "missed_objects.csv"
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["gt_object_id"])
            w.writeheader()
            w.writerows(rows)
        counts = Counter(r["kind"] for r in rows)
        print(f"\n{scene} [{tag or 'baseline'}]: {len(found)} found, {len(rows)} missed -> "
              + ", ".join(f"{k} {counts.get(k, 0)}" for k in ("absorbed", "partial", "absent")))
        for k in ("absorbed", "partial", "absent"):
            items = [r for r in rows if r["kind"] == k]
            if items:
                print(f"  {k:9s}: " + ", ".join(
                    f"{r['class']}({r['coverage']:.2f}{'→' + r['covering_node'] if r['covering_node'] and k != 'absent' else ''})"
                    for r in sorted(items, key=lambda r: -r["coverage"])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
