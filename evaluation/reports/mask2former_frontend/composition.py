"""What a scene-graph node's 3D points lie on: the same nearest-GT-point vote the evaluation uses for purity
(evaluation/open_vocab.match_nodes: each voxel takes the GT object of its nearest GT point within 5 cm), but
returning the whole breakdown instead of only the majority share.

    composition("panonly_r1", "results/option2_panoptic/run1", "room_2", {"shelf_0"})
    -> {"shelf_0": [("shelf", "countable", 0.460), ("wall", "structural", 0.423), ...]}
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

MAIN = Path(__file__).resolve().parents[3]  # repo root
GT_ROOT = Path("/home/ravindu/SceneGraphNav/datasets/replica/extracted")
for p in (MAIN, MAIN / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def composition(tag: str, root: str, scene: str, node_ids: set[str], match_dist_m: float = 0.05) -> dict:
    from scipy.spatial import cKDTree
    from object_fusion._dovsg_import import ensure_dovsg_importable
    ensure_dovsg_importable(yaml.safe_load(open(MAIN / "configs/pipeline.yaml"))["external"]["scene_graph_generation_root"])
    from evaluation.replica_gt import load_replica_gt
    from navigation.voxel_map import PersistedVoxelMap
    from scene_graph.storage.serializer import load_scene_graph

    sn = f"replica_{scene}_{tag}"
    payload = load_scene_graph(MAIN / root / scene / "scene_graph" / f"{sn}.pkl")
    vmap = PersistedVoxelMap.from_dict(payload["voxel_map"])
    T = np.array(json.loads((MAIN / "data/processed" / sn / "gravity_alignment.json").read_text())["transform"])
    gt = load_replica_gt(GT_ROOT / scene)
    tree = cKDTree(gt.xyz @ T[:3, :3].T + T[:3, 3])
    out = {}
    for obj in payload["instance_objects"]:
        if obj["class_id"] not in node_ids:
            continue
        pts = vmap.index_to_point(np.asarray(obj["indexes"])) + vmap.resolution / 2
        d, nn = tree.query(pts)
        v = Counter(int(i) for i in gt.object_id[nn[d <= match_dist_m]] if i != 0)
        n = sum(c for o, c in v.items() if gt.kind(o) != "ignored") or 1
        out[obj["class_id"]] = [(gt.label(o), gt.kind(o), c / n) for o, c in v.most_common() if gt.kind(o) != "ignored"]
    return out


if __name__ == "__main__":
    for k, v in composition("panonly_r1", "results/option2_panoptic/run1", "room_2", {"shelf_0"}).items():
        print(k, [(a, b, round(c, 3)) for a, b, c in v[:8]])
