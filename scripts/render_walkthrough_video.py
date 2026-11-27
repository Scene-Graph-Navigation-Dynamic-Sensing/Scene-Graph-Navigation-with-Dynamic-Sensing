"""Render scene_with_object_names.mp4 for a finished Replica evaluation whose per-pixel arrays were already
trimmed (or that ran before the video was added): the nodes with their pipeline names and status colours
next to the ground truth, as evaluation/artifacts.write_video draws it. Depth comes from the prepared
Replica frames (data/raw/replica_<scene>/depth, exact rendered depth) resized to the pipeline frame size.

    python scripts/render_walkthrough_video.py results/replica_disc2/office_0 results/replica_disc2/room_0
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation import artifacts  # noqa: E402
from evaluation.open_vocab import Node  # noqa: E402
from evaluation.replica_gt import load_replica_gt  # noqa: E402

GT_ROOT = Path("/home/ravindu/SceneGraphNav/datasets/replica/extracted")


def render(rd: Path, fps: int = 5) -> Path:
    cfg = yaml.safe_load(open(REPO_ROOT / "configs" / "pipeline.yaml"))
    from object_fusion._dovsg_import import ensure_dovsg_importable
    ensure_dovsg_importable(cfg["external"]["scene_graph_generation_root"])
    from navigation.voxel_map import PersistedVoxelMap
    from scene_graph.storage.serializer import load_scene_graph

    scene = json.loads((rd / "metrics.json").read_text())["info"]["scene"]
    pkl = next((rd / "scene_graph").glob("*.pkl"))
    processed = REPO_ROOT / "data" / "processed" / pkl.stem
    raw = REPO_ROOT / "data" / "raw" / f"replica_{scene}"
    status = {r["node_id"]: r for r in csv.DictReader(open(rd / "nodes.csv"))}
    payload = load_scene_graph(pkl)
    vmap = PersistedVoxelMap.from_dict(payload["voxel_map"])
    nodes = []
    for o in payload["instance_objects"]:
        nid = o["class_id"]
        if nid not in status or len(o.get("indexes", [])) == 0:
            continue
        n = Node(node_id=nid, name=status[nid]["novic_name"], novic_top_k=[], embedding=np.zeros(1),
                 points=vmap.index_to_point(np.asarray(o["indexes"])) + vmap.resolution / 2,
                 detections=[(str(f), np.asarray(b, np.float64).ravel())
                             for f, b in zip(o.get("image_name", []), o.get("xyxy", []))])
        n.status = status[nid]["status"]
        nodes.append(n)
    meta = json.loads((processed / "metadata.json").read_text())
    K = np.array(meta["K"], dtype=np.float64).reshape(3, 3)
    frames = []
    for name in json.loads((processed / "capture.json").read_text())["frame_names"]:
        rgb = cv2.cvtColor(cv2.imread(str(processed / "rgb" / f"{name}.jpg")), cv2.COLOR_BGR2RGB)
        d = cv2.imread(str(raw / "depth" / f"{name}.png"), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
        frames.append({"name": name, "rgb": rgb, "K": K, "c2w": np.loadtxt(processed / "poses" / f"{name}.txt"),
                       "depth": cv2.resize(d, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)})
    gt = load_replica_gt(GT_ROOT / scene)
    T = np.array(json.loads((processed / "gravity_alignment.json").read_text())["transform"])
    colours = {n.node_id: c for n, c in zip(nodes, artifacts.palette(len(nodes)))}
    out = rd / "scene_with_object_names.mp4"
    artifacts.write_video(out, frames, nodes, colours, gt.xyz @ T[:3, :3].T + T[:3, 3], gt, fps=fps)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("result_dir", nargs="+")
    args = ap.parse_args()
    for rd in args.result_dir:
        print(f"wrote {render(Path(rd).resolve())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
