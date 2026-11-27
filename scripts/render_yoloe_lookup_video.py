"""The scene_with_object_names.mp4 walkthrough, with every node named by YOLOE
nearest-noun lookup instead of NOVIC.

For each node of a finished Replica evaluation: its YOLOE object embedding is
averaged over all its detections (anchor matched to each stored box, as in
scripts/eval_yoloe_embedding_acc.py, now for every node, not only the scored
ones), and the node is named by the nearest of NOVIC's 42,919 nouns embedded
with YOLOE's text encoder (cached by scripts/eval_yoloe_open_vocab_lookup.py).

Video as evaluation/artifacts.write_video draws it: left, the nodes re-projected
with their detection boxes and names; right, the ground truth. Box colour for
nodes on one countable object: green = acceptable to the WordNet judge (exact
or more specific), orange = not; red / grey as before (mixed / not counted).

The per-pixel depth arrays of the evaluation runs were deleted after scoring,
so depth comes from the prepared Replica frames (data/raw/replica_<scene>/depth,
exact rendered depth) resized to the pipeline's frame size.

    python scripts/render_yoloe_lookup_video.py --scene room_0 --gpu 1
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
for p in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from evaluation import artifacts  # noqa: E402
from evaluation.naming_judge import grade  # noqa: E402
from evaluation.open_vocab import Node  # noqa: E402
from evaluation.replica_gt import load_replica_gt  # noqa: E402
from perception.image_enhance import enhance  # noqa: E402

GT_ROOT = Path("/home/ravindu/SceneGraphNav/datasets/replica/extracted")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", default="room_0")
    ap.add_argument("--results_root", default="results/replica_final")
    ap.add_argument("--gpu", default="1")
    ap.add_argument("--fps", type=int, default=5)
    args = ap.parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.gpu)
    import torch
    from eval_yoloe_embedding_acc import WEIGHTS, YOLOEEmbedder, class_list, iou

    rd = REPO_ROOT / args.results_root / args.scene
    cfg = yaml.safe_load(open(REPO_ROOT / "configs" / "pipeline.yaml"))
    from object_fusion._dovsg_import import ensure_dovsg_importable
    ensure_dovsg_importable(cfg["external"]["scene_graph_generation_root"])
    from navigation.voxel_map import PersistedVoxelMap
    from scene_graph.storage.serializer import load_scene_graph

    pkl = next((rd / "scene_graph").glob("*.pkl"))
    processed = REPO_ROOT / "data" / "processed" / pkl.stem
    raw = REPO_ROOT / "data" / "raw" / f"replica_{args.scene}"
    payload = load_scene_graph(pkl)
    vmap = PersistedVoxelMap.from_dict(payload["voxel_map"])
    status = {r["node_id"]: r for r in csv.DictReader(open(rd / "nodes.csv"))}

    # 1. every node's averaged YOLOE embedding
    run_cfg = yaml.safe_load(open(rd / "config_used.yaml"))
    width = int(run_cfg["source"]["target_width"])
    enhance_cfg = run_cfg["detection"].get("enhance")
    nodes, per_image = [], defaultdict(list)
    for obj in payload["instance_objects"]:
        nid = obj["class_id"]
        if nid not in status or len(obj.get("indexes", [])) == 0:
            continue
        dets = [(str(f), np.asarray(b, dtype=np.float64).ravel()) for f, b in zip(obj.get("image_name", []),
                                                                                  obj.get("xyxy", []))]
        nodes.append(Node(node_id=nid, name=nid, novic_top_k=[], embedding=np.zeros(1),
                          points=vmap.index_to_point(np.asarray(obj["indexes"])) + vmap.resolution / 2,
                          detections=dets))
        for f, b in dets:
            per_image[f].append((nid, b))
    emb = YOLOEEmbedder(WEIGHTS, 1024, class_list(args.scene))
    sums, counts = defaultdict(lambda: 0), defaultdict(int)
    for name, dets in sorted(per_image.items()):
        bgr = cv2.imread(str(raw / "rgb" / f"{name}.jpg"))
        h0, w0 = bgr.shape[:2]
        bgr = cv2.resize(bgr, (width, int(round(h0 * width / w0))), interpolation=cv2.INTER_AREA)
        e, boxes, conf = emb.image(enhance(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), enhance_cfg))
        for nid, box in dets:
            o = iou(box, boxes)
            fit = np.where(o >= o.max() - 0.05)[0]
            j = int(fit[np.argmax(conf[fit])])
            if o[j] >= 0.5:
                sums[nid] = sums[nid] + e[j]
                counts[nid] += 1
    del emb
    torch.cuda.empty_cache()

    # 2. nearest noun among NOVIC's nouns, YOLOE text embeddings (plain noun)
    nouns = [str(n) for n in np.load(REPO_ROOT / "data" / "cache" / "novic_target_text_embeddings.npz")["nouns"]]
    T = np.load(REPO_ROOT / "data" / "cache" / "yoloe26x_target_text_embeddings_plain.npz")["emb"].astype(np.float32)
    rows = []
    for n in nodes:
        s = status[n.node_id]
        n.status = s["status"]
        if counts[n.node_id] == 0:
            n.name = "(no YOLOE match)"
        else:
            v = sums[n.node_id] / np.linalg.norm(sums[n.node_id])
            n.name = nouns[int(np.argmax(T @ v))]
        g = grade(n.name, s["gt_class"]) if n.status in ("correct", "wrong_name") else ""
        if g:                                  # colour countable nodes by the WordNet judge
            n.status = "correct" if g in ("exact", "specific") else "wrong_name"
        rows.append({"node_id": n.node_id, "status": s["status"], "gt_class": s["gt_class"],
                     "novic_name": s["novic_name"], "yoloe_lookup_name": n.name, "grade": g,
                     "yoloe_detections": counts[n.node_id]})

    # 3. frames: pipeline RGB and poses (map frame), exact Replica depth resized to the frame size
    meta = json.loads((processed / "metadata.json").read_text())
    K = np.array(meta["K"], dtype=np.float64).reshape(3, 3)
    names = json.loads((processed / "capture.json").read_text())["frame_names"]
    frames = []
    for name in names:
        rgb = cv2.cvtColor(cv2.imread(str(processed / "rgb" / f"{name}.jpg")), cv2.COLOR_BGR2RGB)
        d = cv2.imread(str(raw / "depth" / f"{name}.png"), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
        d = cv2.resize(d, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
        frames.append({"name": name, "rgb": rgb, "K": K, "depth": d,
                       "c2w": np.loadtxt(processed / "poses" / f"{name}.txt")})

    # 4. ground truth in the map frame, then the video
    gt = load_replica_gt(GT_ROOT / args.scene)
    T_align = np.array(json.loads((processed / "gravity_alignment.json").read_text())["transform"])
    gt_xyz_map = gt.xyz @ T_align[:3, :3].T + T_align[:3, 3]
    colours = {n.node_id: c for n, c in zip(nodes, artifacts.palette(len(nodes)))}
    out = rd / "scene_with_yoloe_lookup_names.mp4"
    artifacts.write_video(out, frames, nodes, colours, gt_xyz_map, gt, fps=args.fps)
    with open(rd / "yoloe_lookup_all_nodes.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    scored = [r for r in rows if r["grade"]]
    acc = sum(r["grade"] in ("exact", "specific") for r in scored)
    print(f"{args.scene}: {len(nodes)} nodes named by YOLOE lookup; {acc}/{len(scored)} scored nodes acceptable")
    print(f"wrote {out}\nwrote {rd / 'yoloe_lookup_all_nodes.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
