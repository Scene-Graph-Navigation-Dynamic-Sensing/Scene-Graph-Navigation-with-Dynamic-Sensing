"""Does the YOLOE object embedding carry class meaning? (Is a YOLOE-embedding
decoder worth training instead of NOVIC?)

For the nodes of a finished Replica evaluation that matched one countable GT
object cleanly (the nodes Acc@k is scored on), HOV-SG style:

  YOLOE   every detection the node was built from is looked up again in
          YOLOE-26x's text-prompt model (yoloe-26x-seg.pt, same weights family
          as the prompt-free detector the pipeline ran): the anchor whose
          decoded box best overlaps the stored box gives the detection's object
          embedding (the class-branch feature after the contrastive head's
          BatchNorm, i.e. what YOLOE compares with text). Embeddings are
          L2-normalised, averaged over the node, normalised again, and the
          scene's Replica classes are ranked by cosine against YOLOE's own text
          embeddings of the class names (MobileCLIP2 + YOLOE's RepRTA).
  CLIP    the node's stored NOVIC image embedding (DFN5B CLIP ViT-H-14) ranked
          against the same classes with "an image of {}" -- the existing
          Acc@k, read from nodes.csv.

The prompt-free model cannot be used directly: its class branch is fused into
a fixed 4,585-word vocabulary layer, so its embedding can't be compared with
new text. Frames get the same brightness correction the detector saw.

    python scripts/eval_yoloe_embedding_acc.py results/replica_final/office_0 results/replica_final/room_0
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
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.replica_gt import IGNORED_CLASSES, normalize  # noqa: E402
from perception.image_enhance import enhance  # noqa: E402

GT_ROOT = Path("/home/ravindu/SceneGraphNav/datasets/replica/extracted")
WEIGHTS = REPO_ROOT / "external" / "Scene-Graph-Generation" / "yoloe-26x-seg.pt"
KS = (1, 3, 5, 10)


def class_list(scene: str) -> list[str]:
    info = json.loads((GT_ROOT / scene / "habitat" / "info_semantic.json").read_text())
    return sorted({normalize(c["name"]) for c in info["classes"]} - IGNORED_CLASSES)


def iou(box, boxes):
    x1 = np.maximum(box[0], boxes[:, 0]); y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2]); y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    a = (box[2] - box[0]) * (box[3] - box[1])
    b = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / (a + b - inter + 1e-9)


class YOLOEEmbedder:
    """Per-anchor object embeddings and decoded boxes of the text-prompt YOLOE model."""

    def __init__(self, weights: Path, img_size: int, classes: list[str]):
        import torch
        from ultralytics import YOLOE
        self.torch, self.img_size = torch, img_size
        self.yoloe = YOLOE(str(weights))
        self.yoloe.set_classes(classes)                         # the forward pass needs a text prompt
        self.net = self.yoloe.model.cuda().eval().float()
        self.head = self.net.model[-1]
        self.cls = getattr(self.head, "one2one_cv3", None) if self.head.end2end else self.head.cv3
        self.box = getattr(self.head, "one2one_cv2", None) if self.head.end2end else self.head.cv2
        self.con = getattr(self.head, "one2one_cv4", None) if self.head.end2end else self.head.cv4
        self._feats = None
        self.head.register_forward_hook(lambda m, inp, out: setattr(self, "_feats", inp[0]))

    def text(self, names: list[str]) -> np.ndarray:
        pe = self.yoloe.get_text_pe(names)                      # (1, K, D), RepRTA applied
        pe = pe[0].float().cpu().numpy()
        return pe / np.linalg.norm(pe, axis=1, keepdims=True)

    def image(self, rgb: np.ndarray):
        """-> (N, D) unit embeddings, (N, 4) xyxy boxes in original-image pixels, (N,) best class score."""
        torch = self.torch
        h, w = rgb.shape[:2]
        r = self.img_size / max(h, w)
        nh, nw = int(round(h * r)), int(round(w * r))
        canvas = np.full((self.img_size, self.img_size, 3), 114, np.uint8)
        top, left = (self.img_size - nh) // 2, (self.img_size - nw) // 2
        canvas[top:top + nh, left:left + nw] = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
        t = torch.from_numpy(canvas).permute(2, 0, 1)[None].float().cuda() / 255.0
        with torch.no_grad():
            out = self.net(t)
            x = self._feats
            scores = (out[1]["one2one"] if self.head.end2end else out[1])["scores"][0].max(0).values
            emb = torch.cat([self.con[i].norm(self.cls[i](x[i])).flatten(2) for i in range(self.head.nl)], 2)[0]
            boxes = torch.cat([self.box[i](x[i]).view(1, 4 * self.head.reg_max, -1) for i in range(self.head.nl)], 2)
            dbox = self.head._get_decode_boxes({"boxes": boxes, "feats": list(x[:3])})[0]   # (4, N)
        emb = torch.nn.functional.normalize(emb.T, dim=1).cpu().numpy()
        d = dbox.T.cpu().numpy()
        if not self.head.end2end and not getattr(self.head, "xyxy", False):              # xywh -> xyxy
            d = np.concatenate([d[:, :2] - d[:, 2:] / 2, d[:, :2] + d[:, 2:] / 2], 1)
        d[:, [0, 2]] = (d[:, [0, 2]] - left) / r
        d[:, [1, 3]] = (d[:, [1, 3]] - top) / r
        return emb, d, scores.sigmoid().cpu().numpy()


def acc(ranks: np.ndarray) -> dict:
    return {k: float((ranks <= k).mean()) for k in KS}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("result_dir", nargs="+")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--img_size", type=int, default=1024)
    ap.add_argument("--min_iou", type=float, default=0.5, help="drop a detection whose best anchor overlaps less")
    ap.add_argument("--iou_slack", type=float, default=0.05,
                    help="anchors within this IoU of the best fit compete on confidence")
    args = ap.parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.gpu)

    cfg = yaml.safe_load(open(REPO_ROOT / "configs" / "pipeline.yaml"))
    from object_fusion._dovsg_import import ensure_dovsg_importable
    ensure_dovsg_importable(cfg["external"]["scene_graph_generation_root"])
    from scene_graph.storage.serializer import load_scene_graph
    enhance_cfg = yaml.safe_load(open(REPO_ROOT / "configs" / "replica_variants" / "brightness.yaml"))["detection"]["enhance"]

    summary = {}
    for rd in map(Path, args.result_dir):
        metrics = json.loads((rd / "metrics.json").read_text())
        scene = metrics["info"]["scene"]
        classes = class_list(scene)
        index = {c: i for i, c in enumerate(classes)}
        rows = [r for r in csv.DictReader(open(rd / "nodes.csv"))
                if r["status"] in ("correct", "wrong_name") and int(r["gt_rank_in_embedding"] or 0) > 0]
        objs = {o["class_id"]: o for o in load_scene_graph(next((rd / "scene_graph").glob("*.pkl")))["instance_objects"]}
        per_image = defaultdict(list)                       # frame -> [(node, xyxy)]
        for r in rows:
            o = objs[r["node_id"]]
            for name, box in zip(o.get("image_name", []), o.get("xyxy", [])):
                per_image[str(name)].append((r["node_id"], np.asarray(box, dtype=np.float64).ravel()))

        emb = YOLOEEmbedder(WEIGHTS, args.img_size, classes)
        text = {"plain": emb.text(classes), "an image of": emb.text([f"an image of {c}" for c in classes])}
        raw = REPO_ROOT / "data" / "raw" / f"replica_{scene}" / "rgb"
        width = int(yaml.safe_load(open(rd / "config_used.yaml"))["source"]["target_width"])
        sums, counts, ious = defaultdict(lambda: 0), defaultdict(int), []
        for name, dets in sorted(per_image.items()):
            bgr = cv2.imread(str(raw / f"{name}.jpg"))
            # the pipeline resized frames to source.target_width before detecting, so the stored boxes
            # are in that image's pixels
            h0, w0 = bgr.shape[:2]
            bgr = cv2.resize(bgr, (width, int(round(h0 * width / w0))), interpolation=cv2.INTER_AREA)
            rgb = enhance(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), enhance_cfg)
            e, boxes, conf = emb.image(rgb)
            for nid, box in dets:
                o = iou(box, boxes)
                # many neighbouring anchors predict nearly the same box with different classes; YOLOE
                # reports the most confident one, so take that among the anchors that fit the box
                fit = np.where(o >= o.max() - args.iou_slack)[0]
                j = int(fit[np.argmax(conf[fit])])
                ious.append(float(o[j]))
                if o[j] >= args.min_iou:
                    sums[nid] = sums[nid] + e[j]
                    counts[nid] += 1

        out_rows, res = [], {"nodes": 0, "detections_matched": int(sum(counts.values())),
                             "detections_total": len(ious), "median_anchor_iou": float(np.median(ious))}
        ranks = {k: [] for k in text}
        clip_ranks = []
        for r in rows:
            if counts[r["node_id"]] == 0 or r["gt_class"] not in index:
                continue
            v = sums[r["node_id"]] / np.linalg.norm(sums[r["node_id"]])
            row = {"node_id": r["node_id"], "gt_class": r["gt_class"], "detections": counts[r["node_id"]],
                   "clip_rank": int(r["gt_rank_in_embedding"]), "clip_top1": r["embedding_class"]}
            for k, t in text.items():
                order = np.argsort(-(t @ v))
                rank = int(np.where(order == index[r["gt_class"]])[0][0]) + 1
                ranks[k].append(rank)
                row[f"yoloe_{k.replace(' ', '_')}_rank"] = rank
                row[f"yoloe_{k.replace(' ', '_')}_top1"] = classes[order[0]]
            clip_ranks.append(row["clip_rank"])
            out_rows.append(row)
        # averaged YOLOE embedding per node, for other tests (scripts/eval_yoloe_open_vocab_lookup.py)
        ids = [r["node_id"] for r in rows if counts[r["node_id"]] > 0]
        np.savez(rd / "yoloe_node_embeddings.npz", node_ids=np.array(ids),
                 emb=np.stack([sums[i] / np.linalg.norm(sums[i]) for i in ids]).astype(np.float32),
                 detections=np.array([counts[i] for i in ids]))
        res["nodes"] = len(out_rows)
        res["classes"] = len(classes)
        res["clip"] = acc(np.array(clip_ranks))
        res["yoloe"] = {k: acc(np.array(v)) for k, v in ranks.items()}
        res["ranks"] = {"clip": clip_ranks, **{f"yoloe_{k}": v for k, v in ranks.items()}}
        with open(rd / "yoloe_embedding_acc.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
            w.writeheader(); w.writerows(out_rows)
        (rd / "yoloe_embedding_acc.json").write_text(json.dumps(res, indent=1))
        summary[scene] = res
        print(f"\n{scene}: {res['nodes']} nodes, {res['detections_matched']}/{res['detections_total']} detections "
              f"matched to an anchor (median IoU {res['median_anchor_iou']:.2f}), {len(classes)} classes")
        print("  " + " | ".join(f"Acc@{k}" for k in KS))
        print(f"  CLIP (NOVIC DFN5B)          " + "  ".join(f"{res['clip'][k]:.1%}" for k in KS))
        for k in text:
            print(f"  YOLOE ({k:12s})        " + "  ".join(f"{res['yoloe'][k][kk]:.1%}" for kk in KS))
    return 0


if __name__ == "__main__":
    sys.exit(main())
