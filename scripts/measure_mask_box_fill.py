"""Do a detection's mask and box agree? Measures, for every detection behind the
nodes of a finished Replica evaluation, how much of its box the mask fills,
to find a threshold that separates real objects from group / area detections.

The scene graph stores each detection's frame and box but not its mask, so
the pipeline's own tracked detector (same weights, confidence, input size,
tracker and brightness correction, read from the run's config_used.yaml) is
replayed over the frames in order, and every stored box is matched to the
detection with the same box (IoU >= 0.8).

Per detection:
  fill   = mask pixels inside the box / box area
  slack  = 1 - (area of the mask's own bounding rectangle / box area)
           (how much of the box the mask doesn't reach at all)

Grouped by the status of the node the detection fed (nodes.csv): one real
object (correct / wrong_name), mixed (spans several objects), structural
(wall / floor / ceiling), unlabelled. Also reported: the share of each group's
detections a fill threshold would drop, and the real objects (by GT class)
that would lose most detections.

    python scripts/measure_mask_box_fill.py --scenes room_0 office_0 --gpu 1
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

from perception.image_enhance import enhance  # noqa: E402

GROUPS = {"correct": "one real object", "wrong_name": "one real object", "mixed": "mixed",
          "structural": "structural", "unlabelled": "unlabelled", "phantom": "phantom"}
THRESHOLDS = (0.2, 0.3, 0.4, 0.5, 0.6)


def iou(box, boxes):
    x1 = np.maximum(box[0], boxes[:, 0]); y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2]); y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    a = (box[2] - box[0]) * (box[3] - box[1])
    b = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / (a + b - inter + 1e-9)


def fill_and_slack(box, mask):
    h, w = mask.shape
    x1, y1 = max(int(np.floor(box[0])), 0), max(int(np.floor(box[1])), 0)
    x2, y2 = min(int(np.ceil(box[2])), w), min(int(np.ceil(box[3])), h)
    area = max((x2 - x1) * (y2 - y1), 1)
    inside = mask[y1:y2, x1:x2]
    if not inside.any():
        return 0.0, 1.0, 0.0
    ys, xs = np.nonzero(inside)
    tight = (xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1)
    return float(inside.sum() / area), float(1 - tight / area), float(area / (w * h))


def measure(scene: str, rd: Path, gpu_ok: bool) -> list[dict]:
    cfg = yaml.safe_load(open(rd / "config_used.yaml"))
    det_cfg = cfg["detection"]
    from object_fusion._dovsg_import import ensure_dovsg_importable
    ensure_dovsg_importable(cfg["external"]["scene_graph_generation_root"])
    from perception.detection.tracked_yoloe_detector import TrackedYOLOEDetector
    from scene_graph.storage.serializer import load_scene_graph

    status = {r["node_id"]: r for r in csv.DictReader(open(rd / "nodes.csv"))}
    pkl = next((rd / "scene_graph").glob("*.pkl"))
    stored = defaultdict(list)                       # frame -> [(node, box)]
    for o in load_scene_graph(pkl)["instance_objects"]:
        if o["class_id"] in status:
            for f, b in zip(o.get("image_name", []), o.get("xyxy", [])):
                stored[str(f)].append((o["class_id"], np.asarray(b, np.float64).ravel()))

    detector = TrackedYOLOEDetector(weights_path=det_cfg["weights"], conf_thresh=det_cfg["conf_thresh"],
                                    iou_thresh=det_cfg["iou_thresh"], img_size=det_cfg["img_size"],
                                    device="cuda" if gpu_ok else "cpu", tracker=det_cfg.get("tracker", "botsort.yaml"))
    raw = REPO_ROOT / "data" / "raw" / f"replica_{scene}" / "rgb"
    width = int(cfg["source"]["target_width"])
    names = json.loads((REPO_ROOT / "data" / "processed" / pkl.stem / "capture.json").read_text())["frame_names"]
    rows = []
    for name in names:                               # in order: the tracker needs the sequence
        bgr = cv2.imread(str(raw / f"{name}.jpg"))
        h0, w0 = bgr.shape[:2]
        bgr = cv2.resize(bgr, (width, int(round(h0 * width / w0))), interpolation=cv2.INTER_AREA)
        dets = detector.track(enhance(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), det_cfg.get("enhance")))
        if not dets or name not in stored:
            continue
        boxes = np.stack([d.bbox for d in dets])
        for nid, box in stored[name]:
            o = iou(box, boxes)
            j = int(np.argmax(o))
            s = status[nid]
            row = {"scene": scene, "frame": name, "node_id": nid, "status": s["status"],
                   "group": GROUPS.get(s["status"], s["status"]), "gt_class": s["gt_class"],
                   "node_voxels": int(s["voxels"]), "node_purity": float(s["purity"]), "match_iou": float(o[j])}
            if o[j] >= 0.8:
                row["fill"], row["slack"], row["box_frac_of_image"] = fill_and_slack(box, dets[j].mask)
            rows.append(row)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenes", nargs="+", default=["room_0", "office_0"])
    ap.add_argument("--results_root", default="results/replica_final")
    ap.add_argument("--out", default="results/mask_box_fill")
    ap.add_argument("--gpu", default="1")
    args = ap.parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.gpu)
    import torch
    out = REPO_ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for s in args.scenes:
        r = measure(s, REPO_ROOT / args.results_root / s, torch.cuda.is_available())
        matched = [x for x in r if "fill" in x]
        print(f"{s}: {len(matched)}/{len(r)} stored detections matched to a replayed detection (IoU >= 0.8)")
        rows += r
    with open(out / "detections.csv", "w", newline="") as f:
        keys = sorted({k for r in rows for k in r}, key=lambda k: list(rows[0]).index(k) if k in rows[0] else 99)
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    m = [r for r in rows if "fill" in r]
    L = [f"# Mask vs box fill: {', '.join(args.scenes)}", "",
         f"{len(m)} of {len(rows)} stored detections matched (IoU >= 0.8). fill = mask pixels / box area; "
         "slack = share of the box the mask's own bounding rectangle doesn't reach.", "",
         "| Node type | Detections | Fill p10 | Fill median | Fill p90 | Slack median |", "|---|---|---|---|---|---|"]
    groups = ["one real object", "mixed", "structural", "unlabelled"]
    for g in groups:
        f = np.array([r["fill"] for r in m if r["group"] == g])
        sl = np.array([r["slack"] for r in m if r["group"] == g])
        if len(f):
            L.append(f"| {g} | {len(f)} | {np.percentile(f, 10):.2f} | {np.median(f):.2f} | {np.percentile(f, 90):.2f} | "
                     f"{np.median(sl):.2f} |")
    L += ["", "Share of each type's detections a fill threshold would drop:", "",
          "| Threshold | " + " | ".join(groups) + " |", "|---|" + "---|" * len(groups)]
    for t in THRESHOLDS:
        cells = []
        for g in groups:
            f = np.array([r["fill"] for r in m if r["group"] == g])
            cells.append(f"{(f < t).mean():.0%}" if len(f) else "–")
        L.append(f"| fill < {t} | " + " | ".join(cells) + " |")
    L += ["", "Real objects losing the most detections at fill < 0.4 (GT class: dropped / total):", ""]
    by_cls = defaultdict(lambda: [0, 0])
    for r in m:
        if r["group"] == "one real object":
            by_cls[r["gt_class"]][1] += 1
            by_cls[r["gt_class"]][0] += r["fill"] < 0.4
    worst = sorted(by_cls.items(), key=lambda kv: -kv[1][0] / kv[1][1])[:15]
    L.append(", ".join(f"{c} {d}/{n}" for c, (d, n) in worst if d))
    (out / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main())
