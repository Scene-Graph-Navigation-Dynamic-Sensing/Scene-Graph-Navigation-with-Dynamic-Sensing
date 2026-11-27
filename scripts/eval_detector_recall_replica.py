"""How many Replica objects does a 2D proposal model find at all?

Isolates the detector from tracking, fusion and naming: project the Replica
ground-truth objects into each frame (depth-tested, so hidden parts don't
count), run a class-agnostic mask model on the same frame, and check which
visible GT objects get a predicted mask with IoU >= --iou (one object per
frame is an "occurrence").

Reported per detector:
  object recall   visible countable GT objects found in at least one frame
  occurrence rec. share of (object, frame) occurrences found
  masks / frame, seconds / frame
  object recall vs frame stride (every 20th / 10th / 5th NICE-SLAM frame), to
  see whether denser frames find objects sparser ones miss

Frames are read straight from the NICE-SLAM release and resized to the
pipeline's processing width (640), so detector inputs match the pipeline's.

    python scripts/eval_detector_recall_replica.py --scene office_0 \\
        --detectors yoloe:models/checkpoints/yoloe-26s-seg-pf.pt yoloe:models/checkpoints/yoloe-26l-seg-pf.pt
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.replica_gt import load_replica_gt, project_depth_test  # noqa: E402
from perception.image_enhance import enhance  # noqa: E402

NICE = Path("/home/ravindu/SceneGraphNav/datasets/replica/nice_slam/Replica")
GT_ROOT = Path("/home/ravindu/SceneGraphNav/datasets/replica/extracted")
WIDTH = 640


def gt_id_image(gt, pts, K, c2w, depth, shape):
    ok = project_depth_test(pts, K, c2w, depth, 0.05)
    w2c = np.linalg.inv(c2w)
    cam = pts[ok] @ w2c[:3, :3].T + w2c[:3, 3]
    u = np.round(K[0, 0] * cam[:, 0] / cam[:, 2] + K[0, 2]).astype(int)
    v = np.round(K[1, 1] * cam[:, 1] / cam[:, 2] + K[1, 2]).astype(int)
    img = np.zeros(shape, np.int32)
    order = np.argsort(-cam[:, 2])
    img[v[order], u[order]] = gt.object_id[ok][order]
    return cv2.dilate(img.astype(np.float32), np.ones((3, 3), np.uint8)).astype(np.int32)


class YOLOE:
    def __init__(self, weights, conf=0.25, imgsz=640):
        from ultralytics import YOLO
        self.m, self.conf, self.imgsz = YOLO(weights), conf, imgsz

    def __call__(self, rgb):
        r = self.m.predict(rgb[..., ::-1], conf=self.conf, imgsz=self.imgsz, verbose=False, retina_masks=True)[0]
        if r.masks is None:
            return np.zeros((0,) + rgb.shape[:2], bool)
        return r.masks.data.cpu().numpy().astype(bool)


class FastSAM:
    def __init__(self, weights, conf=0.4, imgsz=640):
        from ultralytics import FastSAM as F
        self.m, self.conf, self.imgsz = F(weights), conf, imgsz

    def __call__(self, rgb):
        r = self.m(rgb[..., ::-1], conf=self.conf, imgsz=self.imgsz, verbose=False, retina_masks=True)[0]
        if r.masks is None:
            return np.zeros((0,) + rgb.shape[:2], bool)
        return r.masks.data.cpu().numpy().astype(bool)


def build(spec: str):
    """kind:weights[:conf[:imgsz]]"""
    parts = spec.split(":")
    kind, weights = parts[0], parts[1]
    conf = float(parts[2]) if len(parts) > 2 else None
    imgsz = int(parts[3]) if len(parts) > 3 else 640
    if kind == "yoloe":
        return YOLOE(weights, conf if conf is not None else 0.25, imgsz)
    if kind == "fastsam":
        return FastSAM(weights, conf if conf is not None else 0.4, imgsz)
    raise ValueError(kind)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--detectors", nargs="+", required=True, help="kind:weights[:conf[:imgsz]]")
    ap.add_argument("--stride", type=int, default=5, help="finest NICE-SLAM frame stride evaluated")
    ap.add_argument("--max_frames", type=int, default=400)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--min_pixels", type=int, default=100, help="GT object must cover this many pixels in a frame")
    ap.add_argument("--enhance", default="none", help="none | gamma (the brightness variant)")
    ap.add_argument("--out", default="results/detector_recall")
    args = ap.parse_args()

    src = NICE / args.scene.replace("_", "")
    cam = json.loads((NICE / "cam_params.json").read_text())["camera"]
    s = WIDTH / cam["w"]
    H = int(round(cam["h"] * s))
    K = np.array([[cam["fx"] * s, 0, cam["cx"] * s], [0, cam["fy"] * s, cam["cy"] * s], [0, 0, 1]])
    poses = np.loadtxt(src / "traj.txt").reshape(-1, 4, 4)
    idx = list(range(0, len(poses), args.stride))[: args.max_frames]

    gt = load_replica_gt(GT_ROOT / args.scene)
    countable = {o for o, v in gt.objects.items() if v["kind"] == "countable"}

    frames = []
    for i in idx:
        rgb = cv2.cvtColor(cv2.imread(str(src / "results" / f"frame{i:06d}.jpg")), cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (WIDTH, H), interpolation=cv2.INTER_AREA)
        d = cv2.imread(str(src / "results" / f"depth{i:06d}.png"), cv2.IMREAD_UNCHANGED) / cam["scale"]
        d = cv2.resize(d, (WIDTH, H), interpolation=cv2.INTER_NEAREST)
        ids = gt_id_image(gt, gt.xyz, K, poses[i], d, (H, WIDTH))
        present = {int(o): int(n) for o, n in zip(*np.unique(ids, return_counts=True))
                   if o in countable and n >= args.min_pixels}
        frames.append({"src": i, "rgb": enhance(rgb, {"method": args.enhance} if args.enhance != "none" else None),
                       "ids": ids, "present": present})
    visible = set().union(*[f["present"] for f in frames])
    print(f"{args.scene}: {len(frames)} frames (stride {args.stride}), {len(visible)} countable GT objects visible")

    out_dir = REPO_ROOT / args.out / args.scene
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {"scene": args.scene, "frames": len(frames), "stride": args.stride, "iou": args.iou,
              "enhance": args.enhance, "visible_objects": len(visible), "detectors": {}}
    for spec in args.detectors:
        det = build(spec)
        det(frames[0]["rgb"])                                   # warm-up
        found_at: dict[int, list[int]] = {}
        occ, hit, n_masks, t = 0, 0, 0, 0.0
        for f in frames:
            t0 = time.time()
            masks = det(f["rgb"])
            t += time.time() - t0
            if masks.shape[1:] != f["ids"].shape:
                masks = np.stack([cv2.resize(m.astype(np.uint8), (WIDTH, H), interpolation=cv2.INTER_NEAREST)
                                  for m in masks]).astype(bool) if len(masks) else masks
            n_masks += len(masks)
            for o in f["present"]:
                occ += 1
                g = f["ids"] == o
                if len(masks):
                    inter = (masks & g).sum((1, 2))
                    iou = inter / ((masks | g).sum((1, 2)) + 1e-9)
                    if iou.max() >= args.iou:
                        hit += 1
                        found_at.setdefault(o, []).append(f["src"])
        by_stride = {}
        for k in (4, 2, 1):
            keep = {f["src"] for f in frames[::k]}
            seen = {o for f in frames[::k] for o in f["present"]}
            got = {o for o, srcs in found_at.items() if any(x in keep for x in srcs)}
            by_stride[args.stride * k] = {"visible": len(seen), "found": len(got & seen),
                                          "recall": len(got & seen) / max(len(seen), 1)}
        missed = sorted(gt.label(o) for o in visible - set(found_at))
        r = {"object_recall": len(found_at) / max(len(visible), 1), "objects_found": len(found_at),
             "occurrence_recall": hit / max(occ, 1), "masks_per_frame": n_masks / len(frames),
             "seconds_per_frame": t / len(frames), "recall_by_stride": by_stride, "missed": missed}
        report["detectors"][spec] = r
        print(f"  {spec:60s} objects {r['objects_found']}/{len(visible)} ({100 * r['object_recall']:.1f}%)  "
              f"occurrences {100 * r['occurrence_recall']:.1f}%  {r['masks_per_frame']:.1f} masks/frame  "
              f"{1000 * r['seconds_per_frame']:.0f} ms/frame  by stride "
              + ", ".join(f"{k}: {100 * v['recall']:.0f}%" for k, v in sorted(by_stride.items(), reverse=True)))
        del det
    tag = "" if args.enhance == "none" else f"_{args.enhance}"
    (out_dir / f"report{tag}.json").write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
