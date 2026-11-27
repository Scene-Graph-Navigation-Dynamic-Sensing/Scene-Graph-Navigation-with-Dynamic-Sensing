"""Preview: SAM2 refining YOLOE's masks, on one Replica scene (no mapping).

For every frame, the pipeline's tracked YOLOE detector (the final Replica
configuration's settings, read from that run's config_used.yaml) gives boxes,
masks and track ids. Each box is then given to SAM2 (hiera-large, DovSG's
checkpoint) as a prompt, which returns its own mask for the same object.

Writes results/sam2_refine/<scene>/:
  yoloe_vs_sam2_masks.mp4   left: YOLOE masks + boxes; right: SAM2 masks for the same boxes
                            (same colour per track). Header: per-frame spill of each.
  stats.json                per-detection agreement and "spill": the share of a mask's pixels
                            that land on wall / floor / ceiling in Replica's ground truth, for
                            detections whose mask mostly covers one countable object

Ground truth per pixel comes from projecting Replica's semantic-mesh samples
into each frame with the exact depth (as evaluation/artifacts.write_video does).

    python scripts/preview_sam2_refine.py --scene room_0 --gpu 1
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
SAM2_SRC = REPO_ROOT / "external" / "DovSG" / "third_party" / "segment-anything-2"
SAM2_CKPT = REPO_ROOT / "external" / "Scene-Graph-Generation" / "checkpoints" / "segment-anything-2" / "sam2_hiera_large.pt"
GT_ROOT = Path("/home/ravindu/SceneGraphNav/datasets/replica/extracted")
for p in (REPO_ROOT, SAM2_SRC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from evaluation import artifacts  # noqa: E402
from evaluation.replica_gt import load_replica_gt  # noqa: E402
from perception.image_enhance import enhance  # noqa: E402


def colour(tid: int) -> tuple:
    rng = np.random.default_rng(abs(tid) * 7919 + 13)
    return tuple(int(c) for c in rng.integers(60, 255, 3))


def overlay(rgb, masks, boxes, tids):
    out = rgb.copy()
    order = np.argsort([-m.sum() for m in masks])              # big masks first, small ones on top
    for i in order:
        c = np.array(colour(tids[i]), np.float32)
        m = masks[i]
        out[m] = (0.45 * out[m] + 0.55 * c).astype(np.uint8)
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, colour(tids[i]), 1)
    for b, t in zip(boxes, tids):
        x1, y1, x2, y2 = [int(round(v)) for v in b]
        cv2.rectangle(out, (x1, y1), (x2, y2), colour(t), 1)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", default="room_0")
    ap.add_argument("--run", default="results/replica_final", help="run whose detector settings and poses to use")
    ap.add_argument("--gpu", default="1")
    ap.add_argument("--fps", type=int, default=5)
    args = ap.parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.gpu)
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from perception.detection.tracked_yoloe_detector import TrackedYOLOEDetector

    rd = REPO_ROOT / args.run / args.scene
    cfg = yaml.safe_load(open(rd / "config_used.yaml"))
    det_cfg, width = cfg["detection"], int(cfg["source"]["target_width"])
    processed = REPO_ROOT / "data" / "processed" / next((rd / "scene_graph").glob("*.pkl")).stem
    raw = REPO_ROOT / "data" / "raw" / f"replica_{args.scene}"
    out = REPO_ROOT / "results" / "sam2_refine" / args.scene
    out.mkdir(parents=True, exist_ok=True)

    detector = TrackedYOLOEDetector(weights_path=det_cfg["weights"], conf_thresh=det_cfg["conf_thresh"],
                                    iou_thresh=det_cfg["iou_thresh"], img_size=det_cfg["img_size"], device="cuda",
                                    tracker=det_cfg.get("tracker", "botsort.yaml"))
    sam = SAM2ImagePredictor(build_sam2("sam2_hiera_l.yaml", str(SAM2_CKPT.resolve()), device="cuda"))

    # ground truth in the map frame, for per-pixel labels
    gt = load_replica_gt(GT_ROOT / args.scene)
    T_align = np.array(json.loads((processed / "gravity_alignment.json").read_text())["transform"])
    gt_xyz_map = gt.xyz @ T_align[:3, :3].T + T_align[:3, 3]
    kind_id = {"countable": 1, "structural": 2}
    meta = json.loads((processed / "metadata.json").read_text())
    K = np.array(meta["K"], dtype=np.float64).reshape(3, 3)
    names = json.loads((processed / "capture.json").read_text())["frame_names"]

    h = int(round(680 * width / 1200))
    W, H = 2 * width, h + 30
    video = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
         "-r", str(args.fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23",
         "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", str(out / "yoloe_vs_sam2_masks.mp4")], stdin=subprocess.PIPE)
    font = cv2.FONT_HERSHEY_SIMPLEX
    stats = {"yoloe_spill": [], "sam2_spill": [], "iou": [], "area_ratio": [], "sam2_seconds": [], "detections": 0}

    for name in names:
        bgr = cv2.imread(str(raw / "rgb" / f"{name}.jpg"))
        h0, w0 = bgr.shape[:2]
        bgr = cv2.resize(bgr, (width, int(round(h0 * width / w0))), interpolation=cv2.INTER_AREA)
        rgb_det = enhance(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), det_cfg.get("enhance"))
        rgb_show = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        dets = detector.track(rgb_det)
        boxes = np.stack([d.bbox for d in dets]) if dets else np.zeros((0, 4))
        ymasks = [d.mask for d in dets]
        tids = [d.track_id for d in dets]
        smasks = []
        if len(dets):
            t = time.time()
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                sam.set_image(rgb_det)
                m, _, _ = sam.predict(box=boxes.astype(np.float32), multimask_output=False)
            torch.cuda.synchronize()
            stats["sam2_seconds"].append(time.time() - t)
            m = np.asarray(m)
            smasks = [mm[0] > 0 for mm in (m if m.ndim == 4 else m[None])]

        # per-pixel ground truth: 0 unknown, 1 countable object, 2 wall / floor / ceiling
        depth = cv2.imread(str(raw / "depth" / f"{name}.png"), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
        depth = cv2.resize(depth, (rgb_show.shape[1], rgb_show.shape[0]), interpolation=cv2.INTER_NEAREST)
        c2w = np.loadtxt(processed / "poses" / f"{name}.txt")
        ok, uv, z = artifacts._project(gt_xyz_map, K, c2w, depth, 0.05)
        oid_img = artifacts._splat(rgb_show.shape[:2], uv, gt.object_id[ok], z)
        kinds = np.zeros(oid_img.max() + 1 if oid_img.size else 1, np.uint8)
        for oid, v in gt.objects.items():
            if oid < len(kinds):
                kinds[oid] = kind_id.get(v["kind"], 0)
        kimg = kinds[oid_img]
        fy, fs = [], []
        for ym, sm in zip(ymasks, smasks):
            stats["detections"] += 1
            inter, union = (ym & sm).sum(), (ym | sm).sum()
            stats["iou"].append(float(inter / union) if union else 0.0)
            stats["area_ratio"].append(float(sm.sum() / max(ym.sum(), 1)))
            labelled = (kimg[ym] > 0).sum()
            if labelled and (kimg[ym] == 1).sum() > 0.5 * labelled:            # mostly one countable object
                y_sp = float((kimg[ym] == 2).sum() / labelled)
                s_lab = (kimg[sm] > 0).sum()
                s_sp = float((kimg[sm] == 2).sum() / s_lab) if s_lab else 0.0
                stats["yoloe_spill"].append(y_sp)
                stats["sam2_spill"].append(s_sp)
                fy.append(y_sp)
                fs.append(s_sp)

        canvas = np.full((H, W, 3), 255, np.uint8)
        canvas[30:30 + rgb_show.shape[0], :width] = overlay(rgb_show, ymasks, boxes, tids)
        canvas[30:30 + rgb_show.shape[0], width:] = overlay(rgb_show, smasks, boxes, tids)
        sp = lambda v: f"{100 * np.mean(v):.0f}%" if v else "-"
        cv2.putText(canvas, f"YOLOE masks  frame {name}  wall/floor/ceiling spill {sp(fy)}", (6, 20), font, 0.45,
                    (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"SAM2 masks (YOLOE boxes as prompts)  spill {sp(fs)}", (width + 6, 20), font, 0.45,
                    (0, 0, 0), 1, cv2.LINE_AA)
        video.stdin.write(canvas.tobytes())
    video.stdin.close()
    video.wait()

    ys, ss = np.array(stats["yoloe_spill"]), np.array(stats["sam2_spill"])
    summary = {"scene": args.scene, "frames": len(names), "detections": stats["detections"],
               "median_iou_yoloe_vs_sam2": float(np.median(stats["iou"])),
               "median_area_sam2_over_yoloe": float(np.median(stats["area_ratio"])),
               "object_detections_scored_for_spill": int(len(ys)),
               "spill_mean": {"yoloe": float(ys.mean()), "sam2": float(ss.mean())},
               "spill_median": {"yoloe": float(np.median(ys)), "sam2": float(np.median(ss))},
               "masks_with_spill_over_20pct": {"yoloe": float((ys > 0.2).mean()), "sam2": float((ss > 0.2).mean())},
               "sam2_seconds_per_frame": float(np.median(stats["sam2_seconds"]))}
    (out / "stats.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    print(f"wrote {out / 'yoloe_vs_sam2_masks.mp4'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
