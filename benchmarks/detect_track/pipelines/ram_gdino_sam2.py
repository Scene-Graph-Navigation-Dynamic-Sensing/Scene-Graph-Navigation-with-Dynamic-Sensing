#!/usr/bin/env python3
"""
RAM++ -> Grounding DINO (SwinT-OGC) -> SAM2.1-L, with two tracking back-ends:

  --tracker botsort : RAM++ + GDINO every frame, BoT-SORT (with camera-motion compensation) on the
                      GDINO boxes, SAM2 image predictor segments every tracked box every frame.
                      Track label = label of its highest-confidence detection (as in run_dovsg_pipeline.py).
  --tracker sam2    : RAM++ + GDINO + SAM2 image masks every --det-every frames; in between, the SAM2
                      video predictor propagates the masks online. New detections are matched to the
                      tracked masks by mask IoU to keep IDs (as in grounded_sam2_tracking_camera_with_continuous_id.py).

GDINO / SAM2 / tracking work on a 1280x720 copy of the frame (GDINO resizes to ~1333x750 and SAM2 to
1024x1024 internally anyway); boxes are reported in original-resolution pixels. Run with locate-env.
"""

from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np
import torch
import torchvision

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
GSAM2 = "/home/ravindu/rashmika/Grounded-SAM-2"
sys.path.insert(0, os.path.join(GSAM2, "grounding_dino"))
sys.path.insert(0, GSAM2)

from common.bench import Bench  # noqa: E402
from common_ram import RamTagger  # noqa: E402

GDINO_CFG = os.path.join(GSAM2, "grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py")
GDINO_CKPT = os.path.join(GSAM2, "gdino_checkpoints/groundingdino_swint_ogc.pth")
SAM2_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"
SAM2_CKPT = os.path.join(GSAM2, "checkpoints/sam2.1_hiera_large.pt")
WORK = (1280, 720)
BOX_T, TEXT_T, NMS_T = 0.3, 0.3, 0.5


def gdino_detect(gd, frame_work_bgr, classes):
    det = gd.predict_with_classes(image=frame_work_bgr, classes=classes, box_threshold=BOX_T, text_threshold=TEXT_T)
    keep = [i for i, c in enumerate(det.class_id) if c is not None and c != -1 and int(c) < len(classes)]
    xyxy, conf = det.xyxy[keep], det.confidence[keep]
    labels = [classes[int(det.class_id[i])] for i in keep]
    if len(keep):
        idx = torchvision.ops.nms(torch.from_numpy(xyxy).float(), torch.from_numpy(conf).float(), NMS_T).numpy()
        xyxy, conf, labels = xyxy[idx], conf[idx], [labels[i] for i in idx]
    return xyxy.reshape(-1, 4), conf.reshape(-1), labels


def sam2_masks(predictor, boxes):
    if len(boxes) == 0:
        return np.zeros((0, WORK[1], WORK[0]), bool)
    masks, _, _ = predictor.predict(point_coords=None, point_labels=None, box=boxes, multimask_output=False)
    if masks.ndim == 4:
        masks = masks[:, 0]
    return masks.astype(bool)


def mask_box(m: np.ndarray):
    ys, xs = np.nonzero(m)
    if len(xs) == 0:
        return None
    return [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]


class BotSort:
    def __init__(self):
        import yaml
        import ultralytics
        from ultralytics.trackers import BOTSORT
        from ultralytics.utils import IterableSimpleNamespace
        with open(os.path.join(os.path.dirname(ultralytics.__file__), "cfg/trackers/botsort.yaml")) as f:
            cfg = yaml.safe_load(f)
        cfg["with_reid"] = False
        self.tracker = BOTSORT(IterableSimpleNamespace(**cfg))
        self.best: dict[int, tuple[str, float]] = {}

    def update(self, img, xyxy, conf, labels):
        from ultralytics.engine.results import Boxes
        data = torch.zeros((len(xyxy), 6))
        if len(xyxy):
            data[:, :4] = torch.from_numpy(xyxy).float()
            data[:, 4] = torch.from_numpy(conf).float()
        tracks = self.tracker.update(Boxes(data, orig_shape=img.shape[:2]), img)
        out = []
        for row in tracks:
            x1, y1, x2, y2, tid, score, _, i = row[:8]
            tid, i = int(tid), int(i)
            lab, c = labels[i], float(conf[i])
            if tid not in self.best or c > self.best[tid][1]:
                self.best[tid] = (lab, c)
            out.append((tid, np.array([x1, y1, x2, y2]), self.best[tid][0], c))
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracker", choices=["botsort", "sam2"], required=True)
    ap.add_argument("--det-every", type=int, default=10, help="sam2 tracker: detection interval in frames")
    ap.add_argument("--amp", action="store_true", help="run SAM2 under fp16 autocast")
    ap.add_argument("--frames", type=int, default=0)
    ap.add_argument("--source", default=None, help="video file or folder of images (default: IMG_6100.mp4)")
    ap.add_argument("--out-root", default=None, help="output root (default: pipeline_benchmark/outputs)")
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    def amp():
        return torch.autocast("cuda", dtype=torch.float16, enabled=args.amp)

    from groundingdino.util.inference import Model as GDModel
    from sam2.build_sam import build_sam2, build_sam2_video_predictor
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    name = args.name or ("p3a_ram_gdino_sam2_botsort" if args.tracker == "botsort" else "p3b_ram_gdino_sam2_videotrack")
    cfg = {"tagger": "RAM++ swin_l 384", "detector": "GroundingDINO SwinT-OGC", "box_thr": BOX_T, "text_thr": TEXT_T,
           "segmenter": f"SAM2.1 hiera-L ({'fp16 autocast' if args.amp else 'fp32'})", "tracker": args.tracker, "work_res": list(WORK)}
    if args.tracker == "sam2":
        cfg["det_every"] = args.det_every
    bench = Bench(name, cfg, max_frames=args.frames, video=args.source, out_root=args.out_root)

    ram = RamTagger()
    gd = GDModel(model_config_path=GDINO_CFG, model_checkpoint_path=GDINO_CKPT, device="cuda")
    sam_img = SAM2ImagePredictor(build_sam2(SAM2_CFG, SAM2_CKPT, device="cuda"))
    if args.tracker == "botsort":
        tracker = BotSort()
    else:
        video_pred = build_sam2_video_predictor(SAM2_CFG, SAM2_CKPT, device="cuda")
        state = None
        tracked: dict[int, dict] = {}        # obj_id -> {"label", "score", "mask"(torch bool, work res)}
        n_objects = 0
    bench.models_loaded()

    for idx, frame in bench.frames():
        t = bench.timer()
        H, W = frame.shape[:2]
        sx, sy = W / WORK[0], H / WORK[1]
        work = cv2.resize(frame, WORK, interpolation=cv2.INTER_AREA)
        work_rgb = cv2.cvtColor(work, cv2.COLOR_BGR2RGB)
        objects, masks = [], {}

        if args.tracker == "botsort":
            with t("ram"):
                classes = ram.tags(frame)
            with t("gdino"):
                xyxy, conf, labels = gdino_detect(gd, work, classes)
            with t("botsort"):
                tracks = tracker.update(work, xyxy, conf, labels)
            if tracks:
                with t("sam2"), amp():
                    sam_img.set_image(work_rgb)
                    ms = sam2_masks(sam_img, np.stack([b for _, b, _, _ in tracks]))
                for (tid, box, lab, c), m in zip(tracks, ms):
                    objects.append({"id": tid, "label": lab, "score": c,
                                    "box": [box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy]})
                    masks[tid] = m
        else:
            detect_now = idx % args.det_every == 0
            det_boxes = np.zeros((0, 4))
            if detect_now:
                with t("ram"):
                    classes = ram.tags(frame)
                with t("gdino"):
                    det_boxes, det_conf, det_labels = gdino_detect(gd, work, classes)
            if len(det_boxes):
                with t("sam2_image"), amp():
                    sam_img.set_image(work_rgb)
                    new_masks = torch.from_numpy(sam2_masks(sam_img, det_boxes)).cuda()
                with t("id_match"):
                    new_tracked = {}
                    prev = [(oid, o["mask"]) for oid, o in tracked.items()]
                    for m, lab, c in zip(new_masks, det_labels, det_conf):
                        if m.sum() == 0:
                            continue
                        oid = None
                        for pid, pm in prev:
                            inter = (m & pm).sum().item()
                            union = (m | pm).sum().item()
                            if union and inter / union > 0.3 and pid not in new_tracked:
                                oid = pid
                                break
                        if oid is None:
                            n_objects += 1
                            oid = n_objects
                        new_tracked[oid] = {"label": lab, "score": float(c), "mask": m}
                    tracked = new_tracked
                with t("sam2_video"), amp():
                    state = video_pred.init_state()
                    state["video_height"], state["video_width"] = WORK[1], WORK[0]
                    fidx = video_pred.add_new_frame(state, work_rgb)
                    for oid, o in tracked.items():
                        video_pred.add_new_mask(state, fidx, oid, o["mask"])
                    _, obj_ids, vmasks = video_pred.infer_single_frame(state, fidx)
            elif state is not None and tracked:
                with t("sam2_video"), amp():
                    fidx = video_pred.add_new_frame(state, work_rgb)
                    _, obj_ids, vmasks = video_pred.infer_single_frame(state, fidx)
            else:
                obj_ids, vmasks = [], None

            if len(obj_ids):
                with t("sam2_video"), amp():
                    vm = (vmasks[:, 0] > 0.0)
                    for oid, m in zip(obj_ids, vm):
                        tracked[oid]["mask"] = m
                    vm_np = vm.cpu().numpy()
                for oid, m in zip(obj_ids, vm_np):
                    b = mask_box(m)
                    if b is None:
                        continue
                    o = tracked[oid]
                    objects.append({"id": oid, "label": o["label"], "score": o["score"],
                                    "box": [b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy]})
                    masks[oid] = m
        bench.record(idx, frame, objects, t, masks=masks)

    bench.finish()


if __name__ == "__main__":
    main()
