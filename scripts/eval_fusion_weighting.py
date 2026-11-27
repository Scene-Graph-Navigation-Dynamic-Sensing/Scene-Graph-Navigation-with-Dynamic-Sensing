#!/usr/bin/env python3
"""Which per-view weighting of the running-average embedding gives the most
accurate object fusion?

Fusion matches tracks to objects by 0.3 geometric + 0.7 cosine, where an
object's embedding is the running average of its detections' CLIP embeddings,
every view weighted equally (DoVSG merge_obj_to_obj, weighted by detection
count). Here the same track-level fusion, clean-up and naming run once per
weighting of that average:

  uniform (current)        every view counts once
  sqrt mask area           closer / fuller views count more
  not truncated            views cut by the image border count 0.2
  closer (1/depth^2)       by the median depth of the mask
  YOLOE confidence
  area x uncut x sharp     sqrt(area) x not-truncated x sqrt(sharpness)
  outlier rejection        robust mean of the member views: views far from
                           their object's consensus are left out
  best view only           the object's embedding is its largest uncut view

and the fused objects are scored as in scripts/eval_crop_mode.py (ground truth
= the labelled tracks): objects found, fused objects that are one real object,
splits, mixed objects, voxel purity / recall, naming. Box-crop embeddings
from outputs/eval/crop_mode/<video>/cache.pkl; no detection is re-run.

    python scripts/eval_fusion_weighting.py --videos initial,updated_same_angle
"""
from __future__ import annotations

import argparse
import copy
import itertools
import json
import logging
import pickle
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logger = logging.getLogger("eval_fusion_weighting")
ROOT = REPO_ROOT / "outputs" / "eval" / "crop_mode"
SCHEMES = ["uniform (current)", "sqrt mask area", "not truncated", "closer (1/depth^2)", "YOLOE confidence",
           "area x uncut x sharp", "outlier rejection", "best view only"]


def robust_mean(E: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
    """Mean of unit embeddings without the views far from their consensus
    (cosine to the median direction below median - 2.5 MAD)."""
    En = E / np.linalg.norm(E, axis=1, keepdims=True)
    if len(En) >= 3:
        med = np.median(En, axis=0)
        cos = En @ (med / np.linalg.norm(med))
        mad = np.median(np.abs(cos - np.median(cos))) + 1e-6
        keep = cos >= np.median(cos) - 2.5 * mad
        En = En[keep]
    m = En.mean(0)
    return m / np.linalg.norm(m)


class WeightedMerge:
    """Wraps fusion.merge_obj_to_obj so the merged embedding follows `scheme`
    instead of the detection-count weighting. Every fused dict carries
    'eval_uid' (a list, so DoVSG's merge concatenates it and its first entry
    always names the surviving object)."""

    def __init__(self, fusion, scheme: str, feats: dict):
        import torch
        import torch.nn.functional as F

        self.torch, self.F = torch, F
        self.fusion, self.scheme, self.feats = fusion, scheme, feats
        self.orig = fusion.merge_obj_to_obj
        self.weight, self.best, self.members = {}, {}, {}
        sharp_med = np.median([f["sharp"] for f in feats.values()]) or 1.0
        for uid, f in feats.items():
            full = 0.2 if f["truncated"] else 1.0
            self.weight[uid] = {
                "uniform (current)": 1.0, "sqrt mask area": np.sqrt(f["area"]), "not truncated": full,
                "closer (1/depth^2)": 1.0 / max(f["depth"], 0.1) ** 2, "YOLOE confidence": f["conf"],
                "area x uncut x sharp": np.sqrt(f["area"]) * full * np.sqrt(f["sharp"] / sharp_med),
                "outlier rejection": 1.0, "best view only": 1.0}[scheme]
            self.best[uid] = (f["area"] * (1.0 if not f["truncated"] else 1e-3), f["emb"])
            self.members[uid] = [f["emb"]]
        fusion.merge_obj_to_obj = self

    def __call__(self, obj1, obj2, run_dbscan=True):
        u1, u2 = obj1["eval_uid"][0], obj2["eval_uid"][0]
        to_np = lambda c: (c.detach().cpu().numpy() if hasattr(c, "detach") else np.asarray(c)).astype(np.float32).ravel()  # noqa: E731
        c1, c2 = to_np(obj1["clip_ft"]), to_np(obj2["clip_ft"])
        w1, w2 = self.weight.get(u1, 1.0), self.weight.get(u2, 1.0)
        out = self.orig(obj1, obj2, run_dbscan=run_dbscan)
        if self.scheme == "uniform (current)":
            return out  # DoVSG's own detection-count weighting
        if self.scheme == "best view only":
            b = max(self.best.get(u1, (0, c1)), self.best.get(u2, (0, c2)), key=lambda x: x[0])
            self.best[u1], emb = b, b[1]
        elif self.scheme == "outlier rejection":
            self.members[u1] = self.members.get(u1, [c1]) + self.members.get(u2, [c2])
            emb = robust_mean(np.stack(self.members[u1]))
        else:
            emb = c1 * w1 + c2 * w2
        emb = emb / np.linalg.norm(emb)
        out["clip_ft"] = self.torch.from_numpy(np.asarray(emb, np.float32))
        out["text_ft"] = out["clip_ft"]
        self.weight[u1] = w1 + w2
        return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", default="initial,updated_same_angle,updated_diff_angle,update_in_same_video,img6100")
    ap.add_argument("--schemes", default=",".join(SCHEMES))
    ap.add_argument("--tag", default="", help="suffix of the result file, for runs in parallel")
    ap.add_argument("--keep_low_confidence", action="store_true",
                    help="keep objects NOVIC names below the confidence cutoff, as 'unknown', so fusion is "
                         "scored on every fused object and a weak name only counts against naming")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    import cv2
    from PIL import Image

    from object_fusion.instance_fusion import build_instance_fusion_class
    from object_fusion.streaming import prepare_native_libraries
    from scripts.eval_crop_mode import Judge, build_voxel_map, load_models
    from scripts.run_static_scene_graph import build_graph_from_objects, load_config, load_everyday_names

    cfg = load_config(REPO_ROOT / "configs" / "pipeline_fyp_videos.yaml")
    fc = cfg["object_fusion"]
    if args.keep_low_confidence:
        cfg["label_mapper"]["low_score_action"] = "unknown"
    prepare_native_libraries()
    _, novic = load_models(cfg, tracker=False)
    judge, everyday = Judge(), load_everyday_names(cfg)
    InstanceFusion = build_instance_fusion_class(cfg["external"]["scene_graph_generation_root"])
    schemes = args.schemes.split(",")
    uid_counter = itertools.count()

    for video in args.videos.split(","):
        out_path = ROOT / video / f"fusion_weighting{args.tag}.json"
        cache = pickle.loads((ROOT / video / "cache.pkl").read_bytes())
        labels = {int(k): v for k, v in json.loads((ROOT / video / "labels.json").read_text()).items()}
        ds = Path(cache["dataset_dir"])
        vd = build_voxel_map(cfg, ds)
        frames = cache["frames"]
        for fr in frames:
            fr["masks"] = (np.unpackbits(fr["masks"], axis=-1, count=fr["shape"][1]).astype(bool)
                           if len(fr["boxes"]) else np.zeros((0, *fr["shape"]), bool))

        # ground truth: each labelled object's voxel set = its detections' mask voxels
        inst_vox = {}
        for f, fr in enumerate(frames):
            pim, pmask = vd.pixel_index_mappings[f], vd.pixel_index_masks[f]
            for i, t in enumerate(fr["tids"]):
                g = labels.get(int(t), "ignore")
                if g != "ignore":
                    inst_vox.setdefault(g, set()).update(np.unique(pim[fr["masks"][i] & pmask]).tolist())
        det_count = {}
        for fr in frames:
            for t in fr["tids"]:
                g = labels.get(int(t), "ignore")
                det_count[g] = det_count.get(g, 0) + 1
        present = [g for g in inst_vox if det_count.get(g, 0) >= fc["obj_min_detections"]]

        results = json.loads(out_path.read_text()) if out_path.exists() else {}
        for scheme in schemes:
            t0 = time.time()
            fusion = InstanceFusion(downsample_voxel_size=fc["downsample_voxel_size"],
                                    spatial_weight=fc["spatial_weight"], vis_weight=fc["vis_weight"],
                                    sim_threshold=fc["sim_threshold"])
            fusion.start_new_scan(vd)
            per_frame, feats = [], {}
            for f, fr in enumerate(frames):
                dl = fusion.build_detection_list_from_frame(
                    boxes=fr["boxes"], masks=fr["masks"], confs=fr["confs"], embeddings=fr["emb"]["bbox"],
                    pixel_indexes=vd.pixel_index_mappings[f], pixel_indexes_mask=vd.pixel_index_masks[f],
                    image_name=vd.names[f], track_ids=fr["tids"])
                if len(dl):
                    h, w = fr["shape"]
                    gray = cv2.cvtColor(np.asarray(Image.open(ds / "rgb" / f"{f:06d}.jpg")), cv2.COLOR_RGB2GRAY)
                    z = np.load(ds / "point" / f"{f:06d}.npy")[..., 2]
                    valid = np.load(ds / "mask" / f"{f:06d}.npy").astype(bool)
                for det in dl:
                    uid = next(uid_counter)
                    det["eval_uid"] = [uid]
                    m = det["mask"][0]
                    ys, xs = np.nonzero(m)
                    x1, y1, x2, y2 = np.asarray(det["xyxy"][0]).astype(int)
                    patch = gray[max(0, y1):y2, max(0, x1):x2]
                    zz = z[m & valid]
                    feats[uid] = {
                        "area": float(m.sum() / (h * w)),
                        "truncated": bool(len(xs) and (xs.min() <= 2 or ys.min() <= 2 or xs.max() >= w - 3
                                                      or ys.max() >= h - 3)),
                        "depth": float(np.median(zz)) if len(zz) else 1.0, "conf": float(det["conf"]),
                        "sharp": float(cv2.Laplacian(patch, cv2.CV_64F).var()) if patch.size > 16 else 0.0,
                        "emb": np.asarray(det["clip_ft"], np.float32).ravel()}
                per_frame.append(dl)
            WeightedMerge(fusion, scheme, feats)
            objects, _ = fusion.get_instances(view_dataset=vd, per_frame_detection_lists=per_frame,
                                              obj_min_detections=fc["obj_min_detections"],
                                              decode_fn=novic.decode_embeddings, match_level="track",
                                              track_end_frames=fc.get("track_end_frames", 10))
            objects, _, _ = build_graph_from_objects(cfg, copy.copy(objects), vd, novic, everyday, verbose=False,
                                                     relations=False)
            fused = []
            for o in objects:
                comp = {}
                for t in o["track_ids"]:
                    g = labels.get(int(t), "ignore")
                    comp[g] = comp.get(g, 0) + 1
                real = {g: n for g, n in comp.items() if g != "ignore"}
                g = max(real, key=real.get) if real else "ignore"
                tot = sum(comp.values()) or 1
                vox = set(np.asarray(o["indexes"]).tolist())
                gv = inst_vox.get(g, set())
                s, l_ = judge(o["class_name"], g.split("|")[0]) if g != "ignore" else (False, False)
                fused.append({"name": o["class_name"], "object": g, "purity": comp.get(g, 0) / tot,
                              "inside": comp, "voxel_purity": len(vox & gv) / max(len(vox), 1),
                              "voxel_recall": len(vox & gv) / max(len(gv), 1), "strict": s, "lenient": l_})
            real_objs = [x for x in fused if x["object"] != "ignore"]
            good = [x for x in real_objs if x["purity"] >= 0.5]
            per_obj = {}
            for x in good:
                per_obj[x["object"]] = per_obj.get(x["object"], 0) + 1
            mixed = [x for x in real_objs if x["purity"] < 0.5 or sum(
                1 for gg, n in x["inside"].items() if gg != "ignore" and n >= 0.2 * sum(x["inside"].values())) > 1]
            results[scheme] = {
                "objects_in_video": len(present), "fused_objects": len(fused),
                "found": len({x["object"] for x in good} & set(present)),
                "one_real_object": len(good),
                "split_objects": sum(n - 1 for n in per_obj.values()),
                "mixed_objects": len(mixed),
                "voxel_purity": float(np.mean([x["voxel_purity"] for x in good])) if good else 0.0,
                "voxel_recall": float(np.mean([x["voxel_recall"] for x in good])) if good else 0.0,
                "named_strict": sum(x["strict"] for x in good), "named_lenient": sum(x["lenient"] for x in good),
                "objects": [{k: v for k, v in x.items() if k != "inside"} | {"inside": x["inside"]} for x in fused],
                "seconds": round(time.time() - t0, 1)}
            r = results[scheme]
            logger.info(f"[{video} / {scheme}] found {r['found']}/{r['objects_in_video']}, "
                        f"{r['one_real_object']}/{r['fused_objects']} fused objects are one object, "
                        f"{r['split_objects']} splits, {r['mixed_objects']} mixed, named {r['named_lenient']} "
                        f"({r['seconds']}s)")
            out_path.write_text(json.dumps(results, indent=2, default=str))
    novic.close()


if __name__ == "__main__":
    main()
