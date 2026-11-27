#!/usr/bin/env python3
"""Bounding-box crops vs segmentation crops for the NOVIC (CLIP) embeddings,
on our own videos.

YOLOE-26-seg-pf gives each detection a box and a mask. The embedding can come
from the padded box crop (crop_mode "bbox", what NOVIC was trained on) or from
the same crop with everything outside the mask blacked out ("segmented").
Fusion lifts the MASK into the voxel map in both modes; only the embedding
differs, and through it matching, fusion and naming.

The videos have no labels, so the ground truth is per BoT-SORT track: every
track is labelled with the real object it follows ("chair|wooden",
"laptop", "ignore" for wall/floor/junk). Detection and tracking run once, so
both modes see identical detections and track ids.

  prepare  python scripts/eval_crop_mode.py prepare --dataset_dir data/processed/fyp_initial_v4 --name initial
           -> outputs/eval/crop_mode/<name>/cache.pkl, tracks.png (one tile per track, to label)
  score    python scripts/eval_crop_mode.py score --name initial
           (reads outputs/eval/crop_mode/<name>/labels.json: {"<track id>": "<class>|<instance>", ...})
           -> results.json

Scored per mode:
  naming      NOVIC's decode of each detection's embedding, and of each
              track's average embedding (the moving average fusion keeps),
              raw and after the everyday-names mapping, against the class,
              judged with WordNet: strict = same synset; lenient = also a more
              specific name ("armchair" for "chair") or the direct parent.
  matching    cosine of two detections (and two track averages) of the same
              object vs of different objects: ROC AUC, and the share of
              different-object pairs above fusion's cosine for a merge.
  fusion      the pipeline's track-level fusion, clean-up and naming; each
              fused object's detections (via its track ids) give its object,
              purity, and whether objects were split or merged; recall /
              precision of objects; naming of the fused objects.
  voxels      an object's voxel set = the mask voxels of all its detections.
              Per detection: share of its mask / box voxels in its own
              object's set, in another object's set, or in neither
              (background). Per fused object: voxel purity and recall
              against its object's set.
"""
from __future__ import annotations

import argparse
import copy
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

logger = logging.getLogger("eval_crop_mode")
# name -> (extract_crop mode, padding as a fraction of the box size)
MODE_SPECS = {
    "bbox": ("bbox", 0.15),            # the pipeline's crop: padded box
    "segmented": ("segmented", 0.15),  # outside the mask black
    "tight": ("bbox", 0.05),           # box with a third of the padding
    "blur": ("blur", 0.15),            # padded box, outside the mask blurred
}
MODES = ("bbox", "segmented")          # computed by `prepare`; `embed` adds others
OUT_ROOT = REPO_ROOT / "outputs" / "eval" / "crop_mode"


def load_models(cfg, tracker=True):
    from perception.detection.tracked_yoloe_detector import TrackedYOLOEDetector
    from semantic_decoding.novic.novic_decoder import NOVICDecoder

    det_cfg, sem_cfg = cfg["detection"], cfg["semantic_decoding"]
    detector = (TrackedYOLOEDetector(weights_path=det_cfg["weights"], conf_thresh=det_cfg["conf_thresh"],
                                     iou_thresh=det_cfg["iou_thresh"], img_size=det_cfg["img_size"],
                                     device=det_cfg["device"], tracker=det_cfg.get("tracker", "botsort.yaml"))
                if tracker else None)
    novic = NOVICDecoder(novic_dir=sem_cfg["novic_dir"], checkpoint_path=sem_cfg["novic_checkpoint"],
                         gencfg=sem_cfg["gencfg"], guide_targets=sem_cfg["guide_targets"], device=sem_cfg["device"],
                         embedding_batch_size=sem_cfg.get("embedding_batch_size", 8))
    return detector, novic


def build_voxel_map(cfg, ds: Path):
    from object_fusion.view_dataset import build_view_dataset_parallel
    from scene_graph.builder.gravity_alignment import gravity_align_dataset

    gravity_align_dataset(ds)  # as the pipeline does; idempotent (always from poses_unaligned/)
    fc = cfg["object_fusion"]
    return build_view_dataset_parallel(str(ds), cfg["external"]["scene_graph_generation_root"], fc["voxel_resolution"],
                                       workers=fc.get("voxel_map_workers", 8))


# ─── prepare ───

def prepare(args, cfg):
    import cv2
    from PIL import Image

    from object_fusion.streaming import detect_frame
    from semantic_decoding.novic.crop_utils import extract_crop

    ds = Path(args.dataset_dir)
    ds = ds if ds.is_absolute() else REPO_ROOT / ds
    out = OUT_ROOT / args.name
    out.mkdir(parents=True, exist_ok=True)
    detector, novic = load_models(cfg)
    n = json.loads((ds / "metadata.json").read_text())["length"]
    frames, best = [], {}
    for f in range(n):
        image = np.asarray(Image.open(ds / "rgb" / f"{f:06d}.jpg"), dtype=np.uint8)
        boxes, masks, confs, tids, _, _ = detect_frame(cfg, image, detector, tracked=True)
        emb = {}
        for mode in MODES:
            crop_mode, pad = MODE_SPECS[mode]
            crops = [extract_crop(image, boxes[i], masks[i], mode=crop_mode, padding_frac=pad)
                     for i in range(len(boxes))]
            emb[mode] = novic.embed_crops(crops) if crops else np.zeros((0, 1024), np.float32)
        frames.append({"boxes": boxes, "masks": np.packbits(masks, axis=-1) if len(masks) else masks,
                       "shape": image.shape[:2], "confs": confs, "tids": tids, "emb": emb})
        for i, t in enumerate(tids):  # the largest view of each track, for labelling
            area = int(masks[i].sum())
            if t >= 0 and area > best.get(int(t), (0,))[0]:
                crop = extract_crop(image, boxes[i], masks[i], mode="bbox", padding_px=15, padding_frac=0.15).copy()
                x1, y1, x2, y2 = boxes[i].astype(int)
                pw, ph = int(round((x2 - x1) * 0.15)), int(round((y2 - y1) * 0.15))
                m = masks[i][max(0, y1 - ph):y2 + ph, max(0, x1 - pw):x2 + pw].astype(np.uint8)
                cs, _ = cv2.findContours(m[:crop.shape[0], :crop.shape[1]], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(crop, cs, -1, (255, 0, 255), 2)
                best[int(t)] = (area, crop, f)
        if f % 25 == 0:
            logger.info(f"frame {f + 1}/{n}: {len(boxes)} detections")

    counts, first, last = {}, {}, {}
    for f, fr in enumerate(frames):
        for t in fr["tids"]:
            t = int(t)
            counts[t] = counts.get(t, 0) + 1
            first.setdefault(t, f)
            last[t] = f
    with open(out / "cache.pkl", "wb") as fh:
        pickle.dump({"dataset_dir": str(ds), "frames": frames}, fh, protocol=4)

    # contact sheet: one tile per track
    tiles = []
    for t in sorted(best):
        _, crop, f = best[t]
        h, w = crop.shape[:2]
        s = 150 / max(h, w)
        tile = np.full((180, 160, 3), 255, np.uint8)
        c = cv2.resize(crop, (max(1, int(w * s)), max(1, int(h * s))))
        tile[28:28 + c.shape[0], 5:5 + c.shape[1]] = c
        cv2.putText(tile, f"t{t} n{counts[t]} f{first[t]}-{last[t]}", (3, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (0, 0, 0), 1)
        tiles.append(tile)
    cols = 10
    while len(tiles) % cols:
        tiles.append(np.full((180, 160, 3), 255, np.uint8))
    sheet = np.vstack([np.hstack(tiles[k:k + cols]) for k in range(0, len(tiles), cols)])
    cv2.imwrite(str(out / "tracks.png"), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    (out / "tracks.json").write_text(json.dumps({t: {"detections": counts[t], "frames": [first[t], last[t]]}
                                                 for t in sorted(counts)}, indent=1))
    novic.close()
    logger.info(f"{len(best)} tracks; sheet {out / 'tracks.png'}")


def embed(args, cfg):
    """Adds the embeddings of more crop modes to a prepared cache, on the same
    detections (no detection or tracking is re-run)."""
    from PIL import Image

    from semantic_decoding.novic.crop_utils import extract_crop

    out = OUT_ROOT / args.name
    cache = pickle.loads((out / "cache.pkl").read_bytes())
    ds = Path(cache["dataset_dir"])
    modes = args.modes.split(",")
    _, novic = load_models(cfg, tracker=False)
    for f, fr in enumerate(cache["frames"]):
        if not len(fr["boxes"]):
            for m in modes:
                fr["emb"][m] = np.zeros((0, 1024), np.float32)
            continue
        image = np.asarray(Image.open(ds / "rgb" / f"{f:06d}.jpg"), dtype=np.uint8)
        masks = np.unpackbits(fr["masks"], axis=-1, count=fr["shape"][1]).astype(bool)
        for m in modes:
            crop_mode, pad = MODE_SPECS[m]
            crops = [extract_crop(image, fr["boxes"][i], masks[i], mode=crop_mode, padding_frac=pad)
                     for i in range(len(fr["boxes"]))]
            fr["emb"][m] = novic.embed_crops(crops)
    with open(out / "cache.pkl", "wb") as fh:
        pickle.dump(cache, fh, protocol=4)
    novic.close()
    logger.info(f"Added {modes} embeddings to {out / 'cache.pkl'}")


# ─── scoring helpers ───

class Judge:
    def __init__(self):
        from nltk.corpus import wordnet as wn
        self.wn = wn

    def synsets(self, name: str):
        name = name.strip().lower()
        s = self.wn.synsets(name.replace(" ", "_").replace("'", ""), pos=self.wn.NOUN)
        if not s and " " in name:  # head noun of a phrase
            s = self.wn.synsets(name.split()[-1], pos=self.wn.NOUN)
        return s

    def __call__(self, pred: str, gt: str) -> tuple[bool, bool]:
        """(strict, lenient) correctness of name `pred` for class `gt`."""
        p_, g_ = pred.strip().lower(), gt.strip().lower()
        if p_ == g_:
            return True, True
        ps, gs = set(self.synsets(p_)), set(self.synsets(g_))
        if ps & gs:
            return True, True
        for p in ps:
            if set(p.closure(lambda s: s.hypernyms())) & gs:
                return False, True  # more specific: armchair for chair
            if any(p in set(g.hypernyms()) for g in gs):
                return False, True  # direct parent
        return False, False


def roc_auc(pos: np.ndarray, neg: np.ndarray) -> float:
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    ranks = np.concatenate([pos, neg]).argsort().argsort() + 1
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def pair_stats(emb: np.ndarray, inst: np.ndarray, merge_cos: float) -> dict:
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    cos = emb @ emb.T
    iu = np.triu_indices(len(emb), 1)
    c, same = cos[iu], (inst[:, None] == inst[None, :])[iu]
    return {"auc_same_vs_different": round(roc_auc(c[same], c[~same]), 4),
            "mean_cos_same": round(float(c[same].mean()), 4) if same.any() else None,
            "mean_cos_different": round(float(c[~same].mean()), 4),
            f"different_pairs_above_{merge_cos:.3f}": round(float((c[~same] >= merge_cos).mean()), 4),
            f"same_pairs_above_{merge_cos:.3f}": round(float((c[same] >= merge_cos).mean()), 4) if same.any() else None,
            "pairs_same": int(same.sum()), "pairs_different": int((~same).sum())}


def rate(xs):
    return round(float(np.mean(xs)), 4) if len(xs) else None


# ─── score ───

def score(args, cfg):
    from object_fusion.instance_fusion import build_instance_fusion_class
    from object_fusion.instance_tagger import to_everyday_names
    from object_fusion.streaming import prepare_native_libraries
    from scripts.run_static_scene_graph import build_graph_from_objects, load_everyday_names

    out = OUT_ROOT / args.name
    cache = pickle.loads((out / "cache.pkl").read_bytes())
    modes = args.modes.split(",")
    labels = {int(k): v for k, v in json.loads((out / "labels.json").read_text()).items()}
    frames = cache["frames"]
    for fr in frames:
        fr["masks"] = (np.unpackbits(fr["masks"], axis=-1, count=fr["shape"][1]).astype(bool)
                       if len(fr["boxes"]) else np.zeros((0, *fr["shape"]), bool))
    prepare_native_libraries()
    vd = build_voxel_map(cfg, Path(cache["dataset_dir"]))
    _, novic = load_models(cfg, tracker=False)
    judge, everyday = Judge(), load_everyday_names(cfg)
    fc = cfg["object_fusion"]
    merge_cos = fc["sim_threshold"] / fc["vis_weight"]  # cosine that merges on its own (no overlap)

    # per detection: object, class, mask / box voxels
    dets = []
    for f, fr in enumerate(frames):
        pim, pmask = vd.pixel_index_mappings[f], vd.pixel_index_masks[f]
        h, w = pim.shape
        for i, t in enumerate(fr["tids"]):
            lab = labels.get(int(t), "ignore")
            if lab == "ignore":
                continue
            x1, y1, x2, y2 = fr["boxes"][i].astype(int)
            box = np.zeros((h, w), bool)
            box[max(0, y1):y2, max(0, x1):x2] = True
            dets.append({"f": f, "i": i, "tid": int(t), "inst": lab, "cls": lab.split("|")[0],
                         "mask_vox": np.unique(pim[fr["masks"][i] & pmask]), "box_vox": np.unique(pim[box & pmask]),
                         "emb": {m: fr["emb"][m][i] for m in modes}})
    instances = sorted({d["inst"] for d in dets})
    inst_vox = {g: set(np.concatenate([d["mask_vox"] for d in dets if d["inst"] == g]).tolist()) for g in instances}
    owner = {}
    for g, vs in inst_vox.items():
        for v in vs:
            owner.setdefault(v, set()).add(g)

    def voxel_split(vox, g):
        own = other = bg = 0
        for v in vox.tolist():
            o = owner.get(v)
            if o is None:
                bg += 1
            elif g in o:
                own += 1
            else:
                other += 1
        n = max(len(vox), 1)
        return own / n, other / n, bg / n

    vm = [voxel_split(d["mask_vox"], d["inst"]) for d in dets]
    vb = [voxel_split(d["box_vox"], d["inst"]) for d in dets]
    results = {"video": args.name, "dataset_dir": cache["dataset_dir"], "frames": len(frames),
               "labelled_detections": len(dets), "objects": instances,
               "voxels_per_detection": {
                   "mask": {"own_object": rate([a for a, _, _ in vm]), "other_object": rate([b for _, b, _ in vm]),
                            "background": rate([c for _, _, c in vm]),
                            "voxels_mean": rate([len(d["mask_vox"]) for d in dets])},
                   "box": {"own_object": rate([a for a, _, _ in vb]), "other_object": rate([b for _, b, _ in vb]),
                           "background": rate([c for _, _, c in vb]),
                           "voxels_mean": rate([len(d["box_vox"]) for d in dets])},
                   "note": "fusion lifts the mask in both crop modes; mask 'own_object' is 1 - overlap with others by "
                           "construction, the box row shows what lifting the box would add"},
               "modes": {}}

    InstanceFusion = build_instance_fusion_class(cfg["external"]["scene_graph_generation_root"])
    inst_arr = np.array([d["inst"] for d in dets])
    tracks = sorted({d["tid"] for d in dets})
    previous = json.loads((out / "results.json").read_text()) if (out / "results.json").exists() else {}
    results["modes"] = previous.get("modes", {})
    for mode in modes:
        E = np.stack([d["emb"][mode] for d in dets]).astype(np.float32)
        # naming: every detection, and every track's average (the moving average)
        track_E = np.stack([E[[k for k, d in enumerate(dets) if d["tid"] == t]].mean(0) for t in tracks])
        track_cls = [next(d["cls"] for d in dets if d["tid"] == t) for t in tracks]
        track_inst = np.array([next(d["inst"] for d in dets if d["tid"] == t) for t in tracks])
        naming = {}
        for level, emb, classes in (("detection", E, [d["cls"] for d in dets]), ("track_average", track_E, track_cls)):
            # chunks: one beam-search batch over ~1000 embeddings needs > 12 GB
            raw = [r.label for k in range(0, len(emb), 64) for r in novic.decode_embeddings(emb[k:k + 64])]
            mapped = (to_everyday_names(raw, emb, everyday, novic.embed_text,
                                        cfg["label_mapper"].get("everyday_name_weight", 0.2)) if everyday else raw)
            jr = [judge(p, c) for p, c in zip(raw, classes)]
            jm = [judge(p, c) for p, c in zip(mapped, classes)]
            per_class = {}
            for (s, l_), c in zip(jm, classes):
                pc = per_class.setdefault(c, [0, 0, 0])
                pc[0] += 1
                pc[1] += int(s)
                pc[2] += int(l_)
            naming[level] = {"n": len(classes), "raw_strict": rate([s for s, _ in jr]),
                             "raw_lenient": rate([l_ for _, l_ in jr]), "mapped_strict": rate([s for s, _ in jm]),
                             "mapped_lenient": rate([l_ for _, l_ in jm]),
                             "per_class_mapped[n,strict,lenient]": per_class}
            if level == "track_average":
                naming[level]["tracks"] = [{"track": t, "object": ti, "raw": r, "mapped": m}
                                           for t, ti, r, m in zip(tracks, track_inst, raw, mapped)]
        matching = {"detections": pair_stats(E, inst_arr, merge_cos),
                    "track_averages": pair_stats(track_E, track_inst, merge_cos)}

        # the pipeline's fusion + clean-up + naming on this mode's embeddings
        fusion = InstanceFusion(downsample_voxel_size=fc["downsample_voxel_size"], spatial_weight=fc["spatial_weight"],
                                vis_weight=fc["vis_weight"], sim_threshold=fc["sim_threshold"])
        fusion.start_new_scan(vd)
        per_frame = [fusion.build_detection_list_from_frame(
            boxes=fr["boxes"], masks=fr["masks"], confs=fr["confs"], embeddings=fr["emb"][mode],
            pixel_indexes=vd.pixel_index_mappings[f], pixel_indexes_mask=vd.pixel_index_masks[f],
            image_name=vd.names[f], track_ids=fr["tids"]) for f, fr in enumerate(frames)]
        objects, _ = fusion.get_instances(view_dataset=vd, per_frame_detection_lists=per_frame,
                                          obj_min_detections=fc["obj_min_detections"],
                                          decode_fn=novic.decode_embeddings, match_level="track",
                                          track_end_frames=fc.get("track_end_frames", 10))
        objects, _, _ = build_graph_from_objects(cfg, copy.copy(objects), vd, novic, everyday, verbose=False)
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
            fused.append({"name": o["class_name"], "object": g, "purity": round(comp.get(g, 0) / tot, 3),
                          "objects_inside": {k: v for k, v in sorted(comp.items(), key=lambda kv: -kv[1])},
                          "voxel_purity": round(len(vox & gv) / max(len(vox), 1), 3),
                          "voxel_recall": round(len(vox & gv) / max(len(gv), 1), 3),
                          "name_strict": s, "name_lenient": l_})
        real_objs = [x for x in fused if x["object"] != "ignore"]
        good = [x for x in real_objs if x["purity"] >= 0.5]
        found = {x["object"] for x in good}
        min_det = fc["obj_min_detections"]
        present = [g for g in instances if sum(1 for d in dets if d["inst"] == g) >= min_det]
        per_obj = {}
        for x in good:
            per_obj[x["object"]] = per_obj.get(x["object"], 0) + 1
        fusion_scores = {
            "fused_objects": len(fused),
            "objects_in_video": len(present),
            "recall(object found, purity>=0.5)": round(len([g for g in present if g in found]) / max(len(present), 1), 3),
            "precision(fused object is one real object)": round(len(good) / max(len(fused), 1), 3),
            "missed_objects": [g for g in present if g not in found],
            "split_objects(>1 fused object)": {g: n for g, n in per_obj.items() if n > 1},
            "merged_objects(purity<0.5 or 2 objects >=20%)": [
                x["name"] for x in real_objs
                if x["purity"] < 0.5 or sum(1 for g, n in x["objects_inside"].items()
                                            if g != "ignore" and n >= 0.2 * sum(x["objects_inside"].values())) > 1],
            "mean_purity": rate([x["purity"] for x in real_objs]),
            "voxel_purity": rate([x["voxel_purity"] for x in good]),
            "voxel_recall": rate([x["voxel_recall"] for x in good]),
            "naming_strict": rate([x["name_strict"] for x in good]),
            "naming_lenient": rate([x["name_lenient"] for x in good]),
            "objects": fused,
        }
        results["modes"][mode] = {"naming": naming, "matching": matching, "fusion": fusion_scores}
        logger.info(f"[{args.name} / {mode}] naming det {naming['detection']['mapped_lenient']}, track-avg "
                    f"{naming['track_average']['mapped_lenient']}; AUC det {matching['detections']['auc_same_vs_different']}"
                    f"; fusion recall {fusion_scores['recall(object found, purity>=0.5)']}, precision "
                    f"{fusion_scores['precision(fused object is one real object)']}, naming "
                    f"{fusion_scores['naming_lenient']}")
    novic.close()
    (out / "results.json").write_text(json.dumps(results, indent=2, default=str))
    logger.info(f"Saved {out / 'results.json'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["prepare", "embed", "score"])
    ap.add_argument("--modes", default="bbox,segmented",
                    help=f"embed/score: comma-separated crop modes from {sorted(MODE_SPECS)}")
    ap.add_argument("--dataset_dir", help="prepare: the video's processed capture (data/processed/<run>)")
    ap.add_argument("--name", required=True, help="short name of the video, e.g. initial")
    ap.add_argument("--config", default="configs/pipeline_fyp_videos.yaml")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    from scripts.run_static_scene_graph import load_config

    cfg = load_config(REPO_ROOT / args.config)
    t0 = time.time()
    {"prepare": prepare, "embed": embed, "score": score}[args.step](args, cfg)
    logger.info(f"{args.step} took {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
