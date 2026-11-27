"""Front-end test bench: compare detection / segmentation / association variants on Replica without running the
rest of the pipeline (no fusion thread, no DoVSG, no scene graph).

Every variant sees the same 200 frames the pipeline sees (data/raw/replica_<scene>, resized to 640 px, the
brightness correction of the final configuration applied) and is scored against a per-pixel ground-truth instance
map ray-cast from Replica's semantic mesh with the ground-truth pose. A predicted mask is graded by the GT pixels
under it:

  object      >= 50% of its labelled pixels on ONE countable object
  stuff       >= 50% on wall / floor / ceiling
  group       neither, and >= 2 countable objects each hold >= 15% of it (a box around several objects)
  mixed       neither (object + background)
  unlabelled  mostly on pixels Replica leaves unlabelled (left out of precision)

Detection variants (--methods):
  pf          the current front end: YOLOE-26x prompt-free, 1024 px, conf 0.10 + BoT-SORT (pipeline call, which
              hands Ultralytics an RGB array although Ultralytics expects BGR for numpy input)
  pf_rgb      the same, with the channel order fixed (frame passed as BGR)
  pf_val      pf + the detection validation (border_stuff, label_stuff, group_test) + pixel ownership
  pf_ade      pf, rejecting a detection whose mask lies >= --ade_stuff_frac on ADE20K wall / floor / ceiling
              pixels (Mask2Former Swin-L ADE20K run in parallel)                                         [test c]
  tag         tag-then-detect: NOVIC tags each frame (whole frame + a 2x3 grid of tiles), physical nouns minus
              stuff words accumulate into a scene vocabulary, and YOLOE-26x (text-prompt model) is prompted with
              it instead of running prompt-free                                                           [test a]
  panoptic    Mask2Former Swin-L ADE20K panoptic: every pixel belongs to exactly one segment; the "thing" segments
              are the detections, stuff segments (wall, floor, ceiling, ...) are dropped                  [test c]

Association (--assoc_on, test b): BoT-SORT's track ids (pf / pf_rgb / pf_val) are compared with pose-aware
association in the style of ConceptGraphs: each map object (2.5 cm voxels + mean NOVIC embedding) is projected into
the frame with the pose and depth, and a detection joins the object with the best
max(2D IoU with the projection, 3D voxel overlap) + embedding cosine >= --pa_threshold (ConceptGraphs: 1.2);
objects overlapping >= 70% with cosine >= 0.7 are merged every 20 frames. Its objects are then scored in 3D like
the full pipeline (eval_replica_scene.py): found / duplicates / non-objects / object precision / F1, and the NOVIC
name of each object with the WordNet judge, next to the full pipeline's numbers (results/replica_final).

    HF_HUB_OFFLINE=1 python scripts/eval_frontend_variants.py --scene office_0 --gpu 1
    python scripts/eval_frontend_variants.py --combine          # results/frontend_tests/comparison.md

Outputs, results/frontend_tests/<scene>/: report.json, report.md, overlays_<method>.jpg (4 frames, each mask's
outline coloured by grade), pose_assoc_<method>_nodes.csv.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logger = logging.getLogger("frontend")
GT_ROOT = Path("/home/ravindu/SceneGraphNav/datasets/replica/extracted")
OUT_ROOT = REPO_ROOT / "results" / "frontend_tests"
FINAL_OVERRIDES = ["merge_fixes", "brightness", "fusion_naming", "discovery"]
WIDTH = 640
KIND = {"none": 0, "countable": 1, "structural": 2, "ignored": 3}
GRADE_COLORS = {"object": (40, 200, 40), "stuff": (230, 40, 40), "group": (255, 150, 0), "mixed": (200, 0, 200),
                "unlabelled": (150, 150, 150)}
ADE_STRUCTURAL = {"wall", "floor", "ceiling"}
# ADE20K classes that are surfaces / outdoor regions rather than objects (dropped from the panoptic detections)
ADE_STUFF = ADE_STRUCTURAL | {"building", "sky", "tree", "road", "grass", "sidewalk", "earth", "mountain", "water",
                              "sea", "field", "sand", "river", "hill", "land", "skyscraper", "path", "runway",
                              "dirt track", "lake", "waterfall", "house", "hovel", "tower", "bridge", "pier"}
# words NOVIC returns for whole views that name a place or a surface, not an object
TAG_STOP = {"office", "room", "interior", "home", "house", "building", "apartment", "space", "scene", "workplace",
            "lounge", "hall", "hallway", "corridor", "lobby", "studio", "kitchen", "bedroom", "bathroom", "corner",
            "wall", "floor", "ceiling", "flooring", "parquet", "carpeting", "surface", "interior design", "furnishing",
            "room access", "doorway", "world", "wallpaper", "panelling", "paneling", "molding", "moulding", "tile"}


# ─── data ───────────────────────────────────────────────────────────────────────────────────────────────────────

def deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in (over or {}).items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def final_config() -> dict:
    cfg = yaml.safe_load(open(REPO_ROOT / "configs" / "pipeline.yaml"))
    for v in FINAL_OVERRIDES:
        cfg = deep_merge(cfg, yaml.safe_load(open(REPO_ROOT / "configs" / "replica_variants" / f"{v}.yaml")) or {})
    return cfg


def load_frames(scene: str, enhance_cfg) -> tuple[list[dict], np.ndarray]:
    from perception.image_enhance import enhance
    raw = REPO_ROOT / "data" / "raw" / f"replica_{scene}"
    cam = yaml.safe_load(open(raw / "camera.yaml"))
    s = WIDTH / cam["width"]
    H = int(round(cam["height"] * s))
    K = np.array([[cam["fx"] * s, 0, cam["cx"] * s], [0, cam["fy"] * s, cam["cy"] * s], [0, 0, 1.0]])
    frames = []
    for p in sorted((raw / "rgb").glob("*.jpg")):
        rgb = cv2.resize(cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB), (WIDTH, H), interpolation=cv2.INTER_AREA)
        d = cv2.imread(str(raw / "depth" / f"{p.stem}.png"), cv2.IMREAD_UNCHANGED).astype(np.float64) / cam["depth_scale"]
        frames.append({"name": p.stem, "rgb": enhance(rgb, enhance_cfg), "raw_rgb": rgb,
                       "depth": cv2.resize(d, (WIDTH, H), interpolation=cv2.INTER_NEAREST),
                       "c2w": np.loadtxt(raw / "poses" / f"{p.stem}.txt")})
    return frames, K


def gt_id_maps(scene: str, frames: list[dict], K: np.ndarray) -> np.ndarray:
    """(F, H, W) Replica object id per pixel, ray-cast through the semantic mesh (0 = no surface)."""
    cache = REPO_ROOT / "data" / "cache" / "frontend_gt" / f"{scene}_w{WIDTH}_{len(frames)}.npz"
    if cache.exists():
        return np.load(cache)["ids"]
    import open3d as o3d
    from plyfile import PlyData
    ply = PlyData.read(str(GT_ROOT / scene / "habitat" / "mesh_semantic.ply"))
    v = ply["vertex"]
    verts = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float32)
    faces, fobj = list(ply["face"]["vertex_indices"]), np.asarray(ply["face"]["object_id"], np.int64)
    tris, tobj = [], []
    for k in range(3, max(len(f) for f in faces) + 1):
        sel = [i for i, f in enumerate(faces) if len(f) == k]
        if sel:
            F = np.stack([faces[i] for i in sel]).astype(np.int64)
            for j in range(1, k - 1):
                tris.append(F[:, [0, j, j + 1]])
                tobj.append(fobj[sel])
    tris, tobj = np.concatenate(tris).astype(np.uint32), np.concatenate(tobj)
    rc = o3d.t.geometry.RaycastingScene()
    rc.add_triangles(o3d.core.Tensor(verts), o3d.core.Tensor(tris))
    H, W = frames[0]["depth"].shape
    out, err = np.zeros((len(frames), H, W), np.uint16), []
    for i, f in enumerate(frames):
        rays = rc.create_rays_pinhole(o3d.core.Tensor(K), o3d.core.Tensor(np.linalg.inv(f["c2w"])), W, H)
        hit = rc.cast_rays(rays)
        prim = hit["primitive_ids"].numpy().astype(np.int64)
        valid = prim != o3d.t.geometry.RaycastingScene.INVALID_ID
        out[i][valid] = tobj[prim[valid]]
        # ray length -> depth along the optical axis, to check the mesh lines up with the frame's depth
        dirs = rays.numpy()[..., 3:]
        z = hit["t_hit"].numpy() * (dirs @ np.linalg.inv(f["c2w"])[2, :3])
        ok = valid & (f["depth"] > 0)
        err.append(np.median(np.abs(z[ok] - f["depth"][ok])))
    logger.info("GT id maps: median |mesh depth - frame depth| %.4f m", float(np.median(err)))
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, ids=out)
    return out


# ─── detection variants ─────────────────────────────────────────────────────────────────────────────────────────

def as_dets(masks, boxes=None, confs=None, labels=None, ids=None) -> dict:
    n = len(masks)
    if boxes is None:
        boxes = np.array([mask_box(m) for m in masks]).reshape(-1, 4)
    return {"masks": np.asarray(masks, bool).reshape(n, *(masks[0].shape if n else (0, 0))),
            "boxes": np.asarray(boxes, float).reshape(-1, 4),
            "confs": np.ones(n) if confs is None else np.asarray(confs, float),
            "labels": [""] * n if labels is None else list(labels),
            "ids": np.full(n, -1) if ids is None else np.asarray(ids, int)}


def mask_box(m):
    ys, xs = np.nonzero(m)
    return [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1] if len(xs) else [0, 0, 0, 0]


def yoloe_tracked(cfg, validation=None, bgr=False):
    from perception.detection.tracked_yoloe_detector import TrackedYOLOEDetector
    d = cfg["detection"]
    det = TrackedYOLOEDetector(d["weights"], d["conf_thresh"], d["iou_thresh"], d["img_size"], "cuda",
                               d.get("tracker", "botsort.yaml"), validation=validation, fix_channel_order=bgr)

    def run(f):
        out = det.track(f["rgb"])
        if not out:
            return as_dets(np.zeros((0,) + f["rgb"].shape[:2], bool))
        return as_dets([o.mask for o in out], [o.bbox for o in out], [o.conf for o in out],
                       [o.label for o in out], [o.track_id for o in out])
    return run, det


class Mask2FormerADE:
    """One forward pass per frame, giving both the panoptic segments and the semantic map."""
    NAME = "facebook/mask2former-swin-large-ade-panoptic"

    def __init__(self, threshold=0.5):
        import torch
        from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
        self.torch = torch
        self.proc = AutoImageProcessor.from_pretrained(self.NAME)
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(self.NAME).cuda().eval()
        self.names = {int(k): v.strip() for k, v in self.model.config.id2label.items()}
        self.stuff_ids = {k for k, v in self.names.items() if v in ADE_STUFF}
        self.struct_ids = [k for k, v in self.names.items() if v in ADE_STRUCTURAL]
        self.threshold = threshold
        self.cache = {}
        self.seconds = 0.0

    def __call__(self, f):
        if f["name"] in self.cache:
            return self.cache[f["name"]]
        t0 = time.time()
        torch = self.torch
        H, W = f["raw_rgb"].shape[:2]
        inp = self.proc(images=f["raw_rgb"], return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = self.model(**inp)
        pan = self.proc.post_process_panoptic_segmentation(out, threshold=self.threshold, target_sizes=[(H, W)],
                                                           label_ids_to_fuse=self.stuff_ids)[0]
        sem = self.proc.post_process_semantic_segmentation(out, target_sizes=[(H, W)])[0].cpu().numpy()
        seg = pan["segmentation"].cpu().numpy()
        things = [s for s in pan["segments_info"] if s["label_id"] not in self.stuff_ids]
        masks = [seg == s["id"] for s in things]
        dets = as_dets(masks if masks else np.zeros((0, H, W), bool), confs=[s["score"] for s in things],
                       labels=[self.names[s["label_id"]] for s in things])
        r = {"dets": dets, "structural": np.isin(sem, self.struct_ids),
             "stuff_segments": len(pan["segments_info"]) - len(things)}
        self.seconds += time.time() - t0
        self.cache[f["name"]] = r
        return r


class NovicTagger:
    """NOVIC on the whole frame and a 2x3 grid of overlapping tiles; physical nouns, stuff words removed."""

    def __init__(self, novic, min_score=0.1, rows=2, cols=3, overlap=0.25):
        from evaluation.naming_judge import noun_senses
        from perception.detection.detection_validation import is_stuff_label
        self.novic, self.min_score, self.rows, self.cols, self.overlap = novic, min_score, rows, cols, overlap
        self.noun_senses, self.is_stuff = noun_senses, is_stuff_label
        self.seconds = 0.0

    def views(self, img):
        H, W = img.shape[:2]
        th, tw = H / (self.rows - (self.rows - 1) * self.overlap), W / (self.cols - (self.cols - 1) * self.overlap)
        out = [img]
        for r in range(self.rows):
            for c in range(self.cols):
                y, x = int(r * th * (1 - self.overlap)), int(c * tw * (1 - self.overlap))
                out.append(img[y:min(int(y + th), H), x:min(int(x + tw), W)])
        return out

    def keep(self, word):
        w = word.lower().strip()
        return (w and w not in TAG_STOP and not self.is_stuff(w) and w.split()[-1] not in TAG_STOP
                and bool(self.noun_senses(w)))

    def __call__(self, rgb) -> set[str]:
        t0 = time.time()
        res = self.novic.decode_embeddings(self.novic.embed_crops(self.views(rgb)))
        tags = {lab.lower().strip() for r in res for lab, p in r.top_k if p >= self.min_score and self.keep(lab)}
        self.seconds += time.time() - t0
        return tags


def tag_then_detect(cfg, tagger, conf, bgr=False, max_vocab=300):
    from ultralytics import YOLOE
    weights = cfg["detection"]["weights"].replace("-seg-pf.pt", "-seg.pt")
    model = YOLOE(weights).to("cuda")
    state = {"vocab": [], "counts": Counter(), "changes": 0, "set_s": 0.0}

    def run(f):
        new = tagger(f["rgb"])
        state["counts"].update(new)
        vocab = sorted(w for w, _ in state["counts"].most_common(max_vocab))
        if vocab != state["vocab"]:
            t0 = time.time()
            model.set_classes(vocab, model.get_text_pe(vocab))
            model.predictor = None                       # rebuild with the new class names
            state["vocab"], state["changes"] = vocab, state["changes"] + 1
            state["set_s"] += time.time() - t0
        H, W = f["rgb"].shape[:2]
        if not vocab:
            return as_dets(np.zeros((0, H, W), bool))
        img = np.ascontiguousarray(f["rgb"][..., ::-1]) if bgr else f["rgb"]
        r = model.predict(img, conf=conf, iou=cfg["detection"]["iou_thresh"], imgsz=cfg["detection"]["img_size"],
                          verbose=False, retina_masks=True)[0]
        if r.masks is None or len(r.boxes) == 0:
            return as_dets(np.zeros((0, H, W), bool))
        m = r.masks.data.cpu().numpy().astype(bool)
        if m.shape[1:] != (H, W):
            m = np.stack([cv2.resize(x.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) for x in m]) > 0
        return as_dets(m, r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(),
                       [r.names[int(c)] for c in r.boxes.cls.cpu().numpy()])
    return run, state


def apply_ownership(d: dict, own_cfg: dict) -> dict:
    from perception.detection.detection_validation import pixel_ownership
    if not len(d["masks"]):
        return d
    owned, _ = pixel_ownership(list(d["masks"]), own_cfg.get("containment", 0.8), own_cfg.get("min_owned_frac", 0.1))
    return dict(d, masks=np.stack(owned))


# ─── frame-level scoring ────────────────────────────────────────────────────────────────────────────────────────

def grade_frame(d: dict, gt: np.ndarray, kind_lut: np.ndarray, min_pixels: int = 100) -> dict:
    L = len(kind_lut)
    gt_area = np.bincount(gt.ravel(), minlength=L)
    present = [o for o in np.nonzero(gt_area >= min_pixels)[0] if kind_lut[o] == KIND["countable"]]
    grades, dom, purity, iou = [], [], [], []
    for m in d["masks"]:
        c = np.bincount(gt[m], minlength=L)
        area = int(m.sum())
        lab = c * ((kind_lut == 1) | (kind_lut == 2))
        nlab = lab.sum()
        cnt = c * (kind_lut == 1)
        o = int(cnt.argmax())
        p = cnt[o] / nlab if nlab else 0.0
        if area == 0 or nlab < 0.5 * area:
            g = "unlabelled"
        elif c[kind_lut == 2].sum() / nlab >= 0.5:
            g = "stuff"
        elif p >= 0.5:
            g = "object"
        elif (cnt / nlab >= 0.15).sum() >= 2:
            g = "group"
        else:
            g = "mixed"
        grades.append(g)
        dom.append(o if g == "object" else 0)
        purity.append(p)
        iou.append(cnt[o] / (area + gt_area[o] - cnt[o]) if g == "object" else 0.0)
    hit = {o for o, g, i in zip(dom, grades, iou) if g == "object" and i >= 0.5}
    seen = {o for o, g in zip(dom, grades) if g == "object"}
    # overlap between detections: pixels claimed twice, nested pairs (a mask >= 80% inside a larger one)
    n = len(d["masks"])
    multi, nested, nested_same = 0.0, 0, 0
    if n:
        cover = d["masks"].sum(0)
        multi = float((cover >= 2).sum() / max((cover >= 1).sum(), 1))
        q = np.stack([cv2.resize(m.astype(np.uint8), (WIDTH // 4, m.shape[0] // 4), interpolation=cv2.INTER_AREA)
                      for m in d["masks"]]).reshape(n, -1).astype(np.float32)
        I = q @ q.T
        a = np.diag(I)
        for i in range(n):
            for j in range(n):
                if i != j and a[j] > 0 and I[i, j] >= 0.8 * a[j] and a[j] < 0.8 * a[i]:
                    nested += 1
                    nested_same += int(dom[i] and dom[i] == dom[j])
    return {"grades": grades, "dom": dom, "purity": purity, "iou": iou, "present": present, "hit": hit,
            "seen": seen, "multi": multi, "nested": nested, "nested_same": nested_same}


def summarize(per_frame: list[dict], labels: list[list[str]], gt_label, visible: set[int]) -> dict:
    from evaluation.naming_judge import grade
    g = Counter(x for f in per_frame for x in f["grades"])
    total = sum(g.values())
    scored = total - g["unlabelled"]
    occ = sum(len(f["present"]) for f in per_frame)
    found = set().union(*[f["hit"] for f in per_frame]) & visible
    seen = set().union(*[f["seen"] for f in per_frame]) & visible
    obj_pur = [p for f in per_frame for p, x in zip(f["purity"], f["grades"]) if x == "object"]
    names = Counter()
    for f, labs in zip(per_frame, labels):
        for x, o, lab in zip(f["grades"], f["dom"], labs):
            if x == "object" and lab:
                names[grade(lab, gt_label(o))] += 1
    named = sum(names.values())
    return {
        "detections_per_frame": total / max(len(per_frame), 1),
        "grades": dict(g),
        "non_object_share": (g["stuff"] + g["group"] + g["mixed"]) / max(scored, 1),
        "stuff_per_frame": g["stuff"] / max(len(per_frame), 1),
        "group_per_frame": g["group"] / max(len(per_frame), 1),
        "frame_precision": g["object"] / max(scored, 1),
        "occurrence_recall_iou50": sum(len(f["hit"]) for f in per_frame) / max(occ, 1),
        "objects_found_iou50": len(found), "objects_seen": len(seen), "objects_visible": len(visible),
        "missed": sorted(f"{gt_label(o)}#{o}" for o in visible - seen),
        "object_purity": float(np.mean(obj_pur)) if obj_pur else None,
        "pixels_claimed_twice": float(np.mean([f["multi"] for f in per_frame])),
        "nested_pairs_per_frame": sum(f["nested"] for f in per_frame) / max(len(per_frame), 1),
        "nested_same_object_per_frame": sum(f["nested_same"] for f in per_frame) / max(len(per_frame), 1),
        "label_naming": ({"scored": named, "acceptable": (names["exact"] + names["specific"]) / named,
                          "counts": dict(names)} if named else None),
    }


def association_metrics(per_frame: list[dict], ids: list[np.ndarray]) -> dict:
    """Track purity (share of a track's object detections on its majority GT object) and fragmentation (distinct
    ids per GT object), over detections graded 'object'."""
    by_id, by_obj = defaultdict(Counter), defaultdict(set)
    for f, fid in zip(per_frame, ids):
        for x, o, t in zip(f["grades"], f["dom"], fid):
            if x == "object" and t >= 0:
                by_id[t][o] += 1
                by_obj[o].add(t)
    dets = sum(sum(c.values()) for c in by_id.values())
    pure = sum(c.most_common(1)[0][1] for c in by_id.values())
    return {"ids": len(by_id), "track_purity": pure / max(dets, 1),
            "ids_per_object": float(np.mean([len(v) for v in by_obj.values()])) if by_obj else None,
            "impure_ids": sum(1 for c in by_id.values() if c.most_common(1)[0][1] < 0.8 * sum(c.values()))}


# ─── pose-aware association (ConceptGraphs-style) ───────────────────────────────────────────────────────────────

VOX = 0.025


def vox_keys(pts):
    v = np.floor(pts / VOX).astype(np.int64) + (1 << 20)
    return np.unique((v[:, 0] << 42) | (v[:, 1] << 21) | v[:, 2])


def key_centers(k):
    m = (1 << 21) - 1
    return (np.stack([(k >> 42) & m, (k >> 21) & m, k & m], 1) - (1 << 20) + 0.5) * VOX


def detection_points(mask, depth, K, c2w):
    import open3d as o3d
    v, u = np.nonzero(mask & (depth > 0))
    if len(u) < 20:
        return None
    z = depth[v, u]
    cam = np.stack([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z], 1)
    pts = cam @ c2w[:3, :3].T + c2w[:3, 3]
    pts = key_centers(vox_keys(pts))
    if len(pts) < 10:
        return None
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    lab = np.asarray(pc.cluster_dbscan(eps=0.1, min_points=10))
    if (lab >= 0).sum() < 10:
        return None
    pts = pts[lab == np.bincount(lab[lab >= 0]).argmax()]
    return vox_keys(pts)


class PoseAwareAssociation:
    def __init__(self, K, shape, threshold=1.2, min_geo=0.1, merge_every=20, merge_overlap=0.7, merge_cos=0.7,
                 geo_mode="max"):
        self.K, self.shape, self.geo_mode = K, shape, geo_mode
        self.threshold, self.min_geo = threshold, min_geo
        self.merge_every, self.merge_overlap, self.merge_cos = merge_every, merge_overlap, merge_cos
        self.objs: dict[int, dict] = {}
        self.redirect: dict[int, int] = {}
        self.next_id = 0
        self.seconds = 0.0

    def resolve(self, i):
        while i in self.redirect:
            i = self.redirect[i]
        return i

    def projected(self, ids, depth, c2w, q=4):
        """Quarter-resolution masks of where each object should appear in this frame (occlusion-tested)."""
        H, W = self.shape
        h, w = H // q, W // q
        out = np.zeros((len(ids), h * w), np.float32)
        w2c = np.linalg.inv(c2w)
        for r, i in enumerate(ids):
            p = self.objs[i]["pts"] @ w2c[:3, :3].T + w2c[:3, 3]
            z = p[:, 2]
            ok = z > 0.1
            u = np.round(self.K[0, 0] * p[ok, 0] / z[ok] + self.K[0, 2]).astype(int)
            v = np.round(self.K[1, 1] * p[ok, 1] / z[ok] + self.K[1, 2]).astype(int)
            zz = z[ok]
            inside = (u >= 0) & (u < W) & (v >= 0) & (v < H)
            u, v, zz = u[inside], v[inside], zz[inside]
            d = depth[v, u]
            vis = (d > 0) & (zz < d + 0.1)
            if vis.any():
                img = np.zeros((h, w), np.uint8)
                img[np.minimum(v[vis] // q, h - 1), np.minimum(u[vis] // q, w - 1)] = 1
                out[r] = cv2.dilate(img, np.ones((3, 3), np.uint8)).ravel()
        return out

    def add(self, keys, emb, frame, box):
        i = self.next_id
        self.next_id += 1
        pts = key_centers(keys)
        self.objs[i] = {"keys": keys, "pts": pts, "lo": pts.min(0), "hi": pts.max(0), "emb": emb.copy(), "n": 1,
                        "frames": [frame], "boxes": [box]}
        return i

    def absorb(self, i, keys, emb, n, frames, boxes):
        o = self.objs[i]
        o["keys"] = np.union1d(o["keys"], keys)
        o["pts"] = key_centers(o["keys"])
        o["lo"], o["hi"] = o["pts"].min(0), o["pts"].max(0)
        o["emb"] = (o["emb"] * o["n"] + emb * n) / (o["n"] + n)
        o["n"] += n
        o["frames"] += frames
        o["boxes"] += boxes

    def step(self, fi, frame, masks, boxes, embs, K, c2w, depth, keys=None) -> np.ndarray:
        t0 = time.time()
        out = np.full(len(masks), -1)
        if keys is None:
            keys = [detection_points(m, depth, K, c2w) for m in masks]
        ids = list(self.objs)
        if ids:
            P = self.projected(ids, depth, c2w)
            H, W = self.shape
            D = np.stack([cv2.resize(m.astype(np.uint8), (W // 4, H // 4), interpolation=cv2.INTER_AREA)
                          for m in masks]).reshape(len(masks), -1).astype(np.float32) if len(masks) else None
            inter = D @ P.T if D is not None else None
            iou = inter / (D.sum(1)[:, None] + P.sum(1)[None] - inter + 1e-6) if D is not None else None
            E = np.stack([self.objs[i]["emb"] for i in ids])
            E = E / np.linalg.norm(E, axis=1, keepdims=True)
        for j, k in enumerate(keys):
            if k is None:
                continue
            e = embs[j] / np.linalg.norm(embs[j])
            best, best_s = None, -1
            if ids:
                pk = key_centers(k)
                lo, hi = pk.min(0), pk.max(0)
                cos = E @ e
                for r, i in enumerate(ids):
                    o = self.objs[i]
                    ov = 0.0
                    if np.all(o["lo"] <= hi + VOX) and np.all(lo <= o["hi"] + VOX):
                        ov = np.isin(k, o["keys"], assume_unique=True).mean()
                    geo = ov if self.geo_mode == "3d" else max(ov, iou[j, r])
                    s = geo + cos[r]
                    if geo >= self.min_geo and s >= self.threshold and s > best_s:
                        best, best_s = i, s
            if best is None:
                out[j] = self.add(k, embs[j], frame, boxes[j])
            else:
                self.absorb(best, k, embs[j], 1, [frame], [boxes[j]])
                out[j] = best
        if self.merge_every and (fi + 1) % self.merge_every == 0:
            self.merge()
        self.seconds += time.time() - t0
        return out

    def merge(self):
        changed = True
        while changed:
            changed = False
            ids = sorted(self.objs, key=lambda i: -len(self.objs[i]["keys"]))
            for a_i, a in enumerate(ids):
                if a not in self.objs:
                    continue
                A = self.objs[a]
                ea = A["emb"] / np.linalg.norm(A["emb"])
                for b in ids[a_i + 1:]:
                    if b not in self.objs:
                        continue
                    B = self.objs[b]
                    if not (np.all(A["lo"] <= B["hi"] + VOX) and np.all(B["lo"] <= A["hi"] + VOX)):
                        continue
                    ov = np.isin(B["keys"], A["keys"], assume_unique=True).mean()
                    if ov >= self.merge_overlap and float(ea @ (B["emb"] / np.linalg.norm(B["emb"]))) >= self.merge_cos:
                        self.absorb(a, B["keys"], B["emb"], B["n"], B["frames"], B["boxes"])
                        del self.objs[b]
                        self.redirect[b] = a
                        changed = True


class CommitAssociation(PoseAwareAssociation):
    """Pose-aware association applied at track commit only: BoT-SORT tracks are accumulated (voxel union, mean
    embedding) and, once a track has not been seen for `patience` frames (or at the end), the whole track joins the
    map object with the best 3D overlap (share of the track's voxels inside the object) + cosine >= threshold, or
    becomes a new object. Untracked detections commit at once."""

    def __init__(self, *a, patience=10, **kw):
        super().__init__(*a, **kw)
        self.patience = patience
        self.pending: dict[int, dict] = {}
        self.track_to_obj: dict[int, int] = {}

    def _commit(self, t):
        T = self.pending.pop(t) if t in self.pending else t
        k, e = T["keys"], T["emb"] / np.linalg.norm(T["emb"])
        best, best_s = None, -1
        pk = key_centers(k)
        lo, hi = pk.min(0), pk.max(0)
        for i, o in self.objs.items():
            if not (np.all(o["lo"] <= hi + VOX) and np.all(lo <= o["hi"] + VOX)):
                continue
            geo = np.isin(k, o["keys"], assume_unique=True).mean()
            s = geo + float(e @ (o["emb"] / np.linalg.norm(o["emb"])))
            if geo >= self.min_geo and s >= self.threshold and s > best_s:
                best, best_s = i, s
        if best is None:
            best = self.add(k, T["emb"], T["frames"][0], T["boxes"][0])
            o = self.objs[best]
            o["n"], o["frames"], o["boxes"] = T["n"], list(T["frames"]), list(T["boxes"])
        else:
            self.absorb(best, k, T["emb"], T["n"], T["frames"], T["boxes"])
        return best

    def step_tracks(self, fi, frame, boxes, embs, keys, tids) -> list:
        """Returns, per detection, a handle resolved to an object id by finish()."""
        t0 = time.time()
        out = []
        for j, (k, t) in enumerate(zip(keys, tids)):
            if k is None:
                out.append(None)
                continue
            if t < 0:
                out.append(("obj", self._commit({"keys": k, "emb": embs[j], "n": 1, "frames": [frame],
                                                  "boxes": [boxes[j]]})))
                continue
            T = self.pending.setdefault(t, {"keys": k, "emb": np.zeros_like(embs[j]), "n": 0, "frames": [],
                                            "boxes": [], "last": fi})
            if T["n"]:
                T["keys"] = np.union1d(T["keys"], k)
            T["emb"] = (T["emb"] * T["n"] + embs[j]) / (T["n"] + 1)
            T["n"] += 1
            T["frames"].append(frame)
            T["boxes"].append(boxes[j])
            T["last"] = fi
            out.append(("track", t))
        for t in [t for t, T in self.pending.items() if fi - T["last"] >= self.patience]:
            self.track_to_obj[t] = self._commit(t)
        if self.merge_every and (fi + 1) % self.merge_every == 0:
            self.merge()
        self.seconds += time.time() - t0
        return out

    def finish(self, handles):
        for t in list(self.pending):
            self.track_to_obj[t] = self._commit(t)
        self.merge()
        res = []
        for hs in handles:
            res.append(np.array([-1 if h is None else self.resolve(h[1] if h[0] == "obj" else self.track_to_obj[h[1]])
                                 for h in hs]))
        return res


def embed_detections(novic, f, d, cfg):
    from semantic_decoding.novic.crop_utils import extract_crop
    sem = cfg["semantic_decoding"]
    if not len(d["masks"]):
        return np.zeros((0, 1), np.float32)
    crops = [extract_crop(f["rgb"], b, m, mode=sem["crop_mode"], padding_px=sem["crop_padding_px"],
                          padding_frac=sem.get("crop_padding_frac")) for b, m in zip(d["boxes"], d["masks"])]
    return novic.embed_crops(crops)


def score_objects_3d(pa: PoseAwareAssociation, gt, visible, novic, min_detections=2) -> tuple[dict, list[dict]]:
    """The full-pipeline object metrics on the association's objects (map frame == mesh frame here)."""
    from evaluation.naming_judge import grade
    from evaluation.open_vocab import Node, match_nodes, node_metrics
    objs = [(i, o) for i, o in pa.objs.items() if o["n"] >= min_detections and len(o["keys"]) >= 10]
    nodes = [Node(node_id=f"obj_{i}", name="", novic_top_k=[], embedding=o["emb"], points=o["pts"]) for i, o in objs]
    tags = novic.decode_embeddings(np.stack([o["emb"] for _, o in objs]).astype(np.float32)) if objs else []
    for n, t in zip(nodes, tags):
        n.name = t.label
    match_nodes(nodes, gt.xyz, gt.object_id, gt.kind, gt.label)
    for n in nodes:
        if n.status == "pending":
            n.status = "wrong_name"             # names graded separately below
    m = node_metrics(nodes, sorted(visible), gt.label)
    sc = m["status_counts"]
    real = sc.get("wrong_name", 0)
    fake = sc.get("structural", 0) + sc.get("mixed", 0) + sc.get("phantom", 0)
    prec = m["valid_objects"] / max(real + fake, 1)
    rec = m["node_recall"] or 0
    graded = Counter(grade(n.name, n.gt_label) for n in nodes if n.status == "wrong_name")
    rows = [{"node_id": n.node_id, "name": n.name, "status": n.status, "gt": n.gt_label, "purity": round(n.purity, 2),
             "detections": o["n"], "voxels": len(o["keys"])} for n, (_, o) in zip(nodes, objs)]
    return {"nodes": len(nodes), "non_objects": fake, "scored_nodes": real + fake, "structural": sc.get("structural", 0),
            "mixed": sc.get("mixed", 0), "phantom": sc.get("phantom", 0), "found": m["valid_objects"],
            "visible": m["gt_objects_visible"], "duplicates": m["n_dup"], "object_precision": prec, "recall": rec,
            "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
            "naming_acceptable": (graded["exact"] + graded["specific"]) / max(sum(graded.values()), 1),
            "missed": m["missed_objects"]}, rows


# ─── overlays ───────────────────────────────────────────────────────────────────────────────────────────────────

def overlay(f, d, fr) -> np.ndarray:
    img = (f["rgb"] * 0.6).astype(np.uint8)
    for m, g, lab in zip(d["masks"], fr["grades"], d["labels"]):
        col = GRADE_COLORS[g]
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, cnts, -1, col, 2)
        if lab and cnts:
            x, y, _, _ = cv2.boundingRect(max(cnts, key=cv2.contourArea))
            cv2.putText(img, lab[:18], (x + 2, y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1, cv2.LINE_AA)
    return img


# ─── main ───────────────────────────────────────────────────────────────────────────────────────────────────────

def run_scene(args):
    from evaluation.replica_gt import compute_visibility, load_replica_gt
    from semantic_decoding.novic.novic_decoder import NOVICDecoder
    cfg = final_config()
    out = OUT_ROOT / args.scene
    out.mkdir(parents=True, exist_ok=True)
    frames, K = load_frames(args.scene, cfg["detection"].get("enhance"))
    frames = frames[: args.max_frames] if args.max_frames else frames
    gt_ids = gt_id_maps(args.scene, frames, K)
    gt = load_replica_gt(GT_ROOT / args.scene)
    compute_visibility(gt, [(K, f["c2w"], f["depth"]) for f in frames], np.eye(4))
    visible = set(gt.countable_visible())
    kind_lut = np.zeros(int(max(max(gt.objects), gt_ids.max())) + 1, np.int64)
    for o, v in gt.objects.items():
        kind_lut[o] = KIND[v["kind"]]
    logger.info("%s: %d frames, %d countable objects visible", args.scene, len(frames), len(visible))
    sem = cfg["semantic_decoding"]
    novic = NOVICDecoder(novic_dir=sem["novic_dir"], checkpoint_path=sem["novic_checkpoint"], gencfg=sem["gencfg"],
                         device="cuda", embedding_batch_size=16)
    if novic.is_stub:
        raise SystemExit("NOVIC failed to load (run with HF_HUB_OFFLINE=1)")
    m2f = Mask2FormerADE(args.pan_threshold) if {"pf_ade", "panoptic"} & set(args.methods) else None
    val_cfg = {k: yaml.safe_load(open(REPO_ROOT / "configs" / "replica_variants" / f"{v}.yaml"))["detection"]["validation"]
               for k, v in (("stuff", "val_stuff_detect"), ("group", "val_group_test"), ("own", "val_ownership"))}
    report = {"scene": args.scene, "frames": len(frames), "visible_objects": len(visible), "methods": {},
              "association": {}, "settings": vars(args)}
    show = np.linspace(0, len(frames) - 1, 4).astype(int)
    for method in args.methods:
        logger.info("method %s", method)
        extra = {}
        if method in ("pf", "pf_rgb", "pf_ade"):
            run, det = yoloe_tracked(cfg, bgr=(method == "pf_rgb") or args.bgr_fix)
        elif method == "pf_val":
            run, det = yoloe_tracked(cfg, {**val_cfg["stuff"], **val_cfg["group"]}, bgr=args.bgr_fix)
        elif method == "tag":
            tagger = NovicTagger(novic, args.tag_min_score)
            run, extra = tag_then_detect(cfg, tagger, args.tag_conf, bgr=args.bgr_fix)
        elif method == "panoptic":
            run = lambda f: m2f(f)["dets"]                       # noqa: E731
        else:
            raise SystemExit(f"unknown method {method}")
        per_frame, labels, ids, t_det, dropped = [], [], [], 0.0, Counter()
        keep_for_assoc = method in args.assoc_on
        stored = []
        tiles = []
        for fi, f in enumerate(frames):
            t0 = time.time()
            d = run(f)
            t_det += time.time() - t0
            if method == "pf_val":
                d = apply_ownership(d, val_cfg["own"]["pixel_ownership"])
            if method == "pf_ade" and len(d["masks"]):
                st = m2f(f)["structural"]
                share = np.array([(st & m).sum() / max(m.sum(), 1) for m in d["masks"]])
                keep = share < args.ade_stuff_frac
                dropped["ade_structural"] += int((~keep).sum())
                d = {k: (v[keep] if isinstance(v, np.ndarray) else [x for x, kk in zip(v, keep) if kk])
                     for k, v in d.items()}
            fr = grade_frame(d, gt_ids[fi].astype(np.int64), kind_lut)
            per_frame.append(fr)
            labels.append(d["labels"])
            ids.append(d["ids"])
            if keep_for_assoc:
                stored.append({"packed": np.packbits(d["masks"], axis=-1), "boxes": d["boxes"], "ids": d["ids"]})
            if fi in show:
                tiles.append(overlay(f, d, fr))
        s = summarize(per_frame, labels, gt.label, visible)
        s["seconds_per_frame"] = t_det / len(frames)
        if method == "tag":
            tagger_s = tagger.seconds / len(frames)
            s["tagging_seconds_per_frame"] = tagger_s
            s["vocabulary"] = extra["vocab"]
            s["vocabulary_changes"] = extra["changes"]
        if method in ("panoptic", "pf_ade"):
            s["mask2former_seconds_per_frame"] = m2f.seconds / len(frames)
        if method.startswith("pf"):
            st = det.stats.as_dict() if hasattr(det.stats, "as_dict") else {}
            s["tracker"] = {k: st.get(k) for k in ("detections", "new_tracks", "dropped_by_validation")}
            s["association_botsort"] = association_metrics(per_frame, ids)
        if dropped:
            s["dropped"] = dict(dropped)
        report["methods"][method] = s
        grid = np.concatenate([np.concatenate(tiles[:2], 1), np.concatenate(tiles[2:4], 1)], 0)
        cv2.imwrite(str(out / f"overlays_{method}.jpg"), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
        logger.info("%s: %s", method, json.dumps({k: v for k, v in s.items() if k not in ("missed", "vocabulary")},
                                                 default=float))
        if keep_for_assoc:
            H, W = frames[0]["depth"].shape
            pre, t_emb, t_pts = [], 0.0, 0.0
            for f, st in zip(frames, stored):
                masks = np.unpackbits(st["packed"], axis=-1, count=W).astype(bool)
                t0 = time.time()
                emb = embed_detections(novic, f, {"masks": masks, "boxes": st["boxes"]}, cfg)
                t_emb += time.time() - t0
                t0 = time.time()
                keys = [detection_points(m, f["depth"], K, f["c2w"]) for m in masks]
                t_pts += time.time() - t0
                pre.append((masks, st["boxes"], emb, keys, st["ids"]))
            for mode in args.assoc_modes:
                kind, thr, geo = (mode.split(":") + ["max"])[:3]
                thr = float(thr)
                if kind == "commit":
                    if not method.startswith("pf"):
                        continue                         # needs BoT-SORT ids
                    pa = CommitAssociation(K, (H, W), thr)
                    handles = [pa.step_tracks(fi, f["name"], b, e, k, t)
                               for fi, (f, (m, b, e, k, t)) in enumerate(zip(frames, pre))]
                    pa_ids = pa.finish(handles)
                else:
                    pa = PoseAwareAssociation(K, (H, W), thr, geo_mode=geo)
                    pa_ids = [pa.step(fi, f["name"], m, b, e, K, f["c2w"], f["depth"], keys=k)
                              for fi, (f, (m, b, e, k, t)) in enumerate(zip(frames, pre))]
                    pa.merge()
                    pa_ids = [np.array([pa.resolve(i) if i >= 0 else -1 for i in x]) for x in pa_ids]
                m3d, rows = score_objects_3d(pa, gt, visible, novic)
                key = method if mode == "frame:1.2:max" else f"{method}@{mode}"
                report["association"][key] = {
                    "pose_aware_2d": association_metrics(per_frame, pa_ids),
                    "pose_aware_3d": m3d,
                    "association_seconds_per_frame": (pa.seconds + t_pts) / len(frames),
                    "embedding_seconds_per_frame": t_emb / len(frames)}
                with open(out / f"pose_assoc_{key.replace(':', '_').replace('@', '_')}_nodes.csv", "w") as fh:
                    fh.write(",".join(rows[0]) + "\n" if rows else "")
                    for r in rows:
                        fh.write(",".join(str(v) for v in r.values()) + "\n")
                logger.info("association %s: %s", key, json.dumps(report["association"][key], default=float)[:600])
        if method.startswith("pf"):
            del det
        import torch
        torch.cuda.empty_cache()
    final = REPO_ROOT / "results" / "replica_final" / args.scene / "metrics.json"
    if final.exists():
        report["full_pipeline_reference"] = reference_3d(final)
    (out / "report.json").write_text(json.dumps(report, indent=1, default=float))
    (out / "report.md").write_text(render([report]))
    novic.close()


def reference_3d(path: Path) -> dict:
    m = json.loads(path.read_text())
    n, sc = m["nodes"], m["nodes"]["status_counts"]
    real = sc.get("correct", 0) + sc.get("wrong_name", 0)
    fake = sc.get("structural", 0) + sc.get("mixed", 0) + sc.get("phantom", 0)
    p = n["valid_objects"] / max(real + fake, 1)
    r = n["node_recall"]
    nm = m.get("naming", {}).get("methods", {}).get("novic", {}).get("acceptable")
    return {"nodes": n["nodes_total"], "non_objects": fake, "scored_nodes": real + fake, "found": n["valid_objects"],
            "visible": n["gt_objects_visible"], "duplicates": n["n_dup"], "object_precision": p, "recall": r,
            "f1": 2 * p * r / (p + r) if p + r else 0.0, "naming_acceptable": nm}


def pct(x):
    return "–" if x is None else f"{100 * x:.1f}"


def render(reports: list[dict]) -> str:
    scenes = [r["scene"] for r in reports]
    methods = list(dict.fromkeys(m for r in reports for m in r["methods"]))
    L = [f"# Front-end variants ({', '.join(scenes)})", "",
         "Frame-level: every detection graded against the ray-cast GT instance map (object / stuff = wall-floor-"
         "ceiling / group = >= 2 objects / mixed). Precision = object detections / graded detections. Recall = GT "
         "objects detected with mask IoU >= 0.5 in at least one frame (and in any form, 'seen'). Counts summed over "
         "scenes, rates averaged.", "",
         "| Variant | Dets/frame | **Frame precision** | Stuff/frame | Group/frame | Mixed | **Objects found (IoU≥.5)** | "
         "Seen | Occurrence recall | Object purity | Pixels claimed twice | Nested pairs/frame | Label naming | s/frame |",
         "|" + "---|" * 14]
    for m in methods:
        R = [r["methods"][m] for r in reports if m in r["methods"]]
        av = lambda k: float(np.mean([x[k] for x in R if x.get(k) is not None])) if any(x.get(k) is not None for x in R) else None  # noqa: E731
        sm = lambda k: sum(x[k] for x in R)  # noqa: E731
        mixed = sum(x["grades"].get("mixed", 0) for x in R) / max(sum(r["frames"] for r in reports if m in r["methods"]), 1)
        nam = [x["label_naming"] for x in R if x.get("label_naming")]
        nm = (sum(x["acceptable"] * x["scored"] for x in nam) / sum(x["scored"] for x in nam)) if nam else None
        extra = sum(x.get("mask2former_seconds_per_frame", 0) * (m == "panoptic") for x in R) / len(R)
        L.append(f"| {m} | {av('detections_per_frame'):.1f} | {pct(av('frame_precision'))} | {av('stuff_per_frame'):.2f} | "
                 f"{av('group_per_frame'):.2f} | {mixed:.2f} | {sm('objects_found_iou50')}/{sm('objects_visible')} | "
                 f"{sm('objects_seen')} | {pct(av('occurrence_recall_iou50'))} | {pct(av('object_purity'))} | "
                 f"{pct(av('pixels_claimed_twice'))} | {av('nested_pairs_per_frame'):.2f} | {pct(nm)} | "
                 f"{av('seconds_per_frame') + extra:.2f} |")
    L += ["", "Notes: `tag` time includes NOVIC tagging (also listed alone in report.json); `pf_ade` time excludes the Mask2Former pass "
          "(see report.json `mask2former_seconds_per_frame`). Label naming grades the detector's own class name "
          "(prompt-free vocabulary, NOVIC tags, or ADE20K class) with the WordNet judge, on object detections.", ""]
    # association: one row per association setting (key "method" = per detection, threshold 1.2, geo = max(2D IoU,
    # 3D overlap); "method@frame:<thr>:<max|3d>" / "method@commit:<thr>" for the other settings)
    allkeys = list(dict.fromkeys(k for r in reports for k in r.get("association", {})))
    akeys = lambda m: [k for k in allkeys if k == m or k.startswith(m + "@")]  # noqa: E731

    def alabel(k):
        if "@" not in k:
            return "pose-aware (per detection, 1.2, max(2D,3D))"
        kind, *rest = k.split("@", 1)[1].split(":")
        if kind == "commit":
            return f"pose-aware at track commit ({rest[0]})"
        return f"pose-aware (per detection, {rest[0]}, {'3D only' if rest[1:] == ['3d'] else 'max(2D,3D)'})"
    L += ["## Association (test b)", "",
          "2D: track purity = share of a track's object detections on its majority GT object; ids/object = distinct "
          "ids per GT object (1 is ideal). BoT-SORT ids are per-track before fusion, pose-aware ids are map objects.",
          "", "| Detections | Associator | Ids | Track purity | Ids per object | Impure ids (<80%) | s/frame |",
          "|---|---|---|---|---|---|---|"]
    for m in methods:
        R = [r for r in reports if m in r["methods"]]
        bs = [r["methods"][m].get("association_botsort") for r in R if r["methods"][m].get("association_botsort")]
        if bs:
            L.append(f"| {m} | BoT-SORT | {sum(x['ids'] for x in bs)} | {pct(np.mean([x['track_purity'] for x in bs]))} | "
                     f"{np.mean([x['ids_per_object'] for x in bs]):.1f} | {sum(x['impure_ids'] for x in bs)} | – |")
        for key in akeys(m):
            pa = [r["association"][key] for r in R if key in r.get("association", {})]
            x2 = [p["pose_aware_2d"] for p in pa]
            L.append(f"| {m} | {alabel(key)} | {sum(x['ids'] for x in x2)} | {pct(np.mean([x['track_purity'] for x in x2]))} | "
                     f"{np.mean([x['ids_per_object'] for x in x2]):.1f} | {sum(x['impure_ids'] for x in x2)} | "
                     f"{np.mean([p['association_seconds_per_frame'] for p in pa]):.2f} (+{np.mean([p['embedding_seconds_per_frame'] for p in pa]):.2f} CLIP) |")
    L += ["", "3D objects (same scoring as the full pipeline; object precision = found / (real + non-object nodes)):", "",
          "| Objects from | Nodes | Non-objects | **Obj. precision** | **Recall (found)** | **F1** | Dup | NOVIC named |",
          "|---|---|---|---|---|---|---|---|"]

    def row3(label, X):
        found, vis = sum(x["found"] for x in X), sum(x["visible"] for x in X)
        real_fake = sum(x["scored_nodes"] for x in X)
        p, r = found / max(real_fake, 1), found / max(vis, 1)
        nm = [x["naming_acceptable"] for x in X if x.get("naming_acceptable") is not None]
        L.append(f"| {label} | {sum(x['nodes'] for x in X)} | {sum(x['non_objects'] for x in X)} | {pct(p)} | "
                 f"{pct(r)} ({found}/{vis}) | {2 * p * r / (p + r) if p + r else 0:.3f} | "
                 f"{sum(x['duplicates'] for x in X)} | {pct(np.mean(nm)) if nm else '–'} |")
    ref = [r["full_pipeline_reference"] for r in reports if "full_pipeline_reference" in r]
    if ref:
        row3("full pipeline: pf + BoT-SORT + track fusion + DoVSG (results/replica_final)", ref)
    for m in methods:
        for key in akeys(m):
            row3(f"{m} + {alabel(key)}", [r["association"][key]["pose_aware_3d"] for r in reports
                                           if key in r.get("association", {})])
    L += ["", "Missed objects (never seen as an object detection):", ""]
    for r in reports:
        for m, s in r["methods"].items():
            L.append(f"- {r['scene']} / {m}: {', '.join(s['missed']) or '–'}")
    tag = [(r["scene"], r["methods"]["tag"]["vocabulary"]) for r in reports if "tag" in r["methods"]]
    for s, v in tag:
        L += ["", f"NOVIC tag vocabulary for {s} ({len(v)} words): {', '.join(v)}"]
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene")
    ap.add_argument("--gpu", default="1")
    ap.add_argument("--methods", nargs="+", default=["pf", "pf_rgb", "pf_val", "pf_ade", "tag", "panoptic"])
    ap.add_argument("--assoc_on", nargs="*", default=["pf", "pf_rgb", "pf_val", "pf_ade", "tag", "panoptic"], help="detection variants to run pose-aware association on")
    ap.add_argument("--bgr_fix", action="store_true", help="hand every YOLOE variant a BGR frame")
    ap.add_argument("--max_frames", type=int, default=0)
    ap.add_argument("--tag_min_score", type=float, default=0.1)
    ap.add_argument("--tag_conf", type=float, default=0.10)
    ap.add_argument("--pan_threshold", type=float, default=0.5)
    ap.add_argument("--ade_stuff_frac", type=float, default=0.6)
    ap.add_argument("--pa_threshold", type=float, default=1.2, help="(kept for old reports; see --assoc_modes)")
    ap.add_argument("--assoc_modes", nargs="+", default=["frame:1.2:max"],
                    help="frame:<threshold>:<max|3d> = per-detection pose-aware association (geo = max(2D IoU, 3D "
                         "overlap) or 3D overlap only); commit:<threshold> = association of whole BoT-SORT tracks "
                         "at commit")
    ap.add_argument("--combine", action="store_true", help="only write results/frontend_tests/comparison.md")
    ap.add_argument("--out_root", default=None)
    args = ap.parse_args()
    global OUT_ROOT
    if args.out_root:
        OUT_ROOT = REPO_ROOT / args.out_root
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    if args.combine:
        reports = [json.loads(p.read_text()) for p in sorted(OUT_ROOT.glob("*/report.json"))]
        (OUT_ROOT / "comparison.md").write_text(render(reports))
        print(render(reports))
        return 0
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
    from object_fusion._dovsg_import import ensure_dovsg_importable
    ensure_dovsg_importable(final_config()["external"]["scene_graph_generation_root"])
    run_scene(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
