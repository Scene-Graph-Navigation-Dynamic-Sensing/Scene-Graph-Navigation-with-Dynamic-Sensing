"""Non-Replica training data for the thing/stuff probe (object_fusion.track_validation.stuff_probe), from 3RScan.

The probe scores a track's averaged CLIP (NOVIC DFN5B) embedding at commit. To give it examples that do not share
Replica's look, this runs the pipeline's own front end on real 3RScan captures and labels what it produces:

  1. per scan, every --stride-th frame of the RGB-D sequence, turned upright (Tango frames are stored sideways; the
     rotation is chosen per frame from the pose and the scan's floor), resized to 640 px on the long side
  2. YOLOE-26x prompt-free (the final configuration: conf 0.10, 1024 px, brightness correction, RGB->BGR fixed)
  3. each mask graded against the GT instance map ray-cast through labels.instances.annotated.v2.ply:
     stuff = >= 50% on wall / floor / ceiling, thing = >= 50% on one other labelled object, the rest skipped
  4. the padded box crop of each graded mask embedded with NOVIC's encoder, exactly as the pipeline crops
  5. crops averaged per (scan, GT instance, stuff/thing) -- the analogue of a track's running average -- giving
     one training sample each

    python scripts/build_probe_data_3rscan.py --gpu 2 --scans 80 --out data/cache/probe_3rscan.npz
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import os
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

logger = logging.getLogger("probe3r")
ROOT_3R = Path("/home/ravindu/SceneGraphNav/datasets/3rscan/3rscan_data")
STUFF = {"wall", "floor", "ceiling"}
IGNORE = {"", "object", "objects", "otherprop", "otherstructure", "otherfurniture", "unknown", "item", "stuff"}


def load_mesh(scan: Path):
    from plyfile import PlyData
    ply = PlyData.read(str(scan / "labels.instances.annotated.v2.ply"))
    v = ply["vertex"]
    verts = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float32)
    vobj = np.asarray(v["objectId"], np.int64)
    faces = np.stack(list(ply["face"]["vertex_indices"])).astype(np.uint32)
    fobj = vobj[faces[:, 0]]
    labels = {g["objectId"]: g["label"].strip().lower() for g in json.loads((scan / "semseg.v2.json").read_text())["segGroups"]}
    floor = np.isin(vobj, [o for o, l in labels.items() if l == "floor"])
    up = np.zeros(3)
    if floor.any():
        d = verts.mean(0) - verts[floor].mean(0)
        up[np.argmax(np.abs(d))] = np.sign(d[np.argmax(np.abs(d))])
    else:
        up[2] = 1.0
    return verts, faces, fobj, labels, up


def upright_k(c2w: np.ndarray, up_world: np.ndarray) -> int:
    """np.rot90 k that turns the stored image so world-up points up in the image (x right, y down)."""
    c = c2w[:3, :3].T @ up_world
    if abs(c[1]) >= abs(c[0]):
        return 0 if c[1] < 0 else 2
    return -1 if c[0] < 0 else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu", default="2")
    ap.add_argument("--scans", type=int, default=80)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--max_frames", type=int, default=25)
    ap.add_argument("--out", default="data/cache/probe_3rscan.npz")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    import open3d as o3d
    from eval_frontend_variants import final_config
    from object_fusion._dovsg_import import ensure_dovsg_importable
    from perception.image_enhance import enhance
    from semantic_decoding.novic.crop_utils import extract_crop
    from semantic_decoding.novic.novic_decoder import NOVICDecoder
    from ultralytics import YOLOE

    cfg = final_config()
    ensure_dovsg_importable(cfg["external"]["scene_graph_generation_root"])
    d, sem = cfg["detection"], cfg["semantic_decoding"]
    yolo = YOLOE(d["weights"])
    novic = NOVICDecoder(novic_dir=sem["novic_dir"], checkpoint_path=sem["novic_checkpoint"], gencfg=sem["gencfg"],
                         device="cuda", embedding_batch_size=16)
    scans = sorted(p for p in ROOT_3R.iterdir() if (p / "sequence.zip").exists() and (p / "semseg.v2.json").exists())
    scans = scans[:: max(len(scans) // args.scans, 1)][: args.scans]
    acc = defaultdict(lambda: [np.zeros(0), 0])
    meta = {}
    for si, scan in enumerate(scans):
        try:
            verts, faces, fobj, labels, up = load_mesh(scan)
        except Exception as e:                                   # noqa: BLE001
            logger.warning("%s: %s", scan.name, e)
            continue
        rc = o3d.t.geometry.RaycastingScene()
        rc.add_triangles(o3d.core.Tensor(verts), o3d.core.Tensor(faces))
        z = zipfile.ZipFile(scan / "sequence.zip")
        info = dict(l.split(" = ", 1) for l in z.read("_info.txt").decode().splitlines() if " = " in l)
        Kf = np.array(info["m_calibrationColorIntrinsic"].split(), float).reshape(4, 4)[:3, :3]
        W, H = int(info["m_colorWidth"]), int(info["m_colorHeight"])
        names = sorted(n for n in z.namelist() if n.endswith(".color.jpg"))[:: args.stride][: args.max_frames]
        n_det = 0
        for n in names:
            c2w = np.loadtxt(io.StringIO(z.read(n.replace(".color.jpg", ".pose.txt")).decode()))
            if not np.isfinite(c2w).all():
                continue
            rgb = cv2.cvtColor(cv2.imdecode(np.frombuffer(z.read(n), np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            rays = rc.create_rays_pinhole(o3d.core.Tensor(Kf), o3d.core.Tensor(np.linalg.inv(c2w)), W, H)
            prim = rc.cast_rays(rays)["primitive_ids"].numpy().astype(np.int64)
            ok = prim != o3d.t.geometry.RaycastingScene.INVALID_ID
            gt = np.zeros((H, W), np.int64)
            gt[ok] = fobj[prim[ok]]
            k = upright_k(c2w, up)
            rgb, gt = np.ascontiguousarray(np.rot90(rgb, k)), np.ascontiguousarray(np.rot90(gt, k))
            s = 640 / max(rgb.shape[:2])
            size = (int(round(rgb.shape[1] * s)), int(round(rgb.shape[0] * s)))
            rgb = enhance(cv2.resize(rgb, size, interpolation=cv2.INTER_AREA), d.get("enhance"))
            gt = cv2.resize(gt.astype(np.int32), size, interpolation=cv2.INTER_NEAREST).astype(np.int64)
            r = yolo.predict(np.ascontiguousarray(rgb[..., ::-1]), conf=d["conf_thresh"], iou=d["iou_thresh"],
                             imgsz=d["img_size"], verbose=False, retina_masks=True)[0]
            if r.masks is None or len(r.boxes) == 0:
                continue
            masks = r.masks.data.cpu().numpy().astype(bool)
            if masks.shape[1:] != gt.shape:
                masks = np.stack([cv2.resize(m.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) for m in masks]) > 0
            boxes = r.boxes.xyxy.cpu().numpy()
            keep = []
            for i, m in enumerate(masks):
                ids, cnt = np.unique(gt[m], return_counts=True)
                lab = {int(o): c for o, c in zip(ids, cnt) if o and labels.get(int(o), "") not in IGNORE}
                tot = sum(lab.values())
                if tot < 0.5 * m.sum() or tot == 0:
                    continue
                st = {o: c for o, c in lab.items() if labels[o] in STUFF}
                if sum(st.values()) >= 0.5 * tot:
                    keep.append((i, 1, max(st, key=st.get)))
                else:
                    o = max(lab, key=lab.get)
                    if labels[o] not in STUFF and lab[o] >= 0.5 * tot:
                        keep.append((i, 0, o))
            if not keep:
                continue
            crops = [extract_crop(rgb, boxes[i], masks[i], mode=sem["crop_mode"], padding_px=sem["crop_padding_px"],
                                  padding_frac=sem.get("crop_padding_frac")) for i, _, _ in keep]
            emb = novic.embed_crops(crops)
            emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
            for (i, y, o), e in zip(keep, emb):
                key = (scan.name, o, y)
                a = acc[key]
                a[0] = e if a[1] == 0 else a[0] + e
                a[1] += 1
                meta[key] = labels[o]
            n_det += len(keep)
        logger.info("[%d/%d] %s: %d frames, %d graded detections", si + 1, len(scans), scan.name, len(names), n_det)
    X, y, M = [], [], []
    for key, (s, c) in acc.items():
        X.append(s / np.linalg.norm(s))
        y.append(key[2])
        M.append(f"{key[0]}:{meta[key]}:{c}")
    X, y = np.stack(X).astype(np.float32), np.array(y)
    out = REPO_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, X=X, y=y, meta=np.array(M))
    logger.info("wrote %s: %d samples (%d stuff, %d things) from %d scans", out, len(y), int(y.sum()),
                int((y == 0).sum()), len(scans))
    novic.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
