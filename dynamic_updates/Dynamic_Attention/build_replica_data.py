"""Training / test data for both attention modules, from the 8 Replica scenes the project already evaluated.

Inputs (all already on disk, nothing is re-run):
  results/replica_final/<scene>/scene_graph/replica_<scene>_final.pkl   the pipeline's fused objects (voxels, CLIP
                                                                         embedding, every detection's frame and box)
  results/replica_final/<scene>/nodes.csv                               node -> ground-truth object id, purity
  data/processed/replica_<scene>_final/{rgb,poses,metadata.json}        the frames the pipeline ran on
  data/raw/replica_<scene>/depth                                        ground-truth depth (resized to the frames)

Two visits per scene from one walk-through: memory = frames [0, split_a), new visit = frames [split_b, 200). Each
pipeline object gets, per visit, the voxels that visit's detections really saw (ground-truth depth within 3 cm of the
voxel inside the detection box), a CLIP embedding from up to 6 of that visit's crops (NOVIC's embedder, as the pipeline
does it), its centroid and box. The identity label is the pipeline node itself; nodes that share a ground-truth object
are marked so the loss ignores those pairs.

Presence-head samples, on every 3rd frame of the new visit (YOLOE stride-16 patch features stored for those frames):
  present   the object's footprint, >= 30 % of its voxels visible and < 50 % occluded
  occluded  >= 60 % of the footprint's voxels have something in front
  absent    (a) the footprint shifted onto patches no object owns (bare wall, floor, desk): the object is gone, the
            surface behind it shows; (b) the footprint of a different object (different ground truth, cosine < 0.75):
            something else stands there
The query is always the memory object (its memory-visit embedding and its YOLOE descriptor from its best memory frame).

  python dynamic_updates/Dynamic_Attention/build_replica_data.py [--scenes office_0 ...]
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import time

import numpy as np

import common as C

C.setup_paths()
logger = logging.getLogger("dynamic_attention")
SCENES = ["office_0", "office_1", "office_2", "office_3", "office_4", "room_0", "room_1", "room_2"]


def load_scene(scene: str):
    from PIL import Image

    from navigation.voxel_map import PersistedVoxelMap
    from scene_graph.storage.serializer import load_scene_graph

    res_dir = C.REPO_ROOT / "results" / "replica_final" / scene
    d = load_scene_graph(res_dir / "scene_graph" / f"replica_{scene}_final.pkl")
    vm = PersistedVoxelMap.from_dict(d["voxel_map"])
    gt = {}
    with open(res_dir / "nodes.csv") as f:
        for r in csv.DictReader(f):
            gid = int(r["gt_object_id"]) if r["gt_object_id"] not in ("", "None") else -1
            gt[r["node_id"]] = (gid, r["gt_class"], float(r["purity"] or 0))
    objs = []
    for o in d["instance_objects"]:
        idx = np.asarray(o["indexes"], np.int64)
        if len(idx) < 30:
            continue
        g = gt.get(o["class_id"], (-1, "", 0.0))
        e = np.asarray(o["semantic_embedding"], np.float32)
        objs.append({"id": o["class_id"], "name": o["class_name"], "pts": vm.index_to_point(idx) + vm.resolution / 2,
                     "emb": e / max(np.linalg.norm(e), 1e-8), "dets": list(zip([str(n) for n in o["image_name"]],
                                                                               [np.asarray(b, float) for b in o["xyxy"]])),
                     "gt": g[0] if g[2] >= 0.5 else -1, "gt_class": g[1], "purity": g[2]})
    P = C.REPO_ROOT / "data" / "processed" / f"replica_{scene}_final"
    md = json.loads((P / "metadata.json").read_text())
    K = np.array(md["K"], float).reshape(3, 3)
    w, h = int(md["w"]), int(md["h"])
    names = sorted(p.stem for p in (P / "rgb").glob("*.jpg"))
    poses = np.stack([np.loadtxt(P / "poses" / f"{n}.txt") for n in names])
    raw = C.REPO_ROOT / "data" / "raw" / f"replica_{scene}" / "depth"

    def depth(j):
        a = np.asarray(Image.open(raw / f"{names[j]}.png")).astype(np.float32) / 1000.0
        return np.asarray(Image.fromarray(a).resize((w, h), Image.NEAREST))

    def rgb(j):
        return np.asarray(Image.open(P / "rgb" / f"{names[j]}.jpg").convert("RGB"))

    return objs, names, poses, K, (w, h), depth, rgb


def visible_in_frame(pts, box, c2w, K, depth, tol=0.03, pad=4):
    u, v, z = C.project(pts, c2w, K)
    h, w = depth.shape
    x0, y0, x1, y1 = box
    ok = (z > 0.1) & (u >= max(x0 - pad, 0)) & (u < min(x1 + pad, w)) & (v >= max(y0 - pad, 0)) & (v < min(y1 + pad, h))
    out = np.zeros(len(pts), bool)
    if ok.any():
        d = depth[v[ok].astype(int), u[ok].astype(int)]
        out[np.where(ok)[0]] = (d > 0.05) & (np.abs(d - z[ok]) < tol)
    return out


def crop(img, box, frac=0.15):
    h, w = img.shape[:2]
    x0, y0, x1, y1 = box
    px, py = frac * (x1 - x0), frac * (y1 - y0)
    a, b = int(max(0, x0 - px)), int(max(0, y0 - py))
    c, d = int(min(w, x1 + px)), int(min(h, y1 + py))
    return img[b:max(d, b + 2), a:max(c, a + 2)]


def build(scene: str, embedder, yolo, split_a: int, split_b: int, rng) -> dict:
    import torch

    t0 = time.time()
    objs, names, poses, K, (w, h), depth, rgb = load_scene(scene)
    stem_to_j = {n: j for j, n in enumerate(names)}
    visits = {"A": range(0, split_a), "B": range(split_b, len(names))}
    dcache: dict = {}

    def D(j):
        if j not in dcache:
            dcache[j] = depth(j)
        return dcache[j]

    obs = {"A": [None] * len(objs), "B": [None] * len(objs)}
    crop_jobs = []                                   # (visit, obj, frame, box)
    for vk, rngv in visits.items():
        for i, o in enumerate(objs):
            dets = [(stem_to_j[n], b) for n, b in o["dets"] if n in stem_to_j and stem_to_j[n] in rngv]
            if not dets:
                continue
            pts = o["pts"] if len(o["pts"]) <= 6000 else o["pts"][rng.choice(len(o["pts"]), 6000, replace=False)]
            sel = [dets[k] for k in np.unique(np.linspace(0, len(dets) - 1, min(10, len(dets))).round().astype(int))]
            seen = np.zeros(len(pts), bool)
            best, best_n = None, 0
            for j, b in sel:
                v = visible_in_frame(pts, b, poses[j], K, D(j))
                seen |= v
                if v.sum() > best_n:
                    best, best_n = j, int(v.sum())
            if seen.sum() < 30:
                continue
            p = pts[seen]
            lo, hi = np.percentile(p, 5, 0), np.percentile(p, 95, 0)
            sub = p if len(p) <= 1024 else p[rng.choice(len(p), 1024, replace=False)]
            obs[vk][i] = {"pts": sub.astype(np.float32), "cen": p.mean(0).astype(np.float32),
                          "ext": np.maximum(hi - lo, 0.01).astype(np.float32), "n_det": len(dets), "best_frame": best}
            area = sorted(dets, key=lambda jb: -(jb[1][2] - jb[1][0]) * (jb[1][3] - jb[1][1]))
            pick = area[:3] + [dets[k] for k in np.linspace(0, len(dets) - 1, 3).round().astype(int)]
            for j, b in {(j, tuple(b)): (j, b) for j, b in pick}.values():
                crop_jobs.append((vk, i, j, b))
    # CLIP embeddings of the crops, per visit and object
    crops, keys = [], []
    img_cache: dict = {}
    for vk, i, j, b in crop_jobs:
        if j not in img_cache:
            img_cache[j] = rgb(j)
        c = crop(img_cache[j], b)
        if c.shape[0] >= 4 and c.shape[1] >= 4:
            crops.append(np.ascontiguousarray(c)); keys.append((vk, i))
    t_emb = time.time()
    E = embedder.embed_crops(crops) if crops else np.zeros((0, 1024), np.float32)
    t_emb = time.time() - t_emb
    acc: dict = {}
    for (vk, i), e in zip(keys, E):
        acc.setdefault((vk, i), []).append(e / max(np.linalg.norm(e), 1e-8))
    for (vk, i), es in acc.items():
        if obs[vk][i] is not None:
            m = np.mean(es, 0)
            obs[vk][i]["emb"] = (m / max(np.linalg.norm(m), 1e-8)).astype(np.float32)
    for vk in obs:
        for i in range(len(objs)):
            if obs[vk][i] is not None and "emb" not in obs[vk][i]:
                obs[vk][i] = None

    # memory descriptors: each object's YOLOE features in its best memory frame
    mem_desc = np.zeros((len(objs), C.FEAT_DIM), np.float32)
    mem_ok = np.zeros(len(objs), bool)
    fcache: dict = {}

    def feats(j):
        if j not in fcache:
            fcache[j] = yolo(img_cache[j] if j in img_cache else rgb(j))
        return fcache[j]

    for i, o in enumerate(objs):
        a = obs["A"][i]
        if a is None:
            continue
        j = a["best_frame"]
        fp = C.footprint(o["pts"], poses[j], K, D(j), None)
        if fp is None:
            continue
        keep = fp["vis"] >= 0.5
        if keep.sum() == 0:
            continue
        mem_desc[i] = C.descriptor(feats(j), fp["patches"][keep], (fp["n"] * fp["vis"])[keep])
        mem_ok[i] = True
    for j in [j for j in fcache if j < split_b]:
        del fcache[j]

    # presence samples on every 3rd new-visit frame
    gh, gw = -(-h // C.PATCH), -(-w // C.PATCH)
    frames_B = list(range(split_b, len(names), 3))
    feats_B, samples = [], []
    for slot, j in enumerate(frames_B):
        feats_B.append(feats(j).cpu())
        fps = {}
        owned = np.zeros(gh * gw, bool)
        for i, o in enumerate(objs):
            pts = o["pts"] if len(o["pts"]) <= 3000 else o["pts"][rng.choice(len(o["pts"]), 3000, replace=False)]
            fp = C.footprint(pts, poses[j], K, D(j), None)
            if fp is not None:
                fps[i] = fp
                owned[fp["patches"][fp["vis"] > 0]] = True
        dvalid = np.zeros(gh * gw, bool)
        dd = D(j)
        pv = (dd[::C.PATCH, ::C.PATCH] > 0.05).ravel()
        dvalid[:len(pv)] = pv[:gh * gw]
        for i, fp in fps.items():
            if not mem_ok[i] or len(fp["patches"]) < 3:
                continue
            wts = fp["n"] / fp["n"].sum()
            occ, vis = float((fp["occ"] * wts).sum()), float((fp["vis"] * wts).sum())
            P_, O_ = fp["patches"], fp["occ"]
            if len(P_) > 64:
                k = rng.choice(len(P_), 64, replace=False)
                P_, O_ = P_[k], O_[k]
            if occ >= 0.6:
                samples.append({"slot": slot, "obj": i, "patches": P_.astype(np.int32), "occ": O_.astype(np.float32),
                                "label": 2, "kind": "occluded"})
            elif vis >= 0.3 and occ < 0.5:
                samples.append({"slot": slot, "obj": i, "patches": P_.astype(np.int32), "occ": O_.astype(np.float32),
                                "label": 0, "kind": "present"})
            else:
                continue
            r, c = P_ // gw, P_ % gw
            for _ in range(25):                                    # (a) removed: the same shape on bare surface
                dr, dc = rng.integers(-10, 11, 2)
                if abs(dr) + abs(dc) < 3:
                    continue
                r2, c2 = r + dr, c + dc
                if r2.min() < 0 or c2.min() < 0 or r2.max() >= gh or c2.max() >= gw:
                    continue
                p2 = r2 * gw + c2
                if (~owned[p2] & dvalid[p2]).mean() >= 0.85:
                    samples.append({"slot": slot, "obj": i, "patches": p2.astype(np.int32),
                                    "occ": np.zeros(len(p2), np.float32), "label": 1, "kind": "absent_surface"})
                    break
            others = [k for k, f2 in fps.items() if k != i and len(f2["patches"]) >= 3
                      and ((f2["vis"] * f2["n"]).sum() / f2["n"].sum()) >= 0.3
                      and (objs[k]["gt"] != objs[i]["gt"] or objs[i]["gt"] < 0)
                      and float(objs[k]["emb"] @ objs[i]["emb"]) < 0.75]
            if others and rng.random() < 0.6:                      # (b) something else stands there
                k = others[int(rng.integers(len(others)))]
                P2, O2 = fps[k]["patches"], fps[k]["occ"]
                if len(P2) > 64:
                    s2 = rng.choice(len(P2), 64, replace=False)
                    P2, O2 = P2[s2], O2[s2]
                samples.append({"slot": slot, "obj": i, "patches": P2.astype(np.int32), "occ": O2.astype(np.float32),
                                "label": 1, "kind": "absent_other"})
        dcache.pop(j, None)
    out = {"scene": scene, "grid": (gh, gw), "K": K, "size": (w, h),
           "objects": [{k: o[k] for k in ("id", "name", "gt", "gt_class", "purity")} for o in objs],
           "mem_emb": np.stack([o["emb"] for o in objs]).astype(np.float32),
           "obs": obs, "mem_desc": mem_desc, "mem_ok": mem_ok, "frames_B": frames_B,
           "feats_B": torch.stack(feats_B) if feats_B else torch.zeros(0), "samples": samples,
           "stats": {"objects": len(objs), "obs_A": sum(o is not None for o in obs["A"]),
                     "obs_B": sum(o is not None for o in obs["B"]), "crops": len(crops), "embed_s": round(t_emb, 1),
                     "samples": {k: sum(s["kind"] == k for s in samples) for k in ("present", "occluded", "absent_surface", "absent_other")},
                     "seconds": round(time.time() - t0, 1)}}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenes", nargs="*", default=SCENES)
    ap.add_argument("--config", default="configs/pipeline_lap_landscape_videos.yaml")
    ap.add_argument("--split_a", type=int, default=110)
    ap.add_argument("--split_b", type=int, default=90)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    import torch

    from scripts.run_static_scene_graph import load_config
    from semantic_decoding.novic.novic_decoder import NOVICDecoder

    cfg = load_config(C.REPO_ROOT / args.config)
    sd = cfg["semantic_decoding"]
    embedder = NOVICDecoder(novic_dir=sd["novic_dir"], checkpoint_path=sd["novic_checkpoint"], gencfg=sd["gencfg"],
                            device="cuda", embedding_batch_size=32)
    if embedder.is_stub:
        raise SystemExit("NOVIC embedder did not load")
    yolo = C.YoloeFeatures()
    C.DATA_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    for s in args.scenes:
        d = build(s, embedder, yolo, args.split_a, args.split_b, rng)
        torch.save(d, C.DATA_DIR / f"{s}.pt")
        logger.info(f"[data] {s}: {json.dumps(d['stats'])}")
    embedder.close()


if __name__ == "__main__":
    main()
