"""Files for checking a scene's evaluation by eye.

Everything is written in the map frame (gravity-aligned, z up) so the point
clouds from the run and from the ground truth overlay directly.
"""
from __future__ import annotations

import csv
import json
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

from evaluation.open_vocab import Node
from evaluation.replica_gt import project_depth_test

STATUS_RGB = {
    "correct": (40, 190, 60),
    "wrong_name": (255, 160, 0),
    "mixed": (230, 40, 40),
    "phantom": (230, 40, 40),
    "structural": (150, 150, 150),
    "unlabelled": (150, 150, 150),
}
STATUS_TEXT = {
    "correct": "correct",
    "wrong_name": "wrong name",
    "mixed": "mixes objects",
    "phantom": "not on any surface",
    "structural": "wall/floor/ceiling",
    "unlabelled": "GT has no label",
}


def palette(n: int) -> np.ndarray:
    """n distinct RGB colours (golden-ratio hues)."""
    hues = (np.arange(n) * 0.618033988749895) % 1.0
    hsv = np.stack([hues * 179, np.full(n, 200), np.full(n, 230)], axis=1).astype(np.uint8)[None]
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0].astype(np.uint8)


# ─── tables ───────────────────────────────────────────────────────────────

def write_tables(out: Path, nodes: list[Node], gt, node_m: dict, seg_m: dict) -> None:
    with open(out / "nodes.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["node_id", "status", "novic_name", "name_as_class", "novic_top5", "gt_object_id", "gt_class",
                    "purity", "on_surface", "voxels", "embedding_class", "embedding_top5", "gt_rank_in_embedding",
                    "detections"])
        for n in nodes:
            w.writerow([n.node_id, n.status, n.name, n.name_class, " | ".join(n.novic_top_k[:5]), n.gt_object,
                        n.gt_label, f"{n.purity:.2f}", f"{n.on_surface:.2f}", len(n.points), n.embedding_class,
                        " | ".join(n.embedding_top5), n.rank or "", len(n.detections)])
    matched = {}
    for n in nodes:
        if n.status in ("correct", "wrong_name"):
            matched.setdefault(n.gt_object, []).append(n)
    with open(out / "gt_objects.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["gt_object_id", "class", "kind", "visible", "visible_samples", "found", "matched_nodes",
                    "names_given"])
        vis_counts = {}
        if gt.visible_count is not None:
            ids, c = np.unique(gt.object_id[gt.visible_count > 0], return_counts=True)
            vis_counts = dict(zip(ids.tolist(), c.tolist()))
        for oid in sorted(gt.objects):
            o = gt.objects[oid]
            ns = matched.get(oid, [])
            w.writerow([oid, o["class"], o["kind"], oid in gt.visible_objects, vis_counts.get(oid, 0), bool(ns),
                        " ".join(n.node_id for n in ns), " | ".join(n.name for n in ns)])
    with open(out / "per_class_segmentation.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n_exclude", "class", "points_scored", "iou", "accuracy"])
        for n_ex, m in seg_m.items():
            for r in m["per_class"]:
                w.writerow([n_ex, r["class"], r["gt_points"], f"{r['iou']:.3f}", f"{r['acc']:.3f}"])


# ─── point clouds ─────────────────────────────────────────────────────────

def write_point_clouds(out: Path, nodes: list[Node], all_voxels: np.ndarray, all_colors: np.ndarray,
                       gt_xyz_map: np.ndarray, gt, node_colors: dict[str, np.ndarray]) -> None:
    import open3d as o3d

    def save(path, pts, rgb):
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(np.asarray(pts, dtype=np.float64))
        pc.colors = o3d.utility.Vector3dVector(np.asarray(rgb, dtype=np.float64) / 255.0)
        o3d.io.write_point_cloud(str(path), pc)

    save(out / "voxel_map_rgb.ply", all_voxels, all_colors)
    obj_pts = [n.points for n in nodes if len(n.points)]
    if obj_pts:
        save(out / "voxel_map_objects.ply", np.concatenate(obj_pts),
             np.concatenate([np.tile(node_colors[n.node_id], (len(n.points), 1)) for n in nodes if len(n.points)]))
        save(out / "voxel_map_status.ply", np.concatenate(obj_pts),
             np.concatenate([np.tile(STATUS_RGB[n.status], (len(n.points), 1)) for n in nodes if len(n.points)]))
    ids = sorted(gt.objects)
    gt_pal = {oid: c for oid, c in zip(ids, palette(len(ids)))}
    rgb = np.full((len(gt_xyz_map), 3), 200, dtype=np.uint8)
    for oid in ids:
        if gt.objects[oid]["kind"] == "countable":
            rgb[gt.object_id == oid] = gt_pal[oid]
    keep = np.flatnonzero(gt.object_id > 0)[::4]           # every 4th surface sample is plenty to look at
    save(out / "gt_objects.ply", gt_xyz_map[keep], rgb[keep])


# ─── top-down maps ────────────────────────────────────────────────────────

def write_topdown(out: Path, nodes: list[Node], gt_xyz_map: np.ndarray, gt, node_colors) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(18, 9), sharex=True, sharey=True)
    ax = axes[0]
    for n in nodes:
        if not len(n.points):
            continue
        p = n.points[:: max(len(n.points) // 3000, 1)]
        c = np.array(STATUS_RGB[n.status]) / 255.0
        ax.scatter(p[:, 0], p[:, 1], s=1, color=np.array(node_colors[n.node_id]) / 255.0, alpha=0.5)
        m = n.points.mean(0)
        ax.text(m[0], m[1], n.name, fontsize=7, color=c, weight="bold",
                bbox=dict(facecolor="white", alpha=0.6, lw=0, pad=0.5))
    ax.set_title("Prediction: node names (text colour = status: green correct, orange wrong name, red invalid, "
                 "grey not counted)", fontsize=9)
    ax = axes[1]
    sub = slice(None, None, max(len(gt_xyz_map) // 150000, 1))
    struct = np.isin(gt.object_id, [o for o, v in gt.objects.items() if v["kind"] == "structural"])
    ax.scatter(gt_xyz_map[sub][struct[sub], 0], gt_xyz_map[sub][struct[sub], 1], s=0.2, color="0.85")
    vis = gt.visible_objects
    for oid, o in gt.objects.items():
        if o["kind"] != "countable":
            continue
        sel = gt.object_id == oid
        p = gt_xyz_map[sel][:: max(sel.sum() // 2000, 1)]
        ax.scatter(p[:, 0], p[:, 1], s=1, alpha=0.5, color=("tab:blue" if oid in vis else "0.6"))
        m = gt_xyz_map[sel].mean(0)
        ax.text(m[0], m[1], o["class"], fontsize=7, color=("black" if oid in vis else "0.5"))
    ax.set_title("Ground truth: countable objects (grey = never visible in the walkthrough)", fontsize=9)
    for a in axes:
        a.set_aspect("equal")
        a.set_xlabel("x (m)")
    axes[0].set_ylabel("y (m)")
    fig.tight_layout()
    fig.savefig(out / "topdown_prediction_vs_gt.png", dpi=130)
    plt.close(fig)


# ─── node gallery ─────────────────────────────────────────────────────────

def _box_iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def representative_detection(n: Node, frames_by_name: dict, max_checked: int = 40):
    """The detection whose box best matches the node's own 3D points projected
    into that frame (visible points only), so the crop shows what the node is
    made of rather than whichever box happened to be largest."""
    dets = n.detections
    if len(dets) > max_checked:
        dets = [dets[i] for i in np.linspace(0, len(dets) - 1, max_checked).astype(int)]
    best, best_iou = None, -1.0
    for name, box in dets:
        fr = frames_by_name.get(name)
        if fr is None:
            continue
        ok, uv, _ = _project(n.points, fr["K"], fr["c2w"], fr["depth"], 0.06)
        if ok.sum() < 30:
            continue
        (x1, x2), (y1, y2) = np.percentile(uv[:, 0], [2, 98]), np.percentile(uv[:, 1], [2, 98])
        proj = (x1, y1, x2, y2)
        iou = _box_iou(box, proj)
        if iou > best_iou:
            best, best_iou = (name, box), iou
    if best is None:
        areas = [(d[1][2] - d[1][0]) * (d[1][3] - d[1][1]) for d in n.detections]
        best = n.detections[int(np.argsort(areas)[len(areas) // 2])]
    return best


def write_gallery(out: Path, nodes: list[Node], frame_path, frames_by_name: dict) -> None:
    crops_dir = out / "node_crops"
    crops_dir.mkdir(exist_ok=True)
    tiles = []
    for n in nodes:
        if not n.detections:
            continue
        name, box = representative_detection(n, frames_by_name)
        img = cv2.imread(str(frame_path(name)))
        if img is None:
            continue
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        pad = int(0.15 * max(x2 - x1, y2 - y1))
        crop = img[max(y1 - pad, 0):y2 + pad, max(x1 - pad, 0):x2 + pad]
        if crop.size == 0:
            continue
        cv2.imwrite(str(crops_dir / f"{n.node_id.replace(' ', '_')}.jpg"), crop)
        tile = cv2.resize(crop, (220, 220)) if crop.shape[0] and crop.shape[1] else np.zeros((220, 220, 3), np.uint8)
        bar = np.full((64, 220, 3), 255, np.uint8)
        col = tuple(int(c) for c in STATUS_RGB[n.status][::-1])
        cv2.putText(bar, n.name[:26], (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
        cv2.putText(bar, f"GT: {n.gt_label or '-'}"[:30], (4, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1,
                    cv2.LINE_AA)
        cv2.putText(bar, STATUS_TEXT[n.status], (4, 57), cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1, cv2.LINE_AA)
        tile = np.vstack([tile, bar])
        cv2.rectangle(tile, (0, 0), (219, 283), col, 3)
        tiles.append(tile)
    if not tiles:
        return
    cols = 6
    while len(tiles) % cols:
        tiles.append(np.full_like(tiles[0], 255))
    grid = np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])
    cv2.imwrite(str(out / "nodes_gallery.jpg"), grid)


# ─── video ────────────────────────────────────────────────────────────────

def _splat(shape, uv, ids, z):
    """Nearest-first id image from projected points, holes closed."""
    h, w = shape
    img = np.zeros((h, w), np.int32)
    order = np.argsort(-z)                     # far first, near overwrite
    img[uv[order, 1], uv[order, 0]] = ids[order]
    return cv2.dilate(img.astype(np.float32), np.ones((3, 3), np.uint8)).astype(np.int32)


def _project(pts, K, c2w, depth, tol):
    ok = project_depth_test(pts, K, c2w, depth, tol)
    w2c = np.linalg.inv(c2w)
    cam = pts[ok] @ w2c[:3, :3].T + w2c[:3, 3]
    u = np.round(K[0, 0] * cam[:, 0] / cam[:, 2] + K[0, 2]).astype(np.int64)
    v = np.round(K[1, 1] * cam[:, 1] / cam[:, 2] + K[1, 2]).astype(np.int64)
    return ok, np.stack([u, v], 1), cam[:, 2]


def write_video(out_path: Path, frames: list[dict], nodes: list[Node], node_colors, gt_xyz_map, gt,
                fps: int = 5, tol: float = 0.06) -> None:
    """frames: dicts with name, rgb (HxWx3 RGB), K, c2w (map frame), depth (m).
    Left: prediction (node voxels re-projected, boxes from that frame's
    detections, labelled with the node's name, coloured by status). Right:
    ground truth (countable objects re-projected, labelled with their class)."""
    node_pts = np.concatenate([n.points for n in nodes]) if nodes else np.zeros((0, 3))
    node_ids = np.concatenate([np.full(len(n.points), i + 1) for i, n in enumerate(nodes)]) if nodes else np.zeros(0)
    node_rgb = np.array([[0, 0, 0]] + [node_colors[n.node_id] for n in nodes], np.uint8)
    dets_by_frame: dict[str, list[tuple[Node, np.ndarray]]] = {}
    for n in nodes:
        for fname, box in n.detections:
            dets_by_frame.setdefault(fname, []).append((n, box))

    gt_keep = np.isin(gt.object_id, [o for o, v in gt.objects.items() if v["kind"] in ("countable", "structural")])
    gt_pts, gt_obj = gt_xyz_map[gt_keep], gt.object_id[gt_keep]
    ids = sorted(gt.objects)
    gt_lut = np.zeros((max(ids) + 1 if ids else 1, 3), np.uint8)
    for oid, c in zip(ids, palette(len(ids))):
        gt_lut[oid] = c if gt.objects[oid]["kind"] == "countable" else (170, 170, 170)

    h, w = frames[0]["rgb"].shape[:2]
    W = 2 * w + 2 * 0
    H = h + 30
    proc = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
         "-r", str(fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23",
         "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", str(out_path)],
        stdin=subprocess.PIPE)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for fr in frames:
        rgb, K, c2w, depth = fr["rgb"], fr["K"], fr["c2w"], fr["depth"]
        left, right = rgb.copy(), rgb.copy()
        if len(node_pts):
            ok, uv, z = _project(node_pts, K, c2w, depth, tol)
            idimg = _splat((h, w), uv, node_ids[ok], z)
            m = idimg > 0
            left[m] = (0.45 * left[m] + 0.55 * node_rgb[idimg[m]]).astype(np.uint8)
        for n, box in dets_by_frame.get(fr["name"], []):
            x1, y1, x2, y2 = [int(round(v)) for v in box]
            col = STATUS_RGB[n.status]
            cv2.rectangle(left, (x1, y1), (x2, y2), col, 2)
            label = n.name
            (tw, th), _ = cv2.getTextSize(label, font, 0.45, 1)
            cv2.rectangle(left, (x1, max(y1 - th - 6, 0)), (x1 + tw + 4, max(y1, th + 6)), col, -1)
            cv2.putText(left, label, (x1 + 2, max(y1 - 4, th + 2)), font, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
        ok, uv, z = _project(gt_pts, K, c2w, depth, tol)
        if ok.any():
            idimg = _splat((h, w), uv, gt_obj[ok], z)
            m = idimg > 0
            right[m] = (0.45 * right[m] + 0.55 * gt_lut[idimg[m]]).astype(np.uint8)
            vis_obj = gt_obj[ok]
            for oid in np.unique(vis_obj):
                if gt.objects[int(oid)]["kind"] != "countable":
                    continue
                sel = vis_obj == oid
                if sel.sum() < 40:
                    continue
                cu, cv_ = np.median(uv[sel], axis=0).astype(int)
                label = gt.objects[int(oid)]["class"]
                (tw, th), _ = cv2.getTextSize(label, font, 0.45, 1)
                cv2.rectangle(right, (cu - 2, cv_ - th - 4), (cu + tw + 2, cv_ + 3), (255, 255, 255), -1)
                cv2.putText(right, label, (cu, cv_), font, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
        canvas = np.full((H, W, 3), 255, np.uint8)
        canvas[30:, :w], canvas[30:, w:] = left, right
        cv2.putText(canvas, f"PREDICTION  frame {fr['name']}  (box: green correct / orange wrong name / red invalid "
                    f"/ grey not counted)", (6, 20), font, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(canvas, "GROUND TRUTH (Replica semantic mesh)", (w + 6, 20), font, 0.45, (0, 0, 0), 1,
                    cv2.LINE_AA)
        proc.stdin.write(canvas.tobytes())
    proc.stdin.close()
    proc.wait()


# ─── results.md ───────────────────────────────────────────────────────────

def _fmt(x, pct=False):
    if x is None:
        return "–"
    return f"{100 * x:.1f}%" if pct else f"{x:.3f}"


def write_results_md(out: Path, scene: str, info: dict, nodes: list[Node], ret: dict, node_m: dict, seg_m: dict,
                     files: list[tuple[str, str]]) -> None:
    L = [f"# Open-vocabulary object quality: Replica `{scene}`", ""]
    L += [f"Evaluated {info['evaluated']}. Pipeline run on {info['frames']} frames (every {info['stride']}th of "
          f"NICE-SLAM's 2,000), with ground-truth depth and poses, so these numbers measure detection, fusion and "
          f"naming, not SLAM. Protocol: `docs/evaluation_metrics.md` (group A), code: `evaluation/`.", ""]
    L += ["## Summary", "", "| Metric | Value | Meaning |", "|---|---|---|"]
    L += [f"| Node precision | {_fmt(node_m['node_precision'], True)} | counted nodes that are one real object "
          f"*and* correctly named |",
          f"| Valid node rate | {_fmt(node_m['valid_node_rate'], True)} | counted nodes that are one real object, "
          f"name aside |",
          f"| Node recall | {_fmt(node_m['node_recall'], True)} ({node_m['valid_objects']}/"
          f"{node_m['gt_objects_visible']}) | visible GT objects found by at least one node |",
          f"| Node recall, correctly named | {_fmt(node_m['node_recall_named'], True)} | … and named right |",
          f"| N_dup | {node_m['n_dup']} | extra nodes on an already-found object |"]
    if ret.get("acc_at_k"):
        L += [f"| Acc@{k} (closed set) | {_fmt(v, True)} | GT class in the embedding's top {k} of the "
              f"{ret['vocabulary_size']} Replica classes (chance {_fmt(ret['chance_acc_at_k'][k], True)}) |"
              for k, v in ret["acc_at_k"].items()]
        L += [f"| AUC (Acc@k, k=1..{ret['vocabulary_size']}) | {_fmt(ret['auc'])} | 1.0 = always ranked first; "
              f"embedding quality, not naming (the class list is given) |"]
    for n_ex, s in seg_m.items():
        L += [f"| mIoU / mAcc / F-mIoU, ConceptGraphs exclude-{n_ex} | {_fmt(s['miou'], True)} / "
              f"{_fmt(s['macc'], True)} / {_fmt(s['fmiou'], True)} | {s['num_classes']} classes; excluded: "
              f"{', '.join(s['excluded'])} |"]
    L += ["", f"Node status counts: " + ", ".join(f"{STATUS_TEXT[k]} {v}" for k, v in
                                                   sorted(node_m["status_counts"].items())), ""]
    L += ["## Nodes", "", "| Node | NOVIC name | → class | GT class | Purity | Embedding top-3 | GT rank | Status |",
          "|---|---|---|---|---|---|---|---|"]
    for n in sorted(nodes, key=lambda n: (n.status, n.node_id)):
        L.append(f"| {n.node_id} | {n.name} | {n.name_class} | {n.gt_label or '–'} | {n.purity:.2f} | "
                 f"{', '.join(n.embedding_top5[:3])} | {n.rank or '–'} | {STATUS_TEXT[n.status]} |")
    L += ["", "## Ground-truth objects not found", ""]
    L += [", ".join(node_m["missed_objects"]) or "none", ""]
    if node_m["duplicated_objects"]:
        L += ["Objects with more than one node: " + ", ".join(f"{k} ({v} nodes)" for k, v in
                                                              node_m["duplicated_objects"].items()), ""]
    if not seg_m:
        L += ["## Segmentation", "", "Not computed: ConceptGraphs' ground-truth cloud for this scene is missing.", ""]
    s1 = seg_m[min(seg_m)] if seg_m else {"mprecision": None, "mf1": None, "points_scored": 0, "per_class": [],
                                         "object_classes": []}
    L += [f"## Segmentation per class (ConceptGraphs exclude-{min(seg_m) if seg_m else '-'})", "",
          f"Mean precision {_fmt(s1['mprecision'], True)}, mean F1 {_fmt(s1['mf1'], True)}; "
          f"{s1['points_scored']} reconstruction points scored. Other exclusion settings: "
          f"`per_class_segmentation.csv`.", "",
          "| Class | Points (GT) | IoU | Acc |", "|---|---|---|---|"]
    for r in sorted(s1["per_class"], key=lambda r: -r["gt_points"]):
        L.append(f"| {r['class']} | {r['gt_points']} | {_fmt(r['iou'], True)} | {_fmt(r['acc'], True)} |")
    L += ["", "Class each node got for segmentation: " + ", ".join(
        f"{n.node_id} → {c}" for n, c in zip(nodes, s1["object_classes"])), ""]
    L += ["## How it was measured", "",
          "- **Ground truth:** `habitat/mesh_semantic.ply` (object id per face) and `info_semantic.json` (class per "
          "object), sampled as ~1 point per cm² over the mesh surface plus every face centre. Ignored: `undefined`, `anonymize text/picture`, objects outside the class list. Structural "
          "(wall, floor, ceiling) count for segmentation only.",
          "- **Visible objects:** at least 50 GT surface samples (~50 cm²) seen (in view, not occluded, within 5 cm "
          "of the frame's depth) in some evaluated frame. Only visible objects count for recall.",
          "- **Node → GT:** each node voxel takes the nearest GT surface sample's object within 5 cm; the majority object is "
          "the node's GT, its share is the purity. Under 50% of voxels near the mesh → *not on any surface*; "
          "purity < 0.5 → *mixes objects*.",
          "- **Names:** a node's NOVIC name is mapped to the nearest Replica class by text-embedding similarity "
          f"(prompt `{info['prompt']}`); correct when that equals the GT class.",
          "- **Open-vocabulary naming:** NOVIC's decoded noun and the embedding's nearest noun, both from NOVIC's "
          "target vocabulary, graded by one WordNet judge (`evaluation/naming_judge.py`, "
          "`scripts/eval_open_vocab_naming.py`).",
          "- **Acc@k / AUC:** the Replica classes ranked by cosine between the node's NOVIC image embedding and "
          "each class's text embedding. Scored on nodes matched cleanly to a countable object.",
          "- **Segmentation:** ConceptGraphs' `eval_replica_semseg.py` protocol and ground truth "
          "(`evaluation/conceptgraphs_semseg.py`): their per-point labelled cloud, the 52 Replica classes "
          "restricted to those present in the scene, each node classified among them by its embedding, every "
          "point of our voxel map labelled by its nearest node voxel and scored against its nearest GT point. "
          "Exclude-1 drops `other`; exclude-4 also floor/wall/ceiling; exclude-6 also door/window. Class names "
          "are embedded with NOVIC's CLIP model (DFN5B ViT-H-14), not ConceptGraphs' LAION ViT-H-14, because "
          "our node embeddings live in that space.", ""]
    L += ["## Files", "", "| File | What it shows |", "|---|---|"]
    L += [f"| `{f}` | {d} |" for f, d in files]
    (out / "results.md").write_text("\n".join(L) + "\n")


def copy_if_exists(src: Path, dst: Path) -> bool:
    if src.exists():
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)
        return True
    return False


def dump_json(path: Path, obj) -> None:
    def default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))
    path.write_text(json.dumps(obj, indent=2, default=default))
