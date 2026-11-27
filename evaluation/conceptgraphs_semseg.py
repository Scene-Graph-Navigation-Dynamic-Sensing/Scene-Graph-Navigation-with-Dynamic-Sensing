"""Replica 3D semantic segmentation, following ConceptGraphs exactly.

A port of concept-graphs/conceptgraph/scripts/eval_replica_semseg.py
(https://github.com/concept-graphs/concept-graphs, commit 93277a0), so our
mIoU / mAcc / F-mIoU sit next to the ConceptGraphs paper's numbers:

  * Ground truth: ConceptGraphs' per-point labelled cloud
    (<scene>/Sequence_1/saved-maps-gt, downloaded from the link in their
    README), moved into the trajectory frame by the first pose of
    traj_w_c.txt. Classes: the 52 REPLICA_EXISTING_CLASSES.
  * Classes used per scene: only those present in that scene's ground truth,
    minus an exclusion set (n_exclude = 1: "other"; 4: + floor, wall, ceiling;
    6: + door, window).
  * Each object gets the class whose text embedding ("an image of {class}")
    is closest to its image embedding, among the classes used.
  * Every point of the reconstruction (our whole voxel map, ConceptGraphs'
    "rgb_cloud") takes the class of its nearest object point, with no
    distance limit; each of those points is then matched to its nearest GT
    point. Points whose GT class is not used are dropped.
  * From the confusion matrix (rows GT, columns prediction): IoU, recall,
    precision, F1 per class; mIoU and mAcc (= mean recall) average over the
    used classes; F-mIoU weights IoU by GT frequency.

One unavoidable difference: ConceptGraphs embeds class names with OpenCLIP
ViT-H-14 (laion2b). Our node embeddings come from NOVIC's encoder (OpenCLIP
DFN5B ViT-H-14-378), so class names are embedded with that model's text
tower, the space our embeddings live in.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

# conceptgraph/dataset/replica_constants.py
REPLICA_CLASSES = [
    "other", "backpack", "base-cabinet", "basket", "bathtub", "beam", "beanbag", "bed", "bench", "bike", "bin",
    "blanket", "blinds", "book", "bottle", "box", "bowl", "camera", "cabinet", "candle", "chair", "chopping-board",
    "clock", "cloth", "clothing", "coaster", "comforter", "computer-keyboard", "cup", "cushion", "curtain",
    "ceiling", "cooktop", "countertop", "desk", "desk-organizer", "desktop-computer", "door", "exercise-ball",
    "faucet", "floor", "handbag", "hair-dryer", "handrail", "indoor-plant", "knife-block", "kitchen-utensil",
    "lamp", "laptop", "major-appliance", "mat", "microwave", "monitor", "mouse", "nightstand", "pan", "panel",
    "paper-towel", "phone", "picture", "pillar", "pillow", "pipe", "plant-stand", "plate", "pot", "rack",
    "refrigerator", "remote-control", "scarf", "sculpture", "shelf", "shoe", "shower-stall", "sink",
    "small-appliance", "sofa", "stair", "stool", "switch", "table", "table-runner", "tablet", "tissue-paper",
    "toilet", "toothbrush", "towel", "tv-screen", "tv-stand", "umbrella", "utensil-holder", "vase", "vent",
    "wall", "wall-cabinet", "wall-plug", "wardrobe", "window", "rug", "logo", "bag", "set-of-clothing",
]
REPLICA_EXISTING_CLASSES = [
    0, 3, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 23, 26, 29, 31, 34, 35, 37, 40, 44, 47, 52, 54,
    56, 59, 60, 61, 62, 63, 64, 65, 70, 71, 76, 78, 79, 80, 82, 83, 87, 88, 91, 92, 93, 95, 97, 98,
]
CLASS_NAMES = [REPLICA_CLASSES[i] for i in REPLICA_EXISTING_CLASSES]
EXCLUDE = {
    1: ["other"],
    4: ["other", "floor", "wall", "ceiling"],
    6: ["other", "floor", "wall", "ceiling", "door", "window"],
}
PROMPT = "an image of {}"
CG_GT_ROOT = Path("/home/ravindu/SceneGraphNav/datasets/replica/conceptgraphs_semantic_gt")


def load_gt(scene: str, root: Path = CG_GT_ROOT) -> tuple[np.ndarray, np.ndarray]:
    """GT points in the trajectory (mesh) frame and their class index into CLASS_NAMES."""
    import h5py

    d = root / scene / "Sequence_1"
    pc = d / "saved-maps-gt" / "pointclouds"
    with h5py.File(pc / "pc_points.h5") as f:
        xyz = f["pc_points"][:].astype(np.float64)
    with h5py.File(pc / "pc_embeddings.h5") as f:
        cls_all = f["pc_embeddings"][:].argmax(axis=1)
    all2existing = np.full(len(REPLICA_CLASSES), -1)
    all2existing[REPLICA_EXISTING_CLASSES] = np.arange(len(REPLICA_EXISTING_CLASSES))
    cls = all2existing[cls_all]
    assert cls.min() >= 0, "GT class outside REPLICA_EXISTING_CLASSES"
    T0 = np.loadtxt(d / "traj_w_c.txt").reshape(-1, 4, 4)[0]
    return xyz @ T0[:3, :3].T + T0[:3, 3], cls


def class_text_features(decoder) -> np.ndarray:
    emb = decoder.embed_text([PROMPT.format(c) for c in CLASS_NAMES])
    return emb / np.linalg.norm(emb, axis=1, keepdims=True)


def compute_metrics(conf: np.ndarray) -> dict:
    """conceptgraph/utils/eval.py::compute_metrics."""
    n = len(conf)
    tp = np.diag(conf).astype(float)
    row, col = conf.sum(1).astype(float), conf.sum(0).astype(float)
    iou = tp / np.maximum(1, row + col - tp)
    recall = tp / np.maximum(1, row)
    precision = tp / np.maximum(1, col)
    f1 = 2 * precision * recall / np.maximum(1, precision + recall)
    fmiou = float((iou * row / max(conf.sum(), 1)).sum())
    return {"miou": float(iou.mean()) if n else None, "macc": float(recall.mean()) if n else None,
            "mprecision": float(precision.mean()) if n else None, "mf1": float(f1.mean()) if n else None,
            "fmiou": fmiou, "iou": iou.tolist(), "recall": recall.tolist(), "num_classes": n}


def evaluate(gt_xyz: np.ndarray, gt_cls: np.ndarray, object_points: list[np.ndarray],
             object_embeddings: np.ndarray, class_feats: np.ndarray, recon_xyz: np.ndarray,
             n_exclude: int) -> dict:
    """One scene, one exclusion setting. All points in the same (mesh) frame."""
    from scipy.spatial import cKDTree

    C = len(CLASS_NAMES)
    ignore = {CLASS_NAMES.index(c) for c in EXCLUDE[n_exclude]}
    ignore |= set(range(C)) - set(np.unique(gt_cls).tolist())          # gt_class_only=True
    keep = np.array(sorted(set(range(C)) - ignore))

    emb = object_embeddings / np.linalg.norm(object_embeddings, axis=1, keepdims=True)
    sim = emb @ class_feats.T
    sim[:, sorted(ignore)] = -1e10
    object_class = sim.argmax(1)

    pred_xyz = np.concatenate(object_points)
    pred_cls = np.concatenate([np.full(len(p), c) for p, c in zip(object_points, object_class)])
    # resample onto the reconstruction: each reconstruction point takes its nearest object point's class
    pred_cls = pred_cls[cKDTree(pred_xyz).query(recon_xyz)[1]]
    # each reconstruction point -> nearest GT point
    label_gt = gt_cls[cKDTree(gt_xyz).query(recon_xyz)[1]]
    sel = np.isin(label_gt, keep)
    conf = np.zeros((C, C), dtype=np.int64)
    np.add.at(conf, (label_gt[sel], pred_cls[sel]), 1)
    assert conf[:, sorted(ignore)].sum() == 0 and conf[sorted(ignore), :].sum() == 0

    m = compute_metrics(conf[np.ix_(keep, keep)])
    return {
        "n_exclude": n_exclude, "excluded": EXCLUDE[n_exclude],
        "classes_used": [CLASS_NAMES[i] for i in keep],
        **{k: m[k] for k in ("miou", "macc", "mprecision", "mf1", "fmiou", "num_classes")},
        "per_class": [{"class": CLASS_NAMES[c], "gt_points": int(conf[c].sum()), "iou": m["iou"][i],
                       "acc": m["recall"][i]} for i, c in enumerate(keep)],
        "object_classes": [CLASS_NAMES[c] for c in object_class],
        "points_scored": int(sel.sum()),
        "confusion": conf.tolist(),                 # full 52x52, for pooling scenes like ConceptGraphs' "all"
    }


def pooled(confusions: list[np.ndarray]) -> dict:
    """ConceptGraphs' "all" row: sum the scenes' matrices, keep classes with GT points."""
    conf = sum(np.asarray(c) for c in confusions)
    keep = np.nonzero(conf.sum(1))[0]
    m = compute_metrics(conf[np.ix_(keep, keep)])
    return {k: m[k] for k in ("miou", "macc", "mprecision", "mf1", "fmiou", "num_classes")}
