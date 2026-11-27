"""Open-vocabulary object metrics (docs/evaluation_metrics.md, group A):
Acc@k / AUC, object R@k and node precision / recall / N_dup. Semantic
segmentation (mIoU / mAcc / F-mIoU) follows ConceptGraphs' own protocol and
lives in evaluation/conceptgraphs_semseg.py.

Dataset-independent: callers supply the scene graph's nodes (points in the
map frame, NOVIC embedding, NOVIC names) and a ground truth with labelled
points. Everything happens in the map frame.

Node -> ground truth: each of the node's voxels takes the object id of its
nearest ground-truth point within `match_dist_m`; the majority object is the
node's ground truth and its vote share is the node's purity. Node status:

  correct      matched a countable object, and its name is right
  wrong_name   matched a countable object, name wrong
  mixed        purity < min_purity: one node spanning several objects
  phantom      under half its voxels lie near any ground-truth surface
  structural   matched wall / floor / ceiling       (not counted)
  unlabelled   matched an object with no usable GT label (not counted)

A node's name is "right" when its NOVIC label, mapped to the nearest
vocabulary class by text-embedding similarity, equals the ground-truth class.
A node's *embedding class* (used for Acc@k) is the vocabulary class whose
text embedding is closest to the node's NOVIC image embedding, as
ConceptGraphs assigns classes.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np

KS = (1, 3, 5, 10, 25, 50)
COUNTED = ("correct", "wrong_name", "mixed", "phantom")


@dataclass
class Node:
    node_id: str
    name: str                       # final (label-mapped) NOVIC name
    novic_top_k: list[str]
    embedding: np.ndarray
    points: np.ndarray              # (M, 3) map frame
    detections: list[tuple[str, np.ndarray]] = field(default_factory=list)  # (frame name, xyxy)
    # filled by evaluation
    gt_object: int = 0
    gt_label: str = ""
    purity: float = 0.0
    on_surface: float = 0.0
    median_dist_m: float = 0.0
    status: str = ""
    name_class: str = ""            # NOVIC name mapped to the vocabulary
    embedding_class: str = ""       # embedding's nearest vocabulary class
    embedding_top5: list[str] = field(default_factory=list)
    rank: int = 0                   # rank of the GT class in the embedding ranking
    top_k_classes: list[str] = field(default_factory=list)  # NOVIC top-k mapped to vocabulary


def strict_match(pred: str, gt: str) -> bool:
    pred, gt = pred.lower().strip(), gt.lower().strip()
    return pred == gt or f" {gt} " in f" {pred} " or f" {pred} " in f" {gt} "


def text_embeddings(decoder, names: list[str], prompt: str) -> np.ndarray:
    emb = decoder.embed_text([prompt.format(n) for n in names])
    return emb / np.linalg.norm(emb, axis=1, keepdims=True)


def match_nodes(nodes: list[Node], gt_xyz_map: np.ndarray, gt_object_id: np.ndarray, kind_of, label_of,
                match_dist_m: float = 0.05, min_purity: float = 0.5, min_on_surface: float = 0.5) -> None:
    from scipy.spatial import cKDTree

    tree = cKDTree(gt_xyz_map)
    for n in nodes:
        dist, nn = tree.query(n.points)
        hit = dist <= match_dist_m
        n.on_surface = float(hit.mean()) if len(hit) else 0.0
        n.median_dist_m = float(np.median(dist)) if len(dist) else float("inf")
        votes = Counter(int(i) for i in gt_object_id[nn[hit]] if i != 0)
        if not votes or n.on_surface < min_on_surface:
            n.status = "phantom"
            continue
        oid, k = votes.most_common(1)[0]
        # votes for objects with no usable label (e.g. Replica's anonymised pictures shown on a screen)
        # say nothing about which labelled object the node is, so they don't dilute its purity
        labelled = sum(v for o, v in votes.items() if kind_of(o) != "ignored")
        n.gt_object, n.gt_label = oid, label_of(oid)
        n.purity = k / max(labelled if kind_of(oid) != "ignored" else int(hit.sum()), 1)
        kind = kind_of(oid)
        if kind == "structural":
            n.status = "structural"
        elif kind == "ignored":
            n.status = "unlabelled"
        elif n.purity < min_purity:
            n.status = "mixed"
        else:
            n.status = "pending"            # correct / wrong_name once names are classified


def classify_nodes(nodes: list[Node], vocab: list[str], vocab_emb: np.ndarray, decoder, prompt: str) -> None:
    nouns = sorted({t for n in nodes for t in n.novic_top_k} | {n.name for n in nodes})
    noun_emb = text_embeddings(decoder, nouns, prompt) if nouns else np.zeros((0, vocab_emb.shape[1]))
    noun_to_class = {t: vocab[int(np.argmax(noun_emb[i] @ vocab_emb.T))] for i, t in enumerate(nouns)}
    index = {c: i for i, c in enumerate(vocab)}
    for n in nodes:
        n.name_class = noun_to_class[n.name]
        n.top_k_classes = [noun_to_class[t] for t in n.novic_top_k]
        e = n.embedding / (np.linalg.norm(n.embedding) + 1e-12)
        order = np.argsort(-(vocab_emb @ e))
        n.embedding_class = vocab[order[0]]
        n.embedding_top5 = [vocab[i] for i in order[:5]]
        if n.gt_label in index:
            n.rank = int(np.where(order == index[n.gt_label])[0][0]) + 1
        if n.status == "pending":
            n.status = "correct" if n.name_class == n.gt_label else "wrong_name"


def retrieval_metrics(nodes: list[Node], vocab_size: int) -> dict:
    """Acc@k / AUC over the embedding ranking, object R@k over NOVIC's names.
    Scored on nodes that matched a countable object cleanly."""
    scored = [n for n in nodes if n.status in ("correct", "wrong_name") and n.rank > 0]
    out = {"nodes_scored": len(scored), "vocabulary_size": vocab_size}
    if not scored:
        return out
    ranks = np.array([n.rank for n in scored])
    out["acc_at_k"] = {k: float((ranks <= k).mean()) for k in KS if k <= vocab_size}
    out["auc"] = float(np.mean([(ranks <= k).mean() for k in range(1, vocab_size + 1)]))
    out["mean_rank"] = float(ranks.mean())
    out["chance_acc_at_k"] = {k: k / vocab_size for k in KS if k <= vocab_size}
    out["object_recall_strict"] = {k: float(np.mean([any(strict_match(t, n.gt_label) for t in n.novic_top_k[:k])
                                                      for n in scored])) for k in (1, 3, 5)}
    out["object_recall_mapped"] = {k: float(np.mean([n.gt_label in n.top_k_classes[:k] for n in scored]))
                                   for k in (1, 3, 5)}
    out["final_name_accuracy"] = float(np.mean([n.name_class == n.gt_label for n in scored]))
    out["ranks"] = ranks.tolist()          # kept so scenes can be pooled exactly
    return out


def node_metrics(nodes: list[Node], countable_visible: list[int], label_of) -> dict:
    status = Counter(n.status for n in nodes)
    counted = sum(status[s] for s in COUNTED)
    per_object = defaultdict(list)
    for n in nodes:
        if n.status in ("correct", "wrong_name"):
            per_object[n.gt_object].append(n)
    visible = set(countable_visible)
    found = {o for o in per_object if o in visible}
    found_named = {o for o, ns in per_object.items() if o in visible and any(n.status == "correct" for n in ns)}
    dups = {o: len(ns) - 1 for o, ns in per_object.items() if len(ns) > 1}
    return {
        "nodes_total": len(nodes),
        "status_counts": dict(status),
        "nodes_counted": counted,
        "node_precision": status["correct"] / counted if counted else None,
        "valid_node_rate": (status["correct"] + status["wrong_name"]) / counted if counted else None,
        "valid_objects": len(found),
        "gt_objects_visible": len(visible),
        "node_recall": len(found) / len(visible) if visible else None,
        "node_recall_named": len(found_named) / len(visible) if visible else None,
        "n_dup": int(sum(dups.values())),
        "duplicated_objects": {label_of(o) + f"#{o}": k + 1 for o, k in dups.items()},
        "missed_objects": sorted(f"{label_of(o)}#{o}" for o in visible - found),
    }
