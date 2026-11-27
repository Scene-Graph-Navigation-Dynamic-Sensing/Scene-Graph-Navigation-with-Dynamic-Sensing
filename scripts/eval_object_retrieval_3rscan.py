"""Open-vocabulary object retrieval metrics on a 3RScan scene.

Scores a saved scene graph (.pkl from run_static_scene_graph.py, run on a
prepare_3rscan_dataset.py folder with reference poses) against the scan's
ground-truth instance labels, the way two groups of papers report it:

  * ReasoningGraph-style Acc@k / AUC: rank every class name of the label
    vocabulary by cosine similarity between the node's NOVIC image embedding
    and the class name's text embedding (same CLIP text tower); Acc@k is the
    share of nodes whose ground-truth class is in the top k. AUC is the mean
    of Acc@k over k = 1..|vocabulary| (1.0 = always rank 1).
  * Open3DSG-style object R@k: is the ground-truth class among NOVIC's own
    top-k decoded nouns? "strict" needs a word match (exact, or one name
    contained in the other, e.g. "floor lamp" ~ "lamp"); "mapped" first maps
    each NOVIC noun to its nearest vocabulary class by text similarity.

Ground truth per node: the node's voxels are moved back into the scan frame
(undoing gravity alignment), each voxel takes the objectId of its nearest
labels.instances.annotated.v2.ply vertex within --max_dist, and the majority
objectId gives the node's class. Nodes whose voxels mostly miss the mesh, or
whose majority share is below --min_purity, are reported as unmatched and
left out of the metrics. So are nodes that land on something 3RScan does not
annotate: 3RScan labels furniture and structure but rarely small items (a
candle on a shelf, a picture on a wall), so such a node's voxels vote for the
shelf or wall behind it. A node counts as "unannotated" when its ground truth
is wall/floor/ceiling, or when the matched object is at least 1 m across and
the node's largest extent is under --min_size_ratio of it. --keep_unannotated
scores those nodes anyway (a pessimistic lower bound).
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logger = logging.getLogger(__name__)

SCANS_ROOT = Path("/home/ravindu/SceneGraphNav/datasets/3rscan/3rscan_data")
KS = (1, 3, 5, 10, 25, 50, 100)


STRUCTURAL = {"wall", "floor", "ceiling"}


def load_gt(scan_dir: Path) -> tuple[np.ndarray, np.ndarray, dict[int, str], dict[int, float]]:
    from plyfile import PlyData

    v = PlyData.read(str(scan_dir / "labels.instances.annotated.v2.ply"))["vertex"]
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)
    obj_ids = np.asarray(v["objectId"], dtype=np.int64)
    groups = json.load(open(scan_dir / "semseg.v2.json"))["segGroups"]
    labels = {g["objectId"]: g["label"] for g in groups}
    sizes = {g["objectId"]: float(max(g["obb"]["axesLengths"])) for g in groups}
    return xyz, obj_ids, labels, sizes


def vocabulary(scans_root: Path) -> list[str]:
    names = set()
    for f in glob.glob(str(scans_root / "*" / "semseg.v2.json")):
        names.update(g["label"] for g in json.load(open(f))["segGroups"])
    return sorted(names)


def strict_match(pred: str, gt: str) -> bool:
    pred, gt = pred.lower().strip(), gt.lower().strip()
    return pred == gt or f" {gt} " in f" {pred} " or f" {pred} " in f" {gt} "


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene_graph", required=True, help="outputs/scene_graphs/<scene>.pkl")
    ap.add_argument("--scan_id", required=True, help="3RScan scan UUID")
    ap.add_argument("--processed_dir", required=True,
                    help="data/processed/<scene> (for gravity_alignment.json)")
    ap.add_argument("--scans_root", default=str(SCANS_ROOT))
    ap.add_argument("--max_dist", type=float, default=0.10, help="voxel-to-mesh match radius, metres")
    ap.add_argument("--min_purity", type=float, default=0.5)
    ap.add_argument("--min_size_ratio", type=float, default=0.35)
    ap.add_argument("--keep_unannotated", action="store_true")
    ap.add_argument("--prompt", default="an image of a {}", help="text template for class names")
    ap.add_argument("--out", default=None, help="JSON report path (default: next to the .pkl)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    from scipy.spatial import cKDTree

    import yaml

    from navigation.voxel_map import PersistedVoxelMap
    from object_fusion._dovsg_import import ensure_dovsg_importable
    from scene_graph.storage.serializer import load_scene_graph
    from semantic_decoding.novic.novic_decoder import NOVICDecoder

    cfg = yaml.safe_load(open(REPO_ROOT / "configs" / "pipeline.yaml"))
    # the pickled scene graph holds dovsg ObjectNode instances
    ensure_dovsg_importable(cfg["external"]["scene_graph_generation_root"])
    payload = load_scene_graph(Path(args.scene_graph))
    vmap = PersistedVoxelMap.from_dict(payload["voxel_map"])
    embeddings = payload["node_embeddings"]

    align_file = Path(args.processed_dir) / "gravity_alignment.json"
    T_align = np.array(json.load(open(align_file))["transform"]) if align_file.exists() else np.eye(4)
    T_inv = np.linalg.inv(T_align)

    gt_xyz, gt_obj, gt_labels, gt_sizes = load_gt(Path(args.scans_root) / args.scan_id)
    tree = cKDTree(gt_xyz)
    vocab = vocabulary(Path(args.scans_root))
    vocab_index = {c: i for i, c in enumerate(vocab)}

    # --- ground-truth class per node -------------------------------------
    nodes = []
    for obj in payload["instance_objects"]:
        node_id = obj["class_id"]
        if node_id not in embeddings or not obj.get("indexes"):
            continue
        pts = vmap.index_to_point(np.asarray(obj["indexes"])) + vmap.resolution / 2
        pts = pts @ T_inv[:3, :3].T + T_inv[:3, 3]
        dist, nn = tree.query(pts)
        hit = dist <= args.max_dist
        votes = Counter(int(i) for i in gt_obj[nn[hit]] if i != 0)
        entry = {
            "node_id": node_id,
            "novic_label": obj.get("class_name"),
            "novic_top_k": [str(t[0]) for t in (obj.get("novic_top_k") or [])],
            "voxels": int(len(pts)),
            "on_mesh": float(hit.mean()),
            "median_dist_m": float(np.median(dist)),
            "extent_m": float(np.ptp(pts, axis=0).max()),
        }
        if votes:
            gt_id, n = votes.most_common(1)[0]
            entry.update(gt_object_id=gt_id, gt_label=gt_labels.get(gt_id, "?"), purity=float(n / max(hit.sum(), 1)))
        if not votes or entry["on_mesh"] < 0.5 or entry["purity"] < args.min_purity:
            entry["status"] = "unmatched"
        elif not args.keep_unannotated and (
            entry["gt_label"] in STRUCTURAL
            or (gt_sizes[gt_id] >= 1.0 and entry["extent_m"] < args.min_size_ratio * gt_sizes[gt_id])
        ):
            entry["status"] = "unannotated"
        else:
            entry["status"] = "matched"
        entry["matched"] = entry["status"] == "matched"
        nodes.append(entry)

    matched = [n for n in nodes if n["matched"] and n["gt_label"] in vocab_index]
    logger.info("%d nodes with embeddings, %d matched to a ground-truth object", len(nodes), len(matched))

    # --- text embeddings (same CLIP text tower NOVIC uses) ----------------
    novic_cfg = cfg["semantic_decoding"]
    decoder = NOVICDecoder(novic_dir=novic_cfg["novic_dir"], checkpoint_path=novic_cfg["novic_checkpoint"],
                           gencfg=novic_cfg["gencfg"], device=novic_cfg.get("device", "cuda"))
    if decoder.is_stub:
        raise SystemExit("NOVIC failed to load (set HF_HUB_OFFLINE=1 if it is a Hub 401)")
    text = decoder.embed_text([args.prompt.format(c) for c in vocab])
    text /= np.linalg.norm(text, axis=1, keepdims=True)
    novic_nouns = sorted({t for n in matched for t in n["novic_top_k"]} | {n["novic_label"] for n in matched})
    noun_emb = decoder.embed_text([args.prompt.format(t) for t in novic_nouns])
    noun_emb /= np.linalg.norm(noun_emb, axis=1, keepdims=True)
    noun_to_class = {t: vocab[int(np.argmax(noun_emb[i] @ text.T))] for i, t in enumerate(novic_nouns)}
    decoder.close()

    # --- Acc@k / AUC from embedding ranking -------------------------------
    ranks = []
    for n in matched:
        e = np.asarray(embeddings[n["node_id"]], dtype=np.float32)
        sims = text @ (e / np.linalg.norm(e))
        order = np.argsort(-sims)
        n["rank"] = int(np.where(order == vocab_index[n["gt_label"]])[0][0]) + 1
        n["embedding_top5"] = [vocab[i] for i in order[:5]]
        ranks.append(n["rank"])
    ranks = np.asarray(ranks)
    V = len(vocab)
    acc = {f"Acc@{k}": float((ranks <= k).mean()) for k in KS if k <= V}
    auc = float(np.mean([(ranks <= k).mean() for k in range(1, V + 1)]))
    chance = {f"Acc@{k}": k / V for k in KS if k <= V}

    # --- object R@k from NOVIC's decoded nouns ----------------------------
    def recall(match_fn, k):
        return float(np.mean([any(match_fn(t, n["gt_label"]) for t in n["novic_top_k"][:k]) for n in matched]))

    strict = {f"R@{k}": recall(strict_match, k) for k in (1, 3, 5)}
    mapped = {f"R@{k}": recall(lambda t, g: noun_to_class[t] == g, k) for k in (1, 3, 5)}
    final_label = {
        "strict": float(np.mean([strict_match(n["novic_label"], n["gt_label"]) for n in matched])),
        "mapped": float(np.mean([noun_to_class[n["novic_label"]] == n["gt_label"] for n in matched])),
    }

    report = {
        "scene_graph": args.scene_graph, "scan_id": args.scan_id, "prompt": args.prompt,
        "vocabulary_size": V, "nodes_with_embedding": len(nodes), "nodes_matched": len(matched),
        "nodes_unannotated": sum(n["status"] == "unannotated" for n in nodes),
        "nodes_unmatched": sum(n["status"] == "unmatched" for n in nodes),
        "gt_objects_in_scan": len(gt_labels), "gt_objects_hit": len({n["gt_object_id"] for n in matched}),
        "embedding_retrieval": {**acc, "AUC": auc, "mean_rank": float(ranks.mean()), "chance": chance},
        "novic_object_recall": {"strict": strict, "mapped_to_vocab": mapped, "final_node_label_R@1": final_label},
        "nodes": nodes,
    }
    suffix = "_retrieval_eval_all.json" if args.keep_unannotated else "_retrieval_eval.json"
    out = Path(args.out) if args.out else Path(args.scene_graph).with_name(Path(args.scene_graph).stem + suffix)
    out.write_text(json.dumps(report, indent=2))

    print(f"\n{Path(args.scene_graph).stem}: {len(matched)}/{len(nodes)} nodes matched, "
          f"{report['gt_objects_hit']}/{len(gt_labels)} GT objects hit, vocabulary {V}")
    print("Acc@k  " + "  ".join(f"{k}={v:.3f}" for k, v in acc.items()) + f"  AUC={auc:.3f}")
    print("chance " + "  ".join(f"{k}={v:.3f}" for k, v in chance.items()))
    print("R@k strict  " + "  ".join(f"{k}={v:.3f}" for k, v in strict.items()))
    print("R@k mapped  " + "  ".join(f"{k}={v:.3f}" for k, v in mapped.items()))
    print(f"final label R@1 strict={final_label['strict']:.3f} mapped={final_label['mapped']:.3f}")
    print(f"{'node':28s} {'gt':16s} {'rank':>4s}  novic top-3 | embedding top-3")
    for n in sorted(matched, key=lambda n: n["rank"]):
        print(f"{n['node_id'][:28]:28s} {n['gt_label'][:16]:16s} {n['rank']:4d}  "
              f"{', '.join(n['novic_top_k'][:3])} | {', '.join(n['embedding_top5'][:3])}")
    for n in nodes:
        if not n["matched"]:
            print(f"{n['status']}: {n['node_id']} on_mesh={n['on_mesh']:.2f} purity={n.get('purity', 0):.2f} "
                  f"extent={n['extent_m']:.2f}m gt={n.get('gt_label')}")
    print(f"report: {out}")


if __name__ == "__main__":
    main()
