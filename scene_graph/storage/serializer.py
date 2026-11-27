"""Persists the static scene graph: a pickle carrying everything (including
NOVIC embeddings, for a future dynamic-update stage to diff against without
re-running fusion), and a human-readable nested JSON summary (embeddings
omitted, children nested under their parent) for quick inspection.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np


# DoVSG spells the part-of relation "belong"; the graph uses three in total.
RELATION_NAMES = {"on": "on", "inside": "inside", "belong": "belongs"}


def scene_graph_to_relations(scene_graph) -> list[dict]:
    """Flattens the tree into a list of {parent, child, relation} edges via
    the same BFS graph.py's own visualize() uses. Relations are one of
    "on" (supported by), "inside" (contained in) or "belongs" (part of)."""
    relations = []
    queue = [scene_graph.root]
    while queue:
        node = queue.pop(0)
        for child in node.children.values():
            relations.append({
                "parent": node.node_id,
                "child": child.node_id,
                "child_class": child.node_class,
                "relation": RELATION_NAMES.get(child.parent_relation, child.parent_relation),
                "is_part": child.is_part,
            })
            queue.append(child)
    return relations


def scene_graph_to_next_to(scene_graph) -> list[dict]:
    """The lateral "next to" edges, {a, b, gap_m} with a < b. Kept apart from
    scene_graph_to_relations() because they are not parent/child edges of the
    tree. Empty for graphs built without them (old pickles, flat fallback)."""
    return list(getattr(scene_graph, "next_to", []))


def node_naming_info(instance_objects, top_k: int = 5) -> dict[str, dict]:
    """node_id -> NOVIC score and its top-k beam guesses, for the JSON tree."""
    info = {}
    for obj in instance_objects:
        if "class_id" not in obj or "novic_score" not in obj:
            continue
        info[obj["class_id"]] = {
            "novic_score": round(float(obj["novic_score"]), 3),
            "name_confidence": (round(float(obj["name_confidence"]), 3)
                                if "name_confidence" in obj else None),
            "novic_label": obj.get("novic_label"),
            "top_k": [{"label": label, "score": round(float(score), 3)}
                      for label, score in (obj.get("novic_top_k") or [])[:top_k]],
        }
    return info


def scene_graph_to_tree(scene_graph, node_embeddings: dict, node_centroids: dict | None = None,
                        node_naming: dict | None = None) -> dict:
    """Nests the graph into its actual shape: each node embeds its children
    directly (ObjectNode.children), instead of the flat (parent, child) edge
    list scene_graph_to_relations() produces. The root has relation=None.
    Each node also lists `next_to`: its lateral neighbours (not tree edges)."""
    node_centroids = node_centroids or {}
    node_naming = node_naming or {}
    neighbours: dict[str, list[dict]] = {}
    for edge in scene_graph_to_next_to(scene_graph):
        neighbours.setdefault(edge["a"], []).append({"node_id": edge["b"], "gap_m": edge["gap_m"]})
        neighbours.setdefault(edge["b"], []).append({"node_id": edge["a"], "gap_m": edge["gap_m"]})

    def build(node) -> dict:
        naming = node_naming.get(node.node_id, {})
        return {
            "node_id": node.node_id,
            "node_class": node.node_class,
            "novic_score": naming.get("novic_score"),
            "name_confidence": naming.get("name_confidence"),
            "novic_label": naming.get("novic_label"),
            "top_k": naming.get("top_k", []),
            "has_embedding": node.node_id in node_embeddings,
            "centroid": node_centroids.get(node.node_id),
            "relation": RELATION_NAMES.get(node.parent_relation, node.parent_relation),
            "is_part": node.is_part,
            "next_to": neighbours.get(node.node_id, []),
            "children": [build(child) for child in node.children.values()],
        }

    return build(scene_graph.root)


def compute_node_centroids(instance_objects, view_dataset) -> dict[str, list[float]]:
    """World-space centroid per node, in metres, for every node that has a
    matching fused instance (join key: ObjectNode.node_id == instance
    'class_id', the same key graph_builder.py uses to attach embeddings).
    Nodes with no matching instance (e.g. the synthetic 'floor_0' root) are
    simply absent from the result.

    Needs view_dataset because a node itself carries no position -- only the
    instance's voxel 'indexes' do, and those only resolve to real-world XYZ
    through view_dataset.index_to_point(). view_dataset is never persisted,
    so this must run at save time, while the caller's pipeline run still has
    it in scope (see scripts/run_static_scene_graph.py's save_scene_graph
    call).
    """
    centroids: dict[str, list[float]] = {}
    for obj in instance_objects:
        indexes = obj.get("indexes")
        if indexes is None or len(indexes) == 0:
            continue
        points = view_dataset.index_to_point(indexes)
        centroids[obj["class_id"]] = np.asarray(points).mean(axis=0).tolist()
    return centroids


def build_voxel_map_snapshot(view_dataset) -> dict:
    """Minimal, self-contained snapshot of a ViewDataset's voxel index --
    resolution, world bounds, and the occupied-voxel -> color mapping --
    persisted alongside the scene graph so navigation/ can rebuild a 2D
    occupancy grid from a saved .pkl without needing the raw recorder
    frames or a live ViewDataset (which is never itself persisted; see
    compute_node_centroids's docstring above for why).

    Kept in sync by contract with navigation/voxel_map.py::PersistedVoxelMap,
    which reimplements just enough of ViewDataset's read-only voxel-index API
    (index_to_point etc.) to consume this snapshot -- if either side's
    encoding changes, the other must change with it.
    """
    return {
        "resolution": view_dataset.resolution,
        "bounds": (
            view_dataset.bounds.xmin, view_dataset.bounds.xmax,
            view_dataset.bounds.ymin, view_dataset.bounds.ymax,
            view_dataset.bounds.zmin, view_dataset.bounds.zmax,
        ),
        "indexes_colors_mapping_dict": view_dataset.indexes_colors_mapping_dict,
    }


def save_scene_graph(
    out_dir: Path,
    scene_name: str,
    scene_graph,
    instance_objects,
    node_embeddings: dict,
    view_dataset=None,
) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    instance_objects = list(instance_objects)
    node_centroids = (
        compute_node_centroids(instance_objects, view_dataset)
        if view_dataset is not None else {}
    )
    voxel_map = build_voxel_map_snapshot(view_dataset) if view_dataset is not None else None

    pkl_path = out_dir / f"{scene_name}.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(
            {
                "scene_graph": scene_graph,
                "instance_objects": instance_objects,
                "node_embeddings": node_embeddings,
                "node_centroids": node_centroids,
                "voxel_map": voxel_map,
            },
            f,
        )

    tree = scene_graph_to_tree(scene_graph, node_embeddings, node_centroids, node_naming_info(instance_objects))
    json_path = out_dir / f"{scene_name}.json"
    with open(json_path, "w") as f:
        json.dump(tree, f, indent=2)

    return pkl_path, json_path


def load_scene_graph(pkl_path: Path) -> dict:
    with open(pkl_path, "rb") as f:
        payload = pickle.load(f)
    # Pickles saved before navigation/ was added have no "voxel_map" key --
    # tolerate that instead of a KeyError so existing callers (task_planning)
    # that never touch it are unaffected; navigation/planner.py surfaces a
    # clear, actionable error for the None case instead.
    payload.setdefault("voxel_map", None)
    return payload
