"""DoVSG's original relation logic — on / belong / inside — reimplemented here.

Why this file exists: the vendored Scene-Graph-Generation checkout has
uncommitted edits to `scene_graph_processer.py` that replace DoVSG's support
test with a two-stage ray-cast variant. This module reproduces the *committed*
DoVSG algorithm (`git show HEAD:dovsg/memory/scene_graph/scene_graph_processer.py`)
so the pipeline uses the published method and is unaffected by experiments in
that working tree.

The three relations, exactly as DoVSG defines them:

  on      Take each object's 2D footprint — the alpha shape (concave hull) of
          its points projected onto the ground. B supports A when
              area(A ∩ B) / area(A) > alpha_shape_overleaf_rate_threshold
          and B's lowest point is strictly below A's lowest point. Among all
          such B, the one whose base is closest to A's base wins. If no B
          qualifies, A rests on the floor. Objects whose base is already within
          stand_floor_threshold of z=0 go straight onto the floor.

  belong  A part-level object (handle, drawer, …) attaches to whichever object
          shares the most points with it, within resolution * 10 metres.

  inside  After the tree is built, a child is re-labelled "inside" its parent
          when at least inside_threshold of its points fall within the parent's
          3D convex hull (Delaunay), falling back to the parent's bounding box.
          Added here: the child's base must also sit more than
          support_max_penetration_m below the parent's top. A desk's hull
          spans its cables and edges, so it encloses a mouse or wallet lying
          flat on it; those rest on the top and stay "on".

One addition to DoVSG's "on": B's top (95th-percentile height) must also be
within `support_max_gap_m` below, or `support_max_penetration_m` above, A's
base (5th-percentile height; its single lowest point is too noisy), or A's
base must be within `support_max_gap_m` above B's base, i.e. A stands on B's
bottom, as a cup inside a box does. DoVSG only compares bases, so an object floating 20 cm above
another (a chair over a flip flop on the floor) or sunk 20 cm into it (a
laptop inside a person's footprint) was still "on" it. Among the B that pass,
the one whose top is nearest A's base wins. When objects lie beneath A but
none touches it, A is attached to the floor with no relation (support
unknown); with nothing beneath it at all, A is "on" the floor as in DoVSG
(legs and bases the camera never saw).

next_to  Added here; a lateral edge, not a tree edge (an object still has exactly
          one parent). Stored as `graph.next_to`, a list of
          {"a", "b", "gap_m"} with a < b (the relation is symmetric). A and B
          are next to each other when ALL of these hold:
            - same parent: they rest on / sit in the same support, so a cup on
              a desk is not "next to" the chair beside the desk;
            - side by side: the gap between their ground footprints (alpha
              shapes) is at most next_to_max_gap_m, and the footprints overlap
              by at most alpha_shape_overleaf_rate_threshold of the smaller
              one (an object tucked under or over another is not beside it);
            - same height band: their [base, top] ranges (5th / 95th
              percentile height) overlap by at least
              next_to_min_vertical_overlap of the shorter one, so a lamp on a
              shelf is not next to the sofa below it.
          Tree ancestors / descendants never qualify; part-level objects are
          skipped.

One property worth naming: "B's base is strictly below A's base" is
antisymmetric, so A and B can never each support the other. The support graph
is therefore a DAG and resolution always terminates. The fork's variant
compares B's *top* against A's base plus a 0.15 m tolerance, which is not
antisymmetric — that is what let two objects each claim the other and spin the
resolution loop forever.
"""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def _footprint(points: np.ndarray, alpha: float):
    """DoVSG's 2D footprint: alpha shape of the unique XY points, jittered to
    avoid colinearity, taking the largest polygon of a MultiPolygon."""
    import alphashape
    from shapely.geometry import MultiPolygon

    xy = np.unique(points[:, :2], axis=0)
    xy = xy + np.random.normal(0, 4e-3, xy.shape)
    shape = alphashape.alphashape(xy, alpha=alpha)
    if isinstance(shape, MultiPolygon):
        shape = max(shape.geoms, key=lambda s: s.area)
    return shape


class _ConvexRegion:
    """The convex hull of a point set as half-spaces, for "how many of these
    points lie inside it". DoVSG triangulates the parent (scipy Delaunay) and
    counts find_simplex() >= 0; a Delaunay triangulation covers exactly the
    convex hull, so this is the same region. On voxel-grid points find_simplex
    is very slow for points outside the hull: 61 s for one 3,880-voxel object
    against a 73,702-voxel parent on lap_in. The hull is 0.02 s and the test
    one matrix product. Points on the hull's surface count as inside, as with
    find_simplex. Raises scipy's QhullError for flat or tiny sets, like Delaunay."""

    def __init__(self, points: np.ndarray):
        from scipy.spatial import ConvexHull

        hull = ConvexHull(points)
        self.normals, self.offsets = hull.equations[:, :3], hull.equations[:, 3]
        self.tolerance = 1e-9 * max(1.0, float(np.abs(points).max()))

    def count_inside(self, points: np.ndarray, chunk: int = 200_000) -> int:
        inside = 0
        for a in range(0, len(points), chunk):
            d = points[a:a + chunk] @ self.normals.T + self.offsets
            inside += int(np.all(d <= self.tolerance, axis=1).sum())
        return inside


def _find_support(class_id, info, mapping, overlap_threshold, max_gap_m, max_penetration_m):
    """DoVSG's support test plus a contact check. Returns (the instance
    beneath or None, whether anything lies beneath this object at all)."""
    alpha_shape, lower_z = info["alpha_shape"], info["lower_z"]
    candidates = []
    anything_beneath = False
    for other_id, other in mapping.items():
        if other_id == class_id:
            continue
        inter_rate = (alpha_shape.intersection(other["alpha_shape"]).area / alpha_shape.area
                      if alpha_shape.area > 0 else 0)
        if inter_rate <= overlap_threshold or other["lower_z"] >= lower_z:
            continue
        anything_beneath = True
        on_top = other["upper_z"] - info["base_z"]
        on_bottom = info["base_z"] - other["base_z"]
        distances = []
        if -max_gap_m <= on_top <= max_penetration_m:
            distances.append(abs(on_top))
        if 0.0 <= on_bottom <= max_gap_m:
            distances.append(on_bottom)
        if distances:
            candidates.append((other["ins_obj"], min(distances)))
    if not candidates:
        return None, anything_beneath
    return min(candidates, key=lambda x: x[1])[0], True


def _ancestor_ids(node) -> set[str]:
    ids = set()
    while node.parent is not None:
        node = node.parent
        ids.add(node.node_id)
    return ids


def compute_next_to(graph, mapping, max_gap_m, min_vertical_overlap, max_footprint_overlap):
    """The "next to" edges for every pair of placed, non-part objects in
    `mapping`; see the module docstring for the rules."""
    ids = sorted(cid for cid in mapping if cid in graph.object_nodes)
    edges = []
    for i, a_id in enumerate(ids):
        a, a_node = mapping[a_id], graph.object_nodes[a_id]
        for b_id in ids[i + 1:]:
            b, b_node = mapping[b_id], graph.object_nodes[b_id]
            if a_node.parent is None or a_node.parent is not b_node.parent:
                continue
            if a_id in _ancestor_ids(b_node) or b_id in _ancestor_ids(a_node):
                continue
            gap = float(a["alpha_shape"].distance(b["alpha_shape"]))
            if gap > max_gap_m:
                continue
            smaller = min(a["alpha_shape"].area, b["alpha_shape"].area)
            if smaller > 0 and a["alpha_shape"].intersection(b["alpha_shape"]).area / smaller > max_footprint_overlap:
                continue
            overlap = min(a["upper_z"], b["upper_z"]) - max(a["base_z"], b["base_z"])
            shorter = min(a["upper_z"] - a["base_z"], b["upper_z"] - b["base_z"])
            if overlap <= 0 or overlap < min_vertical_overlap * max(shorter, 1e-6):
                continue
            edges.append({"a": a_id, "b": b_id, "gap_m": round(gap, 3)})
    return edges


def build_scene_graph_dovsg(
    view_dataset,
    instance_objects,
    scene_graph_generation_root: str,
    resolution: float = 0.01,
    stand_floor_threshold: float = 0.15,
    alphashape_alpha: float = 1.0,
    alpha_shape_overleaf_rate_threshold: float = 0.1,
    inside_threshold: float = 0.95,
    part_level_classes: list[str] | None = None,
    support_max_gap_m: float = 0.10,
    support_max_penetration_m: float = 0.15,
    next_to_max_gap_m: float = 0.30,
    next_to_min_vertical_overlap: float = 0.5,
    history_graph=None,
):
    """With `history_graph`, only the instances that have no node in it yet
    are placed, onto the existing graph, and existing nodes are left exactly
    as they are -- DoVSG's update_scene_graph() (SceneGraphProcesser.
    build_scene_graph(instance_scene_graph=history)). Existing instances still
    count as supports. Remove changed nodes first with remove_nodes()."""
    from object_fusion._dovsg_import import ensure_dovsg_importable

    ensure_dovsg_importable(scene_graph_generation_root)
    from dovsg.memory.scene_graph.graph import ObjectNode, SceneGraph

    if part_level_classes is None:
        part_level_classes = ["handle", "drawer", "door", "knob", "button", "lid", "cover", "wheel"]

    if history_graph is None:
        root = ObjectNode(parent=None, node_class="floor", node_id="floor_0")
        graph = SceneGraph(root_node=root)
    else:
        graph, root = history_graph, history_graph.root
    new_ids = {o["class_id"] for o in instance_objects} - set(graph.object_nodes)
    by_class_id = {o["class_id"]: o for o in instance_objects}

    # ── footprint + base height for every non-part object ──
    mapping = {}
    for obj in instance_objects:
        if obj["class_name"] in part_level_classes:
            continue
        points = view_dataset.index_to_point(obj["indexes"])
        mapping[obj["class_id"]] = {
            "ins_obj": obj,
            "lower_z": float(np.min(points, axis=0)[2]),
            "base_z": float(np.percentile(points[:, 2], 5)),
            "upper_z": float(np.percentile(points[:, 2], 95)),
            "alpha_shape": _footprint(points, alphashape_alpha),
        }
    mapping = dict(sorted(mapping.items(), key=lambda kv: kv[1]["lower_z"]))

    # ── objects already at floor height rest on the floor ──
    for class_id, info in mapping.items():
        if class_id in graph.object_nodes:
            continue
        if info["lower_z"] < stand_floor_threshold:
            graph.add_node(parent=root, node_class=info["ins_obj"]["class_name"],
                           node_id=class_id, parent_relation="on", is_part=False)

    # ── everything else: find what is beneath it ──
    # Strictly-lower-base makes this a DAG, and processing in ascending base
    # order means a support is always placed before the object it carries, so
    # one pass suffices. The guard is kept so a pathological input degrades to
    # "support unknown" rather than looping.
    pending = [cid for cid in mapping if cid not in graph.object_nodes]
    while pending:
        progressed = False
        for class_id in list(pending):
            info = mapping[class_id]
            support, anything_beneath = _find_support(
                class_id, info, mapping, alpha_shape_overleaf_rate_threshold,
                support_max_gap_m, support_max_penetration_m)
            if support is not None and support["class_id"] not in graph.object_nodes:
                continue  # its support is not placed yet; try again next pass
            if support is None and anything_beneath:
                logger.info("Objects lie beneath '%s' (base z=%.2f m) but none touches it; attached "
                            "to the floor with support unknown.", class_id, info["base_z"])
                parent, relation = root, None
            elif support is None:
                parent, relation = root, "on"
            else:
                parent, relation = graph.object_nodes[support["class_id"]], "on"
            graph.add_node(parent=parent, node_class=info["ins_obj"]["class_name"],
                           node_id=class_id, parent_relation=relation, is_part=False)
            pending.remove(class_id)
            progressed = True

        if not progressed:
            for class_id in pending:
                logger.warning("No resolvable support for '%s'; attached to the floor "
                               "with no relation claimed.", class_id)
                graph.add_node(parent=root, node_class=mapping[class_id]["ins_obj"]["class_name"],
                               node_id=class_id, parent_relation=None, is_part=False)
            break

    # ── part-level objects: "belong" to whatever they touch most ──
    for part in [o for o in instance_objects
                 if o["class_name"] in part_level_classes and o["class_id"] in new_ids]:
        import faiss

        part_points = view_dataset.index_to_point(part["indexes"])
        index = faiss.IndexFlatL2(part_points.shape[1])
        index.add(part_points)
        threshold = resolution * 10

        best, best_rate = None, 0.0
        for class_id, info in mapping.items():
            parent_points = view_dataset.index_to_point(info["ins_obj"]["indexes"])
            distances, _ = index.search(parent_points, 1)
            rate = float(np.sum(distances < threshold ** 2) / len(distances))
            if rate > best_rate:
                best, best_rate = class_id, rate

        if best is not None and best in graph.object_nodes:
            graph.add_node(parent=graph.object_nodes[best], node_class=part["class_name"],
                           node_id=part["class_id"], parent_relation="belong", is_part=True)
        else:
            logger.warning("Part-level object '%s' has no parent object; skipped.", part["class_id"])

    # ── re-label containment as "inside" ──
    for parent_node in [n for cid, n in list(graph.object_nodes.items())
                        if cid != root.node_id and not n.is_part and cid in by_class_id]:
        children = [c for c in parent_node.children.values()
                    if not c.is_part and c.node_id in by_class_id and c.node_id in new_ids]
        if not children:  # nothing to test: skip building the parent's hull
            continue
        parent_points = view_dataset.index_to_point(by_class_id[parent_node.node_id]["indexes"])
        if len(parent_points) < 4:
            continue
        try:
            hull = _ConvexRegion(parent_points)
        except Exception:
            hull = None
        lo, hi = np.min(parent_points, axis=0), np.max(parent_points, axis=0)

        for child in children:
            child_points = view_dataset.index_to_point(by_class_id[child.node_id]["indexes"])
            if len(child_points) == 0:
                continue
            if hull is not None:
                contained = hull.count_inside(child_points)
            else:
                contained = np.all((child_points >= lo) & (child_points <= hi), axis=1).sum()
            if contained >= inside_threshold * len(child_points):
                parent_info, child_info = mapping.get(parent_node.node_id), mapping.get(child.node_id)
                if parent_info is not None and child_info is not None:
                    depth = parent_info["upper_z"] - child_info["base_z"]
                    if depth <= support_max_penetration_m:
                        logger.info("'%s' lies within the hull of '%s' but rests on its top "
                                    "(%.2f m below it); kept as 'on'.", child.node_id, parent_node.node_id, depth)
                        continue
                child.parent_relation = "inside"
                logger.info("'%s' is inside '%s'", child.node_id, parent_node.node_id)

    # ── lateral "next to" edges, recomputed over every object so edges to
    #    removed nodes vanish and updated objects are re-measured ──
    graph.next_to = compute_next_to(graph, mapping, next_to_max_gap_m, next_to_min_vertical_overlap,
                                    alpha_shape_overleaf_rate_threshold)
    logger.info("%d 'next to' edges", len(graph.next_to))

    return graph


def remove_nodes(graph, class_ids) -> list[str]:
    """DoVSG's update_scene_graph() deletion step: removes the nodes of
    `class_ids` from `graph` together with everything they carried, and returns
    the ids of those carried nodes that were not themselves removed -- they
    lost their support and must be placed again
    (build_scene_graph_dovsg(history_graph=)). Upstream re-places only direct
    children; here every descendant is, so nothing is silently dropped."""
    removed = {cid for cid in class_ids if cid in graph.object_nodes}
    orphans = []
    for cid in removed:
        node = graph.object_nodes.get(cid)
        if node is None:  # already gone with an ancestor
            continue
        stack = list(node.children.values())
        while stack:
            n = stack.pop()
            stack.extend(n.children.values())
            if n.node_id not in removed:
                orphans.append(n.node_id)
            graph.object_nodes.pop(n.node_id, None)
        graph.object_nodes.pop(cid, None)
        node.delete()
    return orphans
