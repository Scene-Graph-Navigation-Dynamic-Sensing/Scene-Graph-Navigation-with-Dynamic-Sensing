"""Static scene graph construction — thin re-export of DoVSG's
SceneGraphProcesser + ObjectNode/SceneGraph.

build_scene_graph() only ever reads an instance's 'indexes', 'class_name',
and 'class_id' (recomputing all geometry from view_dataset.index_to_point()
on the fly) — no coupling to bbox, clip_ft, or any detector-specific field,
so it's reused completely unchanged. Only the *dynamic* variant
(SceneGraphProcesser.update_scene_graph) is intentionally not used here,
per the plan's static-only scope.

ObjectNode/SceneGraph carry no embedding field, so the NOVIC embedding each
instance was tagged with (object_fusion/instance_tagger.py) is tracked in a
separate {node_id: embedding} side-table built alongside the graph — see
build_static_scene_graph() below and scene_graph/storage/serializer.py,
which persists both together.

KNOWN LIMITATION (confirmed by running the real pipeline against real data,
not just a stub-mode artifact — see README "Known limitation found during
testing"): build_scene_graph() can loop indefinitely when two instances end
up mutually "on" each other (a real geometric-ambiguity case, not merely a
duplicate-detection one — e.g. two spatially-overlapping but genuinely
distinct objects at similar height). This is unmodified upstream DoVSG
code, so it isn't patched here — instead, build_static_scene_graph_safe()
below wraps the call with a timeout and, on hang, drops the more
overlapping/less-confident of the two most spatially-overlapping remaining
instances and retries, falling back to a flat graph (no relations) if it's
still stuck after a few attempts. Use build_static_scene_graph_safe() in
real pipeline runs; build_static_scene_graph() is the raw call, still used
directly for the case where you want the hang to surface rather than be
silently worked around (e.g. investigating the upstream bug itself).
"""
from __future__ import annotations

import logging
import signal

import numpy as np

from object_fusion._dovsg_import import ensure_dovsg_importable

logger = logging.getLogger(__name__)


def build_static_scene_graph(
    view_dataset,
    instance_objects,
    scene_graph_generation_root: str,
    resolution: float = 0.01,
    neighbour_num: int = 5,
    stand_floor_threshold: float = 0.15,
    alphashape_alpha: float = 1.0,
    alpha_shape_overleaf_rate_threshold: float = 0.1,
    part_intersection_rate_threshold: float = 0.2,
    inside_threshold: float = 0.95,
    part_level_classes: list[str] | None = None,
    cycle_patience: int = 50,
    relation_method: str = "dovsg",
    support_max_gap_m: float = 0.10,
    support_max_penetration_m: float = 0.15,
    next_to_max_gap_m: float = 0.30,
    next_to_min_vertical_overlap: float = 0.5,
):
    """Returns (scene_graph, node_embeddings) where node_embeddings maps
    each ObjectNode's node_id (= instance class_id) to its NOVIC embedding.
    """
    if relation_method == "dovsg":
        # published DoVSG algorithm, reimplemented in scene_graph/builder/
        # dovsg_relations.py so the vendored checkout's local edits don't change
        # what "on" means
        from scene_graph.builder.dovsg_relations import build_scene_graph_dovsg

        scene_graph = build_scene_graph_dovsg(
            view_dataset=view_dataset,
            instance_objects=instance_objects,
            scene_graph_generation_root=scene_graph_generation_root,
            resolution=resolution,
            stand_floor_threshold=stand_floor_threshold,
            alphashape_alpha=alphashape_alpha,
            alpha_shape_overleaf_rate_threshold=alpha_shape_overleaf_rate_threshold,
            inside_threshold=inside_threshold,
            part_level_classes=part_level_classes,
            support_max_gap_m=support_max_gap_m,
            support_max_penetration_m=support_max_penetration_m,
            next_to_max_gap_m=next_to_max_gap_m,
            next_to_min_vertical_overlap=next_to_min_vertical_overlap,
        )
        node_embeddings: dict[str, np.ndarray] = {
            obj["class_id"]: np.asarray(obj["semantic_embedding"], dtype=np.float32)
            for obj in instance_objects if "semantic_embedding" in obj
        }
        return scene_graph, node_embeddings

    ensure_dovsg_importable(scene_graph_generation_root)
    from dovsg.memory.scene_graph.scene_graph_processer import SceneGraphProcesser

    processor = SceneGraphProcesser(
        resolution=resolution,
        neighbour_num=neighbour_num,
        stand_floor_threshold=stand_floor_threshold,
        alphashape_alpha=alphashape_alpha,
        alpha_shape_overleaf_rate_threshold=alpha_shape_overleaf_rate_threshold,
        part_intersection_rate_threshold=part_intersection_rate_threshold,
        inside_threshold=inside_threshold,
        part_level_classes=part_level_classes,
    )
    scene_graph = processor.build_scene_graph(
        view_dataset=view_dataset,
        instance_objects=instance_objects,
        # supply the graph ourselves so its node registry can break support
        # cycles, which upstream's loop would otherwise spin on forever
        instance_scene_graph=_new_scene_graph(
            scene_graph_generation_root, cycle_patience,
            # base height per instance, so a cycle is broken in favour of the
            # object low enough to actually be holding the other one up
            bottom_z={obj["class_id"]: float(view_dataset.index_to_point(obj["indexes"])[:, 2].min())
                      for obj in instance_objects}),
    )

    if isinstance(scene_graph.object_nodes, _CycleBreakingNodes):
        scene_graph.object_nodes.resolve_unverified_parents()

    node_embeddings: dict[str, np.ndarray] = {}
    for obj in instance_objects:
        if "semantic_embedding" in obj:
            node_embeddings[obj["class_id"]] = np.asarray(obj["semantic_embedding"], dtype=np.float32)

    return scene_graph, node_embeddings


def _instance_bbox(view_dataset, obj) -> tuple[np.ndarray, np.ndarray]:
    points = view_dataset.index_to_point(obj["indexes"])
    return points.min(axis=0), points.max(axis=0)


def _overlap_ratio(box_a, box_b) -> float:
    """3D overlap volume relative to the smaller of the two boxes (0-1)."""
    min_a, max_a = box_a
    min_b, max_b = box_b
    inter_size = np.clip(np.minimum(max_a, max_b) - np.maximum(min_a, min_b), 0, None)
    inter_vol = float(np.prod(inter_size))
    vol_a = float(np.prod(np.clip(max_a - min_a, 1e-6, None)))
    vol_b = float(np.prod(np.clip(max_b - min_b, 1e-6, None)))
    return inter_vol / min(vol_a, vol_b)


def _drop_most_overlapping_instance(view_dataset, objects: list, dropped_log: list) -> list | None:
    """Finds the pair of instances with the highest 3D overlap and removes
    the less-confident one (lower NOVIC score). Returns None if nothing is
    overlapping enough to plausibly be the cause (so the caller knows not
    to keep retrying pointlessly).
    """
    if len(objects) < 2:
        return None

    boxes = [_instance_bbox(view_dataset, obj) for obj in objects]
    worst_ratio, worst_pair = 0.0, None
    for i in range(len(objects)):
        for j in range(i + 1, len(objects)):
            ratio = _overlap_ratio(boxes[i], boxes[j])
            if ratio > worst_ratio:
                worst_ratio, worst_pair = ratio, (i, j)

    if worst_pair is None or worst_ratio < 0.05:
        return None

    i, j = worst_pair
    loser = i if objects[i].get("novic_score", 0) <= objects[j].get("novic_score", 0) else j
    winner = j if loser == i else i
    dropped = objects[loser]
    dropped_log.append(dropped["class_id"])
    logger.warning(
        f"Dropping instance '{dropped['class_id']}' ({dropped['class_name']}, "
        f"novic_score={dropped.get('novic_score', 0):.3f}) — {worst_ratio:.2f} 3D overlap with "
        f"'{objects[winner]['class_id']}' ({objects[winner]['class_name']}) is the likely cause of "
        f"build_scene_graph() hanging (mutual-support cycle)."
    )
    return [obj for idx, obj in enumerate(objects) if idx != loser]


def _run_with_timeout(fn, timeout_s: float, **kwargs):
    """Runs fn(**kwargs), returning None instead of raising if it doesn't
    finish within timeout_s. Main-thread/Unix only (uses SIGALRM) — fine
    for this pipeline's single-threaded CLI use.
    """
    def _handler(signum, frame):
        raise TimeoutError()

    old_handler = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(int(timeout_s))
    try:
        return fn(**kwargs)
    except TimeoutError:
        return None
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


UNRESOLVED_NODE_ID = "__unresolved_support__"


class _CycleBreakingNodes(dict):
    """Node registry that refuses to let the support-resolution loop spin.

    scene_graph_processer.py's "find the object beneath it" loop defers an
    object whose supporting object is not in the graph yet:

        if support["class_id"] not in scene_graph.object_nodes.keys():
            continue                      # never removed from the worklist

    Two objects that each resolve to the other therefore deadlock, and the
    enclosing `while` re-runs the same alpha-shape/ray-cast geometry forever.
    Measured on a real scan: `computer mouse_0` and `computer keyboard_0` each
    claimed support from the other 5,521 times before the watchdog fired.

    After `patience` unsuccessful lookups of the same key, the cycle is broken
    by physics rather than by defaulting: of the two objects, the one whose
    base is **lower** is the supporter, because a higher object cannot hold up
    a lower one. So the requested support is admitted to the graph (placed on
    the floor) and returned as the parent when it really is the lower of the
    pair; the resulting "A on B" is then consistent with the measured heights.

    When the requested support is the *higher* of the two, the direction is
    physically impossible, so it is not invented: the dependent is attached to
    the floor root with no relation, recorded in `unresolved`, and the caller
    can tell it apart from a measured "on".
    """

    def __init__(self, *args, root=None, patience: int = 50, bottom_z: dict | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._root = root
        self._patience = patience
        self._bottom_z = bottom_z or {}
        self._misses: dict = {}
        self._unresolved_parent = None
        self.cycles_broken: list = []
        self.unresolved: list = []

    def resolve_unverified_parents(self) -> list:
        """Lifts every object parked under the marker node onto the floor with
        no relation, and drops the marker. Returns their ids — these are the
        objects whose support could not be determined, as distinct from the
        ones whose "on" was actually measured."""
        holder = self._unresolved_parent
        if holder is None:
            return []
        lifted = []
        for child in list(holder.children.values()):
            child.parent = self._root
            child.parent_relation = None
            self._root.children[child.node_id] = child
            lifted.append(child.node_id)
        self._root.children.pop(UNRESOLVED_NODE_ID, None)
        dict.pop(self, UNRESOLVED_NODE_ID, None)
        self._unresolved_parent = None
        if lifted:
            logger.warning("%d object(s) have no verified support: %s", len(lifted), lifted)
        return lifted

    def keys(self):  # upstream tests membership via .keys()
        return self

    def __contains__(self, key) -> bool:
        if super().__contains__(key):
            return True
        self._misses[key] = self._misses.get(key, 0) + 1
        return self._misses[key] >= self._patience

    def __getitem__(self, key):
        if dict.__contains__(self, key):
            return dict.__getitem__(self, key)

        from dovsg.memory.scene_graph.graph import ObjectNode

        # The deadlocked support is not in the graph yet. Of the objects still
        # stuck in the cycle, the one with the lowest base is the only one
        # physically able to hold up the others, so it is the one admitted.
        stuck = [k for k in self._misses if not dict.__contains__(self, k)]
        lowest = min(stuck, key=lambda k: self._bottom_z.get(k, float("inf")), default=None)

        if key not in self.cycles_broken:
            self.cycles_broken.append(key)

        if key == lowest and self._root is not None:
            # Admit it as a real node under the floor. Its own support was never
            # measured, so it claims no relation — only the object that attaches
            # to it gets a measured "on".
            logger.warning("Support cycle broken at '%s' (base z=%.2f, lowest of %d stuck objects): "
                           "added under the floor with no relation; the object above it attaches to it.",
                           key, self._bottom_z.get(key, float("nan")), len(stuck))
            node = ObjectNode(parent=self._root, node_class=str(key).rsplit("_", 1)[0],
                              node_id=key, parent_relation=None, is_part=False)
            self._root.add_child(node)
            self.unresolved.append(key)
            super().__setitem__(key, node)
            return node

        # Asking for a support that is not the lowest would assert that a higher
        # object holds up a lower one. Refuse to invent it. Upstream hard-codes
        # parent_relation="on" at its add_node call, so the dependent is parked
        # under a marker node; resolve_unverified_parents() below lifts those
        # children to the floor and clears the relation it never measured.
        self.unresolved.append(key)
        logger.warning("Support cycle on '%s' with no physically valid supporter: "
                       "the dependent object's support is left unclaimed.", key)
        if self._unresolved_parent is None:
            self._unresolved_parent = ObjectNode(parent=self._root, node_class="unresolved",
                                                 node_id=UNRESOLVED_NODE_ID, parent_relation=None)
        return self._unresolved_parent


def _new_scene_graph(scene_graph_generation_root: str, cycle_patience: int, bottom_z: dict | None = None):
    """A SceneGraph whose node registry breaks support cycles (see above)."""
    ensure_dovsg_importable(scene_graph_generation_root)
    from dovsg.memory.scene_graph.graph import ObjectNode, SceneGraph

    root = ObjectNode(parent=None, node_class="floor", node_id="floor_0")
    graph = SceneGraph(root_node=root)
    graph.object_nodes = _CycleBreakingNodes(graph.object_nodes, root=root,
                                             patience=cycle_patience, bottom_z=bottom_z)
    return graph


def _build_flat_fallback_graph(objects: list, scene_graph_generation_root: str):
    """Every instance directly under the floor root — used only if
    build_scene_graph() is still hanging after every cycle-breaking retry.
    Guarantees the pipeline always produces *some* scene graph.

    The relation is left as None, not "on". "on" is a measured claim — the
    object's bottom contact slice was ray-cast onto another object's top
    surface. Nothing was measured here, so asserting "on floor" would invent a
    contact and make a guess indistinguishable from the relations that were
    actually verified. A node with relation None reads as "this object exists,
    its support is unknown".
    """
    ensure_dovsg_importable(scene_graph_generation_root)
    from dovsg.memory.scene_graph.graph import ObjectNode, SceneGraph

    root = ObjectNode(parent=None, node_class="floor", node_id="floor_0")
    graph = SceneGraph(root)
    for obj in objects:
        graph.add_node(parent=root, node_class=obj["class_name"], node_id=obj["class_id"],
                       parent_relation=None, is_part=False)
    return graph


def build_static_scene_graph_safe(
    view_dataset,
    instance_objects,
    scene_graph_generation_root: str,
    timeout_s: float = 90.0,
    max_retries: int = 3,
    **scene_graph_kwargs,
):
    """build_static_scene_graph(), defended against the mutual-support-cycle
    hang described in the module docstring: each attempt runs with a
    timeout; on timeout, the most spatially-overlapping instance pair has
    its lower-confidence member dropped and the build is retried. After
    max_retries failed attempts (or once nothing plausible is left to
    drop), falls back to a flat graph so this never hangs the pipeline.

    Use this instead of build_static_scene_graph() in real runs.
    """
    objects = list(instance_objects)
    dropped_ids: list[str] = []

    for attempt in range(max_retries + 1):
        result = _run_with_timeout(
            build_static_scene_graph,
            timeout_s,
            view_dataset=view_dataset,
            instance_objects=objects,
            scene_graph_generation_root=scene_graph_generation_root,
            **scene_graph_kwargs,
        )
        if result is not None:
            if dropped_ids:
                logger.warning(
                    f"Scene graph built after dropping {len(dropped_ids)} instance(s) to break a cycle: {dropped_ids}"
                )
            return result

        logger.warning(f"build_scene_graph() timed out after {timeout_s}s (attempt {attempt + 1}/{max_retries + 1}).")
        if attempt == max_retries:
            break
        reduced = _drop_most_overlapping_instance(view_dataset, objects, dropped_ids)
        if reduced is None:
            logger.warning("No plausible overlapping instance pair left to drop — stopping retries early.")
            break
        objects = reduced

    logger.error(
        f"build_scene_graph() still hanging after cycle-breaking attempts (dropped: {dropped_ids or 'none'}). "
        "Falling back to a flat graph (every instance under the root, no spatial relations)."
    )
    graph = _build_flat_fallback_graph(objects, scene_graph_generation_root)
    node_embeddings = {
        obj["class_id"]: np.asarray(obj["semantic_embedding"], dtype=np.float32)
        for obj in objects
        if "semantic_embedding" in obj
    }
    return graph, node_embeddings
