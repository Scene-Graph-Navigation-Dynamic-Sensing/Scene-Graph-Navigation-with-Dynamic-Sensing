"""Re-evaluate finished Replica runs with a more human-like ground truth.

Replica labels every window pane and every blind panel as its own object, so
one physical window with its blinds is up to a dozen GT objects; a node that
covers a whole blind then scores as "mixed" and the other panels as "missed".
Here:

  - GT instances of the GROUPED_CLASSES (window, blinds) that touch (surface
    samples within --touch_m) are merged into one unit, per class; every
    other class keeps Replica's instances (two cushions side by side stay two).
  - small wall fixtures (wall plug, switch, vent) are reported apart from the
    other objects, since a viewer barely registers them.
  - naming uses the WordNet judge (evaluation/naming_judge.py): "pillow" for
    a cushion is acceptable; the class-match status is shown alongside.

Node -> GT matching is the evaluation's own rule (each voxel takes the nearest
GT sample within 5 cm; majority unit = the node's object; purity >= 0.5),
with units instead of instances. Visibility is the evaluation's: an object
counts if it was visible in that run (gt_objects.csv); a unit is visible if
any member is.

    python scripts/reevaluate_grouped.py results/replica_final/room_0 results/replica_align_group/room_0
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.naming_judge import grade  # noqa: E402
from evaluation.replica_gt import load_replica_gt  # noqa: E402

GT_ROOT = Path("/home/ravindu/SceneGraphNav/datasets/replica/extracted")
GROUPED_CLASSES = {"window", "blinds"}
FIXTURES = {"wall plug", "switch", "vent"}


def group_units(gt, touch_m: float) -> dict[int, int]:
    """object id -> unit id (the smallest member id); touching window panes / blind panels share a unit."""
    from scipy.spatial import cKDTree
    unit = {o: o for o in gt.objects}
    for cls in GROUPED_CLASSES:
        ids = [o for o, v in gt.objects.items() if v["class"] == cls]
        trees = {o: cKDTree(gt.xyz[gt.object_id == o]) for o in ids if (gt.object_id == o).any()}
        parent = {o: o for o in trees}

        def find(o):
            while parent[o] != o:
                parent[o] = parent[parent[o]]
                o = parent[o]
            return o
        keys = list(trees)
        for i, a in enumerate(keys):
            for b in keys[i + 1:]:
                if find(a) != find(b) and trees[a].query(trees[b].data, distance_upper_bound=touch_m)[0].min() <= touch_m:
                    parent[find(a)] = find(b)
        for o in keys:
            unit[o] = find(o)
    return unit


def evaluate(rd: Path, touch_m: float) -> dict:
    cfg = yaml.safe_load(open(REPO_ROOT / "configs" / "pipeline.yaml"))
    from object_fusion._dovsg_import import ensure_dovsg_importable
    ensure_dovsg_importable(cfg["external"]["scene_graph_generation_root"])
    from navigation.voxel_map import PersistedVoxelMap
    from scene_graph.storage.serializer import load_scene_graph
    from scipy.spatial import cKDTree

    scene = json.loads((rd / "metrics.json").read_text())["info"]["scene"]
    pkl = next((rd / "scene_graph").glob("*.pkl"))
    processed = REPO_ROOT / "data" / "processed" / pkl.stem
    T = np.array(json.loads((processed / "gravity_alignment.json").read_text())["transform"])
    gt = load_replica_gt(GT_ROOT / scene)
    unit = group_units(gt, touch_m)
    gt_xyz = gt.xyz @ T[:3, :3].T + T[:3, 3]
    unit_of_sample = np.array([unit.get(int(o), int(o)) for o in gt.object_id])
    tree = cKDTree(gt_xyz)

    visible = {int(r["gt_object_id"]) for r in csv.DictReader(open(rd / "gt_objects.csv")) if r["visible"] == "True"}
    countable = lambda o: gt.objects.get(o, {}).get("kind") == "countable"
    vis_units = {unit[o] for o in visible if countable(o)}
    cls_of_unit = {unit[o]: gt.objects[o]["class"] for o in gt.objects}

    names = {r["node_id"]: r for r in csv.DictReader(open(rd / "nodes.csv"))}
    payload = load_scene_graph(pkl)
    vmap = PersistedVoxelMap.from_dict(payload["voxel_map"])
    nodes = []
    for o in payload["instance_objects"]:
        nid = o["class_id"]
        if nid not in names or len(o.get("indexes", [])) == 0:
            continue
        P = vmap.index_to_point(np.asarray(o["indexes"])) + vmap.resolution / 2
        d, nn = tree.query(P)
        hit = d <= 0.05
        votes = Counter(int(u) for u in unit_of_sample[nn[hit]] if u != 0)
        if not votes or hit.mean() < 0.5:
            status, u, purity = "phantom", None, 0.0
        else:
            u, k = votes.most_common(1)[0]
            labelled = sum(v for x, v in votes.items() if gt.objects.get(x, {}).get("kind") != "ignored")
            purity = k / max(labelled, 1)
            kind = gt.objects.get(u, {}).get("kind")
            status = ("structural" if kind == "structural" else "unlabelled" if kind == "ignored"
                      else "mixed" if purity < 0.5 else "object")
        cls = cls_of_unit.get(u, "") if u is not None else ""
        name = names[nid]["novic_name"]
        nodes.append({"node_id": nid, "name": name, "unit": u, "gt_class": cls, "purity": round(purity, 2),
                      "status": status, "grade": grade(name, cls) if status == "object" else ""})

    per_unit = defaultdict(list)
    for n in nodes:
        if n["status"] == "object":
            per_unit[n["unit"]].append(n)
    found = {u for u in per_unit if u in vis_units}
    fixture_units = {u for u in vis_units if cls_of_unit[u] in FIXTURES}
    other_units = vis_units - fixture_units
    objs = [n for n in nodes if n["status"] == "object"]
    acceptable = sum(n["grade"] in ("exact", "specific") for n in objs)
    named_any = {u for u, ns in per_unit.items() if any(n["grade"] in ("exact", "specific") for n in ns)}
    return {
        "scene": scene, "nodes": len(nodes),
        "real_object_nodes": len(objs),
        "fake_nodes": sum(n["status"] in ("structural", "mixed", "phantom") for n in nodes),
        "fake_detail": dict(Counter(n["status"] for n in nodes if n["status"] in ("structural", "mixed", "phantom"))),
        "unlabelled_nodes": sum(n["status"] == "unlabelled" for n in nodes),
        "visible_objects": len(vis_units), "found": len(found),
        "visible_without_fixtures": len(other_units), "found_without_fixtures": len(found & other_units),
        "fixtures_found": f"{len(found & fixture_units)}/{len(fixture_units)}",
        "duplicates": sum(len(v) - 1 for v in per_unit.values()),
        "duplicated": {f"{cls_of_unit[u]}#{u}": len(v) for u, v in per_unit.items() if len(v) > 1},
        "name_acceptable_nodes": f"{acceptable}/{len(objs)}",
        "objects_with_an_acceptable_name": f"{len(named_any & vis_units)}/{len(found)}",
        "missed": sorted(Counter(cls_of_unit[u] for u in vis_units - found).items(), key=lambda kv: -kv[1]),
        "nodes_detail": nodes,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("result_dir", nargs="+")
    ap.add_argument("--touch_m", type=float, default=0.05)
    args = ap.parse_args()
    for rd in map(Path, args.result_dir):
        r = evaluate(REPO_ROOT / rd if not rd.is_absolute() else rd, args.touch_m)
        (Path(rd) / "reevaluation_grouped.json").write_text(json.dumps(r, indent=1))
        print(f"\n== {rd}")
        for k, v in r.items():
            if k != "nodes_detail":
                print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
