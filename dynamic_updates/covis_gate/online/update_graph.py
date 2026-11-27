"""§8.3 and §9 for the streaming engine: change only the nodes that changed, then learn.

run_covis_update.py does this at the end of its single pass (its steps 6-7). Here it is a function of the decisions
so far, so it can be called while the video is still playing: build_graph() is pure (the memory is not touched, the
move rates are not learned) and can be called again with newer decisions; commit_visit() learns the move rates and
saves the visit once.
"""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import targeted
from scripts.update_scene_graph import renumber, update_scene_graph


def decisions(model, labels: list, assoc: dict | None) -> dict:
    """Final label per memory object: assoc["labels"] (removed / moved / present / deferred) after the association,
    or the evidence labels (absent -> removed) when nothing was observed."""
    if assoc is not None:
        return dict(assoc["labels"])
    return {c: ("removed" if labels[n] == "absent" else labels[n]) for n, c in enumerate(model.names)}


def build_graph(model, cfg: dict, final: dict, assoc: dict | None, observed: list) -> dict:
    t = time.perf_counter()
    rows = assoc["rows"] if assoc else []
    moved = {c: rows[u] for c, u in (assoc["moved"] if assoc else {}).items()}
    removed = [c for c, l in final.items() if l == "removed"]
    new_objects, gone_idx, new_colours = [], [], {}
    for mo in model.objects:
        cid = mo["class_id"]
        if cid in removed:
            gone_idx += list(mo["indexes"])
            continue
        m = dict(mo)
        if cid in moved:
            ob = observed[moved[cid]["index"]]
            gone_idx += list(mo["indexes"])
            m["indexes"], cols = ob.grid_idx, ob.grid_cols
            new_colours.update(cols)
            m["moved_from"] = np.round(model.centroids[cid], 3).tolist()
            m["moved_to"] = moved[cid]["centroid"]
        new_objects.append(m)
    added, added_rows = [], []
    for r in rows:
        if r["decision"] == "added":
            ob = observed[r["index"]]
            o = dict(ob.raw)
            o["indexes"], cols = ob.grid_idx, ob.grid_cols
            new_colours.update(cols)
            if o["indexes"]:
                added.append(o)
                added_rows.append(r)
    renumber(added, set(model.names))
    for o, r in zip(added, added_rows):
        r["added_as"] = o["class_id"]
    new_objects += added
    vd = targeted.ChangedMap(model.map, gone_idx, new_colours)
    if removed or moved or added:
        graph, orphans = update_scene_graph(model.scene["scene_graph"], removed + list(moved), new_objects, vd, cfg)
    else:
        graph, orphans = copy.deepcopy(model.scene["scene_graph"]), []
    return {"graph": graph, "objects": new_objects, "vd": vd, "removed": removed, "moved": moved, "added": added,
            "orphans": orphans, "deferred": [c for c, l in final.items() if l == "deferred"],
            "seconds": round(time.perf_counter() - t, 2)}


def save_built(built: dict, scene_name: str, sg_dir) -> dict:
    """Write the updated scene graph (pkl + json); returns what learning needs from it."""
    from scene_graph.storage.serializer import save_scene_graph, scene_graph_to_relations

    graph = built["graph"]
    new_parent = {r["child"]: r["parent"] for r in scene_graph_to_relations(graph) if r["relation"] in ("on", "inside")}
    objects = built["objects"]
    for o in objects:
        o.pop("mask", None)
    emb = {o["class_id"]: np.asarray(o["semantic_embedding"], np.float32) for o in objects if "semantic_embedding" in o}
    pkl, js = save_scene_graph(sg_dir, scene_name, graph, objects, emb, view_dataset=built["vd"])
    return {"new_parent": new_parent, "saved": [str(pkl), str(js)],
            "added": [{"class_id": o["class_id"], "class_name": o["class_name"], "voxels": len(o["indexes"]),
                       "novic_label": o.get("novic_label")} for o in built["added"]],
            "voxels_after": int(len(built["vd"].indexes_colors_mapping_dict))}


def learn_visit(model, ccfg: dict, scene_name: str, final: dict, saved: dict, sg_dir) -> dict:
    """§9: learn the move rates from this visit and save them next to the scene graph."""
    new_parent, added = saved["new_parent"], saved["added"]
    learned = {}
    for c in model.names:
        learned[c] = model.dyn.learn(c, final[c], model.t_now, ccfg, parent=new_parent.get(c, model.parent_of.get(c)))
        if final[c] == "removed":
            model.dyn.archive(c)
    for o in added:
        model.dyn.add(o, model.t_now, ccfg, parent=new_parent.get(o["class_id"]))
    model.dyn.state["visits"].append({"t_days": model.t_now, "scene_name": scene_name, "memory": model.name,
                                      "labels": final, "added": [o["class_id"] for o in added]})
    model.dyn.state["visit_time_days"] = model.t_now
    model.dyn.save(sg_dir / f"{scene_name}_dynamism.json")
    return {"learned": learned}


# ─── the graph in its own process ───
# build_graph() is a few seconds of Python (alpha shapes, support tests). On a thread of the keyframe loop's process it
# takes the GIL from the loop: keyframes arrived late by seconds. In a process of its own it does not.

@dataclass
class ObsLite:
    raw: dict
    grid_idx: list
    grid_cols: dict


class GraphModel:
    """The parts of MemoryModel that build_graph() reads, loaded in the graph process."""

    def __init__(self, repo_root, cfg: dict, memory: str):
        from navigation.voxel_map import PersistedVoxelMap
        from scene_graph.storage.serializer import load_scene_graph

        self.scene = load_scene_graph(Path(repo_root) / cfg["outputs"]["scene_graphs_dir"] / f"{memory}.pkl")
        self.map = PersistedVoxelMap.from_dict(self.scene["voxel_map"])
        self.objects = self.scene["instance_objects"]
        self.names = [o["class_id"] for o in self.objects]
        res = self.map.resolution
        self.centroids = {}
        for o in self.objects:
            idx = np.asarray([i for i in o["indexes"] if i in self.map.indexes_colors_mapping_dict], np.int64)
            self.centroids[o["class_id"]] = self.map.index_to_point(idx).mean(0) + res / 2


_G: dict = {}


def _graph_init(repo_root: str, config: str, memory: str):
    import sys

    for p in (repo_root,):
        if p not in sys.path:
            sys.path.insert(0, p)
    from object_fusion._dovsg_import import ensure_dovsg_importable
    from scripts.run_static_scene_graph import load_config

    cfg = load_config(Path(repo_root) / config)
    ensure_dovsg_importable(cfg["external"]["scene_graph_generation_root"])
    _G["cfg"] = cfg
    _G["model"] = GraphModel(repo_root, cfg, memory)
    _G["built"] = None
    _install_footprint_cache()
    # Warm the cache while the video plays: every memory object's footprint, as the builder will ask for it.
    m = _G["model"]
    update_scene_graph(m.scene["scene_graph"], [], m.objects, targeted.ChangedMap(m.map, [], {}), cfg)


def _install_footprint_cache():
    """97 % of a graph rebuild (8.2 s of 8.5 on lap_in_1) was the alpha-shape footprint of every object, computed again
    on every rebuild although only the changed objects are placed. A footprint depends only on the object's points,
    so it is kept per (alpha, points). The original jitters the points with unseeded noise on every call, so reusing a
    footprint is exactly as faithful as recomputing one."""
    import hashlib

    import scene_graph.builder.dovsg_relations as R

    if getattr(R._footprint, "cached", False):
        return
    base, cache = R._footprint, {}

    def footprint(points, alpha):
        pts = np.ascontiguousarray(points)
        key = (float(alpha), pts.shape, hashlib.blake2b(pts.tobytes(), digest_size=16).digest())
        if key not in cache:
            cache[key] = base(points, alpha)
        return cache[key]

    footprint.cached = True
    R._footprint = footprint


def _graph_ping() -> bool:
    return True


def _graph_build(final: dict, rows: list, moved: dict, observed: list) -> dict:
    assoc = {"rows": rows, "moved": moved}
    built = build_graph(_G["model"], _G["cfg"], final, assoc, observed)
    _G["built"] = built
    return {"removed": built["removed"], "moved": {c: r["centroid"] for c, r in built["moved"].items()},
            "added": [o["class_id"] for o in built["added"]], "deferred": built["deferred"], "seconds": built["seconds"],
            "orphans": built["orphans"]}


def _graph_save(scene_name: str, sg_dir: str) -> dict:
    return save_built(_G["built"], scene_name, Path(sg_dir))


class GraphWorker:
    """Builds the updated graph in a process; the newest request wins (one running, one waiting)."""

    def __init__(self, repo_root, config: str, memory: str):
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor

        self.pool = ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn"),
                                        initializer=_graph_init, initargs=(str(repo_root), config, memory))
        self.ready = self.pool.submit(_graph_ping)
        self.running = None        # (future, signature, t_submit)
        self.waiting = None        # (final, rows, moved, observed, signature)
        self.built_signature = None
        self.history: list[dict] = []
        self.t0 = time.perf_counter()

    @staticmethod
    def signature(final: dict, assoc: dict | None) -> str:
        rows = [(r["object"], r["decision"], r.get("memory_id")) for r in (assoc["rows"] if assoc else [])
                if r["decision"] in ("added", "moved")]
        return repr((sorted(final.items()), rows))

    def request(self, final: dict, assoc: dict | None):
        sig = self.signature(final, assoc)
        if sig == self.built_signature or (self.running and self.running[1] == sig) or (self.waiting and self.waiting[4] == sig):
            return
        rows = assoc["rows"] if assoc else []
        moved = assoc["moved"] if assoc else {}
        observed = [ObsLite(o.raw, o.grid_idx, o.grid_cols) for o in (assoc["observed"] if assoc else [])]
        self.waiting = (final, rows, moved, observed, sig)
        self.poll()

    def poll(self):
        """Collect a finished build and start the waiting request."""
        if self.running and self.running[0].done():
            fut, sig, t_sub = self.running
            self.running = None
            try:
                res = fut.result()
                self.built_signature = sig
                self.history.append({**{k: res[k] for k in ("removed", "moved", "added", "seconds")},
                                     "latency_s": round(time.perf_counter() - t_sub, 2),
                                     "t_since_start": round(time.perf_counter() - self.t0, 2)})
            except Exception:  # noqa: BLE001 -- a preview that failed must not stop the session
                import logging

                logging.getLogger("covis_gate").exception("[graph] build failed")
        if self.running is None and self.waiting is not None and self.ready.done():
            final, rows, moved, observed, sig = self.waiting
            self.waiting = None
            self.running = (self.pool.submit(_graph_build, final, rows, moved, observed), sig, time.perf_counter())

    def wait(self, timeout: float = 180.0):
        end = time.perf_counter() + timeout
        while (self.running or self.waiting) and time.perf_counter() < end:
            self.poll()
            time.sleep(0.05)

    def save(self, scene_name: str, sg_dir) -> dict:
        return self.pool.submit(_graph_save, scene_name, str(sg_dir)).result()

    def close(self):
        from selective import kill_tree

        for proc in list(getattr(self.pool, "_processes", {}).values()):
            kill_tree(proc.pid)
        self.pool.shutdown(wait=False, cancel_futures=True)


