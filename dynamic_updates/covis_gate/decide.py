"""§8 decision, data association and identity.

  Chow's rule (eq. 19)   absent if P < C_D / C_FR (and the gated through coverage >= v_min),
                         present if P > 1 - C_D / C_MC, deferred otherwise
  explained              a new-video object at the same place as a memory object with the same look
                         (0.3 geometric + 0.7 cosine >= match_threshold) explains it. An absent object
                         explained this way is re-detected in place: present (P1, seeing is self-proving)
  association            absent objects x unexplained new-video objects: s_iu = 0.3 shape + 0.7 cosine,
                         shape = overlap once the two centroids coincide (a moved object is not at its
                         old place, so DovSG's place overlap would be 0). The Hungarian assignment
                         maximises sum s_iu x p(parent_u | i) (eq. 20); a pair is MOVED when s_iu >= delta
  leftovers              absent and unmatched -> REMOVED; unexplained new-video objects in space the
                         memory did not have occupied -> ADDED
"""
from __future__ import annotations

import numpy as np

import theory as T


def _embedding(obj) -> np.ndarray:
    ft = obj.get("semantic_embedding", obj.get("clip_ft"))
    ft = ft.detach().cpu().numpy() if hasattr(ft, "detach") else np.asarray(ft)
    ft = ft.astype(np.float32).ravel()
    return ft / max(np.linalg.norm(ft), 1e-8)


def chow_labels(P: np.ndarray, coverage: np.ndarray, dcfg: dict) -> tuple[list, tuple]:
    lo, hi = T.chow_thresholds(dcfg["cost_false_removal"], dcfg["cost_missed_change"], dcfg["cost_defer"])
    labels = []
    for p, cov in zip(P, coverage):
        if p > hi:
            labels.append("present")
        elif p < lo and cov >= dcfg["v_min"]:
            labels.append("absent")
        else:
            labels.append("deferred")
    return labels, (lo, hi)


def shape_overlap(a: np.ndarray, b: np.ndarray, dist: float) -> float:
    """Share of b's points within dist of a's once both are centred: same shape and size."""
    from scipy.spatial import cKDTree

    d, _ = cKDTree(a - a.mean(0)).query(b - b.mean(0), k=1, distance_upper_bound=dist)
    return float(np.isfinite(d).mean())


def associate(memory_objects, mem_obj_pts, labels, obs_objects, obs_pts, obs_parent_obs, root_id, receptacles_of,
              n_receptacles, remaining_tree, dcfg) -> dict:
    """obs_parent_obs[u]: id (in the new video's graph) of what new-video object u stands on / is in, or None.
    It becomes a memory id through the "explains" matches (the root, the floor, keeps its id).
    receptacles_of[i]: memory object i's receptacle counts.
    Returns rows per new-video object and the moved pairs."""
    from scipy.optimize import linear_sum_assignment
    from scipy.spatial import cKDTree

    trees = [cKDTree(p) for p in mem_obj_pts]
    mem_emb = np.stack([_embedding(o) for o in memory_objects])
    rows, explained_by = [], {}
    for u, (o, pts) in enumerate(zip(obs_objects, obs_pts)):
        cos = mem_emb @ _embedding(o)
        geo = np.array([np.isfinite(t.query(pts, k=1, distance_upper_bound=dcfg["match_distance_m"])[0]).mean()
                        for t in trees])
        score = 0.3 * geo + 0.7 * cos
        score_geo = np.where(geo >= dcfg["min_geometric"], score, -1)
        j = int(np.argmax(score_geo))
        d, _ = remaining_tree.query(pts, k=1, distance_upper_bound=dcfg["novelty_distance_m"])
        novelty = float(np.isinf(d).mean())
        row = {"index": u, "object": o["class_id"], "voxels": int(len(pts)), "detections": int(o["num_detections"]),
               "centroid": np.round(pts.mean(0), 3).tolist(), "novelty": round(novelty, 3),
               "best_memory": memory_objects[j]["class_id"], "cosine": round(float(cos[j]), 3),
               "geometric": round(float(geo[j]), 3), "score": round(float(score[j]), 3)}
        if score_geo[j] >= dcfg["match_threshold"]:
            row["decision"] = "explains"
            row["memory_id"] = memory_objects[j]["class_id"]
            explained_by.setdefault(j, []).append(u)
        elif novelty >= dcfg["min_novelty"]:
            row["decision"] = "unexplained"
        else:
            row["decision"] = "in the memory map already"
            row["inside"] = memory_objects[int(np.argmax(geo))]["class_id"] if geo.max() > 0 else "background"
        rows.append(row)

    obs_to_mem = {r["object"]: r["memory_id"] for r in rows if r["decision"] == "explains"}
    obs_parent = [obs_to_mem.get(p, root_id if p == root_id else None) for p in obs_parent_obs]
    for r in rows:
        r["parent"] = obs_parent[r["index"]]
    redetected = [memory_objects[i]["class_id"] for i in explained_by if labels[i] == "absent"]
    if dcfg["redetection_overrides_absent"]:
        for i in explained_by:
            if labels[i] == "absent":
                labels[i] = "present"
    absent = [i for i, lab in enumerate(labels) if lab == "absent"]
    unexplained = [r["index"] for r in rows if r["decision"] == "unexplained"]
    moved, pair_scores = {}, []
    if absent and unexplained:
        s = np.zeros((len(absent), len(unexplained)))
        weighted = np.zeros_like(s)
        for a, i in enumerate(absent):
            for b, u in enumerate(unexplained):
                cos = float(mem_emb[i] @ _embedding(obs_objects[u]))
                shp = shape_overlap(mem_obj_pts[i], obs_pts[u], dcfg["shape_distance_m"])
                s[a, b] = 0.3 * shp + 0.7 * cos
                prior = T.receptacle_prior(receptacles_of[i], obs_parent[u] or "none", n_receptacles, dcfg["beta"])
                weighted[a, b] = s[a, b] * prior
                pair_scores.append({"memory": memory_objects[i]["class_id"], "new": obs_objects[u]["class_id"],
                                    "cosine": round(cos, 3), "shape": round(shp, 3), "s": round(float(s[a, b]), 3),
                                    "receptacle_prior": round(prior, 3), "new_parent": obs_parent[u]})
        ra, cb = linear_sum_assignment(-weighted)
        for a, b in zip(ra, cb):
            if s[a, b] >= dcfg["delta"]:
                moved[absent[a]] = unexplained[b]
    for i, u in moved.items():
        rows[u]["decision"] = "moved"
        rows[u]["memory_id"] = memory_objects[i]["class_id"]
    for r in rows:
        if r["decision"] == "unexplained":
            r["decision"] = "added"
    final = {}
    for i, lab in enumerate(labels):
        cid = memory_objects[i]["class_id"]
        if lab == "absent":
            final[cid] = "moved" if i in moved else "removed"
        else:
            final[cid] = lab
    return {"rows": rows, "moved": {memory_objects[i]["class_id"]: u for i, u in moved.items()},
            "labels": final, "redetected_in_place": redetected, "pair_scores": pair_scores}
