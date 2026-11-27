"""§5 relocalisation from anchors, without ACE.

  retrieval    a spread of new frames matched (SuperPoint + LightGlue) against a spread of
               memory frames; each new frame's best memory frames give the first matches
  anchor mask  a match is kept only when its memory keypoint, lifted with the memory's
               depth and pose, lands on (within owner_distance_m of) a voxel of an anchor:
               an object with a low learned move rate, or the background (walls, floor)
  RANSAC       eq. 10-12: minimal samples of 3, closed-form Umeyama fit (eq. 11), adaptive
               iteration count; refit on all inliers
  refine       every new frame matched against the memory frames nearest its first pose,
               RANSAC again on all anchor matches
  phi_j        §5.5: share of anchor j's matches that are inliers; phi_j < tau_phi -> the
               anchor moved: demoted to a target for this visit, and the pose refit without it

The new video's frames are tied together by its own DROID-SLAM trajectory
(c2b), so T* is one transform for the whole video, estimated from all its
frames' matches. With monocular depth the two videos' scales differ, so T* is
a Sim3 by default (rigid=True gives the PDF's SE(3)).
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np

import theory as T

logger = logging.getLogger("covis_gate")

BACKGROUND = -1   # owner code of voxels in no object
NO_OWNER = -2     # memory keypoint not on the map


def ransac_umeyama(src: np.ndarray, dst: np.ndarray, thresh: float, with_scale: bool, p: float = 0.99,
                   max_iters: int = 5000, score_n: int = 20000, seed: int = 0):
    """eq. 10-12. Hypotheses are scored on a random subset of the matches
    (there can be 100k+); the iteration count adapts to the best inlier ratio
    so far (eq. 12). Degenerate (near-collinear) samples are rejected (§5.5)."""
    from scripts.compare_video_poses import apply, umeyama

    rng = np.random.default_rng(seed)
    n = len(src)
    sub = rng.choice(n, min(n, score_n), replace=False)
    best, best_count, needed, it = None, -1, max_iters, 0
    while it < min(needed, max_iters):
        it += 1
        idx = rng.choice(n, 3, replace=False)
        if np.linalg.norm(np.cross(src[idx[1]] - src[idx[0]], src[idx[2]] - src[idx[0]])) < 1e-4:
            continue
        Tm = umeyama(src[idx], dst[idx], with_scale=with_scale)
        if with_scale and not 0.66 < np.cbrt(np.linalg.det(Tm[:3, :3])) < 1.5:
            continue
        count = int((np.linalg.norm(apply(Tm, src[sub]) - dst[sub], axis=1) < thresh).sum())
        if count > best_count:
            best, best_count = Tm, count
            needed = T.ransac_iterations(max(count / len(sub), 1e-3), p)
    if best is None:
        return None, np.zeros(n, bool), it
    Tm = best
    for _ in range(3):
        inl = np.linalg.norm(apply(Tm, src) - dst, axis=1) < thresh
        if inl.sum() < 3:
            break
        Tm = umeyama(src[inl], dst[inl], with_scale=with_scale)
    inl = np.linalg.norm(apply(Tm, src) - dst, axis=1) < thresh
    return Tm, inl, it


class AnchorMatcher:
    """LightGlue matches between new and memory frames, lifted to 3D, each
    labelled with the owner (object index / BACKGROUND / NO_OWNER) of the
    memory voxel its memory keypoint lands on."""

    def __init__(self, memory, mem_tree, voxel_owner: np.ndarray, rcfg: dict):
        from dovsg_core import RGBFeatureMatch

        self.memory, self.tree, self.owner, self.cfg = memory, mem_tree, voxel_owner, rcfg
        self.matcher = RGBFeatureMatch(str(Path(rcfg["dovsg_root"]) / "third_party" / "LightGlue"),
                                       rcfg["lightglue_max_keypoints"])
        self.mem_feats = {}

    def mem_feature(self, j: int):
        if j not in self.mem_feats:
            self.mem_feats[j] = self.matcher.extract_feature(self.memory.rgb(j))
        return self.mem_feats[j]

    def new_feature(self, o: dict):
        return self.matcher.extract_feature((o["rgb"] * 255).astype(np.uint8))

    def match(self, f_new, j: int):
        r0, r1, m = self.matcher.match_features(f_new, self.mem_feature(j))
        pairs = m["matches"].cpu().numpy()
        return r0["keypoints"].cpu().numpy()[pairs[:, 0]], r1["keypoints"].cpu().numpy()[pairs[:, 1]]

    def lift_pair(self, o: dict, j: int, kp_new: np.ndarray, kp_mem: np.ndarray):
        """-> (src in the new video's world, dst in the memory world, owner per match, kp_new, kp_mem)."""
        from run_projection_update import lift

        pn, okn = lift(kp_new, o["point"], o["valid"])
        mpt, mvalid = self.memory.geometry(j)
        pm, okm = lift(kp_mem, mpt, mvalid)
        ok = okn & okm
        c2b, mpose = o["c2b"], self.memory.poses[j]
        src = pn[ok] @ c2b[:3, :3].T + c2b[:3, 3]
        dst = pm[ok] @ mpose[:3, :3].T + mpose[:3, 3]
        d, nn = self.tree.query(dst, k=1, distance_upper_bound=self.cfg["owner_distance_m"])
        owner = np.full(len(dst), NO_OWNER, np.int64)
        hit = np.isfinite(d)
        owner[hit] = self.owner[nn[hit]]
        return src, dst, owner, kp_new[ok], kp_mem[ok]


def relocalize(observations: dict, memory, mem_tree, voxel_owner: np.ndarray, anchor_codes: set, rcfg: dict,
               out_dir: Path, viz_keys: list, owner_names: dict) -> dict:
    """Returns {"T": 4x4 new world -> memory world (or None), "phi": {code: ...},
    "demoted": [object indexes], ...numbers}. Sets o["pose"] (camera -> memory
    world) and scales o["point"], o["depth"] when it succeeds."""
    import covis_viz as viz
    from run_projection_update import nearest_memory_frames
    from scripts.compare_video_poses import apply, decompose_sim3

    t0 = time.time()
    am = AnchorMatcher(memory, mem_tree, voxel_owner, rcfg)
    keys = list(observations)
    with_scale = not rcfg["rigid"]

    # ── retrieval: a spread of new frames vs a spread of memory frames ──
    new_sub = sorted({keys[i] for i in np.linspace(0, len(keys) - 1, min(rcfg["retrieval_new_frames"], len(keys)))
                      .round().astype(int)})
    n_mem = len(memory.names)
    mem_sub = sorted(set(np.linspace(0, n_mem - 1, min(rcfg["retrieval_memory_frames"], n_mem)).round().astype(int).tolist()))
    counts = np.zeros((len(new_sub), len(mem_sub)), np.int32)
    feats = {}
    for a, k in enumerate(new_sub):
        feats[k] = am.new_feature(observations[k])
        for b, j in enumerate(mem_sub):
            counts[a, b] = len(am.match(feats[k], j)[0])
    logger.info(f"[1 reloc] retrieval: {len(new_sub)} new x {len(mem_sub)} memory frames ({time.time() - t0:.0f}s)")
    viz.retrieval_matrix(out_dir / "1_retrieval_matches.png", counts, [observations[k]["frame"] for k in new_sub],
                         [memory.names[j] for j in mem_sub])

    def collect(pairs_of: dict):
        S, D, O, F = [], [], [], []
        n_pairs = 0
        for k, js in pairs_of.items():
            f = feats.get(k) if k in feats else am.new_feature(observations[k])
            for j in js:
                kp_new, kp_mem = am.match(f, j)
                n_pairs += 1
                if len(kp_new) < rcfg["min_pair_matches"]:
                    continue
                s, d, ow, _, _ = am.lift_pair(observations[k], j, kp_new, kp_mem)
                S.append(s); D.append(d); O.append(ow); F.append(np.full(len(s), k))
        if not S:
            return None
        return np.concatenate(S), np.concatenate(D), np.concatenate(O), np.concatenate(F), n_pairs

    def anchored(c):
        S, D, O, F, _ = c
        keep = np.isin(O, list(anchor_codes))
        return S[keep], D[keep], O[keep], F[keep]

    first = {k: [mem_sub[b] for b in np.argsort(-counts[a])[:rcfg["retrieval_top_k"]]
                 if counts[a, b] >= rcfg["min_pair_matches"]] for a, k in enumerate(new_sub)}
    c0 = collect(first)
    out = {"retrieval": {"new_frames": len(new_sub), "memory_frames": len(mem_sub),
                         "median_best_matches": int(np.median(counts.max(1))) if counts.size else 0}}
    if c0 is None:
        out.update(T=None, failure="no matches in retrieval")
        return out
    S0, D0, O0, _ = anchored(c0)
    if len(S0) < rcfg["min_inliers"]:
        out.update(T=None, failure=f"only {len(S0)} anchor matches in retrieval")
        return out
    T0, inl0, it0 = ransac_umeyama(S0, D0, rcfg["ransac_thresh_m"], with_scale)
    out["first_fit"] = {"anchor_matches": int(len(S0)), "inliers": int(inl0.sum()), "iterations": it0}
    if T0 is None:
        out.update(T=None, failure="RANSAC found no hypothesis")
        return out

    # ── refine: every (stride-th) new frame vs the memory frames nearest its first pose ──
    s0, R0, t0v = decompose_sim3(T0)
    refine = {}
    for k in keys[::rcfg["pose_frame_stride"]]:
        pose = np.eye(4)
        pose[:3, :3] = R0 @ observations[k]["c2b"][:3, :3]
        pose[:3, 3] = s0 * R0 @ observations[k]["c2b"][:3, 3] + t0v
        refine[k] = [int(j) for j in nearest_memory_frames(pose, memory.poses, rcfg["refine_candidates"],
                                                           rcfg["candidate_angle_weight_m"])]
    c1 = collect(refine)
    S_all, D_all, O_all, F_all, n_pairs = c1
    S, D, O, F = anchored(c1)
    Tm, inl, it = ransac_umeyama(S, D, rcfg["ransac_thresh_m"], with_scale, seed=1)

    # ── phi_j and demotion (§5.5) ──
    def phis(Tm, S, D, O):
        err = np.linalg.norm(apply(Tm, S) - D, axis=1)
        res = {}
        for code in np.unique(O):
            m = O == code
            res[int(code)] = {"matches": int(m.sum()), "phi": round(float((err[m] < rcfg["ransac_thresh_m"]).mean()), 3),
                              "median_error_m": round(float(np.median(err[m])), 4)}
        return res

    phi = phis(Tm, S, D, O)
    demoted = [c for c, v in phi.items() if c != BACKGROUND and v["matches"] >= rcfg["phi_min_matches"]
               and v["phi"] < rcfg["tau_phi"]]
    if demoted:
        keep = ~np.isin(O, demoted)
        Tm, inl_k, it2 = ransac_umeyama(S[keep], D[keep], rcfg["ransac_thresh_m"], with_scale, seed=2)
        it += it2
        inl = np.zeros(len(S), bool)
        inl[keep] = inl_k
        phi = phis(Tm, S, D, O)
    # matches on non-anchor objects (not used): how well they agree with the anchors' pose
    non_anchor = phis(Tm, S_all, D_all, O_all)
    s, R, tr = decompose_sim3(Tm)
    for o in observations.values():
        o["point"] = o["point"] * s
        o["depth"] = o["depth"] * s
        pose = np.eye(4)
        pose[:3, :3] = R @ o["c2b"][:3, :3]
        pose[:3, 3] = s * R @ o["c2b"][:3, 3] + tr
        o["pose"] = o["sim3_pose"] = pose
    inlier_ratio = float(inl.mean()) if len(inl) else 0.0
    out.update({
        "T": Tm, "scale": round(s, 4), "rotation_deg": round(float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))), 2),
        "translation_m": round(float(np.linalg.norm(tr)), 3), "pairs_matched": int(n_pairs),
        "matches_3d": int(len(S_all)), "anchor_matches": int(len(S)), "inliers": int(inl.sum()),
        "inlier_ratio": round(inlier_ratio, 3), "ransac_iterations": int(it),
        "ransac_iterations_eq12": T.ransac_iterations(max(inlier_ratio, 1e-3)),
        "median_error_inliers_mm": round(float(np.median(np.linalg.norm(apply(Tm, S[inl]) - D[inl], axis=1))) * 1000, 1)
        if inl.any() else None,
        "phi": {owner_names[c]: v for c, v in phi.items()},
        "agreement_all_owners": {owner_names.get(c, "not on the map"): v for c, v in non_anchor.items()},
        "demoted": demoted, "time_s": round(time.time() - t0, 1)})
    if inl.sum() < rcfg["min_inliers"] or inlier_ratio < rcfg["min_inlier_ratio"]:
        out["failure"] = (f"{int(inl.sum())} inliers ({inlier_ratio:.0%}): the largest rigid set may not be static "
                          "structure (§5.5 case i)")

    # pictures: matches of a few frames, anchor (green) vs other (red)
    for k in viz_keys:
        if k not in refine or not refine[k]:
            continue
        f = am.new_feature(observations[k])
        kp_new, kp_mem = am.match(f, refine[k][0])
        if len(kp_new) < 8:
            continue
        _, _, ow, kn, km = am.lift_pair(observations[k], refine[k][0], kp_new, kp_mem)
        viz.anchor_match_pair(out_dir / f"1_matches_{observations[k]['frame']}.png",
                              (observations[k]["rgb"] * 255).astype(np.uint8), memory.rgb(refine[k][0]), kn, km,
                              np.isin(ow, list(anchor_codes)),
                              f"new {observations[k]['frame']} vs memory {memory.names[refine[k][0]]}")
    del am
    return out
