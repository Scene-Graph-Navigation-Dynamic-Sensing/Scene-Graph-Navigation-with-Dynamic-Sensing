"""§4, §6, §7: memory voxels projected into every new frame, the signed
residual's three outcomes, per-object log-odds of presence, and the
co-visibility gate that decides how much a frame's absence evidence counts.

Per frame f and object i (object voxels only; background voxels feed the gate):
  n_hit, n_thr        hit / through voxels (eq. 15); occluded ones add nothing
  l+ = alpha n_hit l, l- = -alpha n_thr l                 (eq. 7)
  rho_i^f = 1 - prod_j (1 - w_ij P_j h_j^f)               (eq. 16, h_j^f = hits / projected of neighbour j in f)
  L_i <- clip(L_i + l+ + rho_i^f l-)                      (eq. 17; mixture form eq. 18 as an option)

Two additions the PDF does not have, both switchable:
  - frame correlation: a video gives ~200 nearly identical frames, which the
    PDF's five-frame example does not face. The same Kish design effect as
    eq. 6 is applied across the m_i frames that saw object i:
    beta_i = 1 / (1 + (m_i - 1) rho_f). rho_f = 0 turns it off.
  - local background as a certifier: background voxels (walls, floor, the
    parts of the map in no object) within background_radius_m of a target
    are one more noisy-OR term, P = 1 (structure), w = background_weight.
    Most objects in a room have few graph neighbours; the surface they stand
    on is often background. background_weight = 0 turns it off.
"""
from __future__ import annotations

import logging

import numpy as np

import theory as T

logger = logging.getLogger("covis_gate")

RELATION_WEIGHT = {"belongs": 0.9, "belong": 0.9, "inside": 0.9, "on": 0.8}


# ─── §6 projection + signed residual ───

def classify_frame(o: dict, pts: np.ndarray, cols: np.ndarray, ecfg: dict):
    """One frame: (rows of `pts` with a measurement, rows, cols, outcome).
    Invalid depth and depth discontinuities (o["valid"] is edge-free, §6.3
    guards) are discarded, not treated as measurements. A hit whose colour
    differs (after the frame's exposure gain) is a through for the stored
    object: something else is there now."""
    from change_detection import project

    h, w = o["valid"].shape
    ii, r, c, z = project(pts, o["pose"], o["intrinsic"], h, w)
    keep = o["valid"][r, c] & (z <= ecfg["max_depth_m"])
    ii, r, c, z = ii[keep], r[keep], c[keep], z[keep]
    res = o["depth"][r, c] - z
    sigma = T.depth_sigma(z, ecfg["sigma0_m"], ecfg["sigma_k1"], ecfg["sigma_k2"])
    out = T.classify_residual(res, sigma, signed=ecfg["residual"] == "signed")
    if ecfg["colour_refine"]:
        hit = np.where(out == T.HIT)[0]
        new_rgb, mem_rgb = o["rgb"][r[hit], c[hit]], cols[ii[hit]]
        gain = np.ones(3)
        if len(hit) >= 500:
            gain = np.clip(np.median(new_rgb, 0) / np.maximum(np.median(mem_rgb, 0), 1e-3), 0.5, 2.0)
        dist = np.linalg.norm(new_rgb - np.clip(mem_rgb * gain, 0, 1), axis=1)
        out[hit[dist > ecfg["colour_thresh"]]] = T.THROUGH
    return ii, r, c, out


def gather(frames: list, pts: np.ndarray, cols: np.ndarray, owner: np.ndarray, n_obj: int, bg_groups: dict,
           ecfg: dict) -> dict:
    """Counts per frame and object, background-neighbourhood counts per frame
    and target, through events (for the coverage condition), per-voxel totals."""
    F = len(frames)
    n_valid = np.zeros((F, n_obj), np.int32)
    n_hit = np.zeros((F, n_obj), np.int32)
    n_thr = np.zeros((F, n_obj), np.int32)
    bg_valid = {i: np.zeros(F, np.int32) for i in bg_groups}
    bg_hit = {i: np.zeros(F, np.int32) for i in bg_groups}
    tot = np.zeros((len(pts), 4), np.int32)  # per voxel: [unused, hit, through, occluded]
    through_events = []  # (frame, voxel row) of object voxels
    status = np.zeros(len(pts), np.uint8)
    for f, o in enumerate(frames):
        ii, _, _, out = classify_frame(o, pts, cols, ecfg)
        np.add.at(tot, (ii, out), 1)
        own = owner[ii]
        obj = own >= 0
        n_valid[f] = np.bincount(own[obj], minlength=n_obj)
        n_hit[f] = np.bincount(own[obj & (out == T.HIT)], minlength=n_obj)
        n_thr[f] = np.bincount(own[obj & (out == T.THROUGH)], minlength=n_obj)
        thr_rows = ii[obj & (out == T.THROUGH)]
        through_events.append(np.stack([np.full(len(thr_rows), f, np.int32), thr_rows.astype(np.int32)], 1))
        if bg_groups:
            status[:] = 0
            status[ii] = out
            for i, rows in bg_groups.items():
                st = status[rows]
                bg_valid[i][f] = int((st > 0).sum())
                bg_hit[i][f] = int((st == T.HIT).sum())
        if f and f % 50 == 0:
            logger.info(f"[3 evidence] {f}/{F} frames")
    return {"n_valid": n_valid, "n_hit": n_hit, "n_thr": n_thr, "bg_valid": bg_valid, "bg_hit": bg_hit,
            "through_events": np.concatenate(through_events) if through_events else np.zeros((0, 2), np.int32),
            "voxel_totals": tot[:, 1:]}


# ─── §7 neighbours ───

def neighbours(objects: list, relations: list, next_to: list, ecfg: dict) -> dict:
    """i -> [(j, w_ij, why)]: scene-graph edges both ways (0.9 belongs / inside,
    0.8 on), and "next to" edges as the co-visibility edges, weighted by how
    close the two are: next_to_weight x (1 - gap / next_to_max_gap_m)."""
    idx = {o["class_id"]: n for n, o in enumerate(objects)}
    nb = {n: {} for n in range(len(objects))}

    def put(a, b, w, why):
        if a in idx and b in idx and a != b:
            i, j = idx[a], idx[b]
            if w > nb[i].get(j, (0, ""))[0]:
                nb[i][j] = (w, why)

    for r in relations:
        w = RELATION_WEIGHT.get(r["relation"], 0.0)
        put(r["child"], r["parent"], w, f"{r['relation']} {r['parent']}")
        put(r["parent"], r["child"], w, f"under {r['child']}" if r["relation"] == "on" else f"holds {r['child']}")
    for e in next_to:
        w = ecfg["next_to_weight"] * max(0.0, 1 - e["gap_m"] / ecfg["next_to_max_gap_m"])
        put(e["a"], e["b"], w, f"next to {e['b']} ({e['gap_m']} m)")
        put(e["b"], e["a"], w, f"next to {e['a']} ({e['gap_m']} m)")
    return {i: [(j, w, why) for j, (w, why) in d.items() if w > 0] for i, d in nb.items()}


def background_groups(pts: np.ndarray, owner: np.ndarray, reliable: np.ndarray, targets: list, ecfg: dict) -> dict:
    """target -> rows of reliable background voxels within background_radius_m of it."""
    if ecfg["background_weight"] <= 0:
        return {}
    from scipy.spatial import cKDTree

    bg_rows = np.where((owner == -1) & reliable)[0]
    if len(bg_rows) == 0:
        return {}
    tree = cKDTree(pts[bg_rows])
    out = {}
    for i in targets:
        mine = pts[owner == i]
        if len(mine) > 3000:
            mine = mine[np.random.default_rng(0).choice(len(mine), 3000, replace=False)]
        hits = tree.query_ball_point(mine, ecfg["background_radius_m"])
        rows = np.unique(np.concatenate([np.asarray(h, np.int64) for h in hits])) if len(hits) else np.zeros(0, np.int64)
        if len(rows) >= ecfg["background_min_voxels"]:
            out[i] = bg_rows[rows]
    return out


# ─── §4 + §7 belief sweep ───

def beliefs(counts: dict, prior_L: np.ndarray, is_anchor: np.ndarray, nbrs: dict, ecfg: dict) -> dict:
    """Anchors first with rho = 1 (their beliefs and hit fractions then fixed),
    then targets, gated by their neighbours; targets that neighbour targets are
    swept again with the neighbours' new beliefs until nothing changes (§7.4)."""
    n_valid, n_hit, n_thr = counts["n_valid"], counts["n_hit"], counts["n_thr"]
    F, n = n_hit.shape
    ell = T.voxel_llr(ecfg["p_hit_present"], ecfg["p_hit_absent"])
    informative = n_hit + n_thr
    alpha = (np.full(informative.shape, ecfg["alpha_fixed"]) if ecfg["alpha_fixed"]
             else T.kish_alpha(informative, ecfg["rho_c"]))
    m = (informative > 0).sum(0)
    beta = T.kish_alpha(m, ecfg["rho_f"])  # across frames
    pos = beta * alpha * n_hit * ell                     # l+ per frame (F, n)
    neg = -beta * alpha * n_thr * ell                    # l- per frame (F, n)
    h = np.where(n_valid > 0, n_hit / np.maximum(n_valid, 1), 0.0)
    h_bg = {i: np.where(counts["bg_valid"][i] > 0, counts["bg_hit"][i] / np.maximum(counts["bg_valid"][i], 1), 0.0)
            for i in counts["bg_valid"]}
    lmax = ecfg["l_max"]

    def run(i, rho):
        L = prior_L[i]
        traj = np.empty(F)
        for f in range(F):
            L = float(T.clip_logodds(L + pos[f, i] + T.gated_absence(neg[f, i], rho[f], ecfg["update"]), lmax))
            traj[f] = L
        return L, traj

    def gate(i, P):
        if not ecfg["gate"]:
            return np.ones(F)
        ws, Ps, hs = [], [], []
        for j, w, _ in nbrs[i]:
            ws.append(w); Ps.append(P[j]); hs.append(h[:, j])
        if i in h_bg:
            ws.append(ecfg["background_weight"]); Ps.append(1.0); hs.append(h_bg[i])
        if not ws:
            return np.zeros(F)
        return T.noisy_or(np.array(ws)[None, :], np.array(Ps)[None, :], np.stack(hs, 1))

    L = prior_L.astype(np.float64).copy()
    rho = np.zeros((F, n))
    traj = np.zeros((F, n))
    for i in np.where(is_anchor)[0]:
        rho[:, i] = 1.0
        L[i], traj[:, i] = run(i, rho[:, i])
    targets = np.where(~is_anchor)[0]
    sweeps = 0
    P_prev = T.sigmoid(L)  # targets start from their priors
    for sweeps in range(1, ecfg["max_sweeps"] + 1):
        P_now = P_prev.copy()
        for i in targets:
            rho[:, i] = gate(i, P_prev)
            L[i], traj[:, i] = run(i, rho[:, i])
            P_now[i] = T.sigmoid(L[i])
        if np.max(np.abs(P_now - P_prev), initial=0) < 1e-4:
            P_prev = P_now
            break
        P_prev = P_now
    return {"L": L, "P": T.sigmoid(L), "rho": rho, "trajectory": traj, "h": h, "h_background": h_bg,
            "alpha": alpha, "beta": beta, "frames_informative": m, "ell": ell, "sweeps": sweeps,
            "evidence_pos": pos.sum(0), "evidence_neg_raw": neg.sum(0)}


def gated_coverage(counts: dict, rho: np.ndarray, owner: np.ndarray, reliable: np.ndarray, n_obj: int) -> np.ndarray:
    """§8.1 coverage condition, per object: each judged voxel counts with the
    largest rho of a frame that saw through it; divided by the object's judged
    voxels (in [0, 1], unlike a sum over ~200 frames)."""
    ev = counts["through_events"]
    best = np.zeros(len(owner))
    if len(ev):
        f, row = ev[:, 0], ev[:, 1]
        np.maximum.at(best, row, rho[f, owner[row]])
    judged = np.bincount(owner[(owner >= 0) & reliable], minlength=n_obj).astype(np.float64)
    covered = np.bincount(owner[owner >= 0], weights=best[owner >= 0], minlength=n_obj)
    return covered / np.maximum(judged, 1)
