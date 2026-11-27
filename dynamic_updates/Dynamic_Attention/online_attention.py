"""The attention modules inside the live CoViS engine, by subclassing (covis_gate/online is not edited).

AttentionBelief      OnlineBelief + the presence head's per-frame log-likelihood ratio per object. It goes through the
                     same machinery as the depth evidence: the frame discount (Kish, rho_f) over the frames that produced
                     it, and, when it says absent, the same neighbour gate rho. With no presence term it is exactly
                     OnlineBelief (test_dynamic_attention.py checks it).
AttentionCovis       OnlineCovis + YOLOE patch features of every evidence frame + the presence head on every memory object
                     in view. Memory descriptors (each object's YOLOE features in the memory frame that sees most of it)
                     are computed once at start-up.
AttentionObservationManager
                     ObservationManager with the association (§8.2) done by the attention matcher instead of
                     decide.associate: one assignment for explains / moved / new, with the novelty veto, the size and
                     support filters kept from the rule path.
"""
from __future__ import annotations

import logging
import time

import numpy as np
import torch

import common as C

C.setup_paths()
import theory as T  # noqa: E402
from belief import OnlineBelief  # noqa: E402
from engine import OnlineCovis  # noqa: E402
from observe import ObservationManager  # noqa: E402

logger = logging.getLogger("dynamic_attention")

ATT_DEFAULTS = {
    "presence": True,                 # the presence head's evidence in the belief (ablation: false)
    "matcher": True,                  # the attention matcher for §8.2 (ablation: false = decide.associate)
    "presence_weight": 1.0,           # multiplies the calibrated log-likelihood ratio
    "presence_llr_clip": 3.0,         # |LLR| per frame at most
    "presence_skip_occluded": 0.5,    # P(occluded) above this: the frame says nothing about the object
    "presence_min_patches": 3,
    "presence_occ_tol_m": 0.08,       # depth in front of a voxel by more than this = occluded (monocular depth: wider than Replica's 5 cm)
    "presence_points": 1500,          # voxels per object projected for its footprint
    "matcher_window_m": 2.0,          # memory objects within this (xy) of a track are candidates (plus every absent object)
    "matcher_max_memory": 64,
    "matcher_move_m": 0.30,           # a match to an absent object farther than this is a move
    "att_min_added_detections": 3,    # replaces min_added_detections (8): the matcher's dustbin carries the confidence
    # Fixes after the first live test (2026-11-25):
    "presence_on_anchors": False,     # the head on anchors too? The table (an anchor) got false "absent" votes on every
                                      # frame of the no-change replay; anchors have strong depth evidence anyway
    "matcher_redetect_min_p": 0.8,    # a match may cancel an absent decision ("re-detected in place") only this confident
    "matcher_redetect_min_cos": 0.75, # ... and this similar (the rule's delta). An extension-cord detection matched the
                                      # removed laptop with P 0.62 / cosine 0.57 and kept it in the graph
}


def load_models(device):
    from models import AttentionMatcher, PresenceHead

    mck = torch.load(C.CKPT_DIR / "matcher.pt", map_location=device, weights_only=False)
    matcher = AttentionMatcher().to(device).eval()
    matcher.load_state_dict(mck["state_dict"])
    pck = torch.load(C.CKPT_DIR / "presence.pt", map_location=device, weights_only=False)
    presence = PresenceHead().to(device).eval()
    presence.load_state_dict(pck["state_dict"])
    return matcher, float(mck["tau_match"]), presence


class AttentionBelief(OnlineBelief):
    def _alloc(self, cap: int):
        old = None if self.F == 0 else (self.att[:self.F], self.att_seen[:self.F])
        super()._alloc(cap)
        self.att = np.zeros((cap, self.n), np.float64)
        self.att_seen = np.zeros((cap, self.n), bool)
        if old is not None:
            self.att[:self.F], self.att_seen[:self.F] = old

    def add(self, counts: dict, att: np.ndarray | None = None, seen: np.ndarray | None = None) -> int:
        f = super().add(counts)
        self.att[f] = 0 if att is None else att
        self.att_seen[f] = False if seen is None else seen
        return f

    def solve(self) -> dict:
        """OnlineBelief.solve with one more evidence term per frame: beta_att * LLR, its negative part gated like the
        throughs. The code below is OnlineBelief.solve's, with that term added in `recurrence`."""
        ecfg, F, n = self.cfg, self.F, self.n
        n_valid, n_hit, n_thr = self.n_valid[:F], self.n_hit[:F], self.n_thr[:F]
        ell = T.voxel_llr(ecfg["p_hit_present"], ecfg["p_hit_absent"])
        if F == 0:
            P0 = T.sigmoid(self.prior_L)
            return {"L": self.prior_L.copy(), "P": P0, "rho": np.zeros((0, n)), "trajectory": np.zeros((0, n)),
                    "sweeps": 0, "frames_informative": np.zeros(n, int), "ell": ell, "att_sum": np.zeros(n)}
        informative = n_hit + n_thr
        alpha = (np.full(informative.shape, ecfg["alpha_fixed"]) if ecfg["alpha_fixed"]
                 else T.kish_alpha(informative, ecfg["rho_c"]))
        m = (informative > 0).sum(0)
        beta = T.kish_alpha(m, ecfg["rho_f"])
        pos = beta * alpha * n_hit * ell
        neg = -beta * alpha * n_thr * ell
        att, seen = self.att[:F], self.att_seen[:F]
        beta_att = T.kish_alpha(seen.sum(0), ecfg["rho_f"])
        att_w = beta_att * att * seen
        pos = pos + np.clip(att_w, 0, None)
        neg = neg + np.clip(att_w, None, 0)
        h = np.where(n_valid > 0, n_hit / np.maximum(n_valid, 1), 0.0)
        h_bg = {i: np.where(self.bg_valid[:F, g] > 0, self.bg_hit[:F, g] / np.maximum(self.bg_valid[:F, g], 1), 0.0)
                for g, i in enumerate(self.bg_targets)}
        lmax, mode = ecfg["l_max"], ecfg["update"]

        def recurrence(rho, cols):
            step = pos[:, cols] + T.gated_absence(neg[:, cols], rho[:, cols], mode)
            L = self.prior_L[cols].copy()
            traj = np.empty((F, len(cols)))
            for f in range(F):
                L = np.clip(L + step[f], -lmax, lmax)
                traj[f] = L
            return L, traj

        def gate(i, P):
            if not ecfg["gate"]:
                return np.ones(F)
            ws, Ps, hs = [], [], []
            for j, w, _ in self.nbrs[i]:
                ws.append(w); Ps.append(P[j]); hs.append(h[:, j])
            if i in h_bg:
                ws.append(ecfg["background_weight"]); Ps.append(1.0); hs.append(h_bg[i])
            if not ws:
                return np.zeros(F)
            return T.noisy_or(np.array(ws)[None, :], np.array(Ps)[None, :], np.stack(hs, 1))

        anchors = np.where(self.is_anchor)[0]
        targets = np.where(~self.is_anchor)[0]
        L = self.prior_L.copy()
        rho = np.zeros((F, n))
        traj = np.zeros((F, n))
        if len(anchors):
            rho[:, anchors] = 1.0
            L[anchors], traj[:, anchors] = recurrence(rho, anchors)
        P_prev = T.sigmoid(L)
        sweeps = 0
        for sweeps in range(1, ecfg["max_sweeps"] + 1):
            if len(targets):
                for i in targets:
                    rho[:, i] = gate(i, P_prev)
                L[targets], traj[:, targets] = recurrence(rho, targets)
            P_now = P_prev.copy()
            P_now[targets] = T.sigmoid(L[targets])
            if np.max(np.abs(P_now - P_prev), initial=0) < 1e-4:
                P_prev = P_now
                break
            P_prev = P_now
        return {"L": L, "P": T.sigmoid(L), "rho": rho, "trajectory": traj, "h": h, "h_background": h_bg,
                "alpha": alpha, "beta": beta, "frames_informative": m, "ell": ell, "sweeps": sweeps,
                "evidence_pos": pos.sum(0), "evidence_neg_raw": neg.sum(0), "att_sum": att_w.sum(0),
                "att_frames": seen.sum(0)}


class AttentionCovis(OnlineCovis):
    def __init__(self, model, ccfg, K, presence_net, yolo, device="cuda"):
        super().__init__(model, ccfg, K, device)
        m = model
        self.belief = AttentionBelief(m.prior_L, m.is_anchor, m.nbrs, list(m.bg_groups), ccfg)
        self.sol = self.belief.solve()
        self.presence_net, self.yolo = presence_net, yolo
        self.presence_ms: list[float] = []
        self.presence_log: list[dict] = []
        t = time.time()
        rng = np.random.default_rng(0)
        self.obj_pts = []
        for n in range(m.n_obj):
            p = m.pts[m.owner == n]
            k = ccfg["presence_points"]
            self.obj_pts.append(p if len(p) <= k else p[rng.choice(len(p), k, replace=False)])
        e = np.stack([np.asarray(o["semantic_embedding"], np.float32) for o in m.objects])
        self.q_emb = torch.as_tensor(e / np.linalg.norm(e, axis=1, keepdims=True), device=self.dev)
        self.mem_desc, self.mem_ok, self.mem_frame = self._memory_descriptors()
        self.att_setup_s = round(time.time() - t, 1)

    def _memory_descriptors(self):
        m, K = self.m, self.K
        best, best_n = np.full(m.n_obj, -1), np.zeros(m.n_obj, int)
        for j in range(len(m.frames.names)):
            point, valid = m.frames.geometry(j)
            depth = point[..., 2]
            for n, p in enumerate(self.obj_pts):
                fp = C.footprint(p, m.frames.poses[j], K, depth, valid, tol=self.cfg["presence_occ_tol_m"])
                if fp is None:
                    continue
                v = int((fp["n"] * fp["vis"]).sum())
                if v > best_n[n]:
                    best[n], best_n[n] = j, v
        desc = np.zeros((m.n_obj, C.FEAT_DIM), np.float32)
        ok = np.zeros(m.n_obj, bool)
        feats = {}
        for n in range(m.n_obj):
            j = best[n]
            if j < 0:
                continue
            if j not in feats:
                feats[j] = self.yolo(m.frames.rgb(j))
            point, valid = m.frames.geometry(j)
            fp = C.footprint(self.obj_pts[n], m.frames.poses[j], K, point[..., 2], valid, tol=self.cfg["presence_occ_tol_m"])
            keep = fp["vis"] >= 0.5
            if keep.any():
                desc[n] = C.descriptor(feats[j], fp["patches"][keep], (fp["n"] * fp["vis"])[keep])
                ok[n] = True
        return torch.as_tensor(desc, device=self.dev), ok, best

    def _presence(self, kf, depth, valid, pose, scale):
        t0 = time.perf_counter()
        cfg, n_obj = self.cfg, self.m.n_obj
        att, seen = np.zeros(n_obj), np.zeros(n_obj, bool)
        feats = self.yolo(kf.rgb)
        flat = feats.reshape(-1, feats.shape[-1])
        d = (depth * scale).cpu().numpy()
        vmask = valid.cpu().numpy()
        rows, P_max = [], 64
        for n in range(n_obj):
            if not self.mem_ok[n] or (self.belief.is_anchor[n] and not cfg["presence_on_anchors"]):
                continue
            fp = C.footprint(self.obj_pts[n], pose, self.K, d, vmask, tol=cfg["presence_occ_tol_m"])
            if fp is None or len(fp["patches"]) < cfg["presence_min_patches"]:
                continue
            order = np.argsort(-fp["n"])[:P_max]
            rows.append((n, fp["patches"][order], fp["occ"][order]))
        probs = None
        if rows:
            B = len(rows)
            pf = torch.zeros((B, P_max, C.FEAT_DIM), dtype=flat.dtype, device=self.dev)
            occ = torch.zeros((B, P_max), device=self.dev)
            pad = torch.ones((B, P_max), dtype=torch.bool, device=self.dev)
            for b, (n, p, o) in enumerate(rows):
                pf[b, :len(p)] = flat[torch.as_tensor(p, dtype=torch.long, device=self.dev)]
                occ[b, :len(p)] = torch.as_tensor(o, dtype=torch.float32, device=self.dev)
                pad[b, :len(p)] = False
            idx = torch.as_tensor([r[0] for r in rows], device=self.dev)
            with torch.no_grad():
                probs = torch.softmax(self.presence_net(self.q_emb[idx], self.mem_desc[idx], pf, occ, pad).float(), 1).cpu().numpy()
            for b, (n, _, _) in enumerate(rows):
                pp, pa, po = probs[b]
                if po > cfg["presence_skip_occluded"]:
                    continue
                llr = float(np.log(pp + 1e-6) - np.log(pa + 1e-6))
                att[n] = cfg["presence_weight"] * float(np.clip(llr, -cfg["presence_llr_clip"], cfg["presence_llr_clip"]))
                seen[n] = True
        self.presence_ms.append(1000 * (time.perf_counter() - t0))
        self.presence_log.append({"idx": kf.idx, "objects": {self.m.names[r[0]]: [round(float(x), 3) for x in probs[b]]
                                                             for b, r in enumerate(rows)} if rows else {}})
        return att, seen

    def _evidence_frame(self, kf, rgb, depth, valid, al) -> list:
        """OnlineCovis._evidence_frame with the presence head's term added to the frame's evidence."""
        import decide as D

        t0 = time.perf_counter()
        m = self.m
        busy = self._reloc_busy.is_set()
        pose = al.pose(kf.c2b)
        fit = self._frame_fit(kf, depth, valid, pose, al.scale)
        tf = time.perf_counter()
        if fit < self.cfg["min_frame_fit"]:
            self.frames_skipped.append((kf.idx, round(fit, 3)))
            return []
        f = self.belief.F
        if f >= self.thr_mat.shape[0]:
            self.thr_mat = torch.cat([self.thr_mat, torch.zeros_like(self.thr_mat)])
        counts = self.gm.process(rgb, depth, valid, pose, self.K, al.scale, self.thr_mat[f])
        tp = time.perf_counter()
        att, seen = self._presence(kf, depth, valid, pose, al.scale) if self.cfg["presence"] else (None, None)
        tpe = time.perf_counter()
        self.belief.add(counts, att, seen)
        self.frames_used.append(kf.idx)
        better = counts["n_valid"] > self.best_valid
        self.best_valid = np.where(better, counts["n_valid"], self.best_valid)
        self.best_frame = np.where(better, kf.idx, self.best_frame)
        for cb in self.callbacks:
            cb(kf, depth, valid, pose, al.scale)
        t1 = time.perf_counter()
        is_anchor = m.is_anchor.copy()
        for i in al.demoted:
            if 0 <= i < m.n_obj:
                is_anchor[i] = False
        if (is_anchor != self.belief.is_anchor).any():
            self.belief.set_anchors(is_anchor)
        self.sol = self.belief.solve()
        t2 = time.perf_counter()
        self.coverage = self._coverage(self.sol["rho"])
        t3 = time.perf_counter()
        labels, thr = D.chow_labels(self.sol["P"], self.coverage, self.cfg)
        self.thresholds = thr
        events = self._commit(labels, kf)
        self.labels = labels
        self.p_history.append(self.sol["P"].copy())
        t4 = time.perf_counter()
        self.split.append({"localizer_busy": busy, "fit_ms": 1000 * (tf - t0),
                           "evidence_ms": 1000 * ((tp - tf) + (t1 - tpe)), "presence_ms": 1000 * (tpe - tp),
                           "belief_ms": 1000 * (t2 - t1), "coverage_ms": 1000 * (t3 - t2), "commit_ms": 1000 * (t4 - t3)})
        return events


def _window_tokens(cens_world, exts, pts_world, embs, centre, rng, n_pts=256):
    toks = {"cen": [], "ext": [], "pts": [], "emb": []}
    for cen, ext, p, e in zip(cens_world, exts, pts_world, embs):
        q = p[rng.choice(len(p), n_pts, replace=len(p) < n_pts)] - cen
        toks["cen"].append(cen - centre); toks["ext"].append(ext); toks["pts"].append(q); toks["emb"].append(e)
    return {k: np.asarray(v, np.float32) for k, v in toks.items()}


class AttentionObservationManager(ObservationManager):
    matcher = None
    tau_match = 0.2

    def associate(self, labels: list, dcfg: dict, observed: list | None = None) -> dict:
        if not self.cfg.get("matcher", True) or self.matcher is None:
            return super().associate(labels, dcfg, observed)
        from scipy.spatial import cKDTree

        t0 = time.perf_counter()
        m = self.m
        if observed is None:
            with self._lock:
                observed = list(self.observed)
        self.observed_used = observed
        labels = list(labels)
        absent_rows = np.isin(m.owner, [n for n in range(m.n_obj) if labels[n] == "absent"])
        remaining_tree = cKDTree(m.pts[~absent_rows])
        rng = np.random.default_rng(0)
        mem_pts = [m.pts[m.owner == n] for n in range(m.n_obj)]
        mem_cen = np.array([p.mean(0) if len(p) else np.zeros(3) for p in mem_pts])
        rows, moved, pair_scores, redetected = [], {}, [], []
        if observed:
            t_cen = np.array([o.pts.mean(0) for o in observed])
            centre = t_cen.mean(0)
            near = [n for n in range(m.n_obj) if len(mem_pts[n]) >= 10 and
                    (np.linalg.norm(t_cen[:, :2] - mem_cen[n, :2], axis=1).min() < self.cfg["matcher_window_m"] or labels[n] == "absent")]
            near = sorted(near, key=lambda n: np.linalg.norm(mem_cen[n] - centre))[:self.cfg["matcher_max_memory"]]
            mem_emb = []
            for n in near:
                e = np.asarray(m.objects[n]["semantic_embedding"], np.float32)
                mem_emb.append(e / max(np.linalg.norm(e), 1e-8))
            ext = lambda p: np.maximum(np.percentile(p, 95, 0) - np.percentile(p, 5, 0), 0.01)
            Mt = _window_tokens(mem_cen[near], [ext(mem_pts[n]) for n in near], [mem_pts[n] for n in near], mem_emb, centre, rng)
            Tt = _window_tokens(t_cen, [ext(o.pts) for o in observed], [o.pts for o in observed], [o.emb for o in observed], centre, rng)
            dev = next(self.matcher.parameters()).device
            tt = lambda x: {k: torch.as_tensor(v, device=dev) for k, v in x.items()}
            from models import pair_features

            pair = pair_features([o.pts for o in observed], [mem_pts[n] for n in near], Tt["emb"], Mt["emb"])
            with torch.no_grad():
                P = torch.exp(self.matcher(tt(Tt), tt(Mt), torch.as_tensor(pair, device=dev))).cpu().numpy()
            Mn = len(near)
            claims = {}
            for u, o in enumerate(observed):
                d, _ = remaining_tree.query(o.pts, k=1, distance_upper_bound=dcfg["novelty_distance_m"])
                nov = float(np.isinf(d).mean())
                j = int(np.argmax(P[u, :Mn + 1]))
                best = int(np.argmax(P[u, :Mn])) if Mn else -1
                row = {"index": u, "object": o.uid, "voxels": int(len(o.pts)), "detections": int(o.n_det),
                       "centroid": np.round(t_cen[u], 3).tolist(), "novelty": round(nov, 3),
                       "best_memory": m.names[near[best]] if best >= 0 else None,
                       "p_best": round(float(P[u, best]), 3) if best >= 0 else None, "p_new": round(float(P[u, Mn]), 3),
                       "cosine": round(float(mem_emb[best] @ o.emb), 3) if best >= 0 else None}
                if j < Mn and P[u, j] >= self.tau_match:
                    n = near[j]
                    if labels[n] == "absent" and np.linalg.norm(t_cen[u] - mem_cen[n]) > self.cfg["matcher_move_m"]:
                        if n not in claims or P[u, j] > P[claims[n], near.index(n)]:
                            claims[n] = u
                        row["decision"] = "moved?"
                    else:
                        row["decision"] = "explains"
                    row["memory_id"] = m.names[n]
                    row["p_match"] = round(float(P[u, j]), 3)
                    row["cos_match"] = round(float(mem_emb[j] @ o.emb), 3)
                elif nov >= dcfg["min_novelty"]:
                    row["decision"] = "added"
                else:
                    row["decision"] = "in the memory map already"
                rows.append(row)
            for n, u in claims.items():
                moved[n] = u
                rows[u]["decision"] = "moved"
            for r in rows:
                if r["decision"] == "moved?":
                    r["decision"] = "explains"
            for r in rows:
                if r["decision"] == "explains":
                    n = m.names.index(r["memory_id"])
                    if labels[n] == "absent" and n not in moved:
                        confident = (r["p_match"] >= self.cfg["matcher_redetect_min_p"]
                                     and r["cos_match"] >= self.cfg["matcher_redetect_min_cos"])
                        if not confident:
                            # depth says that object is gone and this match is too weak to say otherwise: the track is
                            # something else standing there (the iron where the laptop was), treated as unmatched
                            r["weak_match"] = r.pop("memory_id")
                            r["decision"] = ("added" if r["novelty"] >= dcfg["min_novelty"]
                                             else "in the memory map already")
                            continue
                        redetected.append(r["memory_id"])
                        if dcfg["redetection_overrides_absent"]:
                            labels[n] = "present"
            for n in range(m.n_obj):
                if labels[n] == "absent" and n in near:
                    k = near.index(n)
                    pair_scores.append({"memory": m.names[n], "p_unmatched": round(float(P[Mn, k]), 3),
                                        "best_track": observed[int(np.argmax(P[:len(observed), k]))].uid,
                                        "p_best_track": round(float(P[:len(observed), k].max()), 3)})
        for r in rows:
            if r["decision"] == "added" and len(observed[r["index"]].grid_idx) < self.cfg["min_added_voxels"]:
                r["decision"] = f"ignored: only {len(observed[r['index']].grid_idx)} voxels"
            elif r["decision"] == "added" and observed[r["index"]].n_det < self.cfg["att_min_added_detections"]:
                r["decision"] = f"tentative: seen {observed[r['index']].n_det} times"
        for r in rows:
            if r["decision"] in ("added", "moved") or r["decision"].startswith(("tentative", "ignored")):
                r["support_share"] = round(self.novelty.support_share(observed[r["index"]].pts), 3)
                if r["decision"] == "added" and r["support_share"] < self.cfg["min_support_share"]:
                    r["decision"] = f"unsupported: {r['support_share']:.0%} on persistent new geometry"
        final = {}
        for n, lab in enumerate(labels):
            final[m.names[n]] = ("moved" if n in moved else "removed") if lab == "absent" else lab
        self.last_match_ms = round(1000 * (time.perf_counter() - t0), 2)
        return {"rows": rows, "moved": {m.names[n]: u for n, u in moved.items()}, "labels": final,
                "redetected_in_place": redetected, "pair_scores": pair_scores, "observed": observed,
                "matcher_ms": self.last_match_ms}
