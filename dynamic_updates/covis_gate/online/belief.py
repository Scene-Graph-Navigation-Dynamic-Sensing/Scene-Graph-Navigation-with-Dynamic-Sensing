"""§4 + §7 as a running belief.

evidence.beliefs() takes the counts of a whole video. This keeps the counts of
the keyframes seen so far and solves the same equations for them after every
keyframe, so the belief is available at any moment and, after the last
keyframe, equals evidence.beliefs() on that video (test_online.py checks it).

Why recompute instead of adding the newest frame's evidence to a running log-odds:
  - the frame discount beta_i = 1 / (1 + (m_i - 1) rho_f) depends on m_i, the number
    of frames that saw object i, so every earlier frame's weight changes when a frame arrives;
  - the gate rho_i^f of an old frame depends on the neighbours' beliefs, which have moved since;
  - L is clipped after every frame, which is path dependent.
Recomputing is F steps of a (n_objects,)-vector recurrence: ~2 ms per sweep at
F = 200, 3 sweeps; the counts per frame are tiny, the images are never revisited.
"""
from __future__ import annotations

import numpy as np

import theory as T


class OnlineBelief:
    def __init__(self, prior_L: np.ndarray, is_anchor: np.ndarray, nbrs: dict, bg_targets: list, ccfg: dict,
                 capacity: int = 256):
        self.cfg, self.prior_L, self.nbrs = ccfg, np.asarray(prior_L, np.float64), nbrs
        self.is_anchor = np.asarray(is_anchor, bool).copy()
        self.bg_targets = list(bg_targets)  # group g <-> object bg_targets[g]
        self.n = len(prior_L)
        self.F = 0
        self._cap = capacity
        self._alloc(capacity)

    def _alloc(self, cap: int):
        n, g = self.n, len(self.bg_targets)
        old = None if self.F == 0 else (self.n_valid[:self.F], self.n_hit[:self.F], self.n_thr[:self.F],
                                        self.bg_valid[:self.F], self.bg_hit[:self.F])
        self.n_valid, self.n_hit, self.n_thr = (np.zeros((cap, n), np.int64) for _ in range(3))
        self.bg_valid, self.bg_hit = (np.zeros((cap, g), np.int64) for _ in range(2))
        if old is not None:
            for dst, src in zip((self.n_valid, self.n_hit, self.n_thr, self.bg_valid, self.bg_hit), old):
                dst[:self.F] = src
        self._cap = cap

    def add(self, counts: dict) -> int:
        """Append one keyframe's counts (gpu_evidence.GpuMemory.process output). Returns its index."""
        if self.F == self._cap:
            self._alloc(self._cap * 2)
        f = self.F
        self.n_valid[f], self.n_hit[f], self.n_thr[f] = counts["n_valid"], counts["n_hit"], counts["n_thr"]
        self.bg_valid[f], self.bg_hit[f] = counts["bg_valid"], counts["bg_hit"]
        self.F += 1
        return f

    def replace(self, f: int, counts: dict) -> None:
        """Overwrite keyframe f (its pose or the alignment changed and it was evaluated again)."""
        self.n_valid[f], self.n_hit[f], self.n_thr[f] = counts["n_valid"], counts["n_hit"], counts["n_thr"]
        self.bg_valid[f], self.bg_hit[f] = counts["bg_valid"], counts["bg_hit"]

    def set_anchors(self, is_anchor: np.ndarray) -> None:
        self.is_anchor = np.asarray(is_anchor, bool).copy()

    def solve(self) -> dict:
        """evidence.beliefs() over the keyframes added so far."""
        ecfg, F, n = self.cfg, self.F, self.n
        n_valid, n_hit, n_thr = self.n_valid[:F], self.n_hit[:F], self.n_thr[:F]
        ell = T.voxel_llr(ecfg["p_hit_present"], ecfg["p_hit_absent"])
        if F == 0:
            P0 = T.sigmoid(self.prior_L)
            return {"L": self.prior_L.copy(), "P": P0, "rho": np.zeros((0, n)), "trajectory": np.zeros((0, n)),
                    "sweeps": 0, "frames_informative": np.zeros(n, int), "ell": ell}
        informative = n_hit + n_thr
        alpha = (np.full(informative.shape, ecfg["alpha_fixed"]) if ecfg["alpha_fixed"]
                 else T.kish_alpha(informative, ecfg["rho_c"]))
        m = (informative > 0).sum(0)
        beta = T.kish_alpha(m, ecfg["rho_f"])
        pos = beta * alpha * n_hit * ell
        neg = -beta * alpha * n_thr * ell
        h = np.where(n_valid > 0, n_hit / np.maximum(n_valid, 1), 0.0)
        h_bg = {i: np.where(self.bg_valid[:F, g] > 0, self.bg_hit[:F, g] / np.maximum(self.bg_valid[:F, g], 1), 0.0)
                for g, i in enumerate(self.bg_targets)}
        lmax, mode = ecfg["l_max"], ecfg["update"]

        def recurrence(rho: np.ndarray, cols: np.ndarray):
            """L_f = clip(L_{f-1} + l+ + gated l-) for the objects `cols`, all at once."""
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
                "evidence_pos": pos.sum(0), "evidence_neg_raw": neg.sum(0)}
