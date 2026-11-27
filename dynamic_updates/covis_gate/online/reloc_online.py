"""§5 relocalisation as the video arrives.

relocalize.relocalize() needs the whole new video: it matches a spread of frames
against a spread of memory frames, fits T*, then matches every 4th frame against
the memory frames nearest its first pose and fits again (15.7 s on lap_remove_2:
420 LightGlue pairs, of which 80 were SuperPoint extractions of the same memory
frames again and again).

Here:
  FeatureBank       the memory side, built once per memory and cached on disk: SuperPoint
                    features of every memory frame, and each keypoint already lifted to the
                    memory's world with the owner (object / background) of the voxel it
                    lands on. A match then costs one LightGlue call and array indexing; no
                    depth maps are read and no KD-tree is queried at run time.
  OnlineLocalizer   one SuperPoint extraction per keyframe it is given, 3 LightGlue pairs
                    (memory frames nearest the current pose estimate), the anchor matches
                    added to what it has, T* refitted from all of them. Until a first T*
                    exists it matches the first keyframe against a spread of memory frames
                    (the batch method's retrieval, for one frame instead of eight).

It does the same estimation as the batch version (anchor-only matches, RANSAC + Umeyama
with the eq. 12 iteration count, phi_j and demotion of moved anchors); only the order of
the work differs: matches arrive over time and T* is refitted as they do.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import torch

import theory as T
from relocalize import BACKGROUND, NO_OWNER, ransac_umeyama

logger = logging.getLogger("covis_gate")


class FeatureBank:
    """SuperPoint features of the memory's frames with their keypoints in the memory's world."""

    def __init__(self, matcher, memory, mem_tree, voxel_owner: np.ndarray, rcfg: dict, cache: Path | None):
        self.matcher, self.memory, self.dev = matcher, memory, matcher.device
        n = len(memory.names)
        tag = f"{n}:{len(voxel_owner)}:{rcfg['lightglue_max_keypoints']}:{rcfg['owner_distance_m']}"
        data = None
        if cache is not None and cache.exists():
            data = torch.load(cache, map_location="cpu")
            if data.get("tag") != tag:
                data = None
        if data is None:
            t = time.time()
            data = self._build(mem_tree, voxel_owner, rcfg, tag)
            logger.info(f"[reloc bank] built for {n} memory frames in {time.time() - t:.1f}s")
            if cache is not None:
                torch.save(data, cache)
        self.kp = [k.to(self.dev) for k in data["kp"]]
        self.desc = [d.to(self.dev) for d in data["desc"]]
        self.scores = [s.to(self.dev) for s in data["scores"]]
        self.size = data["size"].to(self.dev)
        self.world = data["world"].numpy()    # (n, N, 3) float64, NaN where the depth is not trustworthy
        self.owner = data["owner"].numpy()    # (n, N) int16

    def _build(self, mem_tree, voxel_owner, rcfg, tag):
        from run_projection_update import lift

        n = len(self.memory.names)
        N = rcfg["lightglue_max_keypoints"]
        kps, descs, scores = [], [], []
        world = np.full((n, N, 3), np.nan)
        owner = np.full((n, N), NO_OWNER, np.int16)
        size = None
        for j in range(n):
            img = self.memory.rgb(j)
            f = self.matcher.extract_feature(img)
            kp = f["keypoints"][0].cpu().numpy()
            kps.append(f["keypoints"][0].cpu())
            descs.append(f["descriptors"][0].half().cpu())
            scores.append(f["keypoint_scores"][0].half().cpu())
            size = f["image_size"][0].cpu()
            pm, okm = lift(kp, *self.memory.geometry(j))
            pose = self.memory.poses[j]
            w = pm @ pose[:3, :3].T + pose[:3, 3]
            d, nn = mem_tree.query(w, k=1, distance_upper_bound=rcfg["owner_distance_m"])
            m = len(kp)
            world[j, :m][okm] = w[okm]
            o = np.full(m, NO_OWNER, np.int16)
            hit = np.isfinite(d)
            o[hit] = voxel_owner[nn[hit]]
            owner[j, :m] = o
            self.memory._geo.pop(j, None)  # do not keep every depth map in RAM
        return {"tag": tag, "kp": kps, "desc": descs, "scores": scores, "size": size,
                "world": torch.as_tensor(world), "owner": torch.as_tensor(owner)}

    def features(self, j: int) -> dict:
        return {"keypoints": self.kp[j][None], "keypoint_scores": self.scores[j][None].float(),
                "descriptors": self.desc[j][None].float(), "image_size": self.size[None]}


class OnlineLocalizer:
    def __init__(self, bank: FeatureBank, memory_poses: np.ndarray, anchor_codes: set, K: np.ndarray, rcfg: dict,
                 window: int | None = None):
        """window: fit T* from the matches of the last `window` keyframes only (None: from all of them, which is
        what the batch method does). A long video's DROID-SLAM trajectory drifts, so one rigid T* for all of it
        is a compromise; a window follows the drift."""
        self.bank, self.poses, self.anchor_codes, self.K, self.cfg = bank, memory_poses, anchor_codes, K, rcfg
        self.window = window
        self.with_scale = not rcfg["rigid"]
        n_mem = len(memory_poses)
        self.search_set = sorted(set(np.linspace(0, n_mem - 1, min(rcfg["retrieval_memory_frames"], n_mem))
                                     .round().astype(int).tolist()))
        self.S, self.D, self.O, self.F = [], [], [], []   # accumulated matches (all owners)
        self.T = None                                      # 4x4 Sim3, new video's world -> memory world
        self.history = []                                  # (n_keyframes, T) after every refit
        self.n_keyframes = 0
        self.timing = {"extract": 0.0, "match": 0.0, "fit": 0.0}
        self.demoted: list = []
        self.phi: dict = {}

    # ── one keyframe ──
    def _lift_new(self, kp: np.ndarray, depth: np.ndarray, valid: np.ndarray, c2b: np.ndarray):
        h, w = valid.shape
        u = np.clip(np.round(kp[:, 0]).astype(int), 0, w - 1)
        v = np.clip(np.round(kp[:, 1]).astype(int), 0, h - 1)
        z = depth[v, u].astype(np.float64)
        K = self.K
        cam = np.stack([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z], 1)
        return cam @ c2b[:3, :3].T + c2b[:3, 3], valid[v, u]

    def _pair(self, feats, kp_new_all, j: int):
        """LightGlue between the new keyframe and memory frame j -> (idx in new, idx in memory)."""
        t = time.time()
        _, _, m = self.bank.matcher.match_features(feats, self.bank.features(j))
        pairs = m["matches"].cpu().numpy()
        self.timing["match"] += time.time() - t
        return pairs[:, 0], pairs[:, 1]

    def _collect(self, feats, kp_new, pairs_idx, j, src_all, ok_new):
        i0, i1 = pairs_idx
        dst = self.bank.world[j][i1]
        ok = ok_new[i0] & np.isfinite(dst[:, 0])
        return src_all[i0][ok], dst[ok], self.bank.owner[j][i1][ok].astype(np.int64)

    def add_keyframe(self, key: int, rgb_u8: np.ndarray, depth: np.ndarray, valid: np.ndarray, c2b: np.ndarray) -> dict:
        """rgb (H, W, 3) uint8, depth (H, W) in the new video's units, valid the edge-free mask, c2b the video's own
        pose. Returns {"refit": bool, "T": ...}."""
        from run_projection_update import nearest_memory_frames

        cfg = self.cfg
        t = time.time()
        feats = self.bank.matcher.extract_feature(rgb_u8)
        self.timing["extract"] += time.time() - t
        kp = feats["keypoints"][0].cpu().numpy()
        src_all, ok_new = self._lift_new(kp, depth, valid, c2b)
        self.n_keyframes += 1
        S, D, O = [], [], []
        if self.T is None:                                  # SEARCH: this keyframe against a spread of memory frames
            counts = []
            for j in self.search_set:
                counts.append(len(self._pair(feats, kp, j)[0]))
            order = np.argsort(-np.array(counts))[:cfg["retrieval_top_k"]]
            cand = [self.search_set[b] for b in order if counts[b] >= cfg["min_pair_matches"]]
        else:                                               # TRACK: the memory frames nearest the current estimate
            s, R, tr = _decompose(self.T)
            pose = np.eye(4)
            pose[:3, :3] = R @ c2b[:3, :3]
            pose[:3, 3] = s * R @ c2b[:3, 3] + tr
            cand = [int(j) for j in nearest_memory_frames(pose, self.poses, cfg["refine_candidates"],
                                                          cfg["candidate_angle_weight_m"])]
        for j in cand:
            i0, i1 = self._pair(feats, kp, j)
            if len(i0) < cfg["min_pair_matches"]:
                continue
            s_, d_, o_ = self._collect(feats, kp, (i0, i1), j, src_all, ok_new)
            S.append(s_); D.append(d_); O.append(o_)
        if S:
            self.S.append(np.concatenate(S)); self.D.append(np.concatenate(D)); self.O.append(np.concatenate(O))
            self.F.append(np.full(len(self.S[-1]), key))
        return self.refit()

    # ── fit ──
    def _anchored(self):
        lo = 0 if self.window is None else max(0, len(self.S) - self.window)
        S, D, O = np.concatenate(self.S[lo:]), np.concatenate(self.D[lo:]), np.concatenate(self.O[lo:])
        keep = np.isin(O, list(self.anchor_codes))
        return S[keep], D[keep], O[keep], (S, D, O)

    def refit(self) -> dict:
        from scripts.compare_video_poses import apply

        cfg = self.cfg
        if not self.S:
            return {"refit": False, "T": self.T}
        t = time.time()
        S, D, O, everything = self._anchored()
        if len(S) < 30:
            return {"refit": False, "T": self.T}
        thresh = cfg["ransac_thresh_m"]
        Tm = None
        if self.T is not None:                              # warm start: polish the current fit on its inliers
            inl = np.linalg.norm(apply(self.T, S) - D, axis=1) < thresh
            if inl.mean() > 0.5 and inl.sum() >= 30:
                Tm = self.T
                from scripts.compare_video_poses import umeyama
                for _ in range(3):
                    inl = np.linalg.norm(apply(Tm, S) - D, axis=1) < thresh
                    Tm = umeyama(S[inl], D[inl], with_scale=self.with_scale)
        if Tm is None:
            Tm, inl, _ = ransac_umeyama(S, D, thresh, self.with_scale, seed=len(self.history))
            if Tm is None:
                return {"refit": False, "T": self.T}
        inl = np.linalg.norm(apply(Tm, S) - D, axis=1) < thresh

        def phis(Tm, S, D, O):
            err = np.linalg.norm(apply(Tm, S) - D, axis=1)
            out = {}
            for code in np.unique(O):
                m = O == code
                out[int(code)] = {"matches": int(m.sum()), "phi": float((err[m] < thresh).mean()),
                                  "median_error_m": float(np.median(err[m]))}
            return out

        phi = phis(Tm, S, D, O)
        demoted = [c for c, v in phi.items() if c != BACKGROUND and v["matches"] >= cfg["phi_min_matches"]
                   and v["phi"] < cfg["tau_phi"]]
        if demoted:                                          # moved anchors: refit without them
            keep = ~np.isin(O, demoted)
            from scripts.compare_video_poses import umeyama
            Tk = Tm
            for _ in range(3):
                ik = np.linalg.norm(apply(Tk, S[keep]) - D[keep], axis=1) < thresh
                Tk = umeyama(S[keep][ik], D[keep][ik], with_scale=self.with_scale)
            Tm = Tk
            inl = np.linalg.norm(apply(Tm, S) - D, axis=1) < thresh
            phi = phis(Tm, S, D, O)
        self.T, self.demoted, self.phi = Tm, demoted, phi
        self.history.append((self.n_keyframes, Tm.copy()))
        self.timing["fit"] += time.time() - t
        return {"refit": True, "T": Tm, "inliers": int(inl.sum()), "anchor_matches": int(len(S)),
                "inlier_ratio": float(inl.mean())}

    def converged(self, translation_m: float = 0.005, rotation_deg: float = 0.2, scale: float = 0.002,
                  min_keyframes: int = 6) -> bool:
        """The last two fits agree: the video's alignment has stopped moving."""
        if len(self.history) < 2 or self.n_keyframes < min_keyframes:
            return False
        a, b = self.history[-2][1], self.history[-1][1]
        sa, Ra, ta = _decompose(a)
        sb, Rb, tb = _decompose(b)
        ang = np.degrees(np.arccos(np.clip((np.trace(Ra.T @ Rb) - 1) / 2, -1, 1)))
        return bool(abs(sb / sa - 1) < scale and ang < rotation_deg and np.linalg.norm(tb - ta) < translation_m)


def _decompose(T4: np.ndarray):
    s = float(np.cbrt(np.linalg.det(T4[:3, :3])))
    return s, T4[:3, :3] / s, T4[:3, 3]
