"""The co-visibility gate as a streaming engine: push(keyframe) in, beliefs and decisions out.

What runs where, per keyframe (lap_in_1 -> lap_remove_2 numbers, one RTX 5000):

  main thread    upload + edge-free mask (GPU) ......................... ~1 ms
                 frame fit against the dilated occupancy volume ........ <1 ms
                 evidence: project 338k voxels, hit/through/occluded .... ~5 ms   (batch: 64 ms)
                 belief + gate sweeps over the keyframes so far ......... ~4 ms
                 Chow labels, coverage condition, commit rule ........... ~1 ms
  localizer      one SuperPoint + 3 LightGlue pairs per 4th keyframe, T* refitted from the last
  thread         `reloc_window` keyframes' anchor matches ............... ~90 ms, off the main loop

The evidence of a keyframe is computed once, with the alignment current at that moment, and kept as
(n_valid, n_hit, n_thr) per object: images are never revisited. Before the first alignment exists
the keyframes wait (they are on the GPU) and are evaluated, in order, as soon as it does.
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

import decide as D
import evidence as E
import theory as T
from dynamism import Dynamism
from relocalize import BACKGROUND

import gpu_evidence as G
import reloc_online as RO
from belief import OnlineBelief

logger = logging.getLogger("covis_gate")

ONLINE_DEFAULTS = {
    "reloc_stride": 4,              # every 4th keyframe goes to the localizer
    "reloc_window": 24,             # keyframes whose anchor matches T* is fitted from (None-like: 0 = all, as the batch method)
    "min_localized_keyframes": 4,   # keyframes the localizer must have seen before its T* is used
    "commit_frames": 3,
    "max_drain_per_push": 3,        # waiting keyframes evaluated per push once the alignment exists (all of them at once
                                    # made one keyframe take ~1 s); the same evidence, spread over the next few keyframes             # consecutive evidence frames with the same label before a change is committed
    "store_frames": 400,            # raw keyframes kept (detection windows are cut from them)
    "bank_cache": True,             # cache the memory's SuperPoint bank next to its frames
    "frame_fit_radius_m": 0.03,
    "frame_fit_points": 3000,
}


@dataclass
class Keyframe:
    idx: int                        # position in the video
    name: str
    rgb: np.ndarray                 # (H, W, 3) uint8
    depth: np.ndarray               # (H, W) float32, the new video's own units
    mask: np.ndarray                # (H, W) bool, valid depth
    c2b: np.ndarray                 # (4, 4) camera -> the new video's world (its own SLAM)
    t_arrival: float = field(default_factory=time.perf_counter)


@dataclass
class Alignment:
    version: int
    T: np.ndarray
    scale: float
    R: np.ndarray
    t: np.ndarray
    demoted: tuple
    inliers: int
    ratio: float
    keyframes: int

    def pose(self, c2b: np.ndarray) -> np.ndarray:
        p = np.eye(4)
        p[:3, :3] = self.R @ c2b[:3, :3]
        p[:3, 3] = self.scale * self.R @ c2b[:3, 3] + self.t
        return p


class MemoryModel:
    """Everything about the memory that does not change during a session."""

    def __init__(self, repo_root: Path, cfg: dict, ccfg: dict, memory: str, device: str, data_name: str | None = None):
        import run_projection_update as PR
        from navigation.voxel_map import PersistedVoxelMap
        from scene_graph.storage.serializer import load_scene_graph, scene_graph_to_next_to, scene_graph_to_relations
        from scipy.spatial import cKDTree

        self.cfg, self.ccfg, self.name = cfg, ccfg, memory
        sg_dir = repo_root / cfg["outputs"]["scene_graphs_dir"]
        self.sg_dir = sg_dir
        self.dir = repo_root / cfg["outputs"]["data_processed_dir"] / (data_name or memory)   # the memory's frames
        t = time.time()
        self.scene = load_scene_graph(sg_dir / f"{memory}.pkl")
        self.map = PersistedVoxelMap.from_dict(self.scene["voxel_map"])
        self.objects = self.scene["instance_objects"]
        self.names = [o["class_id"] for o in self.objects]
        self.n_obj = len(self.objects)
        res = self.map.resolution
        self.idx = np.fromiter(self.map.indexes_colors_mapping_dict.keys(), dtype=np.int64)
        self.pts = self.map.index_to_point(self.idx) + res / 2
        self.cols = np.array([self.map.indexes_colors_mapping_dict[i] for i in self.idx.tolist()], dtype=np.float64)
        row_of = dict(zip(self.idx.tolist(), range(len(self.idx))))
        self.owner = np.full(len(self.idx), BACKGROUND, np.int64)
        for n, o in enumerate(self.objects):
            self.owner[[row_of[i] for i in o["indexes"] if i in row_of]] = n
        self.relations = scene_graph_to_relations(self.scene["scene_graph"])
        self.next_to = scene_graph_to_next_to(self.scene["scene_graph"])
        self.root_id = self.scene["scene_graph"].root.node_id
        self.parent_of = {r["child"]: r["parent"] for r in self.relations}
        self.n_receptacles = len({r["parent"] for r in self.relations if r["relation"] in ("on", "inside")} | {self.root_id})
        self.centroids = {o["class_id"]: self.pts[self.owner == n].mean(0) for n, o in enumerate(self.objects)}
        self.tree = cKDTree(self.pts)
        self.frames = PR.MemoryFrames(self.dir)

        # §3 prior
        dyn_path = sg_dir / f"{memory}_dynamism.json"
        self.dyn, self.dyn_existed = Dynamism.load_or_init(dyn_path, self.objects, 0.0, ccfg, self.parent_of)
        self.t_mem = self.dyn.state["visit_time_days"]
        self.t_now = self.t_mem + ccfg["delta_t_days"]
        self.S = (np.array([self.dyn.survival(c, self.t_now) for c in self.names]) if ccfg["prior"] == "on"
                  else np.full(self.n_obj, 0.5))
        self.prior_L = np.array([T.logit(s) for s in self.S])
        anchor_ids = set(self.dyn.anchors(self.names, ccfg["anchor_rate"]))
        self.is_anchor = np.array([c in anchor_ids for c in self.names])
        self.anchor_codes = ({n for n in range(self.n_obj) if self.is_anchor[n]}
                             | ({BACKGROUND} if ccfg["background_anchor"] else set()))

        # which memory voxels can be judged (the memory's own frames confirm them); cached by projection_method
        self.reliable = np.ones(len(self.idx), bool)
        if ccfg["reliability_mask"]:
            pcfg = {**PR.DEFAULT_CFG, **(cfg.get("projection_update") or {})}
            self.reliable = PR.memory_stable_voxels(self.dir, self.pts, self.cols, pcfg,
                                                    cfg["external"]["scene_graph_generation_root"])
        self.owner_eff = np.where(self.reliable, self.owner, -3)
        self.nbrs = E.neighbours(self.objects, self.relations, self.next_to, ccfg)
        self.bg_groups = E.background_groups(self.pts, self.owner, self.reliable, list(range(self.n_obj)), ccfg)
        self.judged = np.bincount(self.owner[(self.owner >= 0) & self.reliable], minlength=self.n_obj)
        self.load_s = round(time.time() - t, 1)


class OnlineCovis:
    def __init__(self, model: MemoryModel, ccfg: dict, K: np.ndarray, device: str = "cuda"):
        self.m, self.cfg, self.K = model, ccfg, K
        self.dev = torch.device(device)
        m = model
        t = time.time()
        self.gm = G.GpuMemory(m.pts, m.cols, m.owner_eff, m.n_obj, m.bg_groups, ccfg, self.dev)
        lo = np.asarray(m.map.bounds.lower_bound, np.float64)
        self.occ = G.Occupancy(m.pts, lo, m.map.resolution, np.asarray(m.map.voxel_num), ccfg["frame_fit_radius_m"], self.dev)
        self.belief = OnlineBelief(m.prior_L, m.is_anchor, m.nbrs, list(m.bg_groups), ccfg)
        self.thr_mat = torch.zeros((512, self.gm.n_slots), dtype=torch.bool, device=self.dev)
        self.rng = torch.Generator(device=self.dev); self.rng.manual_seed(0)

        # localizer (own thread)
        from dovsg_core import RGBFeatureMatch
        matcher = RGBFeatureMatch(str(Path(ccfg["dovsg_root"]) / "third_party" / "LightGlue"), ccfg["lightglue_max_keypoints"])
        cache = (m.dir / "covis_reloc_bank.pt") if ccfg["bank_cache"] else None
        self.bank = RO.FeatureBank(matcher, m.frames, m.tree, m.owner, ccfg, cache)
        window = ccfg["reloc_window"] or None
        self.loc = RO.OnlineLocalizer(self.bank, m.frames.poses, m.anchor_codes, K, ccfg, window=window)
        self._reloc_q: queue.Queue = queue.Queue(maxsize=8)
        self._align: Alignment | None = None
        self._align_lock = threading.Lock()
        self._stop = threading.Event()
        self._reloc_busy = threading.Event()
        self._reloc_thread = threading.Thread(target=self._reloc_loop, name="localizer", daemon=True)
        self._reloc_thread.start()
        self.setup_s = round(time.time() - t, 1)

        # state
        self.n_pushed = 0
        self.pending: deque = deque()
        self.store: deque = deque(maxlen=ccfg["store_frames"])
        self.frames_used: list[int] = []          # keyframe idx of every evidence frame in the belief
        self.frames_skipped: list[tuple] = []
        self.labels = ["deferred"] * m.n_obj
        self.sol = self.belief.solve()
        self.coverage = np.zeros(m.n_obj)
        self.streak = {"absent": np.zeros(m.n_obj, int), "present": np.zeros(m.n_obj, int)}
        self.committed: dict[str, dict] = {}      # object -> {"label", "frame", "wall_s", "P", "coverage"}
        self.events: list[dict] = []
        self.best_frame = np.full(m.n_obj, -1)    # evidence frame (keyframe idx) projecting most of each object
        self.best_valid = np.zeros(m.n_obj, int)
        self.p_history: list[np.ndarray] = []
        self.t_start = time.perf_counter()
        self.callbacks: list = []                 # fn(kf, depth, valid, pose, scale) after every evidence frame that is used
        self.timings: list[dict] = []
        self.split: list[dict] = []
        self.reloc_log: list[dict] = []
        self.lock = threading.Lock()

    # ── localizer thread ──
    def _reloc_loop(self):
        stream = torch.cuda.Stream(device=self.dev)       # its own CUDA stream: its kernels do not queue behind the main loop's
        while not self._stop.is_set():
            try:
                item = self._reloc_q.get(timeout=0.2)
            except queue.Empty:
                continue
            key, rgb, depth, valid, c2b = item
            t = time.perf_counter()
            self._reloc_busy.set()
            try:
                with torch.cuda.stream(stream):
                    r = self.loc.add_keyframe(key, rgb, depth, valid, c2b)
                    stream.synchronize()
            except Exception:  # noqa: BLE001 -- a failed keyframe must not stop the session
                logger.exception("[localizer] keyframe failed")
                continue
            self._reloc_busy.clear()
            dt = time.perf_counter() - t
            if r.get("refit") and self.loc.n_keyframes >= self.cfg["min_localized_keyframes"]:
                ok = r["inliers"] >= self.cfg["min_inliers"] and r["inlier_ratio"] >= self.cfg["min_inlier_ratio"]
                if ok:
                    s, R, tr = RO._decompose(r["T"])
                    with self._align_lock:
                        v = (self._align.version + 1) if self._align else 1
                        self._align = Alignment(v, r["T"], s, R, tr, tuple(self.loc.demoted), r["inliers"],
                                                r["inlier_ratio"], self.loc.n_keyframes)
            self.reloc_log.append({"key": key, "s": round(dt, 3), "inliers": r.get("inliers"),
                                   "ratio": None if r.get("inlier_ratio") is None else round(r["inlier_ratio"], 3)})

    def alignment(self) -> Alignment | None:
        with self._align_lock:
            return self._align

    # ── per keyframe ──
    def push(self, kf: Keyframe) -> dict:
        t0 = time.perf_counter()
        rec = {"idx": kf.idx, "lag_s": t0 - kf.t_arrival}
        n = self.n_pushed
        self.n_pushed += 1
        self.store.append(kf)
        rgb = torch.from_numpy(kf.rgb).to(self.dev, non_blocking=True)
        depth = torch.from_numpy(kf.depth).to(self.dev, non_blocking=True)
        mask = torch.from_numpy(kf.mask).to(self.dev, non_blocking=True)
        valid = G.edge_free_mask_gpu(depth, mask)
        if n % self.cfg["reloc_stride"] == 0:
            item = (kf.idx, kf.rgb, kf.depth, valid.cpu().numpy(), kf.c2b)
            try:
                self._reloc_q.put_nowait(item)
            except queue.Full:                # the localizer is behind: drop the oldest, keep the newest
                try:
                    self._reloc_q.get_nowait()
                except queue.Empty:
                    pass
                self._reloc_q.put_nowait(item)
        if n % self.cfg["evidence_frame_stride"] == 0:
            self.pending.append((kf, rgb, depth, valid))
        new_events = self._drain(self.cfg["max_drain_per_push"])
        rec.update(events=new_events, wall_s=time.perf_counter() - t0, pending=len(self.pending),
                   aligned=self._align is not None)
        self.timings.append(rec)
        return rec

    def _drain(self, limit: int | None = None) -> list:
        out = []
        al = self.alignment()
        n = 0
        while self.pending and al is not None and (limit is None or n < limit):
            n += 1
            kf, rgb, depth, valid = self.pending.popleft()
            ev = self._evidence_frame(kf, rgb, depth, valid, al)
            out += ev
        return out

    def _frame_fit(self, kf, depth, valid, pose, scale) -> float:
        n = self.cfg["frame_fit_points"]
        ys, xs = valid.nonzero(as_tuple=True)
        if len(ys) == 0:
            return 0.0
        sel = torch.randperm(len(ys), generator=self.rng, device=self.dev)[:n]
        ys, xs = ys[sel], xs[sel]
        z = depth[ys, xs].double() * scale
        K = self.K
        cam = torch.stack([(xs.double() - K[0, 2]) / K[0, 0] * z, (ys.double() - K[1, 2]) / K[1, 1] * z, z], 1)
        R = torch.as_tensor(pose[:3, :3], dtype=torch.float64, device=self.dev)
        t = torch.as_tensor(pose[:3, 3], dtype=torch.float64, device=self.dev)
        return float(self.occ.near(cam @ R.T + t).float().mean())

    def _evidence_frame(self, kf, rgb, depth, valid, al: Alignment) -> list:
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
        self.belief.add(counts)
        self.frames_used.append(kf.idx)
        better = counts["n_valid"] > self.best_valid
        self.best_valid = np.where(better, counts["n_valid"], self.best_valid)
        self.best_frame = np.where(better, kf.idx, self.best_frame)
        for cb in self.callbacks:
            cb(kf, depth, valid, pose, al.scale)
        t1 = time.perf_counter()
        is_anchor = m.is_anchor.copy()                      # moved anchors (phi_j < tau) become targets
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
        self.split.append({"localizer_busy": busy, "fit_ms": 1000 * (tf - t0), "evidence_ms": 1000 * (t1 - tf), "belief_ms": 1000 * (t2 - t1),
                           "coverage_ms": 1000 * (t3 - t2), "commit_ms": 1000 * (t4 - t3)})
        return events

    def _coverage(self, rho: np.ndarray) -> np.ndarray:
        F = rho.shape[0]
        slot_owner = self.gm.slot_owner
        rho_t = torch.as_tensor(rho, dtype=torch.float32, device=self.dev)
        best = (self.thr_mat[:F].float() * rho_t[:, slot_owner]).max(0).values
        covered = torch.zeros(self.m.n_obj, device=self.dev).index_add_(0, slot_owner, best)
        judged = torch.bincount(slot_owner, minlength=self.m.n_obj).float().clamp_min(1)
        return (covered / judged).cpu().numpy()

    def _commit(self, labels: list, kf: Keyframe) -> list:
        """A label becomes a decision after commit_frames evidence frames in a row."""
        ev = []
        m = self.m
        for i in range(m.n_obj):
            for lab in ("absent", "present"):
                self.streak[lab][i] = self.streak[lab][i] + 1 if labels[i] == lab else 0
            cur = self.committed.get(m.names[i], {}).get("label", "deferred")
            new = None
            if self.streak["absent"][i] >= self.cfg["commit_frames"]:
                new = "absent"
            elif self.streak["present"][i] >= self.cfg["commit_frames"]:
                new = "present"
            if new is not None and new != cur:
                rec = {"object": m.names[i], "label": new, "frame": kf.idx, "wall_s": round(time.perf_counter() - self.t_start, 2),
                       "P": round(float(self.sol["P"][i]), 4), "coverage": round(float(self.coverage[i]), 3),
                       "previous": cur, "evidence_frames": self.belief.F}
                self.committed[m.names[i]] = rec
                self.events.append(rec)
                ev.append(rec)
        return ev

    def finish(self, timeout: float = 60.0) -> list:
        """End of the stream: let the localizer catch up, then evaluate the keyframes still waiting for an alignment."""
        t = time.perf_counter()
        while (not self._reloc_q.empty() or self._reloc_busy.is_set()) and time.perf_counter() - t < timeout:
            time.sleep(0.01)
        return self._drain()

    def close(self):
        self._stop.set()
        self._reloc_thread.join(timeout=5)
