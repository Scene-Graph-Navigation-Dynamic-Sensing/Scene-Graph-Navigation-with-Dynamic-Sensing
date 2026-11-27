"""Looking at what the evidence cannot explain: when, where, and what came back.

The detector serves one purpose in this method (§8.2): telling moved / removed / added apart. The manager here
starts it only where that is needed, while the video plays:

  - an object is committed absent        -> a window of consecutive keyframes around the keyframe that saw most of it
                                            (is something at its place now, or did it move elsewhere?)
  - new geometry persists in the video   -> a window around the keyframe that saw most of the region
                                            (an added object, or a moved one at its new place)

A window (detect_window consecutive keyframes) is written to a small dataset in tmpfs, with the alignment current at
that moment, and handed to selective.SelectiveWorker: the memory pipeline's own detector, tracker, fusion, naming and
graph, with the embedding policy and the quick voxel map. The jobs run in the worker process; the keyframe loop never
waits for one. A finished job's objects are merged with earlier jobs' (the same object seen in two windows is one)
and associated with the absent objects (decide.associate).
"""
from __future__ import annotations

import json
import logging
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import decide as D
import novelty as N

logger = logging.getLogger("covis_gate")

OBSERVE_DEFAULTS = {
    "observe": True,
    "novel_check_every": 4,            # evidence frames between region checks
    "novel_warmup_keyframes": 8,       # the localizer must have seen this many keyframes before new geometry is counted
    "novel_trigger_frames": 6,         # a region must have been seen by this many keyframes before the detector looks
    "novel_trigger_voxels": 120,       # ... and be at least this big (2 cm cells; the batch method keeps regions of 60)
    "min_added_voxels": 200,           # an unexplained object smaller than this (1 cm voxels) is noise, not an addition
    "novel_min_voxels": 60,
    "novel_min_frames": 3,
    "novel_distance_m": 0.05,
    "novel_voxel_m": 0.02,
    "novel_points_per_frame": 4000,
    "novel_frame_stride": 4,           # keyframes per novelty update (batch: every 4th)
    "observe_workers": 1,              # detection worker processes (each ~4 GB of GPU memory). 2 helps only with GPU to spare:
                                       # on a GPU another job kept at 100 % (2026-10-23) two workers made a window 2x slower
    "detect_window": 6,
    "detect_max_windows": 4,
    "region_same_m": 0.10,             # a region within this of an earlier trigger is the same region
    "embed_every": 5,
    "embed_growth": 1.5,
    "obs_merge_overlap": 0.3,
    "obs_same_place_m": 0.08,
    "min_support_share": 0.0,          # an addition needs this share of its voxels on persistent new geometry (novelty.support_share)
    "min_added_detections": 8,         # an addition must have been detected this often (the memory pipeline keeps objects seen 3 times)
    "job_dir": "/dev/shm/covis_live",
}


@dataclass
class Observed:
    uid: str
    job: int
    raw: dict
    pts: np.ndarray
    emb: np.ndarray
    n_det: int
    parent_obs: str | None
    grid_idx: list
    grid_cols: dict


@dataclass
class Job:
    id: int
    reason: str
    frames: list
    t_submit: float
    future: object
    t_done: float | None = None
    result: dict | None = None
    observed: list = field(default_factory=list)
    loaded: bool = False


def _unit(v) -> np.ndarray:
    v = v.detach().cpu().numpy() if hasattr(v, "detach") else np.asarray(v)
    v = v.astype(np.float32).ravel()
    return v / max(float(np.linalg.norm(v)), 1e-8)


class ObservationManager:
    def __init__(self, model, engine, worker, ocfg: dict, K: np.ndarray, metadata: dict, source_dir: Path | None = None):
        self.m, self.engine, self.worker, self.cfg, self.K = model, engine, worker, ocfg, K
        self.meta, self.source_dir = metadata, source_dir
        self.root = Path(ocfg["job_dir"])
        for old in list(self.root.glob("job*")) + [self.root / "sg"]:   # an earlier session's windows (the worker logs stay)
            shutil.rmtree(old, ignore_errors=True)
        (self.root / "sg").mkdir(parents=True)
        self.jobs: list[Job] = []
        self.observed: list[Observed] = []
        self.triggers: list[dict] = []        # scheduled, waiting for their frames
        self.ready: list[dict] = []           # frames there, waiting for the worker
        self.done_regions: list[np.ndarray] = []
        self.done_centres: list[int] = []
        self.windows_used = 0
        self.log: list[dict] = []
        self._n_ev = 0
        self._pool = ThreadPoolExecutor(1, thread_name_prefix="observe")   # loading results, region extraction
        self._lock = threading.Lock()
        self._region_future = None
        self._loads: list = []
        lo = np.asarray(model.map.bounds.lower_bound, np.float64)
        hi = np.asarray(model.map.bounds.higher_bound, np.float64)
        self.novelty = N.NovelGeometry(model.pts, lo, hi, model.map.resolution, np.asarray(model.map.voxel_num),
                                       ocfg, engine.dev)
        engine.callbacks.append(self._on_evidence)
        h, w = metadata["h"], metadata["w"]
        v, u = np.mgrid[0:h, 0:w]
        self._rays = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1]], -1).astype(np.float32)

    # ── evidence-frame hook: new geometry ──
    def _on_evidence(self, kf, depth, valid, pose, scale):
        if self.engine.loc.n_keyframes < self.cfg["novel_warmup_keyframes"]:
            return                                   # the alignment is still settling: its errors would look like new geometry
        self._n_ev += 1
        stride = max(1, self.cfg["novel_frame_stride"] // max(1, self.engine.cfg["evidence_frame_stride"]))
        if self._n_ev % stride == 0:
            self.novelty.update(kf.idx, depth, valid, pose, scale, self.K)
        if self._n_ev % (self.cfg["novel_check_every"] * stride) == 0 and self._region_future is None:
            snap = self.novelty.snapshot()           # taken here, clustered on the helper thread
            self._region_future = (kf.idx, self._pool.submit(self.novelty.regions_from, *snap))

    def _check_regions(self, now_idx: int, regions: list):
        for r in regions:
            if r["frames_seeing"] < self.cfg["novel_trigger_frames"] or r["voxels"] < self.cfg["novel_trigger_voxels"]:
                continue
            c = np.asarray(r["centroid"])
            if any(np.linalg.norm(c - d) < self.cfg["region_same_m"] for d in self.done_regions):
                continue
            self.done_regions.append(c)
            self._schedule(r["best_frame"], f"new geometry {r['voxels']} voxels at {r['centroid']}", now_idx)

    def on_events(self, events: list, now_idx: int):
        for e in events:
            if e["label"] == "absent":
                i = self.m.names.index(e["object"])
                centre = int(self.engine.best_frame[i]) if self.engine.best_frame[i] >= 0 else now_idx
                self._schedule(centre, f"absent {e['object']}", now_idx)

    def _schedule(self, centre: int, why: str, now_idx: int):
        if self.windows_used + len(self.triggers) + len(self.ready) >= self.cfg["detect_max_windows"]:
            return
        if any(abs(centre - c) < self.cfg["detect_window"] for c in self.done_centres):
            return
        self.done_centres.append(centre)
        self.triggers.append({"centre": centre, "why": why, "scheduled_at": now_idx})
        logger.info(f"[observe] window around keyframe {centre} scheduled: {why}")

    # ── per keyframe ──
    def step(self, now_idx: int, flush: bool = False):
        if self._region_future is not None and self._region_future[1].done():
            idx, fut = self._region_future
            self._region_future = None
            try:
                self._check_regions(now_idx, fut.result())
            except Exception:  # noqa: BLE001
                logger.exception("[observe] region extraction failed")
        half = self.cfg["detect_window"] // 2
        ready, wait = [], []
        for t in self.triggers:
            (ready if flush or now_idx >= t["centre"] + half else wait).append(t)
        self.triggers = wait
        self.ready += ready
        new = self.poll()
        if self.ready and self.worker.free():          # a worker is free
            batch, self.ready = self.ready, []
            self._submit(batch)
        return new

    def _window(self, centre: int) -> list:
        by_idx = {kf.idx: kf for kf in self.engine.store}
        n = self.cfg["detect_window"]
        hi = max(by_idx)
        a = max(min(by_idx), min(centre - n // 2, hi - n + 1))
        return [by_idx[i] for i in range(a, a + n) if i in by_idx]

    def _submit(self, ts: list):
        """One job for the windows of every trigger in ts: while a job runs, the windows that come due wait and are
        detected together, in one fusion (as the batch method fuses all its windows), so commits, naming and the
        graph are paid once."""
        al = self.engine.alignment()
        if al is None or not self.worker.free():
            self.ready = ts + self.ready
            return
        frames = sorted({f.idx: f for t in ts for f in self._window(t["centre"])}.values(), key=lambda f: f.idx)
        t = {"why": "; ".join(x["why"] for x in ts)}
        jid = len(self.jobs)
        d = self.root / f"job{jid}"
        t0 = time.perf_counter()
        self._write_window(d, frames, al)
        job = Job(jid, t["why"], [f.idx for f in frames], time.perf_counter(),
                  self.worker.submit(d, f"obs{jid}", self.root / "sg"))
        self.jobs.append(job)
        self.windows_used += len(ts)
        self.log.append({"event": "submit", "job": jid, "why": t["why"], "windows": len(ts), "frames": job.frames,
                         "prepare_s": round(time.perf_counter() - t0, 3), "t": round(time.perf_counter() - self.engine.t_start, 2)})
        logger.info(f"[observe] job {jid} submitted ({t['why']}): keyframes {job.frames[0]}..{job.frames[-1]}")

    def _write_window(self, d: Path, frames: list, al):
        from PIL import Image

        shutil.rmtree(d, ignore_errors=True)
        for sub in ("rgb", "mask", "point", "poses"):
            (d / sub).mkdir(parents=True)
        for i, kf in enumerate(frames):
            new = f"{i:06d}"
            src = self.source_dir / "rgb" / f"{kf.name}.jpg" if self.source_dir else None
            if src is not None and src.exists():
                (d / "rgb" / f"{new}.jpg").symlink_to(src.resolve())
            else:
                Image.fromarray(kf.rgb).save(d / "rgb" / f"{new}.jpg", quality=95)
            np.save(d / "mask" / f"{new}.npy", kf.mask)
            z = (kf.depth * al.scale).astype(np.float32)
            pts = np.concatenate([self._rays * z[..., None], z[..., None]], -1)
            np.save(d / "point" / f"{new}.npy", pts)
            np.savetxt(d / "poses" / f"{new}.txt", al.pose(kf.c2b))
        meta = dict(self.meta)
        meta["length"] = len(frames)
        (d / "metadata.json").write_text(json.dumps(meta))

    # ── results ──
    def poll(self) -> list:
        """Jobs whose results have been loaded and merged since the last call (loading runs on the helper thread)."""
        for job in self.jobs:
            if job.result is None and job.future.done() and not any(j is job for j, _ in self._loads):
                job.t_done = time.perf_counter()
                try:
                    job.result = job.future.result()
                except Exception as e:  # noqa: BLE001 -- a failed window must not stop the session
                    logger.exception(f"[observe] job {job.id} failed")
                    job.result = {"error": str(e)}
                    job.loaded = True
                    continue
                self._loads.append((job, self._pool.submit(self._ingest, job)))
        new = []
        for job, fut in list(self._loads):
            if fut.done():
                self._loads.remove((job, fut))
                fut.result()
                new.append(job)
        return new

    def settled(self) -> bool:
        return not self.triggers and not self.ready and not self._loads and all(getattr(j, "loaded", False) for j in self.jobs)

    def _ingest(self, job: Job):
        job.observed = self._load(job)
        with self._lock:
            self._merge(job.observed)
        job.loaded = True
        self.log.append({"event": "done", "job": job.id, "s": round(job.t_done - job.t_submit, 2),
                         "objects": [o.uid for o in job.observed], "stages_s": job.result.get("stages_s"),
                         "t": round(time.perf_counter() - self.engine.t_start, 2)})
        logger.info(f"[observe] job {job.id} done in {job.t_done - job.t_submit:.1f}s: "
                    f"{[o.raw['class_id'] for o in job.observed]}")

    def _load(self, job: Job) -> list:
        from navigation.voxel_map import PersistedVoxelMap
        from scene_graph.storage.serializer import load_scene_graph, scene_graph_to_relations
        import targeted

        sg = load_scene_graph(self.root / "sg" / f"obs{job.id}.pkl")
        omap = PersistedVoxelMap.from_dict(sg["voxel_map"])
        root_obs = sg["scene_graph"].root.node_id
        parent = {r["child"]: r["parent"] for r in scene_graph_to_relations(sg["scene_graph"])
                  if r["relation"] in ("on", "inside")}
        out = []
        for o in sg["instance_objects"]:
            cid = o["class_id"]
            p = parent.get(cid)
            idx, cols = targeted.object_in_memory_grid(self.m.map, omap, o)
            out.append(Observed(
                uid=f"j{job.id}:{cid}", job=job.id, raw=o,
                pts=omap.index_to_point(np.asarray(o["indexes"])) + omap.resolution / 2, emb=_unit(o.get("semantic_embedding", o.get("clip_ft"))),
                n_det=int(o["num_detections"]),
                parent_obs=(self.m.root_id if p == root_obs else (f"j{job.id}:{p}" if p else None)),
                grid_idx=idx, grid_cols=cols))
        return out

    def _same(self, a: Observed, b: Observed) -> bool:
        """The memory pipeline's identity rule for tracks (0.3 overlap + 0.7 cosine >= 0.55, with real overlap), and: same
        name and nearly the same place. Two windows see different sides of an object, so their voxels overlap little."""
        from scipy.spatial import cKDTree

        cos = float(a.emb @ b.emb)
        ov_ab = float(np.isfinite(cKDTree(a.pts).query(b.pts, distance_upper_bound=0.02)[0]).mean())
        ov_ba = float(np.isfinite(cKDTree(b.pts).query(a.pts, distance_upper_bound=0.02)[0]).mean())
        geo = max(ov_ab, ov_ba)
        close = float(np.linalg.norm(a.pts.mean(0) - b.pts.mean(0))) < self.cfg["obs_same_place_m"]
        same_name = a.raw.get("class_name") == b.raw.get("class_name")
        return (geo >= self.cfg["obs_merge_overlap"] and 0.3 * geo + 0.7 * cos >= 0.55) or (same_name and close and cos >= 0.6)

    @staticmethod
    def _combine(a: Observed, b: Observed) -> Observed:
        """One object from two sightings: detections add up, embeddings average by detections, voxels join."""
        keep, other = (a, b) if a.n_det >= b.n_det else (b, a)
        emb = keep.emb * keep.n_det + other.emb * other.n_det
        emb = (emb / max(float(np.linalg.norm(emb)), 1e-8)).astype(np.float32)
        raw = dict(keep.raw)
        raw["semantic_embedding"], raw["num_detections"] = emb, keep.n_det + other.n_det
        return Observed(uid=keep.uid, job=keep.job, raw=raw, pts=np.concatenate([keep.pts, other.pts]), emb=emb,
                        n_det=keep.n_det + other.n_det, parent_obs=keep.parent_obs,
                        grid_idx=sorted(set(keep.grid_idx) | set(other.grid_idx)), grid_cols={**other.grid_cols, **keep.grid_cols})

    def _merge(self, fresh: list):
        """The same physical object in two windows is one observed object."""
        for o in fresh:
            twin = next((k for k in self.observed if self._same(o, k)), None)
            if twin is None:
                self.observed.append(o)
                continue
            merged = self._combine(o, twin)
            dropped = o.uid if merged.uid == twin.uid else twin.uid
            self.observed[self.observed.index(twin)] = merged
            for k in self.observed:                     # what stood on / in the dropped one now stands on the kept one
                if k.parent_obs == dropped:
                    k.parent_obs = merged.uid
            self.log.append({"event": "merged", "kept": merged.uid, "dropped": dropped, "detections": merged.n_det})

    # ── §8.2 ──
    def associate(self, labels: list, dcfg: dict, observed: list | None = None) -> dict:
        from scipy.spatial import cKDTree

        m = self.m
        if observed is None:
            with self._lock:
                observed = list(self.observed)
        self.observed_used = observed
        absent_rows = np.isin(m.owner, [n for n in range(m.n_obj) if labels[n] == "absent"])
        remaining_tree = cKDTree(m.pts[~absent_rows])
        mem_obj_pts = [m.pts[m.owner == n] for n in range(m.n_obj)]
        objs = []
        for o in observed:
            d = dict(o.raw)
            d["class_id"] = o.uid
            d["num_detections"] = o.n_det
            objs.append(d)
        assoc = D.associate(m.objects, mem_obj_pts, list(labels), objs, [o.pts for o in observed],
                            [o.parent_obs for o in observed], m.root_id,
                            [m.dyn.rec(c)["receptacles"] for c in m.names], m.n_receptacles, remaining_tree, dcfg)
        for r in assoc["rows"]:
            if r["decision"] == "added" and len(observed[r["index"]].grid_idx) < self.cfg["min_added_voxels"]:
                r["decision"] = f"ignored: only {len(observed[r['index']].grid_idx)} voxels"
            elif r["decision"] == "added" and observed[r["index"]].n_det < self.cfg["min_added_detections"]:
                r["decision"] = f"tentative: seen {observed[r['index']].n_det} times"
        for r in assoc["rows"]:
            if r["decision"] in ("added", "moved") or r["decision"].startswith(("tentative", "ignored")):
                r["support_share"] = round(self.novelty.support_share(observed[r["index"]].pts), 3)
                if r["decision"] == "added" and r["support_share"] < self.cfg["min_support_share"]:
                    r["decision"] = f"unsupported: {r['support_share']:.0%} on persistent new geometry"
        assoc["observed"] = observed
        return assoc
