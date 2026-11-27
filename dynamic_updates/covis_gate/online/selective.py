"""The memory pipeline's detector + embedder on short windows of frames, with far fewer embeddings.

warm_detect.WarmPipeline runs the pipeline's own streaming fusion (YOLOE + BoT-SORT -> CLIP ViT-H crop
embedding -> InstanceFusion -> naming -> scene graph). On lap_remove_2 it spent 25 of its 30 s embedding crops:
17.6 detections a frame, 83 ms a crop, a track's embedding re-computed on every frame it appears in.

Here the embedding step is a policy (PolicyStreamingFusion._embed_loop): a track is embedded the first time it is
seen, and again only when `every` frames have passed or its box has grown by `growth` (a better, closer view);
in between the track's last embedding stands in. Fusion averages a track's embeddings, so this changes each
track's mean embedding very little (earlier measurement: cosine 0.994 to the all-frames mean on lap_in_1 with
`every` = 5, worst track 0.956). Detection, tracking, fusion, naming and the scene graph are the pipeline's own.

The patched class is installed in the worker process only; nothing in object_fusion/ or scripts/ is edited.
"""
from __future__ import annotations

import logging
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

logger = logging.getLogger("covis_gate")

POLICY = {"every": 5, "growth": 1.5, "embed_batch": 32, "streams": True}


def _box_area(b) -> float:
    return float(max(b[2] - b[0], 0) * max(b[3] - b[1], 0))


def make_policy_class(base):
    class PolicyStreamingFusion(base):
        """StreamingFusion whose embedding thread embeds a track on first sight and then only when it is worth it."""

        embedded_crops = 0
        skipped_crops = 0

        def _embed_loop(self, make_embedder, label):
            from object_fusion.streaming import _END, weight_embeddings

            t0 = time.time()
            done = 0
            tracks: dict[int, dict] = {}
            try:
                embedder = make_embedder()
                while True:
                    item = self._to_embed.get()
                    if item is _END or self._errors:
                        break
                    idx, name, boxes, masks, confs, track_ids, crops, weights = item
                    n = len(crops)
                    need = []
                    for d in range(n):
                        tid = int(track_ids[d])
                        st = tracks.get(tid) if tid >= 0 else None
                        if st is None or idx - st["idx"] >= POLICY["every"] or _box_area(boxes[d]) >= POLICY["growth"] * st["area"]:
                            need.append(d)
                    fresh = embedder.embed_crops([crops[d] for d in need]) if need else np.empty((0, 0), np.float32)
                    emb = None
                    if n:
                        dim = fresh.shape[1] if len(fresh) else next(iter(tracks.values()))["emb"].shape[0]
                        emb = np.zeros((n, dim), np.float32)
                        for row, d in zip(fresh, need):
                            emb[d] = row
                            tid = int(track_ids[d])
                            if tid >= 0:
                                tracks[tid] = {"idx": idx, "area": _box_area(boxes[d]), "emb": row}
                        for d in range(n):
                            if d not in need:
                                emb[d] = tracks[int(track_ids[d])]["emb"]
                    else:
                        emb = np.empty((0, 0), np.float32)
                    PolicyStreamingFusion.embedded_crops += len(need)
                    PolicyStreamingFusion.skipped_crops += n - len(need)
                    emb = weight_embeddings(emb, weights)
                    with self._embedded_cv:
                        self._embedded[idx] = (idx, name, boxes, masks, confs, track_ids, emb)
                        self._embedded_cv.notify_all()
                    done += 1
            except BaseException as e:  # noqa: BLE001
                self._errors.append(e)
                with self._embedded_cv:
                    self._embedded_cv.notify_all()
            finally:
                self.frames_per_embedder[label] = done
                self.timings[f"embedding_{label}_s"] = round(time.time() - t0, 1)

    return PolicyStreamingFusion


def fast_view_dataset_for_transfer(workers: int = 1, recorder_dir: str = "", scene_graph_generation_root: str = "",
                                   resolution: float = 0.01, **_):
    """object_fusion.view_dataset.load_view_dataset_for_transfer() for a window of a few frames.

    The original spawns `voxel_map_workers` fresh processes for every call (each imports DoVSG) and removes
    statistical outliers per frame (~1 s a frame): 8 s of waiting for a 6-frame window. Here one process, in order,
    and the depth edge-free mask (compare_video_poses.edge_free_mask, what the change test uses) instead of the
    statistical filter. The voxel map, the pixel -> voxel maps and the colour averaging are the original's."""
    from pathlib import Path

    import numpy as np
    from PIL import Image

    from object_fusion._dovsg_import import ensure_dovsg_importable
    from object_fusion.view_dataset import _frame_voxels
    from scripts.compare_video_poses import edge_free_mask

    ensure_dovsg_importable(scene_graph_generation_root)
    from dovsg.memory.view_dataset import Bounds, ViewDataset

    vd = ViewDataset.__new__(ViewDataset)
    vd.recorder_dir = Path(recorder_dir)
    vd.interval, vd.use_inlier_mask, vd.resolution = 1, True, resolution
    vd.nb_neighbors, vd.std_ratio = 30, 1.5
    vd.metadata = vd.read_metadata()
    frames = list(range(vd.length))
    d = vd.recorder_dir
    masks, lows, highs = [], [], []
    for cnt in frames:
        point = np.load(d / f"point/{cnt:06}.npy", allow_pickle=True).astype(np.float32)
        mask = np.load(d / f"mask/{cnt:06}.npy", allow_pickle=True).astype(np.bool_)
        pose = np.loadtxt(d / f"poses/{cnt:06}.txt").astype(np.float32)
        mask = edge_free_mask(point, mask)
        world = (point @ pose[:3, :3].T + pose[:3, 3])[mask]
        masks.append(mask)
        if len(world):
            lows.append(world.min(0)); highs.append(world.max(0))
    min_b = np.minimum.reduce([np.array([np.inf] * 3), *lows])
    max_b = np.maximum.reduce([np.array([-np.inf] * 3), *highs])
    vd.bounds = Bounds.from_arr(np.round(np.stack([min_b, max_b], axis=1), 2))
    vd.voxel_num = ((vd.bounds.higher_bound - vd.bounds.lower_bound) / resolution).astype(np.int32)
    second = [_frame_voxels(recorder_dir, cnt, masks[cnt], vd.bounds.lower_bound, vd.bounds.higher_bound, resolution,
                            vd.voxel_num) for cnt in frames]
    vd.images, vd.global_points = [], []
    vd.masks = []
    vd.names = [f"{cnt:06}" for cnt in frames]
    vd.append_length_log = [len(frames)]
    vd.pixel_index_mappings = [m for m, _, _, _ in second]
    vd.pixel_index_masks = [k for _, k, _, _ in second]
    all_colours = np.vstack([c for _, _, c, _ in second])
    all_indexes = np.hstack([u for _, _, _, u in second])
    unique, inverse, counts = np.unique(all_indexes, return_inverse=True, return_counts=True)
    summed = np.zeros((unique.size, all_colours.shape[1]))
    np.add.at(summed, inverse, all_colours)
    colours = dict(zip(unique, summed / counts[:, np.newaxis]))
    vd._packed_colours = (np.fromiter(colours.keys(), dtype=np.int64, count=len(colours)),
                          np.asarray(list(colours.values()), dtype=np.float32))
    vd.indexes_colors_mapping_dict = {}
    return vd


def _init_selective(cfg: dict, log_path: str, policy: dict):
    """Worker process start-up: the pipeline's own _init, then the policy class replaces StreamingFusion and the
    voxel map is built by fast_view_dataset_for_transfer()."""
    import warm_detect

    fast = policy.pop("fast_voxel_map", True)
    POLICY.update(policy)
    warm_detect._init(cfg, log_path)
    import scripts.run_static_scene_graph as RS
    from object_fusion.streaming import StreamingFusion

    RS.StreamingFusion = make_policy_class(StreamingFusion)
    if fast:
        RS.load_view_dataset_for_transfer = fast_view_dataset_for_transfer
    pipe = warm_detect._STATE["pipeline"]
    if pipe.error is None:
        # CLIP in batches of 32 instead of 8 (+ empty_cache after each): the same embeddings (max difference 6e-8),
        # 2.1x faster per crop on this GPU.
        pipe.novic.embedding_batch_size = int(POLICY["embed_batch"])
        if POLICY["streams"]:
            # The detection and embedding threads share the default CUDA stream, so YOLOE's small kernels waited behind
            # CLIP's batches: 0.5-0.7 s a frame in a window instead of 83 ms. Each on its own stream, they overlap.
            import torch

            pipe.detector.track = _on_stream(pipe.detector.track, torch.cuda.Stream())
            pipe.novic.embed_crops = _on_stream(pipe.novic.embed_crops, torch.cuda.Stream())


def _on_stream(fn, stream):
    import functools

    import torch

    @functools.wraps(fn)
    def run(*a, **k):
        with torch.cuda.stream(stream):
            out = fn(*a, **k)
            stream.synchronize()
        return out
    return run


def _stats() -> dict:
    import scripts.run_static_scene_graph as RS

    cls = RS.StreamingFusion
    return {"embedded_crops": getattr(cls, "embedded_crops", None), "skipped_crops": getattr(cls, "skipped_crops", None)}


class SelectiveWorker:
    """warm_detect.WarmWorker with the embedding policy. run() blocks until the models are loaded, then runs the
    pipeline on one dataset directory; submit() returns a future instead."""

    def __init__(self, cfg: dict, log_path: Path, policy: dict | None = None):
        import sys

        import warm_detect

        repo = str(Path(__file__).resolve().parents[3])
        if repo not in sys.path:
            sys.path.insert(0, repo)
        self._warm = warm_detect
        self.t0 = time.time()
        self.pool = ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn"),
                                        initializer=_init_selective,
                                        initargs=(cfg, str(log_path), {**POLICY, **(policy or {})}))
        self.loaded = self.pool.submit(self._warm._ping)

    def ready(self) -> bool:
        return self.loaded.done()

    def submit(self, dataset_dir: Path, scene_name: str, scene_graphs_dir: Path):
        return self.pool.submit(self._warm._run, str(dataset_dir), scene_name, str(scene_graphs_dir))

    def stats(self) -> dict:
        return self.pool.submit(_stats).result()

    def run(self, dataset_dir: Path, scene_name: str, scene_graphs_dir: Path) -> dict:
        t = time.time()
        load_s = self.loaded.result()
        waited = round(time.time() - t, 1)
        out = self.submit(dataset_dir, scene_name, scene_graphs_dir).result()
        out["models_loaded_in_background_s"] = load_s
        out["waited_for_models_s"] = waited
        return out

    def close(self):
        """Stop the worker and everything it started (its voxel-map process): terminating only the worker would leave
        that child running, parentless, on the shared server."""
        for proc in list(getattr(self.pool, "_processes", {}).values()):
            kill_tree(proc.pid)
        self.pool.shutdown(wait=False, cancel_futures=True)


class WorkerPool:
    """n SelectiveWorkers. A window job goes to a free one, so a window that comes due while another is being detected
    does not wait for it (each worker is a process: its Python does not share a GIL with the others)."""

    def __init__(self, n: int, cfg: dict, log_dir: Path, policy: dict | None = None):
        self.workers = [SelectiveWorker(cfg, Path(log_dir) / f"worker{i}.log", policy) for i in range(max(1, n))]
        self.running: dict[int, object] = {}

    def wait_loaded(self):
        for w in self.workers:
            w.loaded.result()

    def ready(self) -> bool:
        return any(w.ready() for w in self.workers)

    def _free(self):
        for i, w in enumerate(self.workers):
            f = self.running.get(i)
            if w.ready() and (f is None or f.done()):
                return i
        return None

    def free(self) -> bool:
        return self._free() is not None

    def submit(self, dataset_dir: Path, scene_name: str, scene_graphs_dir: Path):
        i = self._free()
        if i is None:
            raise RuntimeError("no free detection worker")
        self.running[i] = self.workers[i].submit(dataset_dir, scene_name, scene_graphs_dir)
        return self.running[i]

    def stats(self) -> dict:
        out = {}
        for w in self.workers:
            for k, v in w.stats().items():
                out[k] = (out.get(k) or 0) + (v or 0)
        return out

    def close(self):
        for w in self.workers:
            w.close()


def kill_tree(pid: int, grace_s: float = 2.0) -> None:
    """SIGTERM a process and all its descendants, SIGKILL whatever is still there after grace_s."""
    import os
    import signal

    try:
        import psutil

        root = psutil.Process(pid)
        procs = root.children(recursive=True) + [root]
    except Exception:  # noqa: BLE001 -- no psutil or already gone
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        return
    for p in procs:
        try:
            p.terminate()
        except psutil.Error:
            pass
    _, alive = psutil.wait_procs(procs, timeout=grace_s)
    for p in alive:
        try:
            p.kill()
        except psutil.Error:
            pass
