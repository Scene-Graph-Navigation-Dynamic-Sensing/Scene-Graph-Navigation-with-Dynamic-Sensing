"""The memory's object pipeline (scripts/run_static_scene_graph.py, dataset
mode, streaming fusion) in a worker process started with the update, so its
models are loaded by the time the detector is needed.

Run as a subprocess (projection_method.run_memory_pipeline) the pipeline spends
~30 s loading YOLOE, the CLIP ViT-H embedder and NOVIC before it looks at a
frame; here that happens while relocalisation and the change test run. It is a
process, not a thread: loading on a thread of the update's own process slowed
relocalisation by ~10 s (the GIL), and a thread still loading at exit aborted
the interpreter. The
steps are the pipeline's own functions, called in the order its main() calls
them (lines 650-849); only the model loading is moved earlier. Anything the
pipeline does differently in other configurations (no streaming, per-detection
matching, panoptic detector) falls back to the subprocess. The live path (online/selective.py) always uses
WarmPipeline, which builds the detector from detection.backend like run_static_scene_graph.py.
"""
from __future__ import annotations

import copy
import json
import logging
import multiprocessing
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

logger = logging.getLogger("covis_gate")


def _warm_worker(sgg_root: str) -> bool:
    """Runs first in the voxel-map worker process, so its imports are done before the real job arrives."""
    from object_fusion._dovsg_import import ensure_dovsg_importable

    ensure_dovsg_importable(sgg_root)
    import object_fusion.view_dataset  # noqa: F401
    return True


def supported(cfg: dict) -> bool:
    fcfg, dcfg = cfg["object_fusion"], cfg["detection"]
    return (fcfg.get("match_level", "track") == "track" and fcfg.get("streaming", True)
            and dcfg.get("use_tracking", True) and dcfg.get("backend") != "panoptic")


def run_config(cfg: dict, viz_dir: Path) -> dict:
    """As projection_method.run_memory_pipeline(): own visualisation folder, extra
    embedders only on GPUs this process can see."""
    import torch

    run_cfg = copy.deepcopy(cfg)
    run_cfg["outputs"]["visualizations_dir"] = str(viz_dir)
    sem = run_cfg["semantic_decoding"]
    sem["extra_embedding_devices"] = [d for d in sem.get("extra_embedding_devices") or []
                                      if int(d.split(":")[1]) < torch.cuda.device_count()]
    return run_cfg


class WarmPipeline:
    """Lives in the worker process: models loaded once, then run() per dataset."""

    def __init__(self, cfg: dict):
        """cfg: run_config(). Starts the voxel-map process (its imports take ~15 s), then loads the models."""
        self.cfg = cfg
        self.error = None
        self.load_s = None
        self._ready = threading.Event()
        self.pool = ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn"))
        self.pool.submit(_warm_worker, cfg["external"]["scene_graph_generation_root"])
        self._load()

    def _load(self):
        t = time.time()
        try:
            import scripts.run_static_scene_graph as RS

            cfg = self.cfg
            det_cfg = cfg["detection"]
            if det_cfg.get("backend") == "panoptic":     # same choice as run_static_scene_graph.py
                from perception.detection.panoptic_detector import from_config as panoptic_from_config
                self.detector = panoptic_from_config(det_cfg)
            else:
                self.detector = RS.TrackedYOLOEDetector(
                    tracker=det_cfg.get("tracker", "botsort.yaml"), mask_box_filter=det_cfg.get("mask_box_filter"),
                    tighten_box_to_mask=det_cfg.get("tighten_box_to_mask", False), validation=det_cfg.get("validation"),
                    weights_path=det_cfg["weights"], conf_thresh=det_cfg["conf_thresh"], iou_thresh=det_cfg["iou_thresh"],
                    img_size=det_cfg["img_size"], device=det_cfg["device"],
                    fix_channel_order=det_cfg.get("fix_channel_order", True))
            sem = cfg["semantic_decoding"]
            kw = dict(novic_dir=sem["novic_dir"], checkpoint_path=sem["novic_checkpoint"], gencfg=sem["gencfg"],
                      guide_targets=sem["guide_targets"], embedding_batch_size=sem.get("embedding_batch_size", 8))
            self.novic = RS.NOVICDecoder(device=sem["device"], **kw)
            self.extra_embedders = [(lambda device=device: RS.NOVICDecoder(device=device, **kw))
                                    for device in sem.get("extra_embedding_devices") or []]
            self.InstanceFusion = RS.build_instance_fusion_class(cfg["external"]["scene_graph_generation_root"])
            self.everyday_names = RS.load_everyday_names(cfg)
        except BaseException as e:  # noqa: BLE001 -- re-raised by run()
            self.error = e
        finally:
            self.load_s = round(time.time() - t, 1)
            self._ready.set()

    def _fusion(self):
        fcfg = self.cfg["object_fusion"]
        fusion = self.InstanceFusion(downsample_voxel_size=fcfg["downsample_voxel_size"],
                                     spatial_weight=fcfg["spatial_weight"], vis_weight=fcfg["vis_weight"],
                                     sim_threshold=fcfg["sim_threshold"])
        fusion.mask_erode_px = int(fcfg.get("mask_erode_px", 0))
        fusion.min_merge_cosine = float(fcfg.get("min_merge_cosine", 0.0))
        fusion.rematch_min_containment = fcfg.get("rematch_min_containment")
        fusion.track_split_cosine = float(fcfg.get("track_split_cosine", 0.0))
        fusion.small_merge_ratio = float(fcfg.get("small_merge_ratio", 0.0))
        fusion.track_validation = fcfg.get("track_validation") or {}
        fusion.small_merge_cosine = float(fcfg.get("small_merge_cosine", 0.9))
        for key in ("mask_conf_threshold", "max_bbox_area_ratio", "mask_area_threshold", "min_points_threshold"):
            if key in fcfg:
                setattr(fusion, key, type(getattr(fusion, key))(fcfg[key]))
        return fusion

    def run(self, dataset_dir: Path, scene_name: str, scene_graphs_dir: Path) -> dict:
        """The pipeline on dataset_dir (rgb/point/mask/poses/metadata.json, poses
        already in the memory's z-up world). Saves <scene_name>.pkl/.json and
        returns timings."""
        import scripts.run_static_scene_graph as RS

        cfg = self.cfg
        fcfg = cfg["object_fusion"]
        viz_root = Path(cfg["outputs"]["visualizations_dir"])
        timer = RS.StageTimer()
        num_frames = json.loads((dataset_dir / "metadata.json").read_text())["length"]
        t0 = time.time()
        # stage 4 in its own process, as the pipeline does; it builds while the models finish loading
        from object_fusion._dovsg_import import ensure_dovsg_importable

        ensure_dovsg_importable(cfg["external"]["scene_graph_generation_root"])
        job = self.pool.submit(RS.load_view_dataset_for_transfer, workers=fcfg.get("voxel_map_workers", 1),
                          recorder_dir=str(dataset_dir),
                          scene_graph_generation_root=cfg["external"]["scene_graph_generation_root"],
                          resolution=fcfg["voxel_resolution"])
        box, lock = {}, threading.Lock()

        def voxel_map():
            with lock:
                if "vd" not in box:
                    box["vd"] = RS.restore_view_dataset(job.result())
            return box["vd"]

        waited = time.time()
        self._ready.wait()
        waited = round(time.time() - waited, 1)
        if self.error is not None:
            raise self.error
        fusion = self._fusion()

        def prepare():
            vd = voxel_map()
            fusion.start_new_scan(vd)
            fusion.start_stream(vd, decode_fn=self.novic.decode_embeddings,
                                track_end_frames=fcfg.get("track_end_frames", 10),
                                min_merge_overlap=fcfg.get("min_merge_overlap", 0.3))

        stream = RS.StreamingFusion(
            cfg, fusion, self.detector, self.novic, frame_names=[f"{i:06d}" for i in range(num_frames)],
            load_image=lambda i: np.asarray(Image.open(dataset_dir / "rgb" / f"{i:06d}.jpg"), dtype=np.uint8),
            frame_voxels=lambda i: (voxel_map().pixel_index_mappings[i], voxel_map().pixel_index_masks[i]),
            prepare=prepare, on_update=None, det_viz_dir=viz_root / "detection",
            obj_min_detections=fcfg["obj_min_detections"], extra_embedders=self.extra_embedders,
            live_interval_s=0.0)
        with timer.stage("4-6_voxel_map_detect_fuse_streaming"):
            stream.run()
        view_dataset = voxel_map()
        with timer.stage("6_instance_fusion_finalize"):
            objects, _ = fusion.finalize(obj_min_detections=fcfg["obj_min_detections"])
        objects, scene_graph, node_embeddings = RS.build_graph_from_objects(
            cfg, objects, view_dataset, self.novic, self.everyday_names, timer=timer)
        for obj in objects:
            obj.pop("mask", None)
        RS.save_scene_graph(scene_graphs_dir, scene_name, scene_graph, objects, node_embeddings,
                            view_dataset=view_dataset)
        timer.save(dataset_dir / "stage_timings.json")
        return {"models_loaded_in_background_s": self.load_s, "waited_for_models_s": waited,
                "pipeline_s": round(time.time() - t0, 1), "stages_s": {k: round(v, 1) for k, v in timer.stages.items()}}

    def close(self):
        self.pool.shutdown(wait=False, cancel_futures=True)
        if self._ready.is_set() and self.error is None:
            self.novic.close()


# ─── the worker process ───

_STATE = {}


def _init(cfg: dict, log_path: str):
    handler = logging.FileHandler(log_path)
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logging.basicConfig(level=logging.INFO, handlers=[handler])
    from object_fusion.streaming import prepare_native_libraries

    prepare_native_libraries()
    _STATE["pipeline"] = WarmPipeline(cfg)


def _ping() -> float:
    return _STATE["pipeline"].load_s


def _run(dataset_dir: str, scene_name: str, scene_graphs_dir: str) -> dict:
    return _STATE["pipeline"].run(Path(dataset_dir), scene_name, Path(scene_graphs_dir))


class WarmWorker:
    """The update's handle on the worker process. Created at the start of the
    update; run() blocks until the models are loaded, then runs the pipeline;
    close() stops the process (also when it is still loading)."""

    def __init__(self, cfg: dict, log_path: Path):
        import sys

        repo = str(Path(__file__).resolve().parents[2])
        if repo not in sys.path:
            sys.path.insert(0, repo)
        self.t0 = time.time()
        self.pool = ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn"),
                                        initializer=_init, initargs=(cfg, str(log_path)))
        self.loaded = self.pool.submit(_ping)  # starts the process; done when the models are loaded

    def run(self, dataset_dir: Path, scene_name: str, scene_graphs_dir: Path) -> dict:
        t = time.time()
        load_s = self.loaded.result()
        waited = round(time.time() - t, 1)
        out = self.pool.submit(_run, str(dataset_dir), scene_name, str(scene_graphs_dir)).result()
        out["models_loaded_in_background_s"] = load_s
        out["waited_for_models_s"] = waited
        return out

    def close(self):
        for proc in list(getattr(self.pool, "_processes", {}).values()):
            proc.terminate()
        self.pool.shutdown(wait=False, cancel_futures=True)
