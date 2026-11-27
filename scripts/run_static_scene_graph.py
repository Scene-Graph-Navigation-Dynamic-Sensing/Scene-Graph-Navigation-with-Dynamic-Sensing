#!/usr/bin/env python3
"""Static Open-Vocabulary Scene Graph Pipeline — single entry point.

Video feed (DroidCam stream / video file / dataset folder) -> Depth Anything V3
metric (+ optional alignment) -> DROID-SLAM pose -> voxel map -> YOLOE-seg-pf
(class-agnostic detect+segment) + BoT-SORT -> NOVIC crop embedding every frame
-> multi-view instance fusion with a running-average embedding per object ->
NOVIC name decoded from that average as each track ends -> spatial relations
-> static scene graph. Every stage writes its own visualization
under outputs/visualizations/<stage>/. See docs/full_pipeline.md.

Usage:
    python scripts/run_static_scene_graph.py --config configs/pipeline.yaml
    python scripts/run_static_scene_graph.py --source data/raw/my_dataset
"""
from __future__ import annotations

import argparse
import faulthandler
import json
import logging
import multiprocessing
import shutil
import signal
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager, nullcontext
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from perception.frame_source import CameraIntrinsics, FrameSource
from perception.depth_estimation.depth_anything_estimator import DepthAnythingEstimator
from perception.depth_estimation.visualize import save_depth_visualization
from perception.pose_estimation.droid_slam_pose_estimator import DroidSlamPoseEstimator
from perception.pose_estimation.visualize import save_trajectory_comparison, save_trajectory_visualization
from perception.detection.yoloe_seg_detector import YOLOESegDetector
from perception.detection.tracked_yoloe_detector import TrackedYOLOEDetector
from object_fusion.postprocess import (complete_support_surfaces, drop_structural_instances,
                                       find_support_surfaces, merge_fragmented_instances)
from perception.visualization import save_detection_debug_frame
from semantic_decoding.novic.novic_decoder import NOVICDecoder
from object_fusion.streaming import (StreamingFusion, detect_and_embed, prepare_native_libraries,
                                     quiet_loggers, tree_lines,
                                     write_live_graph)
from object_fusion.view_dataset import (load_view_dataset, load_view_dataset_for_transfer, restore_view_dataset,
                                        write_metadata)
from object_fusion.instance_fusion import build_instance_fusion_class
from object_fusion.instance_tagger import tag_instances
from object_fusion.visualize import save_instances_screenshot
from scene_graph.builder.graph_builder import build_static_scene_graph_safe
from scene_graph.builder.gravity_alignment import gravity_align_dataset
from scene_graph.storage.serializer import save_scene_graph
from scene_graph.storage.visualizer import save_scene_graph_visualization
from scripts.eval_pose import (
    align_trajectories,
    evaluate as evaluate_poses,
    format_report as format_pose_report,
    write_per_frame_csv,
)

logger = logging.getLogger("run_static_scene_graph")


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


class StageTimer:
    """Wall-clock time per pipeline stage, logged as each one ends and written
    to stage_timings.json so slow stages are visible without re-reading logs."""

    def __init__(self):
        self.stages: dict[str, float] = {}
        self._t0 = time.time()

    @contextmanager
    def stage(self, name: str):
        start = time.time()
        logger.info(f"[stage] {name}: started")
        try:
            yield
        finally:
            elapsed = time.time() - start
            self.stages[name] = self.stages.get(name, 0.0) + elapsed
            logger.info(f"[stage] {name}: {elapsed:.1f}s")

    @property
    def total(self) -> float:
        return time.time() - self._t0

    def report(self) -> str:
        total = self.total
        width = max((len(n) for n in self.stages), default=10)
        lines = [f"{'stage'.ljust(width)}  {'seconds':>9}  {'share':>6}"]
        for name, secs in sorted(self.stages.items(), key=lambda kv: -kv[1]):
            lines.append(f"{name.ljust(width)}  {secs:9.1f}  {secs / total:5.1%}")
        lines.append(f"{'TOTAL'.ljust(width)}  {total:9.1f}")
        return "\n".join(lines)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(
            {"stages_seconds": {k: round(v, 2) for k, v in self.stages.items()},
             "total_seconds": round(self.total, 2)}, indent=2))


def resolve_intrinsics(cfg: dict, source: FrameSource, native_w: int, native_h: int) -> CameraIntrinsics:
    """Dataset camera.yaml wins; otherwise the config's camera block, whose
    fx/fy/cx/cy are for camera.width x camera.height."""
    if source.intrinsics is not None:
        logger.info("Intrinsics from dataset camera.yaml")
        return source.intrinsics
    cam = cfg["camera"]
    logger.warning("Intrinsics from configs/pipeline.yaml camera block — metric depth scales with "
                   "focal length, so calibrate your camera rather than trusting the defaults.")
    return CameraIntrinsics(fx=cam["fx"], fy=cam["fy"], cx=cam["cx"], cy=cam["cy"],
                            width=cam.get("width", native_w), height=cam.get("height", native_h))


def reference_depth(frame, K: CameraIntrinsics) -> np.ndarray:
    """The dataset's own sensor depth, for the `reference` depth backend —
    the ground-truth baseline to compare an estimated-depth run against."""
    if frame.ref_depth_m is None:
        raise RuntimeError(f"depth.backend='reference' but frame {frame.name} has no depth/ file")
    return cv2.resize(frame.ref_depth_m, (K.width, K.height), interpolation=cv2.INTER_NEAREST)


def estimate_depth(depth_estimator: DepthAnythingEstimator, rgb: np.ndarray, K: CameraIntrinsics) -> np.ndarray:
    depth, is_metric = depth_estimator.estimate(rgb, K.fx, K.fy)
    if not is_metric:
        raise RuntimeError("depth model returned non-metric depth — use a DA3METRIC-* checkpoint "
                           "(depth.model_id) so the map and DROID-SLAM get metres")
    if depth.shape != (K.height, K.width):
        # nearest, not cubic: interpolating across a depth edge invents
        # surfaces between foreground and background
        depth = cv2.resize(depth, (K.width, K.height), interpolation=cv2.INTER_NEAREST)
    return depth


def compute_global_depth_scale(cfg: dict, source: FrameSource, depth_estimator: DepthAnythingEstimator,
                               K_native: CameraIntrinsics, target_w: int) -> float:
    """One sequence-wide median(reference)/median(predicted) ratio, estimated
    from a spread sample of frames that have reference depth."""
    depth_cfg = cfg["depth"]
    lo, hi = depth_cfg["min_valid_depth_m"], depth_cfg["max_valid_depth_m"]
    scales = []
    for fr in source.sample(depth_cfg.get("align_sample_frames", 10)):
        if fr.ref_depth_m is None:
            continue
        h0, w0 = fr.rgb.shape[:2]
        K = K_native.scaled_to(target_w, round(h0 * target_w / w0))
        pred = estimate_depth(depth_estimator, cv2.resize(fr.rgb, (K.width, K.height), interpolation=cv2.INTER_AREA), K)
        ref = cv2.resize(fr.ref_depth_m, (K.width, K.height), interpolation=cv2.INTER_NEAREST)
        m = (ref > lo) & (ref < hi) & (pred > 0)
        if m.sum() > 100:
            scales.append(float(np.median(ref[m]) / np.median(pred[m])))
    if not scales:
        raise RuntimeError("depth.align=global found no frames with usable reference depth "
                           "(needs a depth/ folder in the dataset)")
    scale = float(np.median(scales))
    logger.info(f"Global depth alignment: scale={scale:.4f} from {len(scales)} frames "
                f"(spread {min(scales):.3f}-{max(scales):.3f})")
    return scale


def capture_frames(cfg: dict, source: FrameSource, dataset_dir: Path,
                   depth_estimator: DepthAnythingEstimator, pose_estimator: DroidSlamPoseEstimator) -> dict:
    """Reads the source, resizes each frame to source.target_width, predicts
    metric depth (optionally globally aligned to the dataset's reference
    depth), and stages the rgb/point/mask/depth/calib inputs DROID-SLAM and
    ViewDataset need. Frames are renumbered contiguously; frame_names maps
    each processed index back to its source frame.
    """
    depth_cfg = cfg["depth"]
    target_w = cfg["source"]["target_width"]
    align = depth_cfg.get("align", "none") if depth_estimator is not None else "none"
    if align == "global" and not source.has_reference_depth():
        raise RuntimeError("depth.align=global needs a dataset folder with a depth/ subfolder")

    rgb_dir = dataset_dir / "rgb"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    depth_viz_dir = REPO_ROOT / cfg["outputs"]["visualizations_dir"] / "depth"

    K: CameraIntrinsics | None = None
    scale = 1.0
    frame_names: list[str] = []

    for frame in source:
        if K is None:
            h0, w0 = frame.rgb.shape[:2]
            K_native = resolve_intrinsics(cfg, source, w0, h0)
            K = K_native.scaled_to(target_w, round(h0 * target_w / w0))
            logger.info(f"Camera intrinsics at {K.width}x{K.height}: "
                        f"fx={K.fx:.1f} fy={K.fy:.1f} cx={K.cx:.1f} cy={K.cy:.1f}")
            if align == "global":
                scale = compute_global_depth_scale(cfg, source, depth_estimator, K_native, target_w)

        rgb = cv2.resize(frame.rgb, (K.width, K.height), interpolation=cv2.INTER_AREA)
        if depth_estimator is None:
            depth = reference_depth(frame, K)
        else:
            depth = estimate_depth(depth_estimator, rgb, K) * scale

        valid = ((depth > depth_cfg["min_valid_depth_m"]) & (depth < depth_cfg["max_valid_depth_m"])
                 & np.isfinite(depth))
        if not valid.any():
            logger.warning(f"Source frame {frame.name}: no valid depth, skipped.")
            continue

        idx = len(frame_names)
        cv2.imwrite(str(rgb_dir / f"{idx:06d}.jpg"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        save_depth_visualization(depth_viz_dir, idx, depth, valid)
        pose_estimator.save_frame_inputs(dataset_dir, idx, depth, K.fx, K.fy, K.cx, K.cy)
        frame_names.append(frame.name)
        if idx % 20 == 0:
            logger.info(f"Captured {idx + 1} frames (source frame {frame.name})")

    if K is None:
        raise RuntimeError(f"source {source.uri} produced no frames")

    capture_info = {
        "source": source.uri,
        "source_kind": source.kind,
        "num_frames": len(frame_names),
        "fx": K.fx, "fy": K.fy, "cx": K.cx, "cy": K.cy,
        "width": K.width, "height": K.height,
        "depth_model": depth_cfg["model_id"],
        "depth_align": align,
        "depth_scale": scale,
        "frame_names": frame_names,
    }
    (dataset_dir / "capture.json").write_text(json.dumps(capture_info, indent=2))
    return capture_info


def write_reference_poses(source: FrameSource, frame_names: list[str], dataset_dir: Path) -> None:
    """Writes the dataset's ground-truth poses as the trajectory, for the
    `reference` pose backend — the baseline a DROID-SLAM run is compared to."""
    reference = source.reference_poses()
    if not reference:
        raise RuntimeError("pose.backend='reference' but the dataset has no poses/ folder")
    poses_dir = dataset_dir / "poses"
    poses_dir.mkdir(parents=True, exist_ok=True)
    missing = [n for n in frame_names if n not in reference]
    if missing:
        raise RuntimeError(f"{len(missing)} captured frames have no reference pose (e.g. {missing[:3]})")
    for idx, name in enumerate(frame_names):
        np.savetxt(poses_dir / f"{idx:06d}.txt", reference[name])
    logger.info(f"Wrote {len(frame_names)} ground-truth poses to {poses_dir}")


def evaluate_against_reference_poses(source: FrameSource, frame_names: list[str],
                                     poses: dict[int, np.ndarray], out_path: Path, viz_dir: Path) -> None:
    """Scores the estimated trajectory when the dataset ships ground-truth poses,
    and saves the side-by-side trajectory plot plus a per-frame error table."""
    reference = source.reference_poses()
    if not reference:
        logger.info("Dataset ships no ground-truth poses — skipping pose evaluation.")
        return
    gt = {i: reference[name] for i, name in enumerate(frame_names) if name in reference}
    est = {i: poses[i] for i in gt if i in poses}
    if len(est) < 3:
        logger.warning(f"Only {len(est)} frames have both estimated and reference poses — skipping pose evaluation.")
        return

    result = evaluate_poses(gt, est)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    logger.info("Camera pose accuracy vs dataset ground truth:\n" + format_pose_report(result))

    aligned = align_trajectories(gt, est)
    csv_path = out_path.with_name(out_path.stem + "_per_frame.csv")
    write_per_frame_csv(csv_path, aligned)
    plot_path = save_trajectory_comparison(
        viz_dir, aligned,
        title=f"DROID-SLAM vs ground truth — {len(aligned['frames'])} frames, "
              f"ATE {result['ate_sim3']['translation']['rmse']*100:.1f} cm, scale {aligned['scale']:.4f}",
    )
    logger.info(f"Pose evaluation : {out_path}")
    logger.info(f"Per-frame errors: {csv_path}")
    logger.info(f"Comparison plot : {plot_path}")


def run_perception_and_semantic_decoding(cfg: dict, view_dataset, detector, novic_decoder: NOVICDecoder,
                                         fusion) -> tuple[list, list]:
    """Per frame: YOLOE-seg-pf (+ BoT-SORT track ids) -> padded crops -> NOVIC
    embeddings -> a DetectionList for object_fusion. YOLOE's own class output is
    never read; the name comes only from NOVIC. Sequential; the default
    pipeline streams the same steps in parallel (object_fusion/streaming.py).

    Returns (per_frame_detection_lists, frame_records); frame_records feeds the
    semantic-decoding visualization once tagging is done.
    """
    det_viz_dir = REPO_ROOT / cfg["outputs"]["visualizations_dir"] / "detection"
    tracked = callable(getattr(detector, "track", None))   # YOLOE + BoT-SORT or panoptic + BoT-SORT
    fusion.start_new_scan(view_dataset)

    per_frame_detection_lists, frame_records = [], []
    for idx, name in enumerate(view_dataset.names):
        image_rgb = view_dataset.images[idx]
        boxes, masks, confs, track_ids, embeddings = detect_and_embed(cfg, image_rgb, detector, novic_decoder,
                                                                      tracked)
        save_detection_debug_frame(det_viz_dir, idx, image_rgb, boxes, confs, list(masks))
        per_frame_detection_lists.append(fusion.build_detection_list_from_frame(
            boxes=boxes, masks=masks, confs=confs, embeddings=embeddings,
            pixel_indexes=view_dataset.pixel_index_mappings[idx],
            pixel_indexes_mask=view_dataset.pixel_index_masks[idx], image_name=name, track_ids=track_ids))
        frame_records.append({"name": name, "image_rgb": image_rgb, "boxes": boxes, "masks": masks, "confs": confs})
        if idx % 20 == 0 or idx == len(view_dataset.names) - 1:
            extra = f", {len(set(track_ids[track_ids >= 0]))} tracks" if tracked else ""
            logger.info(f"Frame {idx + 1}/{len(view_dataset.names)} ({name}): {len(boxes)} detections{extra}")

    return per_frame_detection_lists, frame_records


def load_everyday_names(cfg: dict) -> list[str] | None:
    path = cfg["label_mapper"].get("everyday_names_file")
    if not path:
        return None
    path = Path(path) if Path(path).is_absolute() else REPO_ROOT / path
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def build_graph_from_objects(cfg: dict, objects, view_dataset, novic_decoder: NOVICDecoder, everyday_names,
                             timer: "StageTimer | None" = None, verbose: bool = True, relations: bool = True):
    """Stages 6b-9 on fused objects: clean-up, naming, relations. Used for the
    final scene graph and, on snapshots of the fusion state, for every live
    update while fusion is still running. Returns (objects, graph, node_embeddings)."""
    stage = timer.stage if timer is not None else (lambda name: nullcontext())
    fusion_cfg, label_cfg, sg_cfg = cfg["object_fusion"], cfg["label_mapper"], cfg["scene_graph"]
    say = logger.info if verbose else logger.debug

    # ─── Stage 6b: clean up fusion artefacts before anything is named ───
    with stage("6b_instance_cleanup"):
        if fusion_cfg.get("drop_structural_instances", True):
            objects, _ = drop_structural_instances(
                objects, view_dataset,
                min_area_m2=fusion_cfg.get("structural_min_area_m2", 1.0),
                max_thickness_m=fusion_cfg.get("structural_max_thickness_m", 0.2),
                floor_band_m=fusion_cfg.get("structural_floor_band_m", 0.25))
        if fusion_cfg.get("merge_fragments", True):
            objects = merge_fragmented_instances(
                objects, view_dataset,
                containment=fusion_cfg.get("fragment_containment", 0.5),
                embedding_similarity=fusion_cfg.get("fragment_embedding_similarity", 0.75),
                point_distance_m=fusion_cfg.get("fragment_point_distance_m", 0.05),
                conflict_min_score=fusion_cfg.get("fragment_conflict_min_score", 0.3))
        if sg_cfg.get("gravity_align", True) and fusion_cfg.get("complete_support_surfaces", False):
            top_lo, top_hi = fusion_cfg.get("support_surface_top_range_m", [0.3, 1.3])
            objects = complete_support_surfaces(
                objects, view_dataset, min_top_m=top_lo, max_top_m=top_hi,
                seed_min_area_m2=fusion_cfg.get("surface_seed_min_area_m2", 0.15),
                plane_tol_m=fusion_cfg.get("surface_plane_tol_m", 0.025),
                bridge_m=fusion_cfg.get("surface_bridge_m", 0.06))
        say(f"{len(objects)} instances after cleanup.")

    # ─── Stage 7: final names — track-end decode, re-decoded if merged since ───
    with stage("7_instance_naming"):
        support_surfaces = set()
        if sg_cfg.get("gravity_align", True) and fusion_cfg.get("relabel_support_surfaces", True):
            top_lo, top_hi = fusion_cfg.get("support_surface_top_range_m", [0.3, 1.3])
            support_surfaces = find_support_surfaces(
                objects, view_dataset,
                min_area_m2=fusion_cfg.get("support_surface_min_area_m2", 0.8),
                min_top_m=top_lo, max_top_m=top_hi,
                min_slab_fraction=fusion_cfg.get("support_surface_min_slab_fraction", 0.5))
        objects = tag_instances(objects, novic_decoder,
                                merge_similarity_threshold=label_cfg["merge_similarity_threshold"],
                                min_name_score=label_cfg.get("min_name_score", 0.15),
                                synonym_similarity=label_cfg.get("synonym_similarity", 0.8),
                                everyday_names=everyday_names,
                                everyday_name_weight=label_cfg.get("everyday_name_weight", 0.2),
                                low_score_action=label_cfg.get("low_score_action", "drop"),
                                support_surfaces=support_surfaces,
                                manual_labels=label_cfg.get("manual_labels"))
        snm = label_cfg.get("same_name_merge") or {}
        if snm.get("enabled"):
            from object_fusion.postprocess import merge_same_name_neighbours
            objects, k = merge_same_name_neighbours(objects, view_dataset, snm.get("gap_m", 0.03),
                                                    snm.get("similarity", 0.8), snm.get("containment", 0.2))
            say(f"Same-name merge joined {k} object(s).")
        nns = label_cfg.get("nested_name_suppression") or {}
        if nns.get("enabled"):
            from object_fusion.postprocess import suppress_nested_names
            k = suppress_nested_names(objects, view_dataset, novic_decoder.embed_text, nns.get("similarity", 0.85))
            say(f"Nested-name suppression renamed {k} parent(s).")
    say(f"{len(objects)} instances named.")
    if not relations:  # evaluations that score objects only
        return objects, None, None

    # ─── Stage 9: Scene graph ───
    with stage("9_scene_graph_relations"):
        scene_graph, node_embeddings = build_static_scene_graph_safe(
            view_dataset=view_dataset,
            instance_objects=objects,
            scene_graph_generation_root=cfg["external"]["scene_graph_generation_root"],
            resolution=sg_cfg["resolution"],
            neighbour_num=sg_cfg["neighbour_num"],
            stand_floor_threshold=sg_cfg["stand_floor_threshold"],
            alphashape_alpha=sg_cfg["alphashape_alpha"],
            alpha_shape_overleaf_rate_threshold=sg_cfg["alpha_shape_overleaf_rate_threshold"],
            part_intersection_rate_threshold=sg_cfg["part_intersection_rate_threshold"],
            inside_threshold=sg_cfg["inside_threshold"],
            part_level_classes=sg_cfg.get("part_level_classes"),
            cycle_patience=sg_cfg.get("cycle_patience", 50),
            relation_method=sg_cfg.get("relation_method", "dovsg"),
            support_max_gap_m=sg_cfg.get("support_max_gap_m", 0.10),
            support_max_penetration_m=sg_cfg.get("support_max_penetration_m", 0.15),
            next_to_max_gap_m=sg_cfg.get("next_to_max_gap_m", 0.30),
            next_to_min_vertical_overlap=sg_cfg.get("next_to_min_vertical_overlap", 0.5),
        )
    return objects, scene_graph, node_embeddings


def save_semantic_decoding_visualization(cfg: dict, objects, frame_records: list) -> None:
    """Re-renders each frame's boxes labeled with the final NOVIC tag its
    instance ended up with. This is the first and only place a semantic
    name is shown — the detection stage never assigns one. Instances track
    which (frame, mask_idx) pairs contributed to them via the 'image_name'/
    'mask_idx' list fields DoVSG's own merge already maintains.
    """
    mask_idx_to_label: dict[tuple[str, int], str] = {}
    for obj in objects:
        for name, midx in zip(obj["image_name"], obj["mask_idx"]):
            mask_idx_to_label[(name, midx)] = obj["class_name"]

    sem_viz_dir = REPO_ROOT / cfg["outputs"]["visualizations_dir"] / "semantic_decoding"
    for idx, record in enumerate(frame_records):
        if len(record["confs"]) == 0:
            continue
        name = record["name"]
        labels = [mask_idx_to_label.get((name, i), "unlabeled") for i in range(len(record["confs"]))]
        save_detection_debug_frame(
            sem_viz_dir, idx, record["image_rgb"], record["boxes"], record["confs"], list(record["masks"]), labels=labels
        )


def prepare_dataset_slice(source_dataset: Path, dataset_dir: Path, num_frames: int | None) -> int:
    """Copies rgb/point/mask/poses (+ metadata.json) from an existing
    DoVSG-format dataset (depth/pose already computed — e.g.
    Scene-Graph-Generation/data_example/freiburg1_desk) into dataset_dir,
    optionally truncated to the first num_frames. Used by --dataset_dir mode
    to skip live capture + depth + pose estimation entirely and run
    detection/fusion/tagging/scene-graph on an existing recording instead.
    Returns the number of frames staged.
    """
    with open(source_dataset / "metadata.json") as f:
        metadata = json.load(f)
    available = metadata["length"]
    length = min(num_frames, available) if num_frames else available

    # --dataset_dir data/processed/<scene_name> re-processes a run in place
    # (e.g. after changing fusion or relation settings) without a second copy
    if source_dataset.resolve() == dataset_dir.resolve():
        if length < available:
            raise ValueError("--num_frames cannot truncate a dataset in place; use a new --scene_name.")
        logger.info(f"Re-processing {dataset_dir} in place ({length} frames).")
        return length

    for sub in ["rgb", "point", "mask", "poses"]:
        src_sub = source_dataset / sub
        dst_sub = dataset_dir / sub
        dst_sub.mkdir(parents=True, exist_ok=True)
        ext = ".jpg" if sub == "rgb" else (".txt" if sub == "poses" else ".npy")
        for i in range(length):
            shutil.copy2(src_sub / f"{i:06d}{ext}", dst_sub / f"{i:06d}{ext}")

    metadata["length"] = length
    with open(dataset_dir / "metadata.json", "w") as f:
        json.dump(metadata, f)

    logger.info(f"Staged {length}/{available} frames from {source_dataset} into {dataset_dir}")
    return length


def main():
    # `kill -USR1 <pid>` prints every thread's Python stack (e.g. a hung stream)
    faulthandler.register(signal.SIGUSR1, all_threads=True)
    parser = argparse.ArgumentParser(description="Static open-vocabulary scene graph pipeline")
    parser.add_argument("--config", default="configs/pipeline.yaml")
    parser.add_argument("--scene_name", default=None, help="Defaults to a timestamp-based name")
    parser.add_argument(
        "--dataset_dir", default=None,
        help="Skip live DroidCam capture + depth + pose estimation and run on an existing "
             "DoVSG-format dataset instead (must already have rgb/point/mask/poses/metadata.json). "
             "E.g. Scene-Graph-Generation/data_example/freiburg1_desk",
    )
    parser.add_argument(
        "--num_frames", type=int, default=None,
        help="With --dataset_dir, only use the first N frames (default: all available).",
    )
    parser.add_argument(
        "--source", default=None,
        help="Override source.uri: a DroidCam URL, webcam index, video file, or dataset folder "
             "(see data/raw/README.md).",
    )
    parser.add_argument("--depth_backend", default=None,
                        help="Override depth.backend: 'depth_anything_v3' or 'reference' "
                             "(the dataset's own depth/ files).")
    parser.add_argument("--pose_backend", default=None,
                        help="Override pose.backend: 'droid_slam', 'cpu_icp' or 'reference' "
                             "(the dataset's own poses/ files).")
    parser.add_argument(
        "--pose_only", action="store_true",
        help="Stop after depth + camera pose (and pose evaluation, if the dataset has "
             "ground-truth poses). Skips detection, fusion and the scene graph.",
    )
    parser.add_argument("--skip_gravity_alignment", action="store_true",
                        help="With --dataset_dir: the poses are already in a z-up world (e.g. a new video "
                             "registered to a saved map); keep them as they are.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    cfg = load_config(config_path)
    # deterministic: two runs of the same video should give the same map. Run-to-run differences came from GPU
    # numerics (cuDNN picking algorithms at run time), which can flip a threshold and cascade through fusion.
    if cfg.get("deterministic", True):
        import os
        import random
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")   # before the first cuBLAS call
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)

    if args.depth_backend:
        cfg["depth"]["backend"] = args.depth_backend
    if args.pose_backend:
        cfg["pose"]["backend"] = args.pose_backend

    scene_name = args.scene_name or time.strftime("scene_%Y%m%d_%H%M%S")
    dataset_dir = REPO_ROOT / cfg["outputs"]["data_processed_dir"] / scene_name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    viz_root = REPO_ROOT / cfg["outputs"]["visualizations_dir"]
    ext = cfg["external"]

    # Per-run log file. The DROID-SLAM subprocess writes its progress to the
    # console only; redirect the whole command to capture that too.
    log_path = REPO_ROOT / cfg["outputs"]["logs_dir"] / f"{scene_name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logging.getLogger().addHandler(file_handler)

    timer = StageTimer()
    logger.info(f"=== Static scene graph pipeline: scene '{scene_name}' ===")
    logger.info(f"Config: {config_path}")
    logger.info(f"Log file: {log_path}")
    logger.info(f"Depth model: {cfg['depth']['model_id']} (align={cfg['depth'].get('align', 'none')}) | "
                f"pose backend: {cfg['pose']['backend']}")

    if args.dataset_dir:
        # ─── Dataset mode: skip capture + depth + pose entirely ───
        num_frames = prepare_dataset_slice(Path(args.dataset_dir), dataset_dir, args.num_frames)
        if num_frames < 2:
            raise RuntimeError(f"Only {num_frames} frame(s) available — need at least 2.")
    else:
        # ─── Stages 1-3: frame source -> DA3 metric depth -> DROID-SLAM pose ───
        src_cfg = cfg["source"]
        source = FrameSource(
            uri=args.source or src_cfg["uri"],
            capture_interval_s=src_cfg["capture_interval"],
            stride=src_cfg.get("stride", 1),
            max_frames=src_cfg["max_frames"],
            depth_scale=src_cfg.get("depth_scale", 1000.0),
        )

        depth_cfg = cfg["depth"]
        if depth_cfg["backend"] == "reference":
            # ground-truth baseline: the dataset's own sensor depth, no model
            logger.info("Depth backend 'reference': using the dataset's depth/ files.")
            depth_estimator = None
        else:
            depth_estimator = DepthAnythingEstimator(
                model_id=depth_cfg["model_id"],
                fallback_model_id=depth_cfg["fallback_model_id"],
                device=depth_cfg["device"],
                min_valid_depth_m=depth_cfg["min_valid_depth_m"],
                max_valid_depth_m=depth_cfg["max_valid_depth_m"],
            )

        pose_cfg = cfg["pose"]
        pose_estimator = DroidSlamPoseEstimator(
            backend=pose_cfg["backend"],
            scene_graph_generation_root=ext["scene_graph_generation_root"],
            dovsg_pose_estimation_script=ext["dovsg_pose_estimation_script"],
            dovsg_cpu_pose_estimation_script=ext["dovsg_cpu_pose_estimation_script"],
            stride=pose_cfg["stride"],
        )

        with timer.stage("2_depth_and_capture"):
            capture_info = capture_frames(cfg, source, dataset_dir, depth_estimator, pose_estimator)
        num_frames = capture_info["num_frames"]
        if num_frames < 2:
            raise RuntimeError(f"Only captured {num_frames} usable frame(s) — need at least 2 for pose estimation.")

        # DROID-SLAM runs in a subprocess on the same GPU; release the depth model first.
        del depth_estimator
        torch.cuda.empty_cache()
        depth_estimator = None

        write_metadata(
            dataset_dir,
            width=capture_info["width"],
            height=capture_info["height"],
            K=[capture_info["fx"], 0, capture_info["cx"], 0, capture_info["fy"], capture_info["cy"], 0, 0, 1],
            length=num_frames,
        )

        with timer.stage("3_pose_" + pose_cfg["backend"]):
            if pose_cfg["backend"] == "reference":
                # ground-truth baseline: the dataset's own poses, no SLAM
                write_reference_poses(source, capture_info["frame_names"], dataset_dir)
            else:
                pose_estimator.estimate_trajectory(dataset_dir)
        poses = pose_estimator.load_poses(dataset_dir)
        save_trajectory_visualization(viz_root / "pose", poses)
        logger.info(f"Estimated {len(poses)} camera poses.")
        if len(poses) != num_frames:
            raise RuntimeError(f"Pose estimator returned {len(poses)} poses for {num_frames} frames — "
                               "the voxel map needs one pose per frame (keep pose.stride at 1).")

        evaluate_against_reference_poses(
            source, capture_info["frame_names"], poses,
            REPO_ROOT / cfg["outputs"]["scene_graphs_dir"] / f"{scene_name}_pose_eval.json",
            viz_root / "pose",
        )

        if args.pose_only:
            timer.save(dataset_dir / "stage_timings.json")
            logger.info("Processing time per stage:\n" + timer.report())
            logger.info(f"--pose_only: stopping after pose estimation. Dataset: {dataset_dir}")
            return

    # ─── Stage 4: Global voxel map, in a gravity-aligned (z-up) world ───
    # The relation builder decides "on"/"inside" by z height; DROID-SLAM's world
    # is the first camera's frame, and 3RScan's scan frame isn't z-up either.
    if cfg["scene_graph"].get("gravity_align", True) and not args.skip_gravity_alignment:
        with timer.stage("4a_gravity_alignment"):
            align_info = gravity_align_dataset(dataset_dir)
        (dataset_dir / "gravity_alignment.json").write_text(json.dumps(align_info, indent=2))

    fusion_cfg = cfg["object_fusion"]
    match_level = fusion_cfg.get("match_level", "track")
    if match_level == "track" and not cfg["detection"].get("use_tracking", True):
        logger.warning("object_fusion.match_level 'track' needs detection.use_tracking; using 'detection'.")
        match_level = "detection"
    streaming = match_level == "track" and fusion_cfg.get("streaming", True)
    track_end_frames = fusion_cfg.get("track_end_frames", 10)
    view_dataset_kwargs = dict(recorder_dir=str(dataset_dir),
                               scene_graph_generation_root=ext["scene_graph_generation_root"],
                               resolution=fusion_cfg["voxel_resolution"])

    if streaming:
        # ─── Stage 4 in its own process, started before the models load: the
        # voxel map is CPU work that holds the GIL, so on a thread it only
        # slowed detection down (object_fusion.view_dataset) ───
        voxel_started = time.time()
        # the worker's result holds DoVSG objects: make them importable here before it can arrive (on short
        # sequences it finished before anything else had imported DoVSG, and unpickling it failed)
        from object_fusion._dovsg_import import ensure_dovsg_importable
        ensure_dovsg_importable(ext["scene_graph_generation_root"])
        voxel_pool = ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn"))
        voxel_job = voxel_pool.submit(load_view_dataset_for_transfer, workers=fusion_cfg.get("voxel_map_workers", 1),
                                      **view_dataset_kwargs)
        voxel_box, voxel_lock = {}, threading.Lock()

        def voxel_map():
            with voxel_lock:
                if "vd" not in voxel_box:
                    voxel_box["vd"] = restore_view_dataset(voxel_job.result())
                    logger.info(f"[voxel map] ready {time.time() - voxel_started:.1f}s after it started, built in "
                                "its own process while the models loaded and detection ran")
            return voxel_box["vd"]

    # ─── Models: detection (+ tracking), NOVIC, fusion ───
    det_cfg = cfg["detection"]
    detector_kwargs = dict(
        weights_path=det_cfg["weights"],
        conf_thresh=det_cfg["conf_thresh"],
        iou_thresh=det_cfg["iou_thresh"],
        img_size=det_cfg["img_size"],
        device=det_cfg["device"],
        fix_channel_order=det_cfg.get("fix_channel_order", True),
    )
    if det_cfg.get("backend") == "panoptic":           # perception/detection/panoptic_detector.py
        from perception.detection.panoptic_detector import from_config as panoptic_from_config
        detector = panoptic_from_config(det_cfg)
    elif det_cfg.get("use_tracking", True):
        detector = TrackedYOLOEDetector(tracker=det_cfg.get("tracker", "botsort.yaml"),
                                        mask_box_filter=det_cfg.get("mask_box_filter"),
                                        tighten_box_to_mask=det_cfg.get("tighten_box_to_mask", False),
                                        validation=det_cfg.get("validation"),
                                        **detector_kwargs)
    else:
        detector = YOLOESegDetector(**detector_kwargs)

    sem_cfg = cfg["semantic_decoding"]
    novic_decoder = NOVICDecoder(
        novic_dir=sem_cfg["novic_dir"],
        checkpoint_path=sem_cfg["novic_checkpoint"],
        gencfg=sem_cfg["gencfg"],
        guide_targets=sem_cfg["guide_targets"],
        device=sem_cfg["device"],
        embedding_batch_size=sem_cfg.get("embedding_batch_size", 8),
    )

    # more GPUs embed crops alongside the first (object_fusion/streaming.py)
    extra_embedders = [
        (lambda device=device: NOVICDecoder(
            novic_dir=sem_cfg["novic_dir"], checkpoint_path=sem_cfg["novic_checkpoint"], gencfg=sem_cfg["gencfg"],
            guide_targets=sem_cfg["guide_targets"], device=device,
            embedding_batch_size=sem_cfg.get("embedding_batch_size", 8)))
        for device in sem_cfg.get("extra_embedding_devices") or []]

    InstanceFusion = build_instance_fusion_class(ext["scene_graph_generation_root"])
    fusion = InstanceFusion(
        downsample_voxel_size=fusion_cfg["downsample_voxel_size"],
        spatial_weight=fusion_cfg["spatial_weight"],
        vis_weight=fusion_cfg["vis_weight"],
        sim_threshold=fusion_cfg["sim_threshold"],
    )
    fusion.mask_erode_px = int(fusion_cfg.get("mask_erode_px", 0))
    fusion.min_merge_cosine = float(fusion_cfg.get("min_merge_cosine", 0.0))
    fusion.rematch_min_containment = fusion_cfg.get("rematch_min_containment")
    fusion.track_split_cosine = float(fusion_cfg.get("track_split_cosine", 0.0))
    fusion.small_merge_ratio = float(fusion_cfg.get("small_merge_ratio", 0.0))
    fusion.track_validation = fusion_cfg.get("track_validation") or {}
    fusion.small_merge_cosine = float(fusion_cfg.get("small_merge_cosine", 0.9))
    # DoVSG's per-detection filters (InstanceProcess defaults: conf 0.25, mask > 30% of the image
    # dropped, masks < 25 px dropped, < 10 points dropped). The confidence floor applies after
    # YOLOE's own detection.conf_thresh, so lowering only the latter changes nothing in fusion.
    for key in ("mask_conf_threshold", "max_bbox_area_ratio", "mask_area_threshold", "min_points_threshold"):
        if key in fusion_cfg:
            setattr(fusion, key, type(getattr(fusion, key))(fusion_cfg[key]))
    logger.info("Detection filters in fusion: conf >= %s, mask <= %s of image, mask >= %s px, >= %s points",
                fusion.mask_conf_threshold, fusion.max_bbox_area_ratio, fusion.mask_area_threshold,
                fusion.min_points_threshold)
    everyday_names = load_everyday_names(cfg)
    logger.info(f"Object fusion: {match_level}-level matching"
                + (", streamed (detection, fusion and the scene graph run in parallel)" if streaming else ""))

    if streaming:
        # ─── Stages 4-6 in parallel: detection and embedding (one worker per
        # GPU) run while the voxel map is built; fusion consumes both, and
        # every track that ends is matched, named and shown in a live scene
        # graph (object_fusion/streaming.py) ───
        live_path = REPO_ROOT / cfg["outputs"]["scene_graphs_dir"] / f"{scene_name}_live.json"

        def prepare():
            vd = voxel_map()
            fusion.start_new_scan(vd)
            fusion.start_stream(vd, decode_fn=novic_decoder.decode_embeddings, track_end_frames=track_end_frames,
                                min_merge_overlap=fusion_cfg.get("min_merge_overlap", 0.3))

        def on_update(snapshot, info):
            t = time.time()
            vd = voxel_map()
            with quiet_loggers():
                live_objects, live_graph, _ = build_graph_from_objects(cfg, snapshot, vd, novic_decoder,
                                                                       everyday_names, verbose=False)
            write_live_graph(live_path, live_graph, live_objects, vd, info)
            logger.info(f"[live scene graph #{info['update']}] after {info['frames_fused']}/{num_frames} frames, "
                        f"{info['commits']} tracks ({time.time() - t:.1f}s):\n" + tree_lines(live_graph))

        prepare_native_libraries()  # before any thread starts: see its docstring
        stream = StreamingFusion(
            cfg, fusion, detector, novic_decoder,
            frame_names=[f"{i:06d}" for i in range(num_frames)],
            load_image=lambda i: np.asarray(Image.open(dataset_dir / "rgb" / f"{i:06d}.jpg"), dtype=np.uint8),
            frame_voxels=lambda i: (voxel_map().pixel_index_mappings[i], voxel_map().pixel_index_masks[i]),
            prepare=prepare,
            on_update=on_update if fusion_cfg.get("live_scene_graph", False) else None,
            det_viz_dir=viz_root / "detection",
            obj_min_detections=fusion_cfg["obj_min_detections"],
            extra_embedders=extra_embedders,
            live_interval_s=fusion_cfg.get("live_scene_graph_interval_s", 0.0))
        with timer.stage("4-6_voxel_map_detect_fuse_streaming"):
            stream.run()
        view_dataset = voxel_map()
        voxel_pool.shutdown()
        frame_records = stream.frame_records
        (dataset_dir / "track_commits.json").write_text(json.dumps(fusion.commit_log, indent=2))
        with timer.stage("6_instance_fusion_finalize"):
            objects, _ = fusion.finalize(obj_min_detections=fusion_cfg["obj_min_detections"])
    else:
        # ─── Stage 4: Global voxel map ───
        with timer.stage("4_voxel_map"):
            view_dataset = load_view_dataset(**view_dataset_kwargs)
        # ─── Stage 5: Detection + Segmentation + NOVIC embedding (per frame) ───
        with timer.stage("5_detect_track_embed"):
            per_frame_detection_lists, frame_records = run_perception_and_semantic_decoding(
                cfg, view_dataset, detector, novic_decoder, fusion)
        # ─── Stage 6: Multi-view 3D instance fusion, naming each object as its tracks end ───
        with timer.stage("6_instance_fusion"):
            objects, _ = fusion.get_instances(
                view_dataset=view_dataset,
                per_frame_detection_lists=per_frame_detection_lists,
                obj_min_detections=fusion_cfg["obj_min_detections"],
                decode_fn=novic_decoder.decode_embeddings,
                match_level=match_level,
                track_end_frames=track_end_frames,
            )

    if callable(getattr(detector, "track", None)):
        track_stats = {k: v for k, v in detector.stats.as_dict().items() if not k.startswith("embedding")}
        logger.info(f"Tracker: {json.dumps(track_stats)}")
    logger.info(f"Fused {len(objects)} persistent 3D instances; "
                f"{len(fusion.track_name_log)} track-end NOVIC decodes.")
    if getattr(fusion, "_overlap_index_hits", 0) or getattr(fusion, "_overlap_index_misses", 0):
        hits, misses = fusion._overlap_index_hits, fusion._overlap_index_misses
        logger.info(f"Overlap index cache: {hits} hits, {misses} rebuilds "
                    f"({hits / max(hits + misses, 1):.0%} avoided).")

    # ─── Stages 6b-9: clean-up, naming, relations ───
    objects, scene_graph, node_embeddings = build_graph_from_objects(
        cfg, objects, view_dataset, novic_decoder, everyday_names, timer=timer)

    # Track-end decodes of the instances that survived naming, keyed to the
    # final node id; tracks of dropped instances are left out.
    track_to_node = {int(t): obj["class_id"] for obj in objects for t in obj["track_ids"] if t >= 0}
    (dataset_dir / "track_names.json").write_text(json.dumps(
        [{**{k: v for k, v in r.items() if k not in ("top_k", "object_index")}, "node_id": track_to_node[r["track_id"]]}
         for r in fusion.track_name_log if r["track_id"] in track_to_node], indent=2))
    save_semantic_decoding_visualization(cfg, objects, frame_records)

    # ─── Stage 8: Instance visualization ───
    save_instances_screenshot(objects, view_dataset, viz_root / "instances" / "instances.png")

    # the per-frame masks are never read after fusion and are ~30 MB per object
    for obj in objects:
        obj.pop("mask", None)
    pkl_path, json_path = save_scene_graph(
        REPO_ROOT / cfg["outputs"]["scene_graphs_dir"], scene_name, scene_graph, objects, node_embeddings,
        view_dataset=view_dataset,
    )
    save_scene_graph_visualization(scene_graph, viz_root / "scene_graph")

    novic_decoder.close()
    timer.save(dataset_dir / "stage_timings.json")
    logger.info("Processing time per stage:\n" + timer.report())
    logger.info(f"Done. Scene graph: {pkl_path}  (summary: {json_path})")
    logger.info(f"Visualizations under: {viz_root}")


if __name__ == "__main__":
    main()
