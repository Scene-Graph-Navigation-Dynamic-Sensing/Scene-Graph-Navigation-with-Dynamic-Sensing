"""Track-driven object mapping over a prepared dataset.

Per frame: YOLOE-seg-pf detects, BoT-SORT assigns a persistent track id, and
the image encoder runs ONLY for track ids seen for the first time. A new track
is then cosine-matched against the objects already in the map to decide whether
it is a re-appearance (tracker dropped the id through an occlusion or a view
change) or a genuinely new object. Established tracks skip both the encoder and
the similarity search entirely and just accumulate their mask into the object
they already belong to.

That is the whole point of the tracker here: cosine similarity becomes a
per-new-track cost instead of a per-detection-per-frame one.

Each object accumulates the global voxel indexes its masks covered, so the
result is a voxel map segmented by object, saved as a coloured cloud plus
rendered views.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from object_fusion.view_dataset import load_view_dataset  # noqa: E402
from perception.detection.tracked_yoloe_detector import (  # noqa: E402
    TrackedYOLOEDetector, TrackEmbeddingCache, cosine_similarity_matrix)
from perception.visualization import save_detection_debug_frame  # noqa: E402
from scripts.build_voxel_map import render_views  # noqa: E402

logger = logging.getLogger("tracked_mapping")


class MapObject:
    """One physical object: its voxel support, its embedding, its tracks."""

    def __init__(self, obj_id: int, embedding: np.ndarray, track_id: int, frame: int):
        self.id = obj_id
        self.embedding = np.asarray(embedding, dtype=np.float32)
        self.track_ids = {track_id}
        self.indexes: set[int] = set()
        self.n_detections = 0
        self.first_frame = frame
        self.last_frame = frame

    def absorb(self, embedding: np.ndarray, track_id: int, frame: int) -> None:
        """Running mean of the embeddings that were actually computed."""
        n = len(self.track_ids)
        self.embedding = (self.embedding * n + np.asarray(embedding, np.float32)) / (n + 1)
        self.track_ids.add(track_id)
        self.last_frame = frame

    def add_observation(self, indexes: np.ndarray, frame: int) -> None:
        self.indexes.update(indexes.tolist())
        self.n_detections += 1
        self.last_frame = frame

    def voxel_iou(self, indexes: set[int]) -> float:
        """Overlap between a detection's voxels and the ones already claimed.

        Intersection over the *detection* rather than over the union: an object
        already fused from many views has far more voxels than any single
        masked view of it, so a true union IoU would stay near zero for a
        correct re-appearance and never fire.
        """
        if not self.indexes or not indexes:
            return 0.0
        return len(self.indexes & indexes) / len(indexes)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-dir", required=True, type=Path)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "pipeline.yaml")
    ap.add_argument("--tracker", default="botsort.yaml")
    ap.add_argument("--reid-threshold", type=float, default=None,
                    help="combined score above which a NEW track is treated as an object "
                         "already in the map (default: object_fusion.sim_threshold). The "
                         "score is vis_weight*cosine + spatial_weight*voxel_overlap, the "
                         "same combination object_fusion uses -- raw cosine alone cannot "
                         "separate objects here, since CLIP-style crop embeddings of a "
                         "single room all sit in a narrow high band.")
    ap.add_argument("--min-detections", type=int, default=3,
                    help="drop objects seen in fewer frames than this")
    ap.add_argument("--embedder", choices=["novic", "none"], default="novic",
                    help="none = skip embeddings entirely, identity from the tracker alone")
    ap.add_argument("--no-vis", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    out_dir = args.out_dir or (REPO_ROOT / "outputs" / "tracked_mapping" /
                               f"{args.dataset_dir.name}_{time.strftime('%Y%m%d-%H%M%S')}")
    (out_dir / "frames").mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        handlers=[logging.FileHandler(out_dir / "run.log"), logging.StreamHandler(sys.stdout)],
        force=True,
    )
    logger.info("args: %s", vars(args))

    fusion_cfg = cfg["object_fusion"]
    reid_threshold = args.reid_threshold if args.reid_threshold is not None \
        else fusion_cfg["sim_threshold"]
    vis_weight = fusion_cfg["vis_weight"]
    spatial_weight = fusion_cfg["spatial_weight"]
    logger.info("re-id score = %.2f*cosine + %.2f*voxel_overlap, threshold %.2f",
                vis_weight, spatial_weight, reid_threshold)

    view_dataset = load_view_dataset(
        recorder_dir=str(args.dataset_dir),
        scene_graph_generation_root=cfg["external"]["scene_graph_generation_root"],
        resolution=fusion_cfg["voxel_resolution"],
    )
    logger.info("voxel map: %d frames, %d occupied voxels",
                view_dataset.length, len(view_dataset.indexes_colors_mapping_dict))

    det_cfg = cfg["detection"]
    detector = TrackedYOLOEDetector(
        weights_path=det_cfg["weights"],
        conf_thresh=det_cfg["conf_thresh"],
        iou_thresh=det_cfg["iou_thresh"],
        img_size=det_cfg["img_size"],
        device=det_cfg["device"],
        tracker=args.tracker,
    )
    detector.reset()

    sem_cfg = cfg["semantic_decoding"]
    embed_fn = None
    novic_decoder = None
    if args.embedder == "novic":
        from semantic_decoding.novic.novic_decoder import NOVICDecoder
        novic_decoder = NOVICDecoder(
            novic_dir=sem_cfg["novic_dir"],
            checkpoint_path=sem_cfg["novic_checkpoint"],
            gencfg=sem_cfg["gencfg"],
            guide_targets=sem_cfg["guide_targets"],
            device=sem_cfg["device"],
        )
        embed_fn = novic_decoder.embed_crops
    else:
        embed_fn = lambda crops: np.zeros((len(crops), 8), dtype=np.float32)  # noqa: E731

    from semantic_decoding.novic.crop_utils import extract_crop

    cache = TrackEmbeddingCache(embed_fn, stats=detector.stats)

    objects: list[MapObject] = []
    track_to_object: dict[int, int] = {}
    reid_events, new_object_events = [], []
    n_frames = min(args.limit, view_dataset.length) if args.limit else view_dataset.length

    t0 = time.perf_counter()
    for idx in range(n_frames):
        image_rgb = view_dataset.images[idx]
        detections = detector.track(image_rgb)
        if not detections:
            logger.info("[%03d] no detections", idx)
            continue

        crops = [extract_crop(image_rgb, d.bbox, d.mask,
                              mode=sem_cfg["crop_mode"], padding_px=sem_cfg["crop_padding_px"])
                 for d in detections]
        cache.embed(detections, crops)

        pixel_index_mapping = view_dataset.pixel_index_mappings[idx]
        pixel_index_mask = view_dataset.pixel_index_masks[idx]

        labels = []
        for d in detections:
            valid = d.mask & pixel_index_mask
            det_indexes = set(np.unique(pixel_index_mapping[valid]).tolist()) if valid.any() else set()

            if d.track_id >= 0 and d.track_id in track_to_object:
                # established track: no encoder call, no similarity search
                obj = objects[track_to_object[d.track_id]]
                tag = f"id{d.track_id}->o{obj.id}"
            else:
                # a new track is the ONLY place similarity is computed
                obj = None
                if objects and args.embedder != "none":
                    cos = cosine_similarity_matrix(
                        d.embedding[None, :],
                        np.stack([o.embedding for o in objects]))[0]
                    iou = np.array([o.voxel_iou(det_indexes) for o in objects])
                    score = vis_weight * cos + spatial_weight * iou
                    best = int(np.argmax(score))
                    if score[best] >= reid_threshold:
                        obj = objects[best]
                        obj.absorb(d.embedding, max(d.track_id, 0), idx)
                        reid_events.append({
                            "frame": idx, "track_id": d.track_id, "object_id": obj.id,
                            "score": float(score[best]), "cosine": float(cos[best]),
                            "voxel_overlap": float(iou[best])})
                        tag = f"id{d.track_id}=o{obj.id} reid {score[best]:.2f}"
                if obj is None:
                    obj = MapObject(len(objects), d.embedding, max(d.track_id, 0), idx)
                    objects.append(obj)
                    new_object_events.append({"frame": idx, "track_id": d.track_id,
                                              "object_id": obj.id})
                    tag = f"id{d.track_id}->o{obj.id} NEW"
                if d.track_id >= 0:
                    track_to_object[d.track_id] = obj.id

            if det_indexes:
                obj.add_observation(np.array(sorted(det_indexes)), idx)
            labels.append(tag)

        if not args.no_vis:
            save_detection_debug_frame(
                out_dir / "frames", idx, image_rgb,
                np.stack([d.bbox for d in detections]),
                np.array([d.conf for d in detections]),
                [d.mask for d in detections],
                labels=labels)

        stats = detector.stats
        logger.info("[%03d] det=%2d new_tracks=%2d embeds=%3d objects=%3d",
                    idx, len(detections),
                    sum(1 for d in detections if d.is_new_track),
                    stats.embeddings_computed, len(objects))

    elapsed = time.perf_counter() - t0

    kept = [o for o in objects if o.n_detections >= args.min_detections and o.indexes]
    logger.info("kept %d/%d objects (>= %d detections)", len(kept), len(objects), args.min_detections)

    # colour the voxel map by object
    import matplotlib
    cmap = matplotlib.colormaps.get_cmap("turbo")
    pts, cols = [], []
    for i, o in enumerate(kept):
        idxs = np.array(sorted(o.indexes))
        p = view_dataset.index_to_point(idxs)
        pts.append(np.asarray(p))
        cols.append(np.tile(np.array(cmap(i / max(len(kept) - 1, 1))[:3]), (len(idxs), 1)))

    if pts:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.concatenate(pts).astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector(np.concatenate(cols))
        o3d.io.write_point_cloud(str(out_dir / "objects_voxel_map.ply"), pcd)
        logger.info("wrote %s (%d points)", out_dir / "objects_voxel_map.ply", len(pcd.points))
        if not args.no_vis:
            render_views([pcd], out_dir / "views")

    stats = detector.stats.as_dict()
    result = {
        "dataset_dir": str(args.dataset_dir),
        "tracker": args.tracker,
        "embedder": args.embedder,
        "reid_threshold": reid_threshold,
        "vis_weight": vis_weight,
        "spatial_weight": spatial_weight,
        "frames_processed": n_frames,
        "seconds": round(elapsed, 1),
        "tracking_stats": stats,
        "objects_total": len(objects),
        "objects_kept": len(kept),
        "reid_events": reid_events,
        "new_object_events": len(new_object_events),
        "objects": [{"id": o.id, "track_ids": sorted(o.track_ids), "voxels": len(o.indexes),
                     "detections": o.n_detections,
                     "frames": [o.first_frame, o.last_frame]} for o in kept],
    }
    (out_dir / "tracked_mapping.json").write_text(json.dumps(result, indent=2))

    if novic_decoder is not None:
        novic_decoder.close()

    logger.info("=" * 66)
    logger.info("detections           : %d over %d frames", stats["detections"], stats["frames"])
    logger.info("unique tracks        : %d", stats["unique_tracks"])
    logger.info("encoder calls        : %d (vs %d without tracking)",
                stats["embeddings_computed"], stats["detections"])
    logger.info("encoder calls saved  : %d (%.1f%% reduction)",
                stats["embeddings_saved"], 100.0 * stats["embedding_call_reduction"])
    logger.info("cosine searches      : %d (once per new track)", stats["embeddings_computed"])
    logger.info("re-id merges         : %d", len(reid_events))
    logger.info("objects kept         : %d", len(kept))
    logger.info("results: %s", out_dir)
    logger.info("=" * 66)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
