"""End-to-end integration test using a small slice of the existing
freiburg1_desk RGB-D capture (already in DoVSG's expected rgb/point/mask/
poses/metadata.json layout) instead of a live DroidCam capture — this
exercises everything from detection through scene-graph construction
without needing a camera, Depth Anything, or DROID-SLAM on hand.

Uses the real NOVIC checkpoint (confirmed loadable once `unidecode` is
installed — see HOW_TO_RUN.md) by default; set USE_REAL_NOVIC = False below
to fall back to the stub decoder instead (e.g. if unidecode isn't
installed in your env).

Not collected by a bare `pytest` run automatically (no pytest is installed
in the `scene_graph` conda env this pipeline targets) — run directly:
    python tests/integration/test_static_pipeline_freiburg.py
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

SOURCE_DATASET = Path("/home/ravindu/SceneGraphNav/external/Scene-Graph-Generation/data_example/freiburg1_desk")
SCENE_GRAPH_GENERATION_ROOT = "/home/ravindu/SceneGraphNav/external/Scene-Graph-Generation"
YOLOE_WEIGHTS = "/home/ravindu/SceneGraphNav/models/checkpoints/yoloe-26s-seg-pf.pt"
NUM_FRAMES = 10

USE_REAL_NOVIC = True
NOVIC_DIR = "/home/ravindu/SceneGraphNav/external/novic"
NOVIC_CHECKPOINT = "/home/ravindu/SceneGraphNav/external/novic/outputs/ovod_20240610_105233/ovod_chunk0899_20240612_005748.train"

# With the stub decoder (USE_REAL_NOVIC = False), each crop gets an
# effectively random embedding, so almost nothing merges during fusion —
# unlike a real NOVIC decode, where visually-similar crops of the same
# object get matching embeddings and correctly merge. That leaves far more
# (and much noisier) candidate instances than a real run would, which makes
# DoVSG's O(n^2) support-relation search in build_scene_graph pathologically
# slow — capping detections per frame keeps the stub path a fast wiring
# smoke test rather than a stress test of that unrelated, unmodified
# upstream code. Not needed with the real decoder, but left in either way
# as a sane cap.
MAX_DETECTIONS_PER_FRAME = 6 if not USE_REAL_NOVIC else 15


def _make_frame_slice(tmp_dir: Path, num_frames: int) -> Path:
    dataset_dir = tmp_dir / "freiburg1_desk_slice"
    for sub in ["rgb", "point", "mask", "poses"]:
        (dataset_dir / sub).mkdir(parents=True, exist_ok=True)
        for i in range(num_frames):
            src = SOURCE_DATASET / sub / f"{i:06d}{'.jpg' if sub == 'rgb' else ('.txt' if sub == 'poses' else '.npy')}"
            dst = dataset_dir / sub / src.name
            shutil.copy2(src, dst)

    with open(SOURCE_DATASET / "metadata.json") as f:
        metadata = json.load(f)
    metadata["length"] = num_frames
    with open(dataset_dir / "metadata.json", "w") as f:
        json.dump(metadata, f)

    return dataset_dir


def main():
    import tempfile

    from perception.detection.yoloe_seg_detector import YOLOESegDetector
    from perception.visualization import save_detection_debug_frame
    from semantic_decoding.novic.novic_decoder import NOVICDecoder
    from semantic_decoding.novic.crop_utils import extract_crop
    from object_fusion.view_dataset import load_view_dataset
    from object_fusion.instance_fusion import build_instance_fusion_class
    from object_fusion.instance_tagger import tag_instances
    from scene_graph.builder.graph_builder import build_static_scene_graph_safe
    from scene_graph.storage.serializer import save_scene_graph, scene_graph_to_relations

    with tempfile.TemporaryDirectory() as tmp:
        dataset_dir = _make_frame_slice(Path(tmp), NUM_FRAMES)

        view_dataset = load_view_dataset(
            recorder_dir=str(dataset_dir),
            scene_graph_generation_root=SCENE_GRAPH_GENERATION_ROOT,
            resolution=0.01,
        )
        assert view_dataset.length == NUM_FRAMES

        detector = YOLOESegDetector(weights_path=YOLOE_WEIGHTS, conf_thresh=0.15, device="cuda")
        if USE_REAL_NOVIC:
            novic_decoder = NOVICDecoder(
                novic_dir=NOVIC_DIR, checkpoint_path=NOVIC_CHECKPOINT,
                gencfg="beam_k10_vnone_gp_t1_a0", device="cuda",
            )
            assert not novic_decoder.is_stub, "Expected the real NOVIC checkpoint to load"
        else:
            novic_decoder = NOVICDecoder(checkpoint_path="", device="cuda")
            assert novic_decoder.is_stub

        InstanceFusion = build_instance_fusion_class(SCENE_GRAPH_GENERATION_ROOT)
        fusion = InstanceFusion(downsample_voxel_size=0.01, spatial_weight=0.3, vis_weight=0.7, sim_threshold=0.55)
        fusion.start_new_scan(view_dataset)

        det_viz_dir = Path(tmp) / "viz" / "detection"
        per_frame_detection_lists = []
        for idx, name in enumerate(view_dataset.names):
            image_rgb = view_dataset.images[idx]
            detections = detector.detect(image_rgb)
            detections = sorted(detections, key=lambda d: -d.conf)[:MAX_DETECTIONS_PER_FRAME]
            print(f"frame {name}: {len(detections)} detections (capped)")

            h, w = image_rgb.shape[:2]
            boxes = np.stack([d.bbox for d in detections]) if detections else np.zeros((0, 4))
            masks = np.stack([d.mask for d in detections]) if detections else np.zeros((0, h, w), dtype=bool)
            confs = np.array([d.conf for d in detections])

            crops = [extract_crop(image_rgb, boxes[i], masks[i], mode="segmented", padding_px=10) for i in range(len(detections))]
            embeddings = novic_decoder.embed_crops(crops)

            save_detection_debug_frame(det_viz_dir, idx, image_rgb, boxes, confs, list(masks))

            detection_list = fusion.build_detection_list_from_frame(
                boxes=boxes,
                masks=masks,
                confs=confs,
                embeddings=embeddings,
                pixel_indexes=view_dataset.pixel_index_mappings[idx],
                pixel_indexes_mask=view_dataset.pixel_index_masks[idx],
                image_name=name,
            )
            per_frame_detection_lists.append(detection_list)

        objects, _ = fusion.get_instances(view_dataset, per_frame_detection_lists, obj_min_detections=1)
        print(f"Fused {len(objects)} instances")
        assert len(objects) > 0, "Expected at least one fused 3D instance"

        tag_instances(objects, novic_decoder, merge_similarity_threshold=0.9)
        for obj in objects:
            assert "class_name" in obj and isinstance(obj["class_name"], str)
            assert "semantic_embedding" in obj
            print(f"  instance {obj['class_id']}: '{obj['class_name']}' (novic_score={obj.get('novic_score', 0):.3f}, "
                  f"{obj['num_detections']} detections, {len(obj['indexes'])} points)")

        # build_static_scene_graph_safe() defends against DoVSG's confirmed
        # mutual-support-cycle hang (see graph_builder.py's module docstring
        # and README "Known limitation") by retrying with the most
        # spatially-overlapping instance dropped each time it times out,
        # falling back to a flat graph if it's still stuck.
        scene_graph, node_embeddings = build_static_scene_graph_safe(
            view_dataset=view_dataset,
            instance_objects=objects,
            scene_graph_generation_root=SCENE_GRAPH_GENERATION_ROOT,
            timeout_s=60,
        )

        relations = scene_graph_to_relations(scene_graph)
        print(f"Scene graph: {len(scene_graph.object_nodes)} nodes, {len(relations)} relations")
        assert len(scene_graph.object_nodes) >= 1  # at least the root
        # <= not == : build_static_scene_graph_safe() may have dropped one or
        # more instances to break a mutual-support cycle (see its warnings
        # above), so fewer embeddings than the original fused-instance count
        # is expected and fine — the point is that it didn't hang or crash.
        assert len(node_embeddings) <= len(objects)

        pkl_path, json_path = save_scene_graph(Path(tmp) / "scene_graphs", "freiburg_slice_test", scene_graph, objects, node_embeddings)
        assert pkl_path.exists() and json_path.exists()

        novic_decoder.close()
        print("Integration test passed.")


if __name__ == "__main__":
    main()
