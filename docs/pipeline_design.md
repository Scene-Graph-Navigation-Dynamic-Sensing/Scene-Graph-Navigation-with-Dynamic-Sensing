# Static Open-Vocabulary Scene Graph Pipeline — Design Plan

> **Implementation notes (post-approval):** three things changed from the
> plan below once real assets were found while implementing:
> 1. A real NOVIC checkpoint turned out to already exist at
>    `/home/ravindu/SceneGraphNav/external/novic/outputs/ovod_20240610_105233/ovod_chunk0899_20240612_005748.train`
>    (~11,897-noun vocabulary), plus a working reference integration at
>    `/home/ravindu/rashmika/Novic/YoloE-26/yoloe_novic_pipeline.py` — so
>    `semantic_decoding/novic/novic_decoder.py` wraps the real `NOVICModel`
>    (from that repo's `infer.py`) directly instead of the stub described
>    below. Its `embed_images()` already produces the embedding, so no
>    separate `clip_encoder.py` was needed — `embed_crops()`/
>    `decode_embeddings()` are split methods on the same class instead
>    (cheap per-frame embed vs. expensive once-per-instance decode).
> 2. `SceneGraphProcesser.build_scene_graph()` turned out to only ever read
>    `indexes`/`class_name`/`class_id` off each instance (recomputing all
>    geometry from `view_dataset` itself) — no `bbox`/`clip_ft` coupling —
>    so `scene_graph/builder/graph_builder.py` calls it unmodified rather
>    than needing a new `relation_builder.py`. `ObjectNode` carries no
>    embedding field, so each node's NOVIC embedding is tracked in a
>    separate `{node_id: embedding}` side-table built alongside the graph
>    and persisted together in `scene_graph/storage/serializer.py`.
> 3. `object_fusion/instance_fusion.py` subclasses DoVSG's `InstanceProcess`
>    directly (reusing `compute_spatial_similarities`/`compute_visual_similarities`/
>    `merge_detections_to_objects`/etc. unchanged) rather than a from-scratch
>    port — only `gsam2_obs_to_detection_list()` and the top of
>    `get_instances()` needed replacing, since those are the only parts
>    coupled to GroundingDINO+SAM2's on-disk pickle format. It also applies
>    the same `pytorch3d`-missing fallback `Scene-Graph-Generation/main.py`
>    already monkeypatches onto `compute_3d_iou_accuracte_batch` — confirmed
>    needed, since `pytorch3d` isn't installed in the `scene_graph` conda env.
>
> **Confirmed by running `tests/integration/test_static_pipeline_freiburg.py`
> end-to-end** (real YOLOE-seg-pf detection + fusion + tagging on a slice of
> the existing `freiburg1_desk` capture): perception through instance tagging
> all work correctly. One real finding from that run: `build_scene_graph()`
> (reused unmodified) can loop indefinitely given near-duplicate,
> spatially-coincident instances that end up mutually "on" each other — see
> the README's "Known limitation found during testing" section.

> The rest of this document is the plan as originally approved.

## Context

`Scene-Graph-Navigation-with-Dynamic-Sensing` is currently an empty scaffold (folders + `.gitkeep` only, README has just a title). The goal is to build a **static** scene-graph pipeline: RGB video walkthrough in → open-vocabulary 3D scene graph out. Dynamic updates (`scene_graph/updater`, `dynamic_updates/`), navigation, and task planning are explicitly deferred — this plan only covers perception → naming → 3D fusion → static graph.

Two existing codebases are the source of reusable logic:
- **`Scene-Graph-Generation`** (DoVSG fork): working multi-view 3D fusion + scene-graph-from-alpha-shapes pipeline (`dovsg/memory/view_dataset.py`, `dovsg/memory/instances/instance_process.py`, `dovsg/memory/scene_graph/{scene_graph_processer,graph}.py`), GPU pose estimation via DROID-SLAM (`dovsg/scripts/pose_estimation.py`, backed by the vendored `DROID-SLAM/` repo), and a NOVIC toolkit (`tools/novic/`). Its **`main_live.py`** is the most important reuse target: it *already* implements DroidCam-app live capture, Depth Anything V3 inference (`DepthAnything3.from_pretrained('depth-anything/DA3-BASE')`, lines 666/779-791/1145-1153), DROID-SLAM pose invocation (`run_pose_estimation()`, line 289; `save_droidslam_inputs()`, line 278), and a per-frame debug-visualization convention (`save_perception_debug_outputs()`, line 186; `build_mask_overlay`/`build_mask_contour_overlay`, lines 174-184) that saves numbered inspection images per stage. We adapt this file as the orchestration base rather than writing a new entry point from scratch — only its perception step (`run_perception_pipeline()`, which currently uses YOLOWorld + a fixed class list + CLIP-class-matching) gets swapped for YOLOE-seg-pf + NOVIC.
- **`/home/ravindu/rashmika/dtd_pipeline/pipeline.py`**: the "tag at birth, not per-frame" philosophy — class-agnostic detector → CLIP crop embedding → NOVIC decode → confidence-gated re-tagging. We adopt this efficiency pattern but apply it per **3D instance** instead of per 2D track.

User decisions from clarification, plus a target architecture diagram and further refinements supplied afterward:
- Input: a **DroidCam app stream** (phone camera over the network, e.g. `http://<phone-ip>:4747/video`) — the exact `--source` mode `main_live.py` already supports. Depth comes from **Depth Anything V3** (already integrated and working in `main_live.py`, not a fallback/unverified path — see reuse note above). Camera pose comes from **DROID-SLAM** (see module 1 below), fed by the RGB frames plus the Depth Anything depth maps, exactly as `main_live.py::run_pose_estimation(pose_estimator="droidslam")` already does. A stereo camera (real depth) will replace the monocular depth estimator later — the depth module must be swappable, and DROID-SLAM's `track()` call keeps working with either depth source.
- **Every stage must produce an inspectable visualization**, not just the final scene graph — depth maps, detections/masks, NOVIC tags, fused 3D instances, and the graph itself each get a saved visual artifact so results can be checked stage-by-stage rather than only at the end.
- Detection **and** segmentation: run **YOLOE-26x-seg-pf prompt-free** (`Scene-Graph-Generation/tools/yoloe-26x-seg-pf.pt`, already present) with **no text prompt / class list** — it emits boxes + instance masks in one shot, no separate SAM2 refinement stage needed. This replaces DoVSG's original GroundingDINO+SAM2 text-prompted detector and collapses what was going to be two pipeline stages (detect, then segment) into one.
- The supplied diagram's target end-state also has camera pose coming from **sensor fusion of stereo camera + IMU + LiDAR**, feeding a shared "global location of the object" step that back-projects into a voxel map, and a semantic decoder whose **NOVIC embedding is carried through as a node attribute in the scene graph** (not just a text label) — this matters for object identity/matching later. IMU/LiDAR/stereo aren't available yet, so the pose module in this build phase is RGB + Depth-Anything fed into DROID-SLAM, built behind the same interface so it's a drop-in swap once those sensors exist. The diagram's "Update Changed Objects" loop, Task Planning (LLM), and Navigation Waypoint stages map to the already-scaffolded (currently empty) `scene_graph/updater/`, `task_planning/`, and `navigation/` folders — still out of scope for this plan.

## Non-goals (explicitly out of scope)

- `scene_graph/updater/`, `dynamic_updates/` — dynamic re-scanning/change detection. Left as empty stub packages.
- `navigation/`, `task_planning/` — not touched.
- Real stereo-depth integration — module built swappable, but only the monocular estimator is implemented now.

## Pipeline overview

```
RGB frames ──┬─→ [Depth Estimation]  (Depth Anything V3/V2) ─→ per-frame depth map
             │                                                        │
             ├─→ [Pose Estimation]   (DROID-SLAM, RGB + depth prior;  │
             │    future: stereo+IMU+LiDAR sensor fusion) ─→ per-frame camera pose
             │                                                        │
             └─→ [Detection + Segmentation]  (YOLOE-26x-seg-pf,
                  prompt-free — boxes + masks in one pass, no class list)
                                   │
      depth + pose + masks + boxes ┴──────────────────────────────┐
                                                                    ▼
                                                    [Object Fusion] (multi-view voxel
                                                     back-projection = "global location of
                                                     the object" + spatial/visual clustering
                                                     → persistent 3D instances in a Voxel Map)
                                                                    │
                                              best-view crop per instance
                                                                    ▼
                                          [Semantic Decoding] CLIP encode → NOVIC decode
                                          → open-vocab free-form name + NOVIC embedding
                                          → [Label Mapper] canonicalizes the text label;
                                          the embedding is kept as-is and attached to the node
                                                                    │
                                                                    ▼
                                          [Scene Graph Builder] alpha-shape spatial
                                          relations (on/inside/near) → SceneGraph(nodes carry
                                          label + NOVIC embedding, edges carry relation type)
                                                                    │
                                                                    ▼
                                          [Scene Graph Storage] → outputs/scene_graphs/*.pkl
                                          + outputs/visualizations/*.pdf
```

Key departures from DoVSG's original loop:
1. DoVSG re-runs open-vocab classification (CLIP-vs-text-list) on every detection every frame. We instead **tag once per fused 3D instance**, using the highest-confidence view (mirrors `dtd_pipeline/pipeline.py::_accumulate_and_maybe_tag`'s "best frame by detection confidence" logic) — cuts NOVIC calls from O(frames × detections) to O(instances).
2. DoVSG's graph nodes only carry a class label. Nodes here also carry the **NOVIC embedding vector** for the instance, so later work (task planning queries like "find the keyboard", or a future dynamic-update stage matching re-observed objects) can do embedding similarity lookups without re-running perception.

## Per-stage visualization requirement

Every stage writes a saved visual artifact to `outputs/visualizations/<stage>/`, not just the final graph — mostly by extracting and adapting patterns `main_live.py` already has, rather than inventing a new visualization approach per module:

| Stage | Artifact | Source pattern to adapt |
|---|---|---|
| Depth estimation | Colorized depth map per frame | New (simple `cv2.applyColorMap` on normalized `depth_meters`) — `main_live.py` doesn't currently save one |
| Pose estimation | Camera trajectory plot (3D camera-position path) after a full capture | New (small Open3D `LineSet`/matplotlib 3D plot over `poses/*.txt`) — not currently saved anywhere |
| Detection + segmentation | Per-frame boxes/masks/contours + a summary montage, **labeled only with objectness confidence** (e.g. `obj 0.87`) — no class name exists yet at this stage | `save_perception_debug_outputs()`, `build_mask_overlay()`, `build_mask_contour_overlay()` (`main_live.py:174-276`) — reused directly, fed YOLOE-seg-pf's output; the `classes[class_ids[idx]]` label lookup those functions currently do gets removed since there's no class list |
| Semantic decoding (NOVIC) | Same per-frame overlay, now labeled with **the object's tag as decoded by NOVIC** (then canonicalized by the label mapper) — this is the first and only place a semantic name appears | Same `save_perception_debug_outputs()` pattern, label source swapped from `classes[class_ids[idx]]` to the NOVIC decoder's output |
| Object fusion | Interactive 3D viewer + a saved screenshot, colorable by instance/class/CLIP-similarity | `dovsg/memory/instances/visualize_instances.py::vis_instances()` — reused as-is, it already has `color_by_instance`/`color_by_class`/`color_by_clip_sim` toggles |
| Scene graph | Graph PDF render | `dovsg/memory/scene_graph/graph.py::SceneGraph.visualize()` — reused as-is (already planned in module 9 below) |

## Module-by-module design

### 1. `perception/pose_estimation/` (new subfolder — not in original scaffold, flagging for approval)
The scaffold has no pose/odometry folder. Pose estimation is sensing, not fusion, so it belongs under `perception/`, sitting alongside `depth_estimation/`.

**Use DROID-SLAM**, and reuse it almost verbatim — `main_live.py` already drives it end-to-end, this isn't a new integration:
- `droid_slam_pose_estimator.py`: thin wrapper around the existing `run_pose_estimation(dataset_dir, pose_estimator="droidslam")` (`main_live.py:289-317`), which shells out to `dovsg/scripts/pose_estimation.py` (itself calling `Droid.track(tstamp, image, depth, intrinsics)` at `DROID-SLAM/droid_slam/droid.py:61`, then `Droid.terminate()` for the globally-optimized trajectory) and copies the resulting poses into `dataset_dir/poses/`. Depth input to `track()` comes from `save_droidslam_inputs()` (`main_live.py:278-287`), which is exactly the Depth-Anything-V3-map-as-depth-prior pattern already described above — no new conversion logic needed, just call these functions from our module instead of duplicating them.
- Output: 4x4 pose matrix per frame in `data/processed/<scene>/poses/`, same file layout `main_live.py` already produces.
- Interface: `estimate_trajectory(dataset_dir) -> None` (writes poses to disk, matching the existing subprocess-based contract) — kept swappable via the `pose_backend` config key so a future stereo+IMU+LiDAR fusion module can take this slot without touching downstream code.
- Dependency note: DROID-SLAM requires its compiled CUDA extensions (`droid_backends`, `lietorch`, `pytorch_scatter` under `Scene-Graph-Generation/{DROID-SLAM,lietorch}`) and a GPU. `run_pose_estimation()` already raises a clear error if `import droid` fails, and `main_live.py` supports falling back to `pose_estimator="cpu_icp"` (→ `dovsg/scripts/cpu_pose_estimation.py`) — reuse that same fallback switch rather than building a new one.

### 2. `perception/depth_estimation/`
- `depth_anything_estimator.py`: extract the already-working Depth Anything V3 inference block straight out of `main_live.py` (model load at lines 779-791 via `DepthAnything3.from_pretrained('depth-anything/DA3-BASE')`, per-frame inference at lines 1145-1153: `depth_model.inference([pil_img])` → `prediction.depth[0]`) into a standalone module instead of leaving it inline in the capture loop.
- Defines a small `DepthEstimator` interface (`estimate(rgb) -> depth_map, is_metric: bool`) so a future `stereo_depth_estimator.py` can be dropped in without touching downstream code.

### 3. `perception/detection/`
- `yoloe_seg_detector.py`: wraps YOLOE-26x-seg-pf (`yoloe-26x-seg-pf.pt`) run **prompt-free** — no text prompt, no class list. A single forward pass returns boxes + instance masks + objectness confidence per detection: `[{bbox, mask, conf}, ...]`. **This module never assigns a semantic label** — no class name, no category, nothing beyond geometry + "this is an object" confidence. It replaces both the two-stage "class-agnostic detector → SAM2 refine" design from an earlier draft of this plan, and `main_live.py::run_perception_pipeline`'s current `YOLOWorld`-with-a-class-list detector (which *did* assign a label straight from its fixed class list — that behavior is explicitly not carried over). The object's tag comes exclusively from NOVIC (module 5), downstream of this.

### 4. `perception/segmentation/`
- Not required for the v1 static pipeline — YOLOE-seg-pf already produces usable per-object masks directly. Kept as an empty stub package now; reserved for an optional future `sam2_refiner.py` (box-prompted SAM2, adapting `dovsg/perception/models/mygroundingdinosam2.py`'s SAM2 usage) if YOLOE-pf masks turn out too coarse on a specific scene — not built unless that need actually shows up.

### 5. `semantic_decoding/novic/`
- `clip_encoder.py`: adapt `dtd_pipeline/tagging/clip_encoder.py` (crop → padded → CLIP/SigLIP embedding). Reuse `open_clip` model config pattern from `dtd_pipeline/config.yaml` (`ViT-B-16-SigLIP` / `webli`).
- `novic_decoder.py`: adapt `dtd_pipeline/tagging/novic_decoder.py` + `Scene-Graph-Generation/tools/novic/{infer.py,embedding_decoder.py}` for the actual beam-search decode from embedding → free-form noun phrase. Returns `(label: str, score: float, embedding: np.ndarray)` — **the embedding is returned alongside the label**, not discarded, since the target design attaches it to the object's scene-graph node (see module 8). Needs a NOVIC checkpoint — **none exists anywhere in the workspace currently** (checked `models/novic_weights/` and searched the filesystem); must be obtained/placed at `models/novic_weights/` before this module is usable. Until then, keep dtd_pipeline's stub-decoder fallback (`novic_checkpoint: ""` → dummy label + zero-vector embedding) so the rest of the pipeline is testable.
- Tag-once-per-instance scheduling logic lives in `object_fusion/` (below), not here — this module is a pure `embedding -> (label, score, embedding)` function, reused by both the "provisional first-view tag" and "confirmed best-view tag" calls.

### 6. `semantic_decoding/label_mapper/`
- `label_mapper.py`: canonicalizes NOVIC's free-form text (e.g. "brown wooden office chair" vs "chair") into a stable node label for graph consistency — simple approach: lemmatize + strip leading adjectives, keep the head noun, optionally merge near-duplicate labels within a scene via embedding similarity threshold. This module doesn't exist in either reference codebase; new logic, kept intentionally simple (no external taxonomy/ontology dependency) since scope is a first working static graph.

### 7. `object_fusion/`
The core 3D-fusion adaptation of DoVSG's `dovsg/memory/view_dataset.py` + `dovsg/memory/instances/instance_process.py`:
- `view_dataset.py`: adapt `ViewDataset` — per-frame RGB+depth+mask+pose → global voxel map (`voxelize`, `point_to_voxel`, `calculate_all_global_voxel_indexes_and_colors`). Same voxel resolution config knob as original.
- `instance_fusion.py`: adapt `InstanceProcess.get_instances` / `merge_detections_to_objects` / `compute_spatial_similarities` + `compute_visual_similarities` (drop `compute_text_similarities` — no text classes anymore) to cluster per-frame class-agnostic detections into persistent `MapObjectList` 3D instances using spatial IoU + CLIP visual similarity only (mirrors the "Aggressive Object Merging" tuning already validated in `Scene-Graph-Generation/pipeline.md`: `sim_threshold≈0.55`, `vis_weight` up-weighted vs `spatial_weight`).
- `instance_tagger.py`: new glue — for each fused instance, pick its highest-detection-confidence view's CLIP embedding (reuse the "best-frame" selection logic from `dtd_pipeline/pipeline.py::_accumulate_and_maybe_tag`, lines ~262-271) and call `semantic_decoding/novic` once to get `(label, score, embedding)`, then `label_mapper` to canonicalize the label text only — the embedding itself is stored unmodified on the instance so it can be attached to the corresponding scene-graph node in module 8. Optionally reuse the *provisional label after first observation* idea from dtd_pipeline for fast feedback during a live capture.
- Reuses the CPU-safe 3D-IoU fallback pattern from `Scene-Graph-Generation/main.py`'s `patched_compute_3d_iou_accuracte_batch` monkeypatch (falls back to axis-aligned IoU when `pytorch3d` isn't installed).

### 8. `scene_graph/builder/`
- `graph.py`: adapt `dovsg/memory/scene_graph/graph.py` (`ObjectNode`, `SceneGraph`), with one addition — `ObjectNode` gains a `semantic_embedding: np.ndarray` field alongside `node_class`. **Both fields trace back to NOVIC**: `node_class` is the label_mapper-canonicalized text NOVIC decoded for that instance (module 7's `instance_tagger.py` output), and `semantic_embedding` is the raw NOVIC embedding for the same instance — no other source ever populates either field. This is the concrete implementation of the diagram's "embedding from NOVIC for specific objects" requirement: the scene graph carries both NOVIC's name and its embedding for every object node. No dynamic-update methods needed yet.
- `relation_builder.py`: adapt `dovsg/memory/scene_graph/scene_graph_processer.py::build_scene_graph` (alpha-shape based `on`/`inside` contact-point relations, `root_node_class="floor"`), using the "forgiving" threshold already validated for noisy CPU scans (`alpha_shape_overlap_rate_threshold≈0.1`, from `pipeline.md` refinements). This is the **static** build path only — `update_scene_graph` (the dynamic variant) is intentionally not ported.

### 9. `scene_graph/storage/`
- `serializer.py`: save/load `SceneGraph` + `MapObjectList` to `outputs/scene_graphs/<scene>.pkl`, plus a human-readable JSON summary (node labels + relations, embeddings omitted from the JSON for readability) for quick inspection. The pickle keeps each node's NOVIC embedding intact, so a future dynamic-update stage can diff against this snapshot without re-running fusion from scratch.
- `visualizer.py`: adapt `SceneGraph.visualize()` (graphviz/PDF render) into `outputs/visualizations/`.

### 10. `scene_graph/updater/`
Left empty (`__init__.py` only, or untouched) — explicitly deferred per user request.

### 11. `configs/`
- `pipeline.yaml`: merge `dtd_pipeline/config.yaml`'s structure (detector/CLIP/NOVIC/device sections) with DoVSG's fusion/graph knobs (voxel `resolution`, `sim_threshold`, `vis_weight`/`spatial_weight`, `alpha_shape_overlap_rate_threshold`, `min_height`, `occ_threshold` where still relevant). Includes a `depth_backend: depth_anything_v3` / `pose_backend: droid_slam` switch so both are swappable later (e.g. `pose_backend: cpu_icp` as the documented fallback if DROID-SLAM's CUDA extensions aren't available).

### 12. `scripts/`
- `run_static_scene_graph.py`: adapts `main_live.py` directly as the base rather than writing a new entry point — keeps its DroidCam `--source` capture loop, environment/session directory management, Depth Anything V3 inference loop, and `run_pose_estimation()`/`save_droidslam_inputs()` DROID-SLAM calls untouched. The only real surgery is inside `run_perception_pipeline()`: swap `YOLOWorld` + fixed class list + CLIP-class-matching for `perception/detection/yoloe_seg_detector.py` (prompt-free) → `semantic_decoding/novic` (tag) → `semantic_decoding/label_mapper` (canonicalize), and call each stage's visualizer (previous section) as it runs. Downstream `instance_process`/`scene_graph_processer` calls stay structurally the same, pointed at the new `object_fusion/`/`scene_graph/builder/` modules instead of `dovsg/memory/...` directly.

### 13. `models/`
- `models/novic_weights/` — NOVIC checkpoint (user must supply/download; none found in workspace).
- `models/checkpoints/` — YOLOE-seg-pf weights, SAM2 checkpoint, Depth Anything V3 weights (auto-downloaded via HF on first run, cached here).

### 14. `tests/`
- `tests/unit/`: one test per module using tiny synthetic fixtures — depth estimator output shape/range on a sample image; detector returns boxes on a sample image; NOVIC decoder returns a string given a dummy embedding (stub mode); label_mapper canonicalizes a couple of known synonym pairs; instance fusion merges two overlapping same-class detections across 2 synthetic frames into 1 instance; graph builder produces a "cup on desk" edge on a synthetic point-cloud fixture.
- `tests/integration/`: run the full `run_static_scene_graph.py` on a short existing capture (a 10-20 frame slice of `Scene-Graph-Generation/data_example/freiburg1_desk_raw` or a fresh short phone capture) end-to-end; assert the output graph has a plausible node count and at least one `on`/`inside` edge.

## Risks / caveats to flag now

- **Monocular depth is still scale/consistency-limited** even with DROID-SLAM's optimization — Depth Anything's per-frame metric estimates aren't guaranteed perfectly consistent across views, which can still leave some residual error in the fused 3D map. DROID-SLAM's bundle adjustment reduces this substantially versus plain ICP, but it's not eliminated until the stereo camera is wired in later (per user's own note) — the depth module interface is kept swappable specifically for that reason.
- **DROID-SLAM's CUDA build must be confirmed working** in this environment (compiled `droid_backends`/`lietorch`/`pytorch_scatter` extensions under `Scene-Graph-Generation/`) — `main_live.py::run_pose_estimation()` already raises a clear error and documents the `pose_estimator="cpu_icp"` fallback for exactly this case, so worst case we reuse that switch rather than being blocked.
- **NOVIC checkpoint is missing** — nothing under `models/novic_weights/` and no checkpoint found anywhere on the filesystem. Pipeline will run in dtd_pipeline's stub-decoder mode until one is supplied.
- **`pytorch3d`** (used for accurate 3D IoU in DoVSG) may not be installed — reuse the existing axis-aligned-IoU fallback rather than adding it as a hard dependency.

## Suggested implementation order (for the follow-up build phase)

Each phase below is only considered done once its stage's visualization artifact (per the table above) has actually been produced and eyeballed on a real DroidCam capture — not just once the code runs without errors.

1. Perception: depth + pose (DROID-SLAM) + YOLOE-seg-pf detection/segmentation — check the colorized depth frames, the trajectory plot, and the boxes/masks montage before moving on.
2. Semantic decoding: CLIP encoder + NOVIC decoder + label mapper — check the per-frame overlay now shows NOVIC's decoded names instead of a fixed class list.
3. Object fusion: multi-view 3D instance clustering — open the interactive `vis_instances()` viewer and confirm object count/positions look right.
4. Scene graph: builder + storage — check the graph PDF's nodes/edges look plausible (e.g. "keyboard on desk").
5. `scripts/run_static_scene_graph.py` end-to-end orchestration + `configs/pipeline.yaml` + tests.

## Verification plan

- Unit tests listed above, runnable via `pytest tests/unit`.
- Integration test producing an actual `outputs/scene_graphs/*.pkl` + PDF viz from a short real DroidCam capture, manually sanity-checked (object count, at least one correct spatial relation) before calling any phase "done".
- Because every stage saves its own visualization, sanity-checking a run doesn't require re-running the whole pipeline to debug one stage — inspect that stage's folder under `outputs/visualizations/` directly.
