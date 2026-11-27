# Full pipeline: video feed → 3D scene graph

How the complete pipeline is implemented, how to set it up, and how to run it
on a live camera, a video file, or your own dataset.

```bash
conda activate scene_graph
python scripts/run_static_scene_graph.py --source data/raw/<your_dataset> --scene_name <name>
```

## Pipeline at a glance

| # | Stage | What runs | Code |
|---|---|---|---|
| 1 | Video feed | DroidCam stream, webcam, video file, or dataset folder | `perception/frame_source.py` |
| 2 | Metric depth | Depth Anything V3 `DA3METRIC-LARGE`, converted to metres with the camera focal length, optionally aligned to reference depth | `perception/depth_estimation/depth_anything_estimator.py`, `capture_frames()` in the runner |
| 3 | Camera pose | DROID-SLAM on RGB + metric depth | `perception/pose_estimation/droid_slam_pose_estimator.py` → `Scene-Graph-Generation/dovsg/scripts/pose_estimation.py` |
| 4a | Gravity alignment | World rotated so the dominant horizontal plane is z-up, lowest surface at z = 0 | `scene_graph/builder/gravity_alignment.py` |
| 4 | Voxel map | Every frame's depth back-projected with its pose into one 1 cm voxel grid | `object_fusion/view_dataset.py` (DoVSG `ViewDataset`) |
| 5 | Detection + tracking | YOLOE-26-seg-**pf** (prompt-free) + BoT-SORT: boxes, masks and a persistent track id, no class list | `perception/detection/tracked_yoloe_detector.py` |
| 6 | Crop → NOVIC embedding | Padded crop (15% of the box) embedded by NOVIC's image encoder, every detection on every frame | `semantic_decoding/novic/crop_utils.py`, `novic_decoder.py::embed_crops` |
| 7 | Cosine-similarity fusion | Detections merged across frames into 3D instances by `0.7 × cosine(NOVIC embeddings) + 0.3 × voxel overlap`; each object keeps a running-average embedding, decoded to a name when a track that fed it ends | `object_fusion/instance_fusion.py` |
| 8 | Instance naming | The latest track-end name; objects changed after it are re-decoded from their final average | `object_fusion/instance_tagger.py`, `semantic_decoding/label_mapper/` |
| 9 | Relations + scene graph | `on` / `inside` / `belongs` from alpha-shape contact | `scene_graph/builder/graph_builder.py` |
| 10 | Save | Graph + instances + NOVIC embeddings, plus a PDF render | `scene_graph/storage/` |

Entry point: `scripts/run_static_scene_graph.py`. (`main.py` is a separate 2D
tool — YOLOE + NOVIC tagging on still images, no depth, pose, or 3D.)

### Two deliberate differences from the stage order as first described

**Depth, pose and the voxel map run before detection.** Fusion (stage 7) needs
each detection's 3D location, which needs depth, a pose and the map. The data
flow is the same; only the execution order differs.

**The name is decoded once per track, not once per crop.** Every crop *is*
embedded by NOVIC (stage 6), and each object keeps a running average of those
embeddings (stage 7). The expensive beam-search *decode* into a name runs only
when a track ends, on the average of the object that track fed, so the name
reflects every view of the object rather than whichever one came first.

**YOLOE's own labels are never read.** The `-pf` checkpoint runs prompt-free;
the detector returns only box, mask and objectness confidence
(`yoloe_seg_detector.py`). Until stage 8 every detection is the placeholder
`"object"`, and fusion gives text similarity zero weight.

## 1. One-time setup

Everything runs in the `scene_graph` conda env (Python 3.10). The system
Python is 3.12 and lacks the dependencies.

```bash
conda activate scene_graph
pip install unidecode            # NOVIC's infer.py needs it
```

### DROID-SLAM's CUDA extensions

DROID-SLAM lives in the sibling checkout
`/home/ravindu/SceneGraphNav/external/Scene-Graph-Generation` and needs three compiled
extensions. They must be built **with the env's Python 3.10**, using a
**CUDA 13 `nvcc`** to match torch 2.13+cu130. The system `nvcc` is 12.0 and the
build refuses it.

```bash
conda activate scene_graph
pip install "nvidia-cuda-nvcc~=13.0" ninja numpy-quaternion
export CUDA_HOME=$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cu13
export PATH=$CUDA_HOME/bin:$PATH
nvcc --version                                  # must say 13.x

cd /home/ravindu/SceneGraphNav/external/Scene-Graph-Generation/DROID-SLAM && rm -rf build && python setup.py install
cd /home/ravindu/SceneGraphNav/external/Scene-Graph-Generation/lietorch   && rm -rf build && python setup.py install
MAX_JOBS=$(nproc) pip install --no-build-isolation torch-scatter
```

Check (import `torch` first, or the extensions can't find `libc10.so`):

```bash
cd /home/ravindu/SceneGraphNav/external/Scene-Graph-Generation
python -c "import torch, sys; sys.path.append('./DROID-SLAM/droid_slam'); import droid; print('ok')"
```

`CUDA_HOME` only matters while building. Runs need nothing extra.

### GPU

This machine has two 16 GB GPUs, and GPU 0 is often partly used by other jobs.
Pin a run to a free one with `CUDA_VISIBLE_DEVICES=1`. The depth model is
released before DROID-SLAM starts, so the heaviest single stage is NOVIC's
ViT-H-14-378 encoder.

## 2. Running

### Your dataset

Put it in `data/raw/<name>/`, following **`data/raw/README.md`**:

```
data/raw/<name>/
├── rgb/          frames (sorted by name)   — or one video file instead
├── camera.yaml   fx fy cx cy [width height] [depth_scale]
├── depth/        optional reference depth, same stems as rgb/
└── poses/        optional ground-truth camera-to-world poses, same stems
```

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/run_static_scene_graph.py \
    --source data/raw/<name> --scene_name <name>
```

- With `depth/`, set `depth.align: global` to rescale DA3's depth onto your
  sensor's scale.
- With `poses/`, the DROID-SLAM trajectory is scored automatically against
  them (`outputs/scene_graphs/<name>_pose_eval.json`).
- `source.stride` keeps every Nth frame, and `source.max_frames` caps the count
  (0 = all).

`data/raw/tum_fr1_desk/` is a complete worked example in this layout (TUM RGB-D
freiburg1_desk, with depth and ground-truth poses).

### Live DroidCam

Set `source.uri` to the phone's stream (`http://<phone-ip>:4747/video`) and the
`camera:` block to the phone's calibrated intrinsics, then run without
`--source`. Pan slowly: `capture_interval` takes one frame every 0.3 s, and
DROID-SLAM needs heavy overlap between consecutive frames.

### A video file

`--source path/to/video.mp4`. Frames are sampled by `source.stride` (every Nth
decoded frame), not by wall-clock time. Intrinsics come from the `camera:`
block, or put the video in a folder with a `camera.yaml` beside it.

### A dataset that already has poses (skip depth + DROID-SLAM)

`--dataset_dir data/processed/<scene>` reads a prepared DoVSG-layout dataset
(`rgb/ point/ mask/ poses/ metadata.json`) and runs stages 4–10 only. Use this
for 3RScan: its scans are sparse keyframes that DROID-SLAM cannot track (see
Limitations), so use the shipped poses via `scripts/prepare_3rscan_dataset.py`.

## 3. How each stage is implemented

### Stage 1 — Frame source (`perception/frame_source.py`)

One `FrameSource` class hides the input type:

- **Stream** (URL or webcam index): sampled by wall-clock time, since a live
  source produces frames whether or not they are read. The first 5 frames are
  discarded while exposure settles.
- **Video file**: sampled by frame count. The old capture loop sampled files by
  wall-clock time too, so *which* frames it kept depended on decode speed.
- **Folder**: sorted `rgb/` images. Each `Frame` carries its source name, so
  `depth/` and `poses/` files with the same stem can be matched to it.

Frames are resized to `source.target_width` (aspect kept), and the intrinsics
are rescaled to match (`CameraIntrinsics.scaled_to`). Kept frames are
renumbered contiguously. `capture.json` in the processed dataset records the
map from each processed index back to its source frame.

### Stage 2 — Metric depth (`depth_anything_estimator.py`)

**The key fix.** `DA3METRIC-*` checkpoints predict depth for a canonical 300 px
focal length:

```
metric_depth = raw_output × focal_px / 300
```

`focal_px` is the focal length at the resolution the network actually ran at.
DA3 resizes internally, so it is rescaled by `depth_width / image_width`. The
upstream API applies this only inside its *nested* model, so the standalone
metric checkpoint returned unscaled depth. `estimate(rgb, fx, fy)` now applies
it (`da3_metric_scale()`).

| scene | GT/predicted scale before | after |
|---|---|---|
| 3RScan b05fddcf | 1.500 | **0.975** |
| TUM freiburg1_desk | 1.483 | **1.093** |

The remaining ±10% is genuine per-scene model error. That is what alignment is for:

- `depth.align: none`: trust the metric model. This is the only option for a
  live stream, which has no reference depth.
- `depth.align: global`: sample `align_sample_frames` frames spread over the
  dataset, take `median(reference) / median(predicted)` on each, and multiply
  every frame by the median of those ratios.

The pipeline refuses relative-depth checkpoints (`DA3-BASE` etc.): without
metres, DROID-SLAM's depth prior and the voxel map would silently get the wrong
scale. Depth is upsampled with nearest-neighbour, because cubic interpolation
across an object edge invents surfaces between foreground and background.

### Stage 3 — Camera pose (`droid_slam_pose_estimator.py`)

A thin wrapper around `dovsg/scripts/pose_estimation.py`. It stages
`depth/*.npy` (millimetres) and `calib.txt`, runs DROID-SLAM as a subprocess
(`Droid.track` per frame with the depth prior, then `terminate()` for global
bundle adjustment), and copies the resulting camera-to-world poses into
`poses/`. The depth model is freed first, because both share the GPU. The
pipeline stops with an error if the pose count doesn't match the frame count,
because the voxel map needs a pose for every frame (`pose.stride` must stay 1).

### Checking the camera pose against ground truth

When the dataset has a `poses/` folder, the run scores DROID-SLAM against it
automatically and writes three things:

- **`<scene>_pose_eval.json`** — ATE (Umeyama-aligned, with the scale solved
  and with it fixed at 1) and frame-to-frame RPE.
- **`<scene>_pose_eval_per_frame.csv`** — one row per frame:
  `frame, gt_x/y/z, est_x/y/z, translation_error_m, rotation_error_deg`.
- **`trajectory_vs_groundtruth.png`** — the two trajectories overlaid in 3D and
  as top-down/side views, plus translation, rotation and per-axis error curves.

The estimate is rigidly aligned onto the ground truth first, because SLAM
recovers the trajectory only up to a global transform: its world origin is the
first camera. The solved **scale** is the interesting number on its own — at
1.0 the reconstruction is correctly metric.

To stop after pose and skip the slow fusion stage:

```bash
python scripts/run_static_scene_graph.py --source data/raw/<name> --pose_only
```

To re-score an existing run without re-running anything:

```bash
python scripts/eval_pose.py --gt <gt_poses_dir> --est data/processed/<scene>/poses \
    --out eval.json --csv eval.csv --plot outputs/visualizations/pose
```

Every run also writes `outputs/logs/<scene>.log`. DROID-SLAM's own progress
bar goes to the console only, so redirect the command
(`> outputs/logs/<scene>_console.log 2>&1`) to capture that too.

### Stage 4a — Gravity alignment (`scene_graph/builder/gravity_alignment.py`)

The relation builder (stage 9) assumes a **z-up world with the floor at
z = 0**: it hard-codes `floor_normal = [0, 0, 1]` and decides "on" by comparing
heights. DROID-SLAM's world is the first camera's frame (x right, y **down**,
z forward). Without this step, "on" is judged along the camera's viewing
direction. On the first TUM run, 17 of 18 objects came out "on floor" in a
desk close-up with no floor in view.

Upstream DoVSG levels the world from floor masks found by a *text-prompted*
GroundingDINO + SAM2 detector. That contradicts this pipeline's prompt-free
design, and it fails when no floor is visible. The replacement needs no
detector:

1. **Prior**: a hand-held camera is roughly upright, so the mean of every
   frame's camera-up axis (−y) in world coordinates approximates up.
2. **Refine**: RANSAC-extract the 5 largest planes from the reconstruction,
   and take the biggest within 45° of the prior. That is the floor or a
   tabletop, both horizontal. The prior alone is not enough: a camera looking
   down at a desk tilts it by its pitch (33° on TUM freiburg1_desk). The first
   version used a 30° tolerance and rejected the desk for exactly that reason.
3. **Level**: rotate that normal onto +z, then shift so the 1st percentile of
   height sits at z = 0.

Every pose is rewritten as `T_align @ pose`. The originals are kept in
`poses_unaligned/`, and the step always recomputes from them, so re-running is
harmless. The transform and diagnostics go to `gravity_alignment.json`. On TUM
the desk became a sharp layer at **z = 0.70–0.80 m** (49% of points within
10 cm, a standard desk height), with the camera 1.25–1.73 m up, pitched 43°
down. Set `scene_graph.gravity_align: false` to skip it for data that is
already z-up.

### Stage 4 — Voxel map (`object_fusion/view_dataset.py`)

DoVSG's `ViewDataset`, reused unchanged. It back-projects every frame's
`point/` cloud with its pose into one global grid at
`object_fusion.voxel_resolution` (1 cm), with inlier filtering. Each pixel gets
a global voxel index, which is how a 2D mask becomes a 3D object.

### Stages 5–6 — Detection, tracking, crops, NOVIC embeddings

YOLOE-26s-seg-pf runs prompt-free on each frame, wrapped in Ultralytics'
**BoT-SORT** tracker (`persist=True`), so every detection carries a persistent
**track id** alongside its box, mask and objectness. YOLOE's own class output is
never read.

Each detection is cropped as a padded bounding box, padded by **15% of the box's
own size** (`crop_padding_frac`, dtd_pipeline's `pad_frac`), so context scales
with the object instead of a fixed pixel count. Mask-zeroed crops were tested
and made NOVIC confidently wrong, because its encoder expects real background.

Every crop is embedded by NOVIC's image encoder on every frame. A track's first
frame is often a poor view (partly out of frame, blurred, far away), so its
embedding is not reused for the rest of the track.

### Stage 7 — Instance fusion and track-end naming (`instance_fusion.py`)

A subclass of DoVSG's `InstanceProcess`. A detection whose track id is already
mapped to an object merges straight into it; the tracker is trusted for
identity. An untracked detection, or a track id's first frame, is matched
against every object fused so far by:

```
score = vis_weight × cosine(NOVIC embeddings) + spatial_weight × voxel overlap
      = 0.7 × visual + 0.3 × spatial            (text weight fixed at 0)
```

It merges into its best match when `score ≥ sim_threshold` (0.55); otherwise it
starts a new object. A new track id therefore continues an object the tracker
lost through occlusion. Cosine alone does not work: crops from one room all sit
in a narrow 0.56–0.91 similarity band, so everything would merge. The spatial
term separates look-alikes that are in different places.

Every merge updates the object's embedding as a running average weighted by
detection count. When a track appears for the last time, NOVIC decodes the
running average of the object it fed, and the label, score and the object's
detection count at that moment are appended to the object's `track_names`.
Every decode is also written to `data/processed/<scene>/track_names.json`.
DoVSG's denoise, merge and filter passes then drop objects seen in fewer than
`obj_min_detections` (3) frames.

### Stage 8 — Instance naming (`instance_tagger.py`)

An object takes its latest `track_names` record, as long as its detection count
is unchanged since that decode. Objects that changed afterwards (merged by
DoVSG's `merge_objects` or by the fragment merge in `object_fusion/postprocess.py`,
which also averages the embeddings) and objects built only from untracked detections are decoded
here from their final running average, in one batch. `label_mapper` then
canonicalizes the text ("brown wooden office chair" → "chair") and merges
near-duplicate labels by embedding similarity (0.9). Each instance stores
`class_name`, `novic_score`, `novic_top_k`, `named_by` (`track_end` or
`final_decode`) and its `semantic_embedding`, the running average.

NOVIC spreads its probability over synonyms, so the top score understates a
real object: the second desk on IMG_6100 scored 0.10 for "desk", with
"bureau" and "escritoire" behind it. The name confidence adds the beam scores
of every alternative whose text embedding (NOVIC's CLIP text tower) is at least
0.8 alike to the top label. On IMG_6100 every real object scored at least 0.25
and every guess ("hashish", "eyebrow", "o ring") at most 0.11. An instance
below `min_name_score` (0.15) is removed along with its tracks, track-end names and running-average
embedding; its voxels stay in the map, so it is still an obstacle. With
`low_score_action: unknown` it stays as `unknown_<n>` instead, keeping the
guess in `novic_guess`. Every JSON node lists its score and top 5 guesses
(`top_k`).

NOVIC's vocabulary has 42,919 nouns and it often picks a rare or overly broad
one: IMG_6100's wallet came out as "billfold" and its office chairs as "office
furniture" (0.85, with "chair" at 0.01). `configs/vocab/indoor_objects.txt`
lists about 250 everyday indoor, office, household and warehouse names. A
NOVIC label that isn't on it is replaced by the listed name whose CLIP text
embedding ("a photo of a <name>") is closest to the object's image embedding,
plus 0.2 × the text similarity between that name and NOVIC's own label, so
"earplug" becomes "earphone" rather than the visually similar "charger".
Labels already on the list are kept, and NOVIC's original label is saved as
`novic_label`. The confidence filter still uses NOVIC's own scores.

A table or desk top is decoded after the clutter on it, since that fills its
crop (IMG_6100's desk came out as "laptop"). An instance with a footprint of at
least 0.8 m² and at least half its points in one 10 cm horizontal slab topping
out between 0.3 and 1.3 m is renamed with the first surface noun in its beam
(desk, table, counter, shelf, ...), or `table` if there is none, and is never
dropped for a low score.

### Stages 9–10 — Relations and scene graph (`graph_builder.py`, `storage/`)

DoVSG's `SceneGraphProcesser.build_scene_graph()`, reused unchanged. It finds
the floor, then derives three relation types:

| relation | meaning | how it is decided |
|---|---|---|
| `on` | supported by | the footprints (alpha shapes) overlap by `alpha_shape_overleaf_rate_threshold`, the parent's base is lower, and the parent's top (95th-percentile height) is within 10 cm below / 15 cm above the child's base (5th-percentile height); or the child stands on the parent's bottom (its base within 10 cm above the parent's base, e.g. a cup in a box); an object with things beneath it but none touching gets no relation (support unknown); with nothing beneath it, it is on the floor |
| `inside` | contained in | `inside_threshold` of the child's points fall inside the parent's convex hull, and the child's base is more than `support_max_penetration_m` (15 cm) below the parent's top; flat items lying on a desk stay `on` |
| `belongs` | part of | the child's name is a part-level noun (handle, drawer, door, knob, button, lid, cover, wheel) attached to its host |

Upstream spells the third one `belong`; the serializer normalizes it to
`belongs`. `belongs` only appears when NOVIC actually names something a
part-level noun, so a scene of whole objects yields only `on` and `inside`. Upstream can loop
forever on two objects that each appear to support the other.
`build_static_scene_graph_safe()` runs it with a timeout, and on a hang drops
the lower-confidence member of the most overlapping pair and retries. Each
node's NOVIC embedding is saved in a side table with the graph, so later
queries ("find the keyboard") can match by embedding.

## 4. Outputs

| Path | Contents |
|---|---|
| `outputs/scene_graphs/<scene>.json` | node labels + relations — open this first |
| `outputs/scene_graphs/<scene>.pkl` | full graph, instances and NOVIC embeddings |
| `outputs/scene_graphs/<scene>_pose_eval.json` | ATE/RPE vs ground truth (datasets with `poses/`) |
| `outputs/scene_graphs/<scene>_pose_eval_per_frame.csv` | per-frame ground truth, estimate and error |
| `outputs/visualizations/pose/trajectory_vs_groundtruth.png` | estimate vs ground truth + error curves |
| `outputs/logs/<scene>.log` | run log, including the pose report |
| `outputs/visualizations/depth/` | colorized depth per frame |
| `outputs/visualizations/pose/` | DROID-SLAM trajectory plot |
| `outputs/visualizations/detection/` | YOLOE boxes + masks, objectness only |
| `outputs/visualizations/semantic_decoding/` | the same boxes labelled with NOVIC names |
| `outputs/visualizations/instances/instances.png` | fused 3D instances |
| `outputs/visualizations/scene_graph/` | graph render |
| `data/processed/<scene>/track_names.json` | every track-end decode: track, frame, object, label, score |
| `data/processed/<scene>/` | staged frames, depth, poses, `capture.json` |

`outputs/visualizations/` is shared between runs, so each run overwrites the
previous one's images.

## 5. Key configuration (`configs/pipeline.yaml`)

| Key | Default | Effect |
|---|---|---|
| `source.uri` | DroidCam URL | input; `--source` overrides |
| `source.stride` / `max_frames` | 1 / 300 | frame thinning / cap for files and folders |
| `camera.*` | 525/525/320/240 @ 640×480 | used when the source has no `camera.yaml`; **calibrate this** |
| `depth.model_id` | `DA3METRIC-LARGE` | must be a metric checkpoint |
| `depth.align` | `none` | `global` needs reference depth |
| `pose.backend` | `droid_slam` | `cpu_icp` = Open3D colored-ICP fallback |
| `scene_graph.gravity_align` | `true` | level the world before building relations |
| `detection.use_tracking` / `tracker` | `true` / `botsort.yaml` | BoT-SORT track ids; off = no track ends, so every object is named from its final average in stage 8 |
| `semantic_decoding.crop_padding_frac` | 0.15 | crop context as a fraction of the box; `null` = use `crop_padding_px` |
| `object_fusion.sim_threshold` | 0.55 | higher = fewer merges, more instances |
| `object_fusion.vis_weight` / `spatial_weight` | 0.7 / 0.3 | cosine vs. voxel-overlap weight |
| `object_fusion.obj_min_detections` | 3 | frames an instance needs to survive |
| `object_fusion.fragment_containment` / `fragment_point_distance_m` | 0.5 / 0.05 m | share of a fragment's points that must lie this close to a larger instance's points to merge |
| `object_fusion.fragment_embedding_similarity` | 0.75 | embedding cosine a fragment merge also needs |
| `object_fusion.fragment_conflict_min_score` | 0.3 | two confident, different track-end names block a fragment merge |
| `label_mapper.min_name_score` / `synonym_similarity` / `low_score_action` | 0.15 / 0.8 / `drop` | an instance whose name confidence (top score pooled with its synonyms in the beam) is below this is removed with its tracks and embedding; `unknown` keeps it unnamed |
| `label_mapper.everyday_names_file` / `everyday_name_weight` | `configs/vocab/indoor_objects.txt` / 0.2 | a NOVIC label not in this list is replaced by the listed name best matching the object's image (plus 0.2 × text similarity to NOVIC's label); `null` keeps NOVIC's labels |
| `scene_graph.support_max_gap_m` / `support_max_penetration_m` | 0.10 / 0.15 | how far below / above a child's base its supporter's top may be for "on" |
| `object_fusion.relabel_support_surfaces` | `true` | a large flat instance at furniture height (`support_surface_*`) takes the first surface noun in its beam (desk, table, ...), else `table` |

## 6. Validation

Measured on `data/raw/tum_fr1_desk` (TUM RGB-D freiburg1_desk, motion-capture
ground truth), with `DA3METRIC-LARGE`, `depth.align: global` and DROID-SLAM.

**Depth.** The global alignment scale was 1.0767 (10 frames, spread
1.002–1.104), i.e. the metric model was already within ~8% of the Kinect
before alignment.

**Camera pose vs ground truth** (`--pose_only`, reproduce with
`--source data/raw/tum_fr1_desk --pose_only`):

| metric | full sequence, 596 frames / 9.31 m | 150 frames (stride 2) / 4.53 m |
|---|---|---|
| ATE translation, Sim3-aligned | **2.49 cm RMSE** (median 2.08, max 7.89 cm) | 3.16 cm (median 2.67, max 8.48) |
| ATE solved scale | **1.0219** | **1.0027** |
| ATE translation, SE3 (scale fixed at 1) | 3.11 cm RMSE | 3.17 cm RMSE |
| ATE rotation | 2.79° RMSE (median 2.59°, max 5.08°) | 1.70° (median 1.34°, max 4.73°) |
| RPE per frame | 0.92 cm / 0.57° RMSE | 1.44 cm / 0.88° RMSE |
| runtime (depth + DROID-SLAM) | 4.5 min + 53 s | 80 s + 27 s |

The solved scale is the headline number: **1.0027 and 1.0219**, i.e. within
0.3% and 2.2% of metric with no scale correction applied. That is the
end-to-end confirmation that the `focal/300` depth fix gives DROID-SLAM a
correctly scaled depth prior — before the fix the depth was ~1.5× short. SE3
error (scale forced to 1) is barely worse than Sim3, which is what a genuinely
metric trajectory looks like.

Per-frame errors are in `<scene>_pose_eval_per_frame.csv` and plotted in
`outputs/visualizations/pose/trajectory_vs_groundtruth.png`. Translation error
stays near 2 cm with occasional 5–8 cm excursions during fast turns; rotation
error drifts upward over the sequence (~1.5° early, ~3–4° after frame 400),
the expected slow yaw drift of a monocular-scale SLAM front end.

**Scene graph** (150 frames): 17 fused instances, most with plausible names
(computer mouse, LCD, PC, workstation, highlighter, eraser, earphone, power
sockets). Relations, before and after gravity alignment:

| | object-on-object relations | on "floor" |
|---|---|---|
| without gravity alignment | 1 / 18 | 17 / 18 |
| with gravity alignment | **5 / 17** | 12 / 17 |

After alignment the geometry is right — desk objects sit at z ≈ 0.75 m, monitor
tops at ≈ 1.2 m. The remaining floor attachments come from fusion and naming,
not geometry: the desk surface fused into one 1.4 × 1.5 m instance that NOVIC
named "angoumois moth", and some instances are over-merged ("computer mouse"
1.1 × 1.5 m, "workstation" 3.0 × 2.1 m), so desk items find no clean support.
Raising `object_fusion.sim_threshold` (fewer merges) is the next thing to try.

**Timing** (150 frames, one RTX 5000):

| stage | time |
|---|---|
| depth (DA3METRIC-LARGE) | 80 s |
| DROID-SLAM | 27 s |
| voxel map load | ~2 min |
| YOLOE + NOVIC embeddings | ~4 min |
| instance fusion | **39 min** |
| denoise / merge / filter | 5 min |
| tagging + scene graph | ~1 min |

**Unit tests:** 19 in `tests/unit/`, covering the frame source, the metric
scale formula and gravity alignment on a synthetic scene rotated into an
arbitrary frame. All pass.

## 7. Limitations

- **DROID-SLAM needs continuous motion.** It tracks by optical flow and needs
  roughly < 5° of rotation between consecutive frames. On 3RScan's sparse
  keyframes (median 6.4°, up to 52°) it loses track: 0.93 m ATE and ~138°
  absolute rotation error. Use ground-truth poses for datasets like that.
- **Focal length sets the map scale.** Metric depth is proportional to `fx`.
  With the uncalibrated `camera:` defaults, a phone capture's map will be off
  by the ratio of true to assumed focal length.
- **Monocular depth is inconsistent across frames.** Per-frame scale varies by
  roughly ±10%. One global scale fixes the average, but surfaces still land at
  slightly different depths from frame to frame, which thickens the map (see
  `docs/3rscan_depth_voxel_tracking.md`). A stereo or RGB-D sensor removes this.
- **Prompt-free YOLOE segments room structure too.** Floor, counters and
  cabinets become instances alongside real objects.
- **Instance fusion is the bottleneck.** On 150 TUM frames it took 39 of the
  run's 53 minutes, reaching ~15 s per frame by the end. DoVSG's
  `InstanceProcess` computes spatial overlap against every fused object so far,
  so the cost grows with scene size. For long captures, thin frames with
  `source.stride` (DROID-SLAM itself still wants dense frames, so thin a dense
  video moderately, e.g. 30 fps → stride 2–3).
- **NOVIC names from small crops are noisy.** Most names on the TUM desk were
  right (computer mouse, LCD, PC, highlighter, eraser, earphone, worktable), but
  some were nonsense ("angoumois moth", "comma", "term"). The top-k
  alternatives are saved per instance in the `.pkl`.
