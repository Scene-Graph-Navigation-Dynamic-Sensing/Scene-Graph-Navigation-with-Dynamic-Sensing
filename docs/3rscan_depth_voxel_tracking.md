# 3RScan: depth eval → voxel map → track-driven object mapping

Environment: `conda activate scene_graph`. All commands run from the repo root.

`depth_anything_3` was installed `--no-deps` (its full dependency set wants to
downgrade numpy and pull an xformers-pinned torch, which would break this env);
the eager imports it needs were added individually: `omegaconf`,
`antlr4-python3-runtime==4.9.3`, `einops`, `moviepy==1.0.3`, `decorator==4.4.2`,
`proglog`, `imageio-ffmpeg`, `plyfile`, `pycolmap`, `evo`, `natsort`.

```bash
SEQ=/home/ravindu/SceneGraphNav/datasets/3rscan/3rscan_data/c92fb578-f771-2064-85fc-485dbfba73df/sequence
```

## Two facts about this dataset that drive everything below

**Frames are stored rotated 90°.** The floor runs along the right edge of the
960×540 image. `--rotate` uprights colour and depth, remaps the intrinsics, and
post-multiplies the pose rotation by the matching roll about the optical axis.
Verified: the reconstructed world cloud is bit-identical either way, so rotation
is purely for the models' benefit, not the geometry's.

**Depth and colour share a field of view.** The depth intrinsics are the colour
intrinsics scaled by the resize (756.832 × 224/960 = 176.594), so comparing them
is a resize, not a reprojection.

## 1. Depth Anything V3 vs ground truth

```bash
python scripts/eval_depth_3rscan.py --sequence $SEQ \
  --model-id depth-anything/DA3-BASE --align median --rotate --require-v3
```

`--align` matters: `DA3-BASE` is affine-invariant, so `median` (per-frame) or
`scale_shift` (least squares) is required. `none` is meaningful only for a
metric checkpoint. `--require-v3` aborts rather than silently scoring the V2
fallback in `perception/depth_estimation/depth_anything_estimator.py`.

Results over all 46 frames (`outputs/depth_eval/<run>/summary.json`):

| model | rotated | align | AbsRel | RMSE | δ1 |
|---|---|---|---|---|---|
| DA3-BASE | no | median | 0.122 | 0.346 | 0.846 |
| DA3METRIC-LARGE | no | none | 0.290 | 0.558 | 0.100 |
| DA3METRIC-LARGE | no | median | 0.065 | 0.232 | 0.962 |
| DA3METRIC-LARGE | **yes** | none | 0.263 | 0.506 | 0.165 |
| DA3METRIC-LARGE | **yes** | median | **0.057** | **0.210** | **0.975** |

DA3METRIC-LARGE predicts excellent depth *shape* but its absolute scale is
systematically short by a factor of ~1.35 on this scene (std 0.08 across
frames). Passing the true intrinsics to `inference()` was tested and changes
the output not at all, so this is a model bias, not a calibration error. Since
the factor is near-constant, one global scale recovers metric depth.

> **Correction (2026-09-18):** this is *not* a model bias. DA3METRIC outputs
> depth for a canonical 300 px focal length; metric depth is
> `raw × focal_px / 300`, with focal measured at the resolution the network ran
> at. The upstream API applies that only inside its *nested* model, which is why
> passing intrinsics to `inference()` changed nothing.
> `DepthAnythingEstimator.estimate(rgb, fx, fy)` now applies it. Measured
> GT/predicted scale: b05fddcf **1.500 → 0.975**, TUM freiburg1_desk
> **1.483 → 1.093**. The global-scale numbers below were computed without the
> conversion, so the scales they report (1.35, 1.4311) mostly measure the
> missing `focal/300` factor, not model error. The per-frame *spread* and the
> smearing analysis are unaffected, since one constant factor can't cause them.

## 2. 3RScan → dataset → voxel map

The converter writes the DoVSG layout (`rgb/`, `point/`, `mask/`, `poses/`,
`metadata.json`) that `object_fusion.view_dataset.ViewDataset` already reads.

```bash
# ground-truth depth
python scripts/prepare_3rscan_dataset.py --sequence $SEQ \
  --out data/processed/3rscan_c92fb578_gt --depth-source gt --rotate --rgb-width 480

# DA3 depth, one sequence-wide scale (no per-frame GT needed at run time)
python scripts/prepare_3rscan_dataset.py --sequence $SEQ \
  --out data/processed/3rscan_c92fb578_da3global --depth-source da3 \
  --model-id depth-anything/DA3METRIC-LARGE --da3-align global --rotate --rgb-width 480

python scripts/build_voxel_map.py --dataset-dir data/processed/3rscan_c92fb578_gt
python scripts/build_voxel_map.py --dataset-dir data/processed/3rscan_c92fb578_da3global \
  --out-dir outputs/voxel_maps/da3global
```

Writes `voxel_map.ply`, `trajectory.ply`, `views/{iso,top,front,side}.png` and
`voxel_map_stats.json`. Add `--interactive` for an Open3D window (needs a
display), or `--compare-mesh <scan>/labels.instances.annotated.v2.ply` to
overlay the reference mesh.

Voxel maps at 1 cm, measured against the GT-depth map:

| map | voxels | accuracy (median) | <10 cm | completeness (median) |
|---|---|---|---|---|
| GT depth | 817k | — | — | — |
| DA3 metric, unscaled | 948k | 24.4 cm | 10.6% | 31.0 cm |
| DA3 metric, global scale | 1.70M | **5.4 cm** | **67.7%** | **1.4 cm** (95.1% <10 cm) |

The scaled DA3 map is *denser* than the GT one — DA3 fills the ~30% of pixels
where the Tango sensor returns nothing. It smears across the window, where
glass defeats monocular depth.

## 3. Track-driven object mapping

`perception/detection/tracked_yoloe_detector.py` runs YOLOE-seg-pf with
BoT-SORT (`persist=True`, so ids survive across frames). The image encoder runs
**only for track ids seen for the first time**; `TrackEmbeddingCache` keys the
embedding by track id and reuses it for every later frame of that track.

```bash
python scripts/run_tracked_mapping.py --dataset-dir data/processed/3rscan_c92fb578_gt
```

Measured on the 46-frame sequence: 189 detections, 53 unique tracks,
**82 encoder calls instead of 189 — a 56.6% reduction**, and cosine similarity
is computed 82 times (once per new track) rather than 189.

### Re-ID needs the spatial term, not just cosine

A new track still has to be matched against objects already mapped, since the
tracker drops ids through occlusions. Raw cosine alone does not work here: NOVIC/
OpenCLIP crop embeddings from a single room all sit in a narrow 0.56–0.91 band,
and at the config's 0.55 threshold every detection merged into **one** object.

So the score is `vis_weight*cosine + spatial_weight*voxel_overlap` (0.7/0.3 from
`configs/pipeline.yaml`), the same combination `object_fusion` uses. Overlap is
intersection over the *detection's* voxels, not the union — a well-fused object
has far more voxels than any single view of it, so union IoU would never fire.
That yields 9 objects over 58 re-id merges.

`--embedder none` skips embeddings entirely and takes identity from the tracker
alone, which is the ablation for how much the encoder is contributing.

## 4. Second sequence (b05fddcf): good depth metrics, worse map

Everything above is one scene. Repeating it on `b05fddcf-fca0-2d4f-8a01-c6bdcc9fba8d`
(219 frames, kitchen/hallway) reproduces the depth result but shows that the
depth metrics do not predict map quality.

```bash
SEQ=/home/ravindu/SceneGraphNav/datasets/3rscan/3rscan_data/b05fddcf-fca0-2d4f-8a01-c6bdcc9fba8d/sequence
```

Depth, all 219 frames (`outputs/depth_eval/b05fddcf/*/summary.json`):

| model | rotated | align | AbsRel | RMSE | δ1 |
|---|---|---|---|---|---|
| DA3METRIC-LARGE | yes | none | 0.347 | 0.567 | 0.028 |
| DA3METRIC-LARGE | no | median | 0.076 | 0.166 | 0.947 |
| DA3METRIC-LARGE | **yes** | median | **0.064** | **0.150** | **0.954** |

Same story as sequence 1: shape is excellent, absolute scale is short. `--rotate`
again helps. The global scale here is **1.4311**, estimated from 11 frames.

### The scale is only near-constant, and the map pays for it

On sequence 1 the per-frame scale had std 0.08. Here the 11 sampled frames spread
**1.194–1.662** (`data/processed/3rscan_b05fddcf_da3global/prepare.log`), and
individual frames range at least 1.4311–1.5024 across the eval log. A single
global scale is therefore right on average and wrong on most frames, and since
the GT poses are fixed, each frame's surfaces land at a slightly different depth
in the world. Fused, they thicken rather than coincide.

Three maps from the same 219 frames and the same poses, at 0.01 m
(`outputs/voxel_maps/3rscan_b05fddcf_*`):

| depth source | occupied voxels | extent (m) | voxels / m³ |
|---|---|---|---|
| ground truth | 2,258,699 | 9.93 × 7.72 × 5.28 | 5,581 |
| DA3, global scale 1.4311 | 5,254,098 | 8.01 × 6.46 × 4.14 | 24,528 |
| DA3, unscaled | 2,729,494 | 6.09 × 5.20 × 3.01 | 28,635 |

Two separate failures show up here, and the voxel count alone hides both:

- **Scaling fixes size, not consistency.** The global scale grows the map's
  extent by 1.24–1.38× per axis, toward GT but still ~20% short — the estimate
  is pulled toward near surfaces, where the GT sensor actually returns values,
  while the long ranges that set the room's extent stay under-predicted.
- **Density is the real signal.** The DA3 maps hold 4.4–5.1× more voxels per m³
  than GT. GT depth is 81.9% valid on this sequence, so some of the excess is real
  hole-filling (27% of it, measured below); the rest is putting the *same* surfaces in slightly different places on each
  frame. The scaled map has more than twice GT's voxels inside half GT's volume.

### Pixel → voxel accounting: where the extra voxels come from

`scripts/eval_voxel_stats.py <dataset>` integrates a dataset without rendering
and writes `outputs/voxel_stats/<dataset>.json`. It counts how many pixels were
folded in, how many distinct voxels they produced, and how often a voxel is
re-observed across frames.

| | GT | DA3 global | DA3 unscaled |
|---|---|---|---|
| depth-valid pixels | 73,425,551 (81.9%) | 89,667,360 (100%) | 89,667,360 (100%) |
| integrated after inlier mask | 68,837,642 (76.8%) | 86,364,327 (96.3%) | 86,364,265 (96.3%) |
| final occupied voxels | 2,258,699 | 5,254,098 | 2,729,494 |
| pixels per voxel | 30.48 | 16.44 | 31.64 |
| **observations per voxel** | **2.72** | **1.25** | **1.23** |
| mean share of a frame's voxels that are new | 36.9% | 79.6% | 80.5% |
| new voxels, final frame | 8,127 | 43,846 | 21,004 |
| within-frame pixels per voxel | 11.22 | 13.17 | 25.72 |

The 2.33× voxel excess of DA3-global over GT factors exactly:

    1.255x  more input pixels   (DA3 is 100% valid, GT sensor only 81.9%)
  x 1.854x  worse pixel->voxel collapse
  = 2.326x  (observed 2.326x)

So **27% of the excess is legitimate hole-filling** and **73% is smearing**.
DA3 does fill the ~18% of pixels the sensor drops, and that part is a win.

Which of the two is at fault is settled by splitting the collapse in two:

- *Within* a frame, DA3 is fine — 13.17 px/voxel vs GT's 11.22. A single DA3
  frame is as internally coherent as a GT frame, exactly what AbsRel 0.064 and
  δ1 0.954 predict.
- *Across* frames it fails — an average DA3 voxel is observed **1.25** times
  versus GT's **2.72**, and 80% of the voxels a frame touches have never been
  seen before (GT: 37%). Revisiting known surface should re-confirm existing
  voxels; instead each frame lays down new ones beside them.

That is the whole failure: per-frame geometry is good, frame-to-frame placement
is not, because one global scale cannot track a per-frame factor that ranges
1.194–1.662. The `none` column confirms the diagnosis is scale-independent —
unscaled is *equally* inconsistent (1.23 obs/voxel) and only looks efficient on
`pixels per voxel` (31.64) because the map is squeezed into 95 m³, a quarter of
GT's volume. `pixels per voxel` is confounded by map size; `observations per
voxel` is not, and is the metric to watch.

Note what this means for the depth eval: `--align median` divides out a
per-frame scale before scoring, so it measures shape and is blind to both the
absolute error (δ1 drops 0.954 -> 0.028 at `--align none`) and the frame-to-frame
variation that actually degrades the map. Good median-aligned depth metrics are
necessary but not sufficient for a good fused map.

So `--da3-align global` is enough to get a metrically plausible map out of a
sequence with no depth sensor, but it is not enough to get a *clean* one, and
AbsRel 0.064 / δ1 0.954 gives no warning of that. Per-frame or per-object scale
refinement is the next thing to try; a map built this way should be treated as
approximately sized and heavily over-thickened.

## Caveats

- DA3 scale factors are calibrated against each scene's GT (~1.35 on
  c92fb578, 1.4311 on b05fddcf) and do not transfer between scenes. On a
  sequence without GT depth you would calibrate once against a known distance;
  `--da3-align global` samples ~12 frames to estimate it.
- 3RScan poses are not gravity-aligned, so the "top"/"front" render names are
  viewpoints in the scan frame, not true up.
- Prompt-free YOLOE segments large surfaces (floor, counter, cabinets) as
  readily as objects, so the 9 kept "objects" include room structure.
