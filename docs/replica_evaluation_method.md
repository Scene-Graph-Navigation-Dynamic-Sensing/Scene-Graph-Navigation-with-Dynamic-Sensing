# Replica evaluation: how it was done

How the pipeline was evaluated on the Replica dataset, from the downloaded data to
the final numbers. Results: `results/replica_final/summary.md`. Metric choices and
their rationale: `docs/evaluation_metrics.md` (group A).

There is no real video and no pose estimation in this evaluation. Replica provides
a rendered camera walk-through with exact depth and camera poses, so the numbers
measure the parts this project changes (detection, tracking-aided fusion, naming)
without SLAM or depth errors on top.

## Overview

```
Replica meshes + NICE-SLAM renders + ConceptGraphs GT cloud      (downloaded)
        │
        ▼  scripts/prepare_replica_dataset.py
200 frames per scene: RGB, depth, camera pose            (data/raw/replica_<scene>/)
        │
        ▼  scripts/eval_replica_scene.py  →  scripts/run_static_scene_graph.py
pipeline: detect → track → embed → voxelize → fuse → name → relations
        │                                   (scene graph .pkl + voxel map)
        ▼  evaluation/  (same driver script)
match nodes to ground truth → object, naming, embedding and segmentation metrics
        │
        ▼  scripts/combine_replica_results.py (+ naming scripts)
results/replica_final/summary.md
```

## 1. Data

The 8 Replica scenes ConceptGraphs evaluates on: office_0–4 and room_0–2. Three
sources:

| Source | Contents | Used for |
|---|---|---|
| **Replica dataset** (`dataset/Replica/extracted/<scene>/habitat/`) | the scene mesh with an object id on every face (`mesh_semantic.ply`) and the class of every object (`info_semantic.json`, 101 classes) | object-level ground truth |
| **NICE-SLAM's Replica release** (`dataset/Replica/nice_slam/`) | a rendered camera walk-through per scene: 2,000 RGB frames (1200×680), depth in millimetres, the exact camera pose of every frame (`traj.txt`), intrinsics fx = fy = 600 | pipeline input |
| **ConceptGraphs' ground-truth cloud** (`dataset/Replica/conceptgraphs_semantic_gt/`) | a point cloud per scene, every point labelled with one of 52 classes | segmentation scoring, as ConceptGraphs does it |

The NICE-SLAM trajectories are the input ConceptGraphs used, so the input matches
theirs, except that we keep every 10th frame and they kept every 5th.

## 2. Preparing the frames

`scripts/prepare_replica_dataset.py --scene <scene>` converts each scene into the
layout the pipeline reads (`data/raw/replica_<scene>/`):

- every 10th frame, 200 per scene, renumbered from 0 (`source.json` lists the
  original frame numbers);
- per frame: `rgb/*.jpg`, `depth/*.png` (uint16 millimetres) and `poses/*.txt`
  (4×4 camera-to-world, OpenCV axes); `camera.yaml` holds the intrinsics.

**Alignment check.** The script back-projects the depth of five frames with their
poses and measures the distance to the ground-truth mesh. If the poses, the depth
scale and the mesh did not share one coordinate frame, the points would land far
from the mesh and the script would stop. Every scene passed with a median distance
of 5–6 mm and 100% of points within 5 cm.

## 3. Building the scene graph

`scripts/eval_replica_scene.py --scene <scene>` builds the configuration, runs the
pipeline (`scripts/run_static_scene_graph.py`) and then evaluates. Depth and poses
are read from the dataset (`--depth_backend reference --pose_backend reference`);
Depth Anything and DROID-SLAM do not run.

1. **Frames** are resized to 640 px wide (640×363), intrinsics scaled to match.
2. **Gravity alignment:** the floor plane is fitted and the scene rotated so the
   floor lies at z = 0 (spatial relations such as "on" depend on it).
3. **Streaming stage**, three threads in parallel:
   - *Detection:* YOLOE-26x prompt-free at 1024 px finds objects and their masks
     with no class list; dark frames are brightened first (gamma correction).
     BoT-SORT gives each object a track id across frames.
   - *Embedding:* every detection's padded box crop is embedded with NOVIC's CLIP
     model (DFN5B ViT-H-14).
   - *Fusion:* each mask is lifted into a 1 cm voxel map using depth and pose.
     Detections are accumulated per track; when a track ends it is compared with
     the objects already mapped (0.7 × appearance cosine + 0.3 × voxel overlap, plus
     the merge rules) and merged into one or added as a new object.
4. **Final fusion:** DBSCAN clean-up, a re-match pass between objects, an overlap
   merge, and assignment of voxels claimed by several objects to one owner.
5. **Naming:** NOVIC's decoder turns each object's averaged embedding into a
   free-form noun; the label mapper tidies names. In this evaluation setting
   low-confidence names are kept as `unknown`, so detection can be measured on all
   objects (real videos drop them instead).
6. **Scene graph:** relations between objects ("on", "inside") from their 3D
   geometry.

Output: a pickle with the voxel map (about 1–2 million voxels per scene) and the
object nodes, each with its voxels, embedding, name and source detections.

## 4. Choosing the configuration

The configuration was tuned on two development scenes only, office_0 (dark) and
room_0 (bright); the other six were held out (`results/replica_tuning/`,
`FINDINGS.md`, `comparison.md`).

- Goal: find as many objects as possible; naming was deferred.
- Variants: brightness correction, tracker settings, merge rules, detector size
  and input resolution, fusion's detection filters, minimum detections per object,
  keeping low-confidence names.
- Chosen: variant `disc` = `configs/replica_variants/{merge_fixes,brightness,
  fusion_naming,discovery}.yaml`. It found 55 of 103 development objects against the
  baseline's 27.
- The fixed configuration then ran on all 8 scenes (`results/replica_final/`). The
  held-out scenes did slightly better than the development ones (mean node recall
  67.6% vs 57.3%), so the gains are not fitted to the tuning scenes.

## 5. Evaluation

Code in `evaluation/`; it runs automatically after each scene.

### Ground truth from the mesh (`evaluation/replica_gt.py`)

- The mesh surface is sampled at about one point per cm² plus every face centre;
  each point carries its object id and class.
- **Visibility:** an object counts only if at least 50 of its sample points (about
  50 cm²) were seen in some evaluated frame: in view, not occluded, and within 5 cm
  of that frame's depth. Recall is measured only against objects the camera saw.
- **Object types:** wall, floor and ceiling are *structural* (segmentation only);
  `undefined` and anonymised objects are *ignored*; all others are *countable*.

### Matching nodes to objects (`evaluation/open_vocab.py`)

Each node voxel takes the object of the nearest ground-truth point within 5 cm. The
majority object is the node's match; its share of the voxels is the node's purity.

| Status | Meaning | Counted in Acc@k |
|---|---|---|
| correct / wrong name | one countable object (purity ≥ 0.5), name right / wrong | yes |
| mixed | purity < 0.5: spans several objects | no |
| structural | wall, floor or ceiling | no |
| unlabelled | an object Replica does not label | no |
| phantom | under half its voxels near any surface | no |

### Metrics

| Group | Metric | How |
|---|---|---|
| Objects | node recall | visible countable objects found by at least one node |
| | valid-node rate, N_dup | nodes that are one real object; extra nodes on an already-found object |
| | node precision | valid *and* named right, the name mapped to the nearest class by text similarity; an automatic stand-in for ConceptGraphs' human-judged precision |
| Embedding quality | Acc@k, AUC | the node embedding ranked against text embeddings of the 101 class names ("an image of {}"), as ReasoningGraph and HOV-SG do |
| Naming, open vocabulary | share acceptable | NOVIC's decoded noun vs the nearest of NOVIC's 42,919 nouns to the embedding, both graded by one WordNet judge (exact/synonym, more specific, too generic, wrong); `evaluation/naming_judge.py`, `scripts/eval_open_vocab_naming.py` |
| Naming, class list given | top-1/3/5 | NOVIC restricted to the 101 class names vs ConceptGraphs' rule (nearest class text embedding); exact match only; `scripts/eval_closed_set_naming.py` |
| Segmentation | mIoU, mAcc, F-mIoU | ConceptGraphs' `eval_replica_semseg.py` protocol (`evaluation/conceptgraphs_semseg.py`): their labelled cloud and classes, each node classified by its embedding, every voxel-map point labelled by its nearest node voxel and scored against the nearest ground-truth point. Headline: exclude-6 (other, floor, wall, ceiling, door, window left out), as in their Table II |

## 6. Outputs

- **Per scene** (`results/replica_final/<scene>/`): `results.md`, `metrics.json`,
  a labelled walk-through video, voxel maps and ground truth as `.ply`, the scene
  graph, a node gallery, and per-node, per-object and per-class tables (`nodes.csv`,
  `naming.csv`, `closed_set_naming.csv`, `gt_objects.csv`,
  `per_class_segmentation.csv`), so every number can be checked.
- **All scenes:** `scripts/combine_replica_results.py --results_root
  results/replica_final` writes `results/replica_final/summary.md`, the source
  of the comparison report. Baseline numbers come from the ConceptGraphs and
  ReasoningGraph papers.

To reproduce one scene:

```bash
conda activate scene_graph
python scripts/prepare_replica_dataset.py --scene room_0
V=configs/replica_variants
HF_HUB_OFFLINE=1 python scripts/eval_replica_scene.py --scene room_0 --gpu 0 --tag final \
    --results_root results/replica_final \
    --override $V/merge_fixes.yaml $V/brightness.yaml $V/fusion_naming.yaml $V/discovery.yaml
python scripts/eval_closed_set_naming.py results/replica_final/room_0
python scripts/combine_replica_results.py --results_root results/replica_final
```

## 7. Caveats

- **Exact depth and poses:** the results measure detection, fusion and naming. The
  phone-video setup (Depth Anything V3 + DROID-SLAM) needs its own ablation.
- **Different CLIP model:** class names are embedded with NOVIC's CLIP (DFN5B
  ViT-H-14), not ConceptGraphs' LAION ViT-H-14; an encoder-matched ConceptGraphs
  re-run would remove this difference.
- **Half ConceptGraphs' frame rate:** every 10th frame against their every 5th.
- **Automatic naming judge:** WordNet is strict on near-synonyms; a ~100-node
  human check is still to be done.
- **Tuning scenes:** office_0 and room_0 were used for tuning; the other six were
  held out.
- **Timing:** mapping takes 10–64 min per scene, mostly in object fusion
  (`docs/fusion_timing.md`).
