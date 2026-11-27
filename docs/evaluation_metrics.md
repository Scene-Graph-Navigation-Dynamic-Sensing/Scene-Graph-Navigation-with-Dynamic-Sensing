# Evaluation metrics

Which metrics from the 3D scene graph literature this project should report, and
why. The choice follows from what the project changes relative to its baseline,
DoVSG: each metric below is there to test one of those changes.

## What this project improves (vs DoVSG)

| # | Improvement | What DoVSG does |
|---|---|---|
| **1** | **Open-vocabulary naming with no class list:** YOLOE prompt-free detection plus NOVIC free-form names | GroundingDINO + SAM2 driven by a fixed text prompt, then CLIP matching against a fixed class list |
| **2** | **NOVIC embedding stored on every node**, reused for fusion, naming, update matching and queries | Nodes carry only a class label |
| **3** | **Fusion helped by tracking:** BoT-SORT track ids plus `0.7 × embedding cosine + 0.3 × voxel overlap`, with each object named once per track from its averaged embedding | Re-classifies every detection on every frame |
| **4** | **Phone camera only (RGB):** Depth Anything V3 metric depth, DROID-SLAM poses and gravity alignment | Needs an RGB-D camera |
| **5** | **Scene updates that work with monocular depth:** registration from image matches and point-cloud alignment instead of a pre-trained ACE model, a stricter "seen-through" removal test, a novelty check for new objects, and a limited final merge | ACE relocalization, tuned for RGB-D on a robot |
| 6 | Downstream use: an LLM task planner (`task_planning/`) and a planner/voxel map (`navigation/`) | Similar (LLM planning on a robot) |

Items 1–5 are the project's contributions, so the evaluation focuses on them.

## Tier 1: core claims (must have)

### A. Open-vocabulary object quality (improvements 1, 2, 3)

| Metric | What it proves | Data |
|---|---|---|
| **Open-vocabulary naming: NOVIC vs embedding lookup** (share acceptable / too generic / wrong) | Free-form naming beats naming by nearest noun, with no class list on either side (improvement 1) | Replica (8 scenes) |
| **Acc@k + AUC** (closed set) | The NOVIC embedding stored on nodes separates the classes (improvement 2); embedding quality, not naming | Replica (8 scenes), 3RScan validation |
| **Node precision, valid objects, N_dup, node recall** | Tracking-aided fusion makes one node per real object, with few duplicates or merges (improvement 3) | Replica (directly comparable to ConceptGraphs), 3RScan validation |
| **mAcc / mIoU / F-mIoU** | The open-vocabulary 3D map is right point by point; the headline number for comparing with ConceptGraphs | Replica, 8 scenes |

### B. Dynamic updates (improvement 5)

| Metric | What it proves | Data |
|---|---|---|
| **Change-detection precision / recall / F1**, reported separately for added, removed and moved objects | Updates find real changes and don't invent fake ones from depth noise | **3RScan reference→rescan pairs**: `3RScan.json` labels moved objects (982 rescans), removed objects (240), non-rigid changes (274), and gives the alignment transform. Also our own before/after phone videos. |
| **Scene Graph Accuracy (SGA)** after an update | Only the changed nodes and their `on` / `inside` edges change; the rest of the graph stays intact | Same pairs |
| **Update time vs full rebuild** | A local update is cheaper than rebuilding the map | Same pairs; timings are already logged |

The update method treats a moved object as one removal plus one addition.
"Moved" recall shows whether that holds up and should be reported as is.

### C. Efficiency (improvement 3)

| Metric | What it proves |
|---|---|
| **Per-stage time and FPS** (already logged in `stage_timings.json`) | Overall cost of the pipeline |
| **NOVIC decode calls: one per track vs one per detection** | The "name once per track" saving, measured directly |
| **Scene-graph memory** (MB per scene, with and without embeddings) | The cost of storing embeddings on nodes |

### D. Phone-only sensing (improvement 4)

The literature's metric list covers this only through D-RMSE, but RGB-only
input is one of the main differences from DoVSG, so it needs its own checks.

| Metric | What it proves | Data |
|---|---|---|
| **ATE / RPE** (implemented: `scripts/eval_pose.py`) | DROID-SLAM with Depth Anything depth gives accurate, metric-scale poses | TUM fr1_desk (done), Replica trajectories |
| **Depth error: AbsRel, δ<1.25, D-RMSE** (implemented: `scripts/eval_depth_3rscan.py`) | The metric depth is reliable | 3RScan, Replica |
| **Ablation: ground-truth depth/poses vs estimated**, on the same scenes | How much the A and B metrics drop going from RGB-D to phone only; the key check on the RGB-only claim | Replica, 3RScan |

## Tier 2: worth adding if time allows

| Metric | Why | Data |
|---|---|---|
| **Edge precision** for `on` / `inside` / `belongs` | The relations are rule-based; check them against 3DSSG's support relations (standing on, lying on, supported by, …) or with a small human check | 3RScan + 3DSSG |
| **Object retrieval: O-RSR@1/2/3 m and AMD** | Tests "query by text, get the right node", which directly uses the stored embeddings (improvement 2) | Replica / 3RScan, with queries generated from ground-truth class names |
| **Task grounding: s-acc / t-acc** | Only if the LLM task planner is part of the thesis; needs a small hand-written task set | Own scenes or Replica |

## Skipped (doesn't match what the project builds)

- **PredCls / SGCls / R@K over predicates, mR@K, head/body/tail recall:** these
  assume a learned relation classifier that scores 26+ predicates. The project
  produces three rule-based relations, so these would only show a limitation it
  never claims to address.
- **Room / hierarchy metrics:** the graph has no room layer.
- **PSNR / SSIM / LPIPS, trajectory prediction, Levenshtein distance:** no
  rendering or motion forecasting in the project.
- **SPL, path length, Fréchet distance:** only relevant once real navigation
  trials are run in the Isaac simulator; `navigation/` is a planner, not yet a
  measured navigation loop.

## Data

| Dataset | Location | Used for |
|---|---|---|
| Replica, 8 scenes (office_0–4, room_0–2): meshes with per-face object ids and object boxes | `~/dataset/Replica/extracted/` | A, D |
| Replica RGB-D trajectories (NICE-SLAM release, the version ConceptGraphs uses) | `~/dataset/Replica/nice_slam/` | A, C, D |
| 3RScan scans (validation first, then training) | `~/dataset/3Rscan/3rscan_data/` | A, B, D |
| 3RScan metadata (`3RScan.json`: rescan pairs, moved/removed objects, alignment) | `~/dataset/3Rscan/meta/` | B |
| 3DSSG (objects, relationships, 160 classes, 26 predicates) | `~/dataset/3Rscan/meta/3DSSG/` | Tier 2 edge precision |
| Own before/after phone videos | `data/*.mp4` | B (real phone footage) |
| TUM fr1_desk | `data/raw/tum_fr1_desk` | D |

Tiers 1 and 2 are covered by the data above. Optional extras:

- **ScanNet + ScanRefer:** only needed to compare language grounding with
  ConceptGraphs / HOV-SG on their benchmarks. ScanNet needs a signed licence
  request and is large.
- **HM3DSem:** not needed. ReasoningGraph uses it, but Replica plus 3RScan
  already give comparable Acc@k numbers.

## Order of work

1. Replica node quality and segmentation (A).
2. 3RScan change detection (B), the most distinctive result.
3. Efficiency (C) and the ground-truth vs estimated ablation (D), which mostly
   reuse the runs from steps 1 and 2.
4. Edge precision and object retrieval (Tier 2).

A first check of Acc@k and object R@k on two 3RScan scenes is in
`scripts/eval_object_retrieval_3rscan.py` (reports next to each scene graph as
`*_retrieval_eval.json`).

## Running group A on Replica

One scene at a time (about 20 minutes of pipeline plus 1–2 minutes of
evaluation per scene at 200 frames):

```bash
conda activate scene_graph
HF_HUB_OFFLINE=1 python scripts/eval_replica_scene.py --scene office_0 --gpu 2
# scenes: office_0 … office_4, room_0 … room_2
python scripts/combine_replica_results.py        # -> results/replica/summary.md
```

`--skip_run` re-evaluates a finished run without re-running the pipeline.
Each scene's folder `results/replica/<scene>/` holds `results.md`,
`metrics.json`, the labelled walkthrough video, the voxel maps and GT as
`.ply`, the scene graph, a node gallery, per-node / per-object / per-class
tables, and the pipeline log and config.

Protocol (code in `evaluation/`):

- **Segmentation** follows ConceptGraphs' `eval_replica_semseg.py` exactly
  (their GT cloud, classes, per-scene class restriction, exclude-1/4/6,
  reconstruction-point scoring, metric formulas), except that class names are
  embedded with NOVIC's CLIP model (DFN5B ViT-H-14) instead of LAION
  ViT-H-14, since node embeddings live in that space.
- **Headline setting: exclude-6** (other, floor, wall, ceiling, door, window
  left out). The pipeline does not map background, and ConceptGraphs' own
  Table II numbers use `--n_exclude 6` (their README), so this is the
  like-for-like comparison. Exclude-1 and exclude-4 are still reported.
  Object metrics likewise leave out wall/floor/ceiling (`structural`).
- **Object metrics** need object instances, which ConceptGraphs' GT cloud does
  not have, so they use the Replica semantic mesh, sampled by surface area.
  ConceptGraphs scored node precision with human raters; this is an automatic
  stand-in: a node is correct when it lies on one labelled object (purity ≥ 0.5)
  and its NOVIC name maps to that object's class.
- **Naming is compared on an equal footing.** Ranking the embedding against
  Replica's 101 classes (Acc@k) gives a closed-set method the answer list that
  NOVIC never sees, so Acc@k is reported only as embedding quality and NOVIC's
  R@k is no longer set against it. Instead both methods name each node from the
  same open vocabulary, NOVIC's 42,919 target nouns: NOVIC by decoding, the
  embedding by its nearest noun (same prompt). One WordNet judge grades both
  names as exact/synonym, more specific ("armchair" for chair), too generic
  ("furniture" for chair) or wrong; acceptable = the first two
  (`evaluation/naming_judge.py`, `scripts/eval_open_vocab_naming.py`, run
  automatically by `eval_replica_scene.py`). Each Replica class is pinned to
  the WordNet sense it means; the predicted word may use any physical-object
  sense, equally for both methods. The pipeline's final label is graded too, for
  reference only: it is mapped onto a 245-word everyday list that holds 53 of
  the 101 classes, so it is not open-set.
- **Limits of the judge:** WordNet rejects near-misses people accept ("panel
  light" for a ceiling lamp), so absolute shares are a lower bound and the
  comparison between methods is what it supports. Before quoting absolute
  naming accuracy next to ConceptGraphs' human-rated precision, label ~100
  nodes by hand and report the judge's agreement. OpenLex3D (tiered
  open-vocabulary labels) is the benchmark to adopt if a recognised protocol is
  needed; check that it covers the Replica scenes used here.
