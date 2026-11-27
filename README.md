# Scene-Graph-Navigation-with-Dynamic-Sensing

An open-vocabulary 3D scene graph pipeline: a DroidCam RGB walkthrough goes
in, a 3D scene graph (objects + spatial relations, each object tagged by NOVIC
with both a name and an embedding) comes out. On top of it: dynamic scene
updates (`dynamic_updates/`), LLM task planning (`task_planning/`, Gemini) and
A* navigation (`navigation/`). See `docs/full_pipeline.md` and
`docs/pipeline_design.md` for the architecture and rationale, and
`docs/architecture/` for diagrams.

## Pipeline

```
DroidCam stream
  -> Depth Anything V3           (perception/depth_estimation)
  -> DROID-SLAM pose             (perception/pose_estimation)
  -> Mask2Former Swin-L panoptic  (perception/detection/panoptic_detector.py) — masks, no class list;
                                    wall/floor/ceiling dropped as stuff; BoT-SORT tracks
  -> NOVIC embed + decode        (semantic_decoding/novic) — tag + embedding, once per fused instance
  -> label canonicalization      (semantic_decoding/label_mapper)
  -> DoVSG multi-view 3D fusion  (object_fusion) — reuses Scene-Graph-Generation's
                                    view_dataset.py / instance_process.py directly
  -> DoVSG scene graph builder   (scene_graph/builder) — reuses scene_graph_processer.py directly
  -> pickle + JSON + PDF         (scene_graph/storage)
```

**Detection front end (default since 2026-11-27): Mask2Former only.** On the 8
Replica scenes it gave the highest object precision (86.2%), the fewest
non-object nodes and duplicates, the best naming scores and the shortest run
time, at lower recall (47.3%: it misses wall plugs, switches, vents). Report and
scripts: `evaluation/reports/mask2former_frontend/`. Other front ends:

| Front end | How |
|---|---|
| Mask2Former + YOLOE small objects (previous default, higher recall) | `--override configs/replica_variants/panoptic_yoloe_union.yaml` or `detection.panoptic.yoloe_union.enabled: true` |
| YOLOE-26 prompt-free alone | `detection.backend: yoloe_seg_pf` |

**Scene updates use YOLOE** (`configs/pipeline_lap_*.yaml`). Tested 2026-11-27 on
`lap_in_1` → `lap_remove_2`: with Mask2Former the removed laptop is found but the
added iron is missed (no ADE20K class), and with the YOLOE union two present objects
were wrongly removed. Details: `/home/ravindu/SceneGraphNav/reports/update_detector_test_2026-11-27.md`;
configs: `configs/update_variants/`.

Every stage writes its own visualization to `outputs/visualizations/<stage>/`
so a run can be sanity-checked stage-by-stage, not just at the end.

**Scene updates:** `scripts/update_scene_graph.py` takes a saved scene graph
and a new video of the same scene, finds added or removed objects, and changes
only those nodes, following DoVSG's `update_scene()`. See
`docs/scene_update.md`.

## Setup

Everything the pipeline imports or loads lives inside this folder. The
external repos sit under `external/` and are reused via `sys.path` insertion
rather than copied into the package:

- `external/Scene-Graph-Generation` — DoVSG fork providing
  `ViewDataset`, `InstanceProcess`, `SceneGraphProcesser`, and the
  DROID-SLAM/CPU-ICP pose scripts (plus `DROID-SLAM/droid.pth`). Path is set
  in `configs/pipeline.yaml` under `external.scene_graph_generation_root`.
- `external/novic` — the NOVIC repo (code plus the two trained decoder
  checkpoints under `outputs/`; the pipeline uses
  `ovod_20240628_142131/ovod_chunk0433_20240630_235415.train`). Path is set
  under `semantic_decoding.novic_dir` / `semantic_decoding.novic_checkpoint`.
- `external/DovSG` — upstream DoVSG with its own `.venv`, used only by
  `dynamic_updates/dovsg_method/` (ACE relocalizer, LightGlue, dsacstar).
- `models/checkpoints/` — YOLOE-26 weights (`yoloe-26{s,m,l}-seg-pf.pt`,
  `yoloe-26s-seg.pt`), and the RAM++ / MobileCLIP2 weights used by
  `benchmarks/detect_track/`.

Hugging Face models (Depth Anything V3, the DFN5B CLIP embedder NOVIC uses)
still come from `~/.cache/huggingface`. Run with `HF_HUB_OFFLINE=1`: without
it the embedder's Hub check fails with a 401 and NOVIC silently falls back to
its stub decoder.

Install this pipeline's own dependencies (`pip install -r requirements.txt`)
into whichever environment already has the above two set up — the
`scene_graph` conda env has almost everything already; the one confirmed gap
is `unidecode` (needed by NOVIC's `infer.py`).

DROID-SLAM needs its compiled CUDA extensions (`droid_backends`, `lietorch`,
`pytorch_scatter`, under `Scene-Graph-Generation/{DROID-SLAM,lietorch}`) —
if those aren't built in your environment, set `pose.backend: cpu_icp` in
`configs/pipeline.yaml` to use the geometry-only Open3D colored-ICP fallback
instead (no CUDA required, but noisier).

## Running

See **`HOW_TO_RUN.md`** for the full step-by-step walkthrough (environment
activation, connecting your phone via DroidCam, troubleshooting). Short version:

```bash
conda activate scene_graph
pip install unidecode   # the one confirmed missing dependency
python scripts/run_static_scene_graph.py --config configs/pipeline.yaml
```

Output: `outputs/scene_graphs/<scene_name>.{pkl,json}` and per-stage
visualizations under `outputs/visualizations/`.

## Known limitation found during testing

`SceneGraphProcesser.build_scene_graph()` (reused unmodified from DoVSG) can
loop indefinitely if the fused instance set contains near-duplicate,
spatially-coincident objects that end up mutually "on" each other (A
supports B and B supports A) — confirmed while running
`tests/integration/test_static_pipeline_freiburg.py` in NOVIC's stub-decoder
mode, where near-random per-crop embeddings mean multiple views of the same
real object never merge into one instance during fusion. A real NOVIC
checkpoint should avoid this in practice (visually consistent embeddings
merge those crops before scene-graph building ever sees them), but if
`object_fusion` is under-merging on a noisy real capture (poor `sim_threshold`/
`vis_weight` tuning, heavy occlusion), the same hang could occur. This is
upstream DoVSG behavior, not something introduced here — worth watching for
if a real run seems to hang at the scene-graph stage; lowering `sim_threshold`
or raising `vis_weight` in `configs/pipeline.yaml` (more aggressive merging)
is the first thing to try.

## Project layout

| Folder | What |
|---|---|
| `perception/` | frame source, depth (Depth Anything V3), pose (DROID-SLAM / ICP), detection (Mask2Former, YOLOE) |
| `semantic_decoding/` | NOVIC embedding + naming, label canonicalization, optional vLLM tagger |
| `object_fusion/` | streaming multi-view instance fusion (DoVSG-based), post-processing |
| `scene_graph/` | relation builder, gravity alignment, save / load / render |
| `dynamic_updates/` | scene-graph change detection: `dovsg_method`, `projection_method`, `covis_gate`, `Dynamic_Attention` |
| `task_planning/` | natural-language command -> target object (Gemini + CLIP) |
| `navigation/` | occupancy grid from the voxel map, A* path planning |
| `configs/` | `pipeline.yaml` (defaults), `replica_variants/` (experiment overrides), vocab, stuff probes |
| `scripts/` | entry points: run, evaluate, render, prepare datasets |
| `evaluation/`, `benchmarks/` | metrics, Replica / 3RScan evaluation, published comparisons and reports |
| `tests/` | `pytest tests/unit` |
| `docs/` | design notes, architecture diagrams, stage explainers |

Not in Git (see `.gitignore`): `data/`, `models/`, `external/`, `results/`,
`outputs/`. In this repo they are symlinks to the shared folders in
`/home/ravindu/SceneGraphNav/`; see **`SETUP.md`** for the layout and for setting
up your own clone (`bash scripts/setup_links.sh`).
