m# How to Run the Static Scene Graph Pipeline

Step-by-step instructions to go from a DroidCam phone stream to a saved
static 3D scene graph. **`docs/full_pipeline.md` is the complete reference** —
how each stage is implemented, the DROID-SLAM build, running on your own
dataset (`data/raw/`), and measured results. For the original design
rationale see `docs/pipeline_design.md`.

## 1. Activate an environment

This pipeline reuses code from `Scene-Graph-Generation` and NOVIC directly
(via `sys.path`, not copies), so it needs whatever environment already has
those working. The `scene_graph` conda env has everything except one
package (see step 2):

```bash
conda activate scene_graph
cd /home/ravindu/SceneGraphNav/code
```

Confirmed already present in `scene_graph`: `torch` (CUDA available),
`ultralytics`, `open3d`, `transformers`, `open_clip`, `opencv`, `numpy`,
`faiss`, `scikit-learn`, `graphviz` (Python package), and the system `dot`
binary (needed for the scene-graph PDF render).

## 2. Install the one missing dependency

NOVIC's `infer.py` needs `unidecode`, which isn't in the `scene_graph` env:

```bash
pip install unidecode
```

(`pip install -r requirements.txt` also works, but everything else in that
file is already installed in `scene_graph` — this is the actual gap.)

## 3. Check the two big model dependencies

Both already exist in this workspace and don't need downloading — this
step is just confirming they're where `configs/pipeline.yaml` expects:

- **NOVIC checkpoint**: `configs/pipeline.yaml` already points at
  `/home/ravindu/SceneGraphNav/external/novic/outputs/ovod_20240628_142131/ovod_chunk0433_20240630_235415.train`
  (DFN5B-CLIP-ViT-H-14-378 embedder, 42,919-noun vocabulary). Nothing to do.
- **YOLOE-26-seg-pf weights**: already points at
  `/home/ravindu/SceneGraphNav/models/checkpoints/yoloe-26s-seg-pf.pt` (the
  checkpoint used in `tests/integration/test_static_pipeline_freiburg.py`,
  confirmed working). A larger, more accurate variant is available at
  `/home/ravindu/SceneGraphNav/external/Scene-Graph-Generation/tools/yoloe-26x-seg-pf.pt`
  if you want to trade speed for accuracy — edit `detection.weights` in the
  config to switch.

### DROID-SLAM (camera pose) — check this one before a real run

`configs/pipeline.yaml` defaults to `pose.backend: droid_slam`. That needs
DROID-SLAM's compiled CUDA extensions (`droid_backends`, `lietorch`,
`torch_scatter`) built for the `scene_graph` env's Python 3.10. Check them
(`torch` must be imported first, and the path is relative to that checkout):

```bash
cd /home/ravindu/SceneGraphNav/external/Scene-Graph-Generation
python -c "import torch, sys; sys.path.append('./DROID-SLAM/droid_slam'); import droid; print('ok')"
cd -
```

- **If that prints `ok`**: you're set, DROID-SLAM will run.
- **If it fails**: rebuild them — `docs/full_pipeline.md` §1 has the exact
  commands. Or set `pose.backend: cpu_icp` for a geometry-only Open3D
  colored-ICP fallback that needs no CUDA extensions (noisier trajectory).

## 4. Connect your phone via DroidCam

1. Install the [DroidCam](https://www.dev47apps.com/) app on your phone,
   make sure it's on the same network as this machine.
2. Open the app — it shows an IP and port, e.g. `192.168.8.144:4747`.
3. Edit `configs/pipeline.yaml`:
   ```yaml
   source:
     uri: "http://192.168.8.144:4747/video"   # <- your phone's IP:port
     capture_interval: 0.3   # seconds between captured frames
     max_frames: 100         # how many frames to capture before processing
   ```
   A slow, steady pan around the room while capturing gives DROID-SLAM and
   the 3D fusion stage the best multi-view coverage.

   No phone handy? `source.uri: 0` uses your default webcam instead, or
   point it at a video file or a dataset folder — `--source <path>` on the
   command line does the same without editing the config. Dataset folder
   layout: `data/raw/README.md`.

   Set `camera:` to your phone's calibrated intrinsics: metric depth is
   proportional to the focal length, so the defaults give a wrongly-scaled map.

## 5. Run it

```bash
python scripts/run_static_scene_graph.py --config configs/pipeline.yaml
```

Optional: `--scene_name my_desk_scan` names the run (defaults to a
timestamp). Each run's raw capture is kept at
`data/processed/<scene_name>/` in case you want to re-process it later
without re-capturing.

The script runs through all stages in order and logs progress for each:
connecting to the source, capturing frames, depth estimation, pose
estimation, building the voxel map, detection + NOVIC tagging per frame,
3D instance fusion, and scene graph construction.

## 6. Check the results

- **Final scene graph**: `outputs/scene_graphs/<scene_name>.pkl` (full data:
  graph + instances + NOVIC embeddings) and `<scene_name>.json`
  (human-readable node/relation summary — open this first).
- **Per-stage visualizations**, to sanity-check any stage that looks wrong,
  under `outputs/visualizations/`:
  - `depth/<frame>.jpg` — colorized depth maps
  - `pose/trajectory.png` — camera path plot
  - `detection/<frame>/` — YOLOE boxes/masks, unlabeled (objectness only)
  - `semantic_decoding/<frame>/` — same boxes, now labeled with NOVIC's
    decoded + canonicalized names (the first point a name appears)
  - `instances/instances.png` — fused 3D instances, colored by instance
  - `scene_graph/scene_graph.pdf` — the graph itself (green = "inside",
    orange = part-of, salmon = leaf nodes)

## 7. If something goes wrong

- **Pipeline hangs at the scene-graph stage**: see the "Known limitation"
  section in `README.md` — DoVSG's `build_scene_graph()` can loop on
  near-duplicate, mutually-supporting instances. Try lowering
  `object_fusion.sim_threshold` or raising `object_fusion.vis_weight` in
  the config (more aggressive merging, fewer near-duplicate instances).
- **`import droid` fails / DROID-SLAM errors**: switch
  `pose.backend: cpu_icp` (step 3 above).
- **No objects detected at all**: lower `detection.conf_thresh` in the
  config (default `0.25`).
- **Very few fused instances / everything merged into one blob**: raise
  `object_fusion.sim_threshold` (default `0.55`) or lower
  `object_fusion.vis_weight`.

## Running the tests

Unit tests (pure-Python logic, fast, no GPU/checkpoint needed):
```bash
python -m pytest tests/unit -v   # if pytest is installed
# or, without pytest:
python -c "
import tests.unit.test_label_mapper as m1, tests.unit.test_crop_utils as m2, tests.unit.test_novic_decoder_stub as m3
for mod in (m1, m2, m3):
    for name in dir(mod):
        if name.startswith('test_'):
            getattr(mod, name)()
            print('PASS', name)
"
```

Integration test (runs real YOLOE detection + fusion + tagging against a
slice of the existing `freiburg1_desk` capture, no DroidCam needed —
several minutes, uses GPU):
```bash
python tests/integration/test_static_pipeline_freiburg.py
```
