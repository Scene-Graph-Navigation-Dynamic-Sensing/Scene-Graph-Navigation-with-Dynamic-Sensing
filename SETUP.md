# Setup

## Layout on the lab server

Everything for this project lives in one folder, `/home/ravindu/SceneGraphNav/`:

| Folder | What | In Git? |
|---|---|---|
| `code/` | this repository | yes |
| `datasets/` | Replica (`replica/`), 3RScan (`3rscan/`), synthetic sets (`synthetic/`, links to `~/dataset/synthetic_dataset_0*`, owned by another account), prepared scenes (`raw/`, `processed/`), caches, test videos | no |
| `models/` | YOLOE, RAM++, MobileCLIP2, NOVIC weights | no |
| `external/` | `Scene-Graph-Generation` (team fork), `novic`, `DovSG`: separate Git repos, with their own weights | no |
| `results/` | evaluation runs (`scripts/eval_replica_scene.py`, report inputs) | no |
| `outputs/` | pipeline outputs: scene graphs, visualizations, plans | no |
| `reports/` | finished reports and videos (e.g. `mask2former_replica_results.pdf`) | no |

Inside `code/`, the names `data`, `models`, `external`, `results`, `outputs` are symlinks to those folders, so
every relative path in the code (`data/raw/...`, `results/...`) works unchanged. Configs use absolute paths
under `/home/ravindu/SceneGraphNav/` for the external repos and weights, so every clone shares one copy.

Old locations still work through symlinks: `~/Malshan/Scene-Graph-Navigation-with-Dynamic-Sensing` ->
`code/`, `~/dataset/Replica` -> `datasets/replica`, `~/dataset/3Rscan` -> `datasets/3rscan`. The full list of
moves is in `/home/ravindu/SceneGraphNav/MIGRATION_MANIFEST.tsv`.

## Your own working copy (each team member)

```bash
git clone git@github.com:<org>/<repo>.git ~/work/scene-graph-nav   # anywhere you like
cd ~/work/scene-graph-nav
bash scripts/setup_links.sh                # link the shared data, weights and external repos
#   bash scripts/setup_links.sh --own-results   # keep your results/ and outputs/ separate
git config user.name  "Your Name"
git config user.email "you@example.com"    # the email on your GitHub account
```

## Environment

```bash
conda activate scene_graph                 # /home/ravindu/miniconda3/envs/scene_graph
pip install -r requirements.txt            # only fills gaps; the env already has most of it
export HF_HUB_OFFLINE=1                    # Mask2Former, Depth Anything V3 and DFN5B CLIP come from ~/.cache/huggingface
pytest tests/unit -q                       # quick check
```

## Weights (in `models/` and `external/`)

- Mask2Former `facebook/mask2former-swin-large-ade-panoptic`: Hugging Face cache (default detector)
- `models/checkpoints/yoloe-26{s,m,l}-seg-pf.pt`: YOLOE backend / YOLOE union only
- `external/Scene-Graph-Generation/yoloe-26x-seg-pf.pt`, `.../DROID-SLAM/droid.pth`
- `external/novic/outputs/ovod_20240628_142131/ovod_chunk0433_20240630_235415.train`: NOVIC decoder
