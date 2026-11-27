# Mask2Former front end vs YOLOE on Replica

`mask2former_replica_results.pdf` compares four detection front ends on all 8 Replica scenes (2 runs each):
YOLOE alone, Mask2Former + YOLOE, **Mask2Former alone** and YOLOE + label filter. Mask2Former alone gave the
highest object precision (86.2%), the fewest non-object nodes and duplicates, the best naming and embedding scores
and the shortest run time, at lower recall (47.3%, no wall plugs / switches / vents). It has been the default in
`configs/pipeline.yaml` since 2026-11-27.

| Configuration | Override | Results (not in Git) |
|---|---|---|
| Baseline (YOLOE) | YOLOE backend | `results/replica_rgbfix/run{1,2}` |
| Mask2Former + YOLOE | `configs/replica_variants/panoptic_yoloe_union.yaml` | `results/frontend_choice/panprobe/run{1,2}` |
| Mask2Former (default) | `configs/replica_variants/panoptic_only.yaml` | `results/option2_panoptic/run{1,2}` |
| YOLOE + label filter | `configs/replica_variants/yoloe_labels.yaml` | `results/option3_yoloe_labels/run{1,2}` |

Rebuild the report parts from the run folders (writes `parts/`):

```bash
conda activate scene_graph
python evaluation/reports/mask2former_frontend/make_report.py
python evaluation/reports/mask2former_frontend/make_analysis.py
```

`composition.py` reads the Replica ground truth from `/home/ravindu/SceneGraphNav/datasets/replica/extracted` (server only).
