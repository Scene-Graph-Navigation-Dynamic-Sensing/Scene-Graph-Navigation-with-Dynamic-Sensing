#!/usr/bin/env bash
# Runs every pipeline on IMG_6100.mp4 sequentially on one GPU (fair timing), then evaluates.
set -u
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES=${GPU:-0}
DTD_PY=/home/ravindu/miniconda3/envs/scene_graph/bin/python
LOC_PY=/home/ravindu/miniconda3/envs/locate-env/bin/python
mkdir -p outputs/logs
ln -sf /home/ravindu/SceneGraphNav/models/checkpoints/mobileclip2_b.ts mobileclip2_b.ts

run() { local name=$1; shift; echo "=== $name"; "$@" > "outputs/logs/$name.log" 2>&1 || echo "FAILED: $name (see outputs/logs/$name.log)"; }

run p1_dtd_every_frame        $DTD_PY pipelines/dtd_yolo_clip_novic.py --policy every --fp16
run p4_dtd_confirm_then_track $DTD_PY pipelines/dtd_yolo_clip_novic.py --policy confirm --fp16
run p4b_dtd_confirm_strict    $DTD_PY pipelines/dtd_yolo_clip_novic.py --policy confirm --fp16 --confirm-votes 5 --confirm-frac 0.7 --max-votes 15 --name p4b_dtd_confirm_strict
run p2_ram_yoloe_botsort      $LOC_PY pipelines/ram_yoloe_botsort.py
run p3a_ram_gdino_sam2_botsort    $LOC_PY pipelines/ram_gdino_sam2.py --tracker botsort --amp
run p3b_ram_gdino_sam2_videotrack $LOC_PY pipelines/ram_gdino_sam2.py --tracker sam2 --det-every 10 --amp
echo "all runs done"
$LOC_PY evaluate.py && $LOC_PY make_grid.py
