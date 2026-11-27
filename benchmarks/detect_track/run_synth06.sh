#!/usr/bin/env bash
# All pipelines on the first 100 RGB frames of synthetic_dataset_06, sequentially on GPU $GPU, then evaluate.
set -u
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES=${GPU:-0}
SRC=/home/ravindu/SceneGraphNav/datasets/synthetic/synthetic_dataset_06/rgb
OUTR=outputs_synth06
N=${FRAMES:-100}
DTD_PY=/home/ravindu/miniconda3/envs/scene_graph/bin/python
LOC_PY=/home/ravindu/miniconda3/envs/locate-env/bin/python
COMMON="--source $SRC --out-root $PWD/$OUTR --frames $N"
mkdir -p $OUTR/logs
ln -sf /home/ravindu/SceneGraphNav/models/checkpoints/mobileclip2_b.ts mobileclip2_b.ts

run() { local name=$1; shift; echo "=== $name"; "$@" > "$OUTR/logs/$name.log" 2>&1 || echo "FAILED: $name (see $OUTR/logs/$name.log)"; }

run p1_dtd_every_frame        $DTD_PY pipelines/dtd_yolo_clip_novic.py --policy every --fp16 $COMMON
run p4_dtd_confirm_then_track $DTD_PY pipelines/dtd_yolo_clip_novic.py --policy confirm --fp16 $COMMON
run p4b_dtd_confirm_strict    $DTD_PY pipelines/dtd_yolo_clip_novic.py --policy confirm --fp16 --confirm-votes 5 --confirm-frac 0.7 --max-votes 15 --name p4b_dtd_confirm_strict $COMMON
run p2_ram_yoloe_botsort      $LOC_PY pipelines/ram_yoloe_botsort.py $COMMON
run p3a_ram_gdino_sam2_botsort    $LOC_PY pipelines/ram_gdino_sam2.py --tracker botsort --amp $COMMON
run p3b_ram_gdino_sam2_videotrack $LOC_PY pipelines/ram_gdino_sam2.py --tracker sam2 --det-every 10 --amp $COMMON
echo "all runs done"
$LOC_PY evaluate.py --gt gt_synth06/gt.json --out-root $OUTR | tee $OUTR/results_table.md
$LOC_PY make_grid.py --out-root $OUTR
$LOC_PY export_objects.py --out-root $OUTR --source $SRC
