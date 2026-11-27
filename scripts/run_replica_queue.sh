#!/usr/bin/env bash
# Generic Replica job queue: one worker per GPU pulls lines from a jobs file and runs eval_replica_scene.py.
#   scripts/run_replica_queue.sh <jobs.txt> <log> <gpu> [<gpu> ...]
# Each job line: <tag> <results_root> <scene> <override.yaml> [<override.yaml> ...]
set -u
cd "$(dirname "$0")/.."
Q=$1; LOG=$2; shift 2
PY=/home/ravindu/miniconda3/envs/scene_graph/bin/python
export HF_HUB_OFFLINE=1
next_job() { exec 9>"$Q.lock"; flock 9; local l; l=$(head -n 1 "$Q"); [ -n "$l" ] && sed -i '1d' "$Q"; flock -u 9; echo "$l"; }
worker() {
  local gpu=$1
  while true; do
    local line; line=$(next_job); [ -z "$line" ] && break
    set -- $line; local tag=$1 root=$2 scene=$3; shift 3
    mkdir -p $root/$scene
    echo "$(date '+%F %T') start  $tag $scene (gpu $gpu)" >> $LOG
    if $PY scripts/eval_replica_scene.py --scene $scene --gpu $gpu --tag $tag --results_root $root --trim_processed \
         --override "$@" > $root/$scene/eval.log 2>&1; then
      echo "$(date '+%F %T') done   $tag $scene" >> $LOG
    else
      echo "$(date '+%F %T') FAILED $tag $scene" >> $LOG
    fi
  done
}
for g in "$@"; do worker $g & done
wait
echo "$(date '+%F %T') ALL_DONE" >> $LOG
