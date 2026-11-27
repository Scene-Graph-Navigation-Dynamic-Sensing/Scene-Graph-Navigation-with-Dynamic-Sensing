#!/usr/bin/env bash
# End to end in Isaac Sim: language command -> task plan -> A* path -> Go2 walks it.
#
#   scripts/sim_go_to.sh "go to the forklift"                        # scene synth_stride2_test_v5
#   scripts/sim_go_to.sh "go to the forklift" <scene> [follower args]  # e.g. --stop_distance 0.8
#
# Needs: the isaac-go2-ros2 container running with the env the scene was
# recorded in (synthetic_dataset_01..04 -> full-warehouse in cfg/sim.yaml).
#
# Start point: the robot's current sim pose. If A* cannot start there (e.g.
# the Go2 spawns outside the mapped area), it falls back to the first camera
# pose of the scene's dataset, and the robot first walks straight from where
# it stands to that start point (that first leg is NOT checked against the map).
set -euo pipefail

CMD=${1:?usage: sim_go_to.sh \"<command>\" [scene] [follower args]}
SCENE=${2:-synth_stride2_test_v5}
shift $(( $# < 2 ? $# : 2 ))

REPO=/home/ravindu/SceneGraphNav/code
SIM=/home/ravindu/isaac_ws/src/isaac-go2-ros2
cd "$REPO"
set +u   # conda's activate scripts use unset variables
source /home/ravindu/miniconda3/etc/profile.d/conda.sh
conda activate scene_graph
set -u
export PYTHONPATH=/home/ravindu/SceneGraphNav/external/Scene-Graph-Generation${PYTHONPATH:+:$PYTHONPATH}

SG=outputs/scene_graphs/$SCENE.pkl
ALIGN=data/processed/$SCENE/gravity_alignment.json
[[ -f $SG ]] || { echo "no scene graph $SG"; exit 1; }

echo "== 1/4 task planning: \"$CMD\""
python scripts/run_task_planning.py --scene_graph "$SG" --command "$CMD" | tail -3
TP=$(ls -t outputs/task_plans/${SCENE}_*.json | head -1)

echo "== 2/4 robot pose in sim"
POSE=$(docker exec isaac-go2-ros2 bash -lc "source /opt/ros/humble/setup.bash; \
  timeout 20 /usr/bin/python3 /home/ubuntu/isaac-go2-ros2/waypoint_follower.py --print_pose 2>/dev/null" | tail -1)
[[ -n $POSE ]] || { echo "no /unitree_go2/odom -- is the sim running?"; exit 1; }
START_MAP=$(python - "$POSE" "$ALIGN" <<'EOF'
import json, os, sys
import numpy as np
x, y, z, _ = map(float, sys.argv[1].split())
T = np.asarray(json.load(open(sys.argv[2]))["transform"]) if os.path.exists(sys.argv[2]) else np.eye(4)
p = T @ np.array([x, y, 0.0, 1.0])        # sim floor point -> map frame
print(f"{p[0]:.3f} {p[1]:.3f} 0.0")
EOF
)
echo "robot sim pose (x y z yaw): $POSE  -> map start: $START_MAP"

echo "== 3/4 A* path"
set +e
python scripts/run_navigation.py --scene_graph "$SG" --task_plan "$TP" --start_xyz $START_MAP | tail -5
RC=${PIPESTATUS[0]}
if [[ $RC -eq 2 ]]; then
    START_MAP=$(python -c "import numpy as np; p=np.loadtxt('data/processed/$SCENE/poses/000000.txt')[:3,3]; print(f'{p[0]:.3f} {p[1]:.3f} 0.0')")
    echo "no path from the robot's pose (outside the map?) -- retrying from dataset start $START_MAP"
    echo "robot will first walk straight to that start point"
    python scripts/run_navigation.py --scene_graph "$SG" --task_plan "$TP" --start_xyz $START_MAP | tail -5
    RC=${PIPESTATUS[0]}
fi
set -e
[[ $RC -eq 0 ]] || { echo "A* planning failed (exit $RC)"; exit $RC; }
NP=$(ls -t outputs/navigation_plans/${SCENE}_*.json | head -1)

echo "== 4/4 Go2 follows $NP"
"$SIM/go_to_waypoints.sh" "$REPO/$NP" "$SCENE" "$@"
