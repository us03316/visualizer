#!/usr/bin/env bash
# record_node 대안 실행 (ros2 run이 안 될 때)
# 사용: ./run_record_node.sh   또는  bash run_record_node.sh
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# visualizer/scripts -> visualizer -> src -> ros2_ws
WS_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$WS_ROOT"
source /opt/ros/humble/setup.bash
[ -f install/setup.bash ] && source install/setup.bash
exec python3 -m visualizer.record "$@"
