#!/usr/bin/env bash
# plot_node 대안 실행 (ros2 run이 안 될 때)
# 사용: ./run_plot_node.sh [record.json 경로]
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WS_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$WS_ROOT"
source /opt/ros/humble/setup.bash
[ -f install/setup.bash ] && source install/setup.bash
exec python3 -m visualizer.plot "$@"
