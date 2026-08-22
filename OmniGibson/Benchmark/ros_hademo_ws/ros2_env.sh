#!/bin/bash
# Source this before building or running the ROS 2 side of the benchmark.
#
#   source OmniGibson/Benchmark/ros_hademo_ws/ros2_env.sh
#
# ROS 2 Humble's rclpy / rosidl tooling is compiled against the SYSTEM
# python3.10 (/usr/bin/python3).  A pyenv or conda interpreter earlier on PATH
# shadows it and the build dies with "No module named 'em'" (or rclpy imports
# fail at runtime).  So we push the system interpreter to the front first.

_HADEMO_WS="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export HADEMO_WS="${_HADEMO_WS}"

# Take pyenv/conda shims out of the way for this shell.
export PYENV_VERSION=system
PATH="$(echo "$PATH" | tr ':' '\n' | grep -vE '(\.pyenv|miniconda|anaconda)' | paste -sd:)"
export PATH="/usr/bin:/bin:${PATH}"
unset PYTHONHOME
unset PYTHONPATH

source /opt/ros/humble/setup.bash
if [ -f "${_HADEMO_WS}/install/setup.bash" ]; then
  source "${_HADEMO_WS}/install/setup.bash"
fi

# Keep both sides on the same DDS domain; override if it clashes on your LAN.
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"

# Interpreter the sim-side sidecar is launched with (see action_subscriber.py).
export HADEMO_ROS2_PYTHON="${HADEMO_ROS2_PYTHON:-/usr/bin/python3}"
