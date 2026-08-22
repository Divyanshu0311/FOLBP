#! /bin/bash
# ROS 2 (Humble) launcher for the COHERENT benchmark.
#
# There is no roscore in ROS 2 -- discovery is peer-to-peer over DDS -- so this
# script just starts the two endpoints.  They find each other as long as both
# use the same ROS_DOMAIN_ID (set in ros2_env.sh).
#
# The two sides deliberately run under DIFFERENT interpreters:
#
#   action_publisher.py (LLM side) -> system python3.10, which is the only
#       python ROS 2 Humble's rclpy is built for.
#   sim.py (Isaac Sim side)        -> the `omnigibson` conda env (python3.7,
#       matching Isaac Sim 2022.2.0).  It cannot import rclpy, so it spawns the
#       ROS 2 sidecar in action_subscriber.py automatically.

BENCHMARK_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OMNIGIBSON_ROOT=$(cd "${BENCHMARK_ROOT}/.." && pwd)
WS_ROOT="${BENCHMARK_ROOT}/ros_hademo_ws"

# Scenes
	# house_double_floor_lower
	# Merom_1_int
	# Pomaria_1_int
	# grocery_store_cafe
	# restaurant_brunch

# _Task1, _Task2, ....

# Scene_name_Taskid
#demo1
# task_name="Merom_1_int_Task1"
#demo2
task_name='house_double_floor_lower_Task1'

export COHERENT_PATH="${COHERENT_PATH:-$(cd "${BENCHMARK_ROOT}/../.." && pwd)}"

if [ ! -f "${WS_ROOT}/install/setup.bash" ]; then
	echo "ROS 2 workspace is not built yet. Run:"
	echo "    cd ${WS_ROOT} && source ros2_env.sh && colcon build --symlink-install"
	exit 1
fi

# --- LLM side: real ROS 2 node under system python3.10 ----------------------
LLM_CMD="source '${WS_ROOT}/ros2_env.sh' && \
	export COHERENT_PATH='${COHERENT_PATH}' && \
	python3 '${WS_ROOT}/src/hademo/src/action_publisher.py' --task_name ${task_name}"

# `env -u LD_LIBRARY_PATH`: Isaac Sim's setup_conda_env.sh puts its own libs on
# LD_LIBRARY_PATH, and gnome-terminal then dies with
#   symbol lookup error: .../libpthread.so.0: undefined symbol: __libc_pthread_init
# because it picks up snap's libc. Launch the terminal with a clean loader path.
if command -v gnome-terminal >/dev/null 2>&1; then
	env -u LD_LIBRARY_PATH -u PYTHONPATH -u PYTHONHOME \
		gnome-terminal --tab --title "LLM" -- bash -c "${LLM_CMD}; exec bash" &
else
	# No GUI terminal (headless / ssh): run the LLM side in the background and log it.
	echo "gnome-terminal not found -- running the LLM node in the background."
	echo "  log: ${BENCHMARK_ROOT}/llm_node.log"
	env -u LD_LIBRARY_PATH -u PYTHONPATH -u PYTHONHOME \
		bash -c "${LLM_CMD}" > "${BENCHMARK_ROOT}/llm_node.log" 2>&1 &
fi
sleep 3s

# --- Simulator side: Isaac Sim / OmniGibson under the conda env -------------
conda activate omnigibson

# omnigibson is installed as a PEP 660 editable ("__editable___omnigibson_*_finder.py"),
# which exposes ONLY the `omnigibson` package -- not the source root. The
# 2022-era setuptools the authors used wrote a plain .pth naming the whole source
# directory, which also made `Benchmark` importable. Without that,
# omnigibson/envs/env_base.py's `from Benchmark.agents import *` dies with
#   ModuleNotFoundError: No module named 'Benchmark'
# Isaac's setup_python_env.sh only ever APPENDS to PYTHONPATH, so this survives
# the conda activate hook above.
export PYTHONPATH="${OMNIGIBSON_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# COHERENT ships a custom Isaac extension, omni.isaac.quadrotor, that gets copied
# into isaac_sim-2022.2.0/exts.  Isaac's setup_python_env.sh hard-codes the list
# of extension paths at INSTALL time, so it lists every stock ext but knows
# nothing about ones added later -- hence
#   ModuleNotFoundError: No module named 'omni.isaac.quadrotor'
# (omni.isaac.quadruped needs no such entry: it IS stock, and COHERENT only drops
# extra files -- a1arm_classes.py, qp_arm_controller.py -- inside it.)
ISAAC_ROOT="${ISAAC_PATH:-${HOME}/.local/share/ov/pkg/isaac_sim-2022.2.0}"
QUADROTOR_EXT="${ISAAC_ROOT}/exts/omni.isaac.quadrotor"
if [ -d "${QUADROTOR_EXT}" ]; then
	export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}${QUADROTOR_EXT}"
else
	echo "WARNING: ${QUADROTOR_EXT} not found -- the quadrotor agents will fail to import."
fi

# Call the env's interpreter by path. A pyenv shim earlier on PATH shadows the
# conda env's python3 even after `conda activate`, and the resulting 3.10 then
# tries to load Isaac's cp37 numpy prebundle and dies with
#   ModuleNotFoundError: No module named 'numpy.core._multiarray_umath'
OMNIGIBSON_PY="${CONDA_PREFIX}/bin/python"
if [ ! -x "${OMNIGIBSON_PY}" ]; then
	echo "Could not find the omnigibson env interpreter at ${OMNIGIBSON_PY}."
	echo "Did 'conda activate omnigibson' succeed?"
	exit 1
fi
echo "Simulator interpreter: $("${OMNIGIBSON_PY}" --version 2>&1) (${OMNIGIBSON_PY})"

"${OMNIGIBSON_PY}" "${BENCHMARK_ROOT}/sim.py" --task_name ${task_name}
