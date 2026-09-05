#!/usr/bin/env bash
# Record one demo video: simulator + planner, then compose.
#
#   tools/record_demo.sh <merom|house> <pefa|folbp> [extra sim flags...]
#
# Starts the same three processes the sim demo always uses -- recording adds no
# new plumbing between them, only two extra files on disk:
#
#   1. sim.py              (conda `omnigibson`, py3.7)  --record  -> sim.mp4 + frames.jsonl
#   2. action_publisher.py (system python3.10, rclpy)   --mode ws -> WS <-> ROS 2 bridge
#   3. PEFA|FOLBP main.py  (conda `coherent`, py3.10)   --mode ws -> events.jsonl
#
# The two streams are joined on wall clock afterwards by tools/compose_video.py.
set -euo pipefail

TASK_ARG="${1:?usage: record_demo.sh <merom|house> <pefa|folbp>}"
FRAMEWORK="${2:?usage: record_demo.sh <merom|house> <pefa|folbp>}"
shift 2

export COHERENT_PATH="${COHERENT_PATH:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$COHERENT_PATH"

case "$TASK_ARG" in
  merom) TASK_NAME=Merom_1_int_Task1;              PLANNER_ENV=env_merom ;;
  house) TASK_NAME=house_double_floor_lower_Task1; PLANNER_ENV=env_house ;;
  *) echo "unknown task '$TASK_ARG' (expected merom|house)" >&2; exit 1 ;;
esac
case "$FRAMEWORK" in
  pefa|folbp) : ;;
  *) echo "unknown framework '$FRAMEWORK' (expected pefa|folbp)" >&2; exit 1 ;;
esac

RUN_DIR="$COHERENT_PATH/results/videos/${TASK_ARG}-${FRAMEWORK}-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$RUN_DIR"
echo "[demo] run dir: $RUN_DIR"

PLANNER_PY="${COHERENT_PLANNER_PY:-$HOME/miniconda3/envs/coherent/bin/python}"
SIM_PY="${COHERENT_SIM_PY:-$HOME/miniconda3/envs/omnigibson/bin/python}"
WS_ROOT="$COHERENT_PATH/OmniGibson/Benchmark/ros_hademo_ws"
PORT="${COHERENT_WS_PORT:-8765}"

[ -f "$COHERENT_PATH/setup_env.sh" ] && source "$COHERENT_PATH/setup_env.sh" >/dev/null

# A bridge left over from an earlier run keeps the port, the new bridge fails to
# bind, and the planner then silently drives the *previous* run's simulator (or
# nothing at all). Refuse to start rather than record that.
if ss -ltn 2>/dev/null | grep -q ":$PORT "; then
  echo "port $PORT is already in use -- a bridge from an earlier run is still alive." >&2
  echo "  ps -eo pid=,args= | grep action_publisher" >&2
  exit 1
fi

cleanup() {
  echo "[demo] shutting down"
  for pid in "${SIM_PID:-}" "${BRIDGE_PID:-}"; do
    [ -n "$pid" ] || continue
    kill "$pid" 2>/dev/null || true
  done
  # The bridge sits in an asyncio server loop that does not act on SIGTERM, and a
  # survivor holds the port against the next run. Give them a moment, then insist.
  sleep 3
  for pid in "${SIM_PID:-}" "${BRIDGE_PID:-}"; do
    [ -n "$pid" ] || continue
    kill -0 "$pid" 2>/dev/null && kill -9 "$pid" 2>/dev/null || true
  done
}
trap cleanup EXIT

# --- 2. bridge -------------------------------------------------------------
# Started first: the planner needs something to connect to, and the sim's ROS 2
# handshake is TRANSIENT_LOCAL, so a late joiner still receives the last message.
bash -c "source '$WS_ROOT/ros2_env.sh' >/dev/null 2>&1
         export COHERENT_PATH='$COHERENT_PATH'
         cd '$WS_ROOT/src/hademo/src'
         exec python3 action_publisher.py --task_name $TASK_NAME --mode ws --ws_port $PORT" \
  > "$RUN_DIR/bridge.log" 2>&1 &
BRIDGE_PID=$!
echo "[demo] bridge pid $BRIDGE_PID -> $RUN_DIR/bridge.log"

for _ in $(seq 1 60); do
  ss -ltn 2>/dev/null | grep -q ":$PORT " && break
  sleep 1
done

# --- 1. simulator ----------------------------------------------------------
# The omnigibson env must be *activated*, not merely pointed at: its activate.d
# hook sources Isaac Sim's setup_conda_env.sh, which is what puts the `omni`
# packages on PYTHONPATH.  Calling the interpreter directly gets
# "ModuleNotFoundError: No module named 'omni'".  Everything below mirrors
# Benchmark/run.sh, including the two PYTHONPATH entries it explains at length:
# the source root (so `from Benchmark.agents import *` resolves under the PEP 660
# editable install) and the custom omni.isaac.quadrotor extension.
#
# 1280x720 capture: the 960x540 default upscales badly into a 1920-wide frame.
SIM_CMD="source '$HOME/miniconda3/etc/profile.d/conda.sh'
conda activate omnigibson
export PYTHONPATH='$COHERENT_PATH/OmniGibson'\${PYTHONPATH:+:\$PYTHONPATH}
ISAAC_ROOT=\"\${ISAAC_PATH:-\$HOME/.local/share/ov/pkg/isaac_sim-2022.2.0}\"
QUADROTOR_EXT=\"\$ISAAC_ROOT/exts/omni.isaac.quadrotor\"
[ -d \"\$QUADROTOR_EXT\" ] && export PYTHONPATH=\"\${PYTHONPATH:+\$PYTHONPATH:}\$QUADROTOR_EXT\"
export COHERENT_PATH='$COHERENT_PATH'
# Call the env interpreter by path: a pyenv shim on PATH shadows it even after
# activate, and the resulting 3.10 dies on Isaac's cp37 numpy prebundle.
exec \"\$CONDA_PREFIX/bin/python\" '$COHERENT_PATH/OmniGibson/Benchmark/sim.py' \\
     --task_name $TASK_NAME --record --record_dir '$RUN_DIR' \\
     --viewer_size 1280 720 $*"

bash -c "$SIM_CMD" > "$RUN_DIR/sim.log" 2>&1 &
SIM_PID=$!
echo "[demo] sim pid $SIM_PID -> $RUN_DIR/sim.log  (Isaac Sim takes minutes to boot)"

# The simulator is ready once it has captured its first frame.
for _ in $(seq 1 900); do
  [ -s "$RUN_DIR/frames.jsonl" ] && break
  kill -0 "$SIM_PID" 2>/dev/null || { echo "[demo] sim died, see $RUN_DIR/sim.log" >&2; exit 1; }
  sleep 1
done
echo "[demo] sim is rendering; starting the planner"

# --- 3. planner ------------------------------------------------------------
DIR_UPPER=$(echo "$FRAMEWORK" | tr '[:lower:]' '[:upper:]')

# FOLBP's reported configuration is the `folbp` arm in bench/arms.py, which runs
# --executor_mode deterministic: the Z3-certified plan is dispatched verbatim and
# there is no per-step robot LLM executor at all. That is what makes it ~1-2 LLM
# calls per task rather than PEFA's ~55, and it is the number the README reports.
# args.py defaults to pefa_llm for compatibility with the ablation arms, so the
# demo has to ask for the real configuration explicitly or it films the wrong system.
PLANNER_FLAGS=()
if [ "$FRAMEWORK" = "folbp" ]; then
  PLANNER_FLAGS+=(--executor_mode deterministic --verify z3 --repair symbolic --cdcl)
fi

( cd "$COHERENT_PATH/src/experiment/$DIR_UPPER" && \
  "$PLANNER_PY" main.py --env "$PLANNER_ENV" --task 0 \
      --mode ws --ws_url "ws://127.0.0.1:$PORT" \
      --event_log "$RUN_DIR/events.jsonl" \
      "${PLANNER_FLAGS[@]}" ) 2>&1 | tee "$RUN_DIR/planner.log"

echo "[demo] planner finished; letting the sim settle"
sleep 5
cleanup
trap - EXIT
sleep 2

# --- 4. compose ------------------------------------------------------------
# PEFA's content IS the reasoning -- an Oracle proposal, the executor's
# chain-of-thought, then a judge verdict, every single step, ~55 calls a task.
# Those stretches have to play at real time or the panel is unreadable. FOLBP
# reasons once and then executes, so it can afford 1.5x.
COMPOSE_PACE=(--motion_speed 8 --think_speed 1.5)
[ "$FRAMEWORK" = "pefa" ] && COMPOSE_PACE=(--motion_speed 8 --think_speed 1.0)

"$PLANNER_PY" "$COHERENT_PATH/tools/compose_video.py" \
    --run_dir "$RUN_DIR" --framework "$FRAMEWORK" "${COMPOSE_PACE[@]}"

echo "[demo] done -> $RUN_DIR/final.mp4"
