#!/usr/bin/env bash
# Record task 2 (house_double_floor_lower_Task1) end to end, unattended.
#
#   tools/record_task2.sh [folbp|pefa] [attempts]
#
# Self-contained: cleans up survivors, runs the three processes, retries a failed
# take, composes, and prints where the video landed. Safe to launch with nohup and
# walk away.
set -uo pipefail

export COHERENT_PATH="${COHERENT_PATH:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$COHERENT_PATH"

FRAMEWORK="${1:-folbp}"
ATTEMPTS="${2:-3}"
SCENE=house
TASK_NAME=house_double_floor_lower_Task1
PLANNER_ENV=env_house
PORT="${COHERENT_WS_PORT:-8765}"
PLANNER_PY="${COHERENT_PLANNER_PY:-$HOME/miniconda3/envs/coherent/bin/python}"

[ -f setup_env.sh ] && source setup_env.sh >/dev/null

banner() { echo; echo "======== $* ========"; echo; }

reap() {
  /usr/bin/python3 - <<'PY'
import os, signal, subprocess, time
PATS = ("action_publisher.py", "Benchmark/sim.py", "FOLBP/main.py",
        "PEFA/main.py", "compose_video.py")
me = {os.getpid(), os.getppid()}
out = subprocess.run(["ps", "-eo", "pid=,args="], capture_output=True, text=True).stdout
for line in out.splitlines():
    pid_s, _, args = line.strip().partition(" ")
    try:
        pid = int(pid_s)
    except ValueError:
        continue
    if pid in me or "shell-snapshots" in args:
        continue
    if any(p in args for p in PATS):
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass
time.sleep(3)
PY
}

banner "task 2 / $FRAMEWORK — up to $ATTEMPTS attempt(s)"

# ---------------------------------------------------------------- preflight
# These three have never driven the house scene together, so check the data is
# even present before booting Isaac Sim (which costs minutes).
"$PLANNER_PY" - <<PY || { echo "preflight FAILED"; exit 1; }
import json, sys
env = json.load(open("src/experiment/$([ "$FRAMEWORK" = pefa ] && echo PEFA || echo FOLBP)/env/env_house.json"))[0]
mac = json.load(open("OmniGibson/Benchmark/action_macros.json"))["$TASK_NAME"]
assert env["task_name"] == "$TASK_NAME", env["task_name"]
assert env["name_mapping"], "no name_mapping"
assert mac, "no action macros"
print("preflight OK: %d nodes, %d edges, %d name mappings, %d macros"
      % (len(env["init_graph"]["nodes"]), len(env["init_graph"]["edges"]),
         len(env["name_mapping"]), len(mac)))
PY

RESULT=FAILED
FINAL=""

for attempt in $(seq 1 "$ATTEMPTS"); do
  banner "attempt $attempt of $ATTEMPTS"
  reap

  if ss -ltn 2>/dev/null | grep -q ":$PORT "; then
    echo "port $PORT still held after reap; aborting"
    break
  fi

  tools/record_demo.sh "$SCENE" "$FRAMEWORK"
  rc=$?

  RUN_DIR=$(ls -dt "results/videos/${SCENE}-${FRAMEWORK}-"* 2>/dev/null | head -1)
  echo "[task2] record_demo exit=$rc run_dir=$RUN_DIR"

  if [ -n "$RUN_DIR" ] && grep -qx "success" "$RUN_DIR/planner.log" 2>/dev/null; then
    # record_demo composes on its way out; only re-do it if that did not finish.
    if [ ! -s "$RUN_DIR/final.mp4" ] || \
       ! ffprobe -v error -show_entries format=duration -of csv=p=0 "$RUN_DIR/final.mp4" >/dev/null 2>&1; then
      echo "[task2] composing"
      "$PLANNER_PY" tools/compose_video.py --run_dir "$RUN_DIR" --framework "$FRAMEWORK"
    fi
    RESULT=SUCCESS
    FINAL="$RUN_DIR/final.mp4"
    break
  fi

  echo "[task2] attempt $attempt did not reach success"
  [ -n "$RUN_DIR" ] && tail -25 "$RUN_DIR/planner.log" 2>/dev/null | sed 's/^/    /'
done

reap

banner "task 2 / $FRAMEWORK — $RESULT"
if [ "$RESULT" = SUCCESS ]; then
  echo "  video : $COHERENT_PATH/$FINAL"
  ffprobe -v error -show_entries format=duration -show_entries stream=width,height,codec_name \
          -of default=nw=1 "$FINAL" 2>/dev/null | sed 's/^/  /'
  ls -la "$FINAL" | awk '{printf "  size  : %.1f MB\n", $5/1048576}'
  grep -E "^steps:" "$(dirname "$FINAL")/planner.log" | head -1 | sed 's/^/  /'
  "$PLANNER_PY" -c "
import json
ev=[json.loads(l) for l in open('$(dirname "$FINAL")/events.jsonl')]
c=[e for e in ev if e['stage']=='LLM_CALL']
print('  LLM calls: %d %s' % (len(c), [x['title'] for x in c]))" 2>/dev/null
else
  echo "  no take reached success in $ATTEMPTS attempt(s)"
  echo "  logs: results/videos/${SCENE}-${FRAMEWORK}-*/"
fi
