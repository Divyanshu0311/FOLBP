#!/usr/bin/env bash
# Record every demo video: both scenes x both planners.
#
#   tools/record_all.sh                 # all four
#   tools/record_all.sh merom-folbp house-folbp
#
# Runs strictly one at a time -- there is one Isaac Sim, one GPU and one
# WebSocket port, so these cannot overlap. Each take is retried once, because the
# franka's final placement is occasionally not reachable from where the drone
# happened to land and the skill then stalls until the bridge's 60 s timeout.
set -uo pipefail

export COHERENT_PATH="${COHERENT_PATH:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$COHERENT_PATH"

ALL=(merom-folbp merom-pefa house-folbp house-pefa)
TARGETS=("$@")
[ ${#TARGETS[@]} -eq 0 ] && TARGETS=("${ALL[@]}")

RETRIES="${COHERENT_RETRIES:-1}"
SUMMARY=""

for target in "${TARGETS[@]}"; do
  scene="${target%%-*}"
  framework="${target##*-}"

  ok=no
  for attempt in $(seq 0 "$RETRIES"); do
    [ "$attempt" -gt 0 ] && echo "[all] retrying $target (attempt $((attempt + 1)))"
    echo "[all] === $target ==="

    # A survivor from the previous take holds the port and the next bridge then
    # fails to bind, which record_demo.sh refuses to start on.
    /usr/bin/python3 - <<'PY'
import os, signal, subprocess, time
PATS = ("action_publisher.py", "Benchmark/sim.py", "FOLBP/main.py", "PEFA/main.py")
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

    if tools/record_demo.sh "$scene" "$framework"; then
      run_dir=$(ls -dt "results/videos/${scene}-${framework}-"* 2>/dev/null | head -1)
      if grep -qx "success" "$run_dir/planner.log" 2>/dev/null; then
        ok=yes
        SUMMARY+=$'\n'"  $target  SUCCESS  $run_dir/final.mp4"
        break
      fi
      echo "[all] $target: the planner reported failure, not keeping this take"
    else
      echo "[all] $target: the run itself failed"
    fi
  done

  [ "$ok" = no ] && SUMMARY+=$'\n'"  $target  FAILED after $((RETRIES + 1)) attempt(s)"
done

echo
echo "[all] ============ summary ============"
echo "$SUMMARY"
