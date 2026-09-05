# Demo videos

Records a simulator run with the planner's reasoning shown underneath it:
Isaac Sim on top, a terminal-style transcript band below carrying each stage as
it fires.

```bash
tools/record_demo.sh merom folbp      # -> results/videos/merom-folbp-<ts>/final.mp4
tools/record_demo.sh merom pefa
tools/record_demo.sh house folbp
tools/record_demo.sh house pefa
```

## How it fits together

Recording adds no plumbing between the processes. The same three the sim demo
always used, each now writing one extra file:

| # | process | interpreter | new output |
|---|---|---|---|
| 1 | `Benchmark/sim.py --record` | conda `omnigibson`, py3.7 | `sim.mp4`, `frames.jsonl` |
| 2 | `action_publisher.py --mode ws` | system python3.10 (`rclpy`) | — |
| 3 | `PEFA\|FOLBP/main.py --mode ws --event_log` | conda `coherent`, py3.10 | `events.jsonl` |

The two sides cannot share objects — different interpreters — but they share a
wall clock. Every frame and every stage event carries `time.time()`, and
`compose_video.py` joins them afterwards. That is the entire synchronisation
mechanism, and it is why compositing is a separate offline pass: the band can be
redesigned twenty times without rebooting Isaac Sim.

    sim.py ──► sim.mp4 + frames.jsonl ─┐
                                       ├─► compose_video.py ──► final.mp4
    planner ─► events.jsonl ───────────┘

## Composing on its own

The recording step is expensive; composing is not. To iterate on the band, keep a
run directory and re-run just this:

```bash
python tools/compose_video.py --run_dir results/videos/<run> --framework folbp
python tools/compose_video.py --run_dir results/videos/<run> --framework folbp --limit 400
python tools/compose_video.py --run_dir results/videos/<run> --framework folbp --timelapse
```

`--limit N` renders only the first N frames — seconds per pass while tuning.

Output is 1920x1400 — sim scaled into 1920x1080, a 320px band underneath.

## Pacing

The simulator does not render in real time — the RTX renderer manages about
5 fps here — and the run has two phases that want opposite treatment. A robot
crossing a room is a minute of walking that a viewer takes in within seconds. An
Oracle call is four seconds of wall clock carrying a nine-step plan that needs
time on screen to read. One uniform rate has to fail one of them.

`--pace adaptive` (the default) tells them apart from the bridge's own traffic:
`WS_SEND` is the action arriving at the simulator, the matching `WS_RECV` is it
finishing, and everything between the two is motion. No heuristic involved.

| | default | effect |
|---|---|---|
| `--motion_speed` | 8.0x | through the walking and flying |
| `--think_speed` | 1.5x | through the reasoning, scene static |

On the Merom FOLBP run: 105s of motion becomes 13s, 60s of reasoning becomes 40s,
so the finished video is 53 seconds.

`--pace native` gives one output frame per captured sim frame — exactly
`sim.mp4`'s own rate, uniform, ~6x real time. `--pace realtime` gives true
wall-clock duration, which holds each sim frame for several output frames and
looks like judder. `--timelapse` additionally thins stretches where no stage
fires, and works with either.

Note that animation (the typewriter, the five-second plan dwell) runs on *video*
time, not wall clock. Under native pacing one output frame is ~0.2s of wall
clock, so a dwell expressed in real seconds would flash past in well under one.

## Which configuration gets filmed

`record_demo.sh` runs FOLBP with `--executor_mode deterministic --verify z3
--repair symbolic --cdcl`, which is the `folbp` arm in
[`bench/arms.py`](../src/experiment/bench/arms.py) — the configuration every
number in the README refers to. In it the Z3-certified plan is dispatched
verbatim and **there is no per-step robot LLM executor at all**, which is why the
task costs one or two LLM calls rather than PEFA's ~55.

`FOLBP/args.py` defaults `--executor_mode` to `pefa_llm` instead, because the
ablation arms need that path. So the flags are not optional decoration: without
them the demo films a FOLBP that makes one Oracle call *plus* one executor call
per step, and reports ten calls where the paper reports one.

The header's call counter reads `LLM_CALL` events, emitted from
`bench/llm_meter.py` — the single context manager every physical API call in both
frameworks passes through. Counting stage log lines instead overstates it badly:
FOLBP prints one `ORACLE` line per plan step, so a single Oracle call looks like
eleven.

## What the band shows

PEFA gets one block per stage per step (Oracle → Robot LLM Executor → Judge),
condensed to the decisive line. Steps are never merged: watching each one get
proposed, grounded and judged in turn is the point.

FOLBP returns its whole plan in a single Oracle call, so `ORACLE_PLAN` clears the
band and prints that plan **whole and uncondensed** (two columns when long), holds
it for five seconds, and then lets the Z3 and Validator verdicts stamp onto it.

## Camera

`FOLLOW_CAM_SHOTS` in `Benchmark/sim.py` holds a shot list per task, cutting from
one shot to the next when the carrier picks up the payload. Retune without
editing source:

```bash
tools/record_demo.sh merom folbp --follow_pivot 1.8 6.0 1.7 --follow_smoothing 0.06
tools/record_demo.sh house folbp --follow_trail 2.4 --follow_rise 0.8
tools/record_demo.sh merom folbp --no_follow_camera        # fixed pose from main()
```

Note that `trail` shots walk back along the path the subject has already flown, so
they are wrong immediately after a takeoff — the path is still on the floor and
the camera ends up under the drone looking at the ceiling. Merom uses pivots for
that reason; the house demo's flight is long enough to earn a trail.

## Action macros

`Benchmark/action_macros.json` maps one planner action to the sequence of sim
commands that carries it out. The house demo needs this: its fridge is only
openable from a pose reached via two waypoints, its door is a joint link, and the
flight to the terrace routes through a doorway waypoint. None of that is a
planning decision, so it does not belong in the planner's action space — and
padding the benchmark's step count with it would misreport the count. Merom needs
no macros at all.

## Troubleshooting

**`port 8765 is already in use`** — a bridge from an earlier run survived. It sits
in an asyncio loop that ignores SIGTERM. `ps -eo pid=,args= | grep action_publisher`,
then `kill -9`.

**`ModuleNotFoundError: No module named 'omni'`** — the `omnigibson` env was
pointed at rather than *activated*. Its `activate.d` hook is what sources Isaac
Sim's `setup_conda_env.sh`. `record_demo.sh` handles this; a hand-run `sim.py`
must too.

**Choppy sim motion** — expected. The RTX renderer manages about 5 fps on this
machine, and the composed video plays at true wall-clock rate, so the footage is
genuinely 5 fps while the band animates at 30. `--timelapse` makes it less
noticeable by shortening the walking stretches.

**`moov atom not found`** — the compositor was still running. It writes the
container's index on close; wait for the process to exit.
