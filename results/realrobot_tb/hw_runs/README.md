# Real-robot runs — TB_BallToDroneBasket_Setpoint (env_tb, task 1)

First runs of task 1 dispatched to actual hardware. Everything under
`results/realrobot_tb/runs/` above this directory is planner-side only
(`--mode standalone`, no device ever contacted); these are the runs where the
TurtleBot moved.

## Setup

| | |
|---|---|
| date | 2026-09-20 |
| planners | FOLBP and PEFA, both `--mode tb --tb_urls tb_urls.json` (FOLBP adds `--executor_mode deterministic --verify z3 --repair symbolic --cdcl`) |
| LLM | gemini-2.5-flash, t=0.5, seed 0 |
| TurtleBot3 + OpenMANIPULATOR-X | real, `turtlebot/web_client.py` over HTTP at `192.168.0.102:8080` |
| Crazyflie | **not flown** — `manual` transport, each drone action confirmed by an operator pressing Enter |

`--mode tb` and the per-agent transport map (`http://` / `ws://` / `manual`)
were added to FOLBP on this date; before that FOLBP had no device bridge at all
and only PEFA could drive hardware. The `manual` transport and the timing split
were added to PEFA the same day.

## Runs

| directory | planner | outcome | timing |
|---|---|---|---|
| `20260920-folbp-tb-task1-success_1` | FOLBP | 7/7, `goal_reached` | not instrumented (predates the timing patch) |
| `20260920-pefa-tb-task1-success_1` | PEFA | 7/7, `success!` | device 60.49s \| operator 20.76s \| total 81.25s |
| `20260920-folbp-tb-task1-misrouted` | FOLBP | 15 executed / 4 failed | — |

Both successes are the same physical task on the same hardware, so the two
transcripts are comparable side by side.

## 20260920-folbp-tb-task1-success_1

First run that worked — kept as `success_1`; later successful trials are
numbered alongside it rather than replacing it. 7/7 steps, `success=True`,
`termination_reason=goal_reached`, 0 validator rejections, 0 learned clauses —
the Oracle's first plan was already the ground-truth 7-step plan and Z3 passed
it unmodified.

Steps 1-4 are real TurtleBot motion, confirmed by the bridge's own replies
(`picked ball(300)`, `placed ball(300) at basket(700)`). Steps 5-7 are the
drone's, and were hand-confirmed, not flown.

No `=== Execution Timing ===` block: this run predates the timing
instrumentation. Per-step durations for the FOLBP TurtleBot half can be read
off the three aborted re-runs logged after it (`5.7s` movetowards ball, `26.3s`
grab, `10.1s` movetowards basket, `17.9s` putinto) — close enough to PEFA's
numbers that the device time is the arm, not the planner.

## 20260920-pefa-tb-task1-success_1

PEFA baseline on the same hardware, same day. 7/7 steps, `success!`, the same
ground-truth action sequence. Steps 1-4 real TurtleBot, steps 5-7
operator-confirmed, exactly as above.

This one carries the timing split: **device 60.49s, operator 20.76s, total
81.25s**. Only the 60.49s is machine time. Never quote the 81.25s total as
execution time — 20.76s of it is a human deciding when to press Enter.

The dominant cost is `[grab]` at 26.57s and `[putinto]` at 18.08s, both
OpenMANIPULATOR-X arm motions.

## 20260920-folbp-tb-task1-misrouted

Kept because it is the only recording of FOLBP's CDCL layer reacting to a real
bridge rejection.

The run used `--tb_url` (single device) instead of `--tb_urls`, so the drone's
actions were sent to the TurtleBot bridge too. `web_client.py` no-ops
`[takeoff_from]` and `[land_on]` (`QUADROTOR_NOOP`) but not `[movetowards]`, so
step 5 fell through to the object lookup and came back
`unknown target: setpoint A(600) (not in objects.json)`. FOLBP took that as
physical feedback, learned
`('drone','movetowards','setpoint A','runtime_rejection')`, backtracked to
level 4 and replanned. Final: 15 executed / 4 failed steps, 2 learned clauses.

The rejection was a configuration error rather than a real physical limit, so
the clause is bogus — worth knowing that a misrouted bridge teaches the planner
something false. Clauses are in-memory only and do not persist between runs.

## Not kept

Three further FOLBP attempts were started after `success_1` (they are the tail
of `src/experiment/FOLBP/log/env_tb.txt`, after line 1759). All three were
interrupted part-way — two at the drone takeoff prompt, one mid-`putinto` — so
none is a complete trial and none is saved here. Two aborted PEFA attempts sit
immediately before the PEFA success in its own log for the same reason. An
earlier PEFA loopback smoke test (`127.0.0.1:8080`, `device 1.62s`) is also in
that log and is **not** hardware.

## Files

- `transcript.txt` — the run's block of the planner's `log/env_tb.txt`,
  extracted because that file is append-only across every run and mixes
  loopback tests, aborted attempts and real runs.
- `tb_urls.json` — the routing map used.
- `objects.laptop-copy.json` — the repo's copy of the object positions. **Not
  what the bot was running**: the bot reported `objects_loaded: 2` at `/health`,
  so its own `objects.json` differs from this one. The positions actually used
  for `ball(300)` and `basket(700)` were not captured and are not recoverable
  from here.
