 # ICRA supplementary video — shot script

3:00 · 1920×1080 · 30 fps · no audio. Every caption is burned in and also
shipped as `results/icra_video/folbp_icra2027_reel.srt`.

**To narrate it, read the caption lines below as written.** They are paced at
roughly two words per second, so nothing needs to be rushed. If you record a
voice track, re-render with the caption strip off rather than speaking over
burned-in text.

| # | segment | in | out | length |
|---|---------|----|-----|--------|
| 1 | Title | 0:00 | 0:06 | 6 s |
| 2 | Slide 1 — the baseline loop | 0:06 | 0:21 | 15 s |
| 3 | Slide 2 — FOLBP main path (Fig. 1) | 0:21 | 0:41 | 20 s |
| 4 | Slide 3 — repair, dispatch, learn (Fig. 1) | 0:41 | 1:02 | 21 s |
| 5 | Card — task 2 | 1:02 | 1:06 | 4 s |
| 6 | Simulation, FOLBP ∥ PEFA | 1:06 | 1:58 | 52 s |
| 7 | Card — real robots | 1:58 | 2:01 | 3 s |
| 8 | Hardware, FOLBP ∥ PEFA | 2:01 | 2:45 | 44 s |
| 9 | Results — Table II | 2:45 | 3:00 | 15 s |

Total **3:00**.

---

## Title — 0:00 → 0:06

*No narration — card text only.*

## Slide 1 — the baseline loop — 0:06 → 0:21

> **0:06** Closed-loop LLM planners call the model at every corrective step.
>
> **0:11** Oracle, executor, judge — five calls to take one step.
>
> **0:15** Over the benchmark that is fifty-four calls and two hundred and eleven seconds per task.
>

## Slide 2 — FOLBP main path (Fig. 1) — 0:21 → 0:41

> **0:21** FOLBP restructures the loop around a symbolic core.
>
> **0:23** One LLM call proposes the complete multi-robot plan.
>
> **0:27** The scene graph compiles to ground first-order atoms, with no model involved.
>
> **0:31** Certification forward-simulates the plan and checks every precondition.
>
> **0:35** A validator checks it against each platform's declared capabilities.
>
> **0:38** Execution is a dispatch table. Only one block here is a language model.
>

## Slide 3 — repair, dispatch, learn (Fig. 1) — 0:41 → 1:02

> **0:41** Certification returns the unsatisfied literals as a conflict core.
>
> **0:44** Bounded repair splices in the prerequisite — with no model call.
>
> **0:48** Only conflicts no insertion can fix reach the proposer again.
>
> **0:51** Once certified and validated, the plan is dispatched — in simulation, or to the robots.
>
> **0:54** Reach the goal and the run is finished.
>
> **0:58** A runtime rejection is learned instead, and shapes the next proposal.
>

## Card — task 2 — 1:02 → 1:06

*No narration — card text only.*

## Simulation, FOLBP ∥ PEFA — 1:06 → 1:58

> **1:06** Both planners get the same scene graph and the same goal.
>
> **1:10** FOLBP's plan is certified before the first step runs.
>
> **1:15** PEFA decides one step at a time — five calls each.
>
> **1:22** FOLBP has not called the model since the plan was certified.
>
> **1:30** Same robots, same eleven steps. Only the deliberation differs.
>
> **1:39** FOLBP: eleven of eleven, 326 seconds, one LLM call.
>
> **1:46** PEFA is still executing.
>
> **1:54** PEFA: the same eleven steps, 490 seconds, fifty-five calls.
>

## Card — real robots — 1:58 → 2:01

*No narration — card text only.*

## Hardware, FOLBP ∥ PEFA — 2:01 → 2:45

> **2:01** On real hardware the TurtleBot must put the ball into the drone's basket.
>
> **2:06** Move-towards brings the base partway; the approach finishes inside grab.
>
> **2:11** FOLBP already holds all seven steps. PEFA is still deciding.
>
> **2:16** FOLBP: ball in the basket, handing over to the drone.
>
> **2:21** FOLBP is done — seven of seven, basket down at setpoint A.
>
> **2:27** PEFA is still carrying the ball to the basket.
>
> **2:34** PEFA's drone lifts off.
>
> **2:39** Planning cost: one call against thirty-five, 8.2 seconds against 127.
>

## Results — Table II — 2:45 → 3:00

> **2:45** Across one hundred tasks and two models, success matches.
>
> **2:50** The cost does not: twenty-five to forty-two times fewer calls.
>
> **2:55** Twelve to twenty-one times fewer tokens, and nine to nineteen times lower latency.
>

---

## Numbers on screen, and where they come from

| shown | source |
|-------|--------|
| 54.1 calls · 131.0k tokens · 211.5 s (slide 1) | Table II, PEFA on gemini-2.5-flash |
| 68.3% of conflict episodes closed with no model call (slide 3) | §IV-C |
| 1 call · 11/11 · 326 s and 55 calls · 11/11 · 490 s (sim) | §IV-G, and the run's own `events.jsonl` |
| 7/7 · 89 s and 7/7 · 184 s (hardware) | the recorded demonstration clock |
| 1 call · 3.1k tokens · 8.2 s vs 35 · 65.7k · 127.3 s (hardware panels) | §IV-H |
| the whole results slide | Table II, both models |

The hardware panels quote planner-side cost from §IV-H; the clock on those
panels is the recorded demonstration, which is a different quantity and is
labelled as such in the header.
