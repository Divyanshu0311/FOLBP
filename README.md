# FOLBP — First-Order Logic Bounded Planning for Heterogeneous Multi-Robot Systems

FOLBP is an LLM task planner that puts a **decidable symbolic layer between the
language model and the robots**. The LLM proposes a plan once; a Z3 encoding of
first-order preconditions decides whether that plan is executable; UNSAT cores
drive a symbolic repair; and a CDCL-style conflict store keeps the planner from
re-proposing an already-refuted action.

The baseline is **COHERENT / PEFA** (Liu et al., 2024) — its benchmark, its
simulator, and its planner, unmodified except for the LLM backend and result
instrumentation. This repository is a fork of the COHERENT release; see
[Relationship to COHERENT](#relationship-to-coherent) for exactly what changed.

## Results

100 tasks (20 per env × 5 envs), `gemini-2.5-flash` at `temperature=0`, one seed.
Both FOLBP and PEFA run against the *same* task definitions and the *same* model.

| arm | success | exec/GT | LLM calls/task | latency (s) |
|---|---|---|---|---|
| **FOLBP (full, ours)** | **96/100** | 1.03 | **1.5** | **20.4** |
| &nbsp;&nbsp;− verification | 85/100 | 1.03 | 2.9 | 30.7 |
| &nbsp;&nbsp;repair: LLM re-prompt | 93/99 | 1.03 | 4.0 | 59.5 |
| &nbsp;&nbsp;− CDCL | 94/100 | 1.03 | 2.4 | 25.8 |
| &nbsp;&nbsp;naive one-shot | 65/100 | 1.02 | 1.0 | 11.1 |
| COHERENT / PEFA (baseline) | 95/100 | 1.03 | 54.5 | 201.7 |

FOLBP matches the baseline's success rate while using **~36× fewer LLM calls**
and running **~10× faster** (Wilcoxon, n=100, p≈4e-18 on both). The naive
one-shot arm would have dispatched a physically infeasible action on 53/100
tasks, which is the soundness case for the symbolic layer.

Full table with paired statistics, per-arm ablations and all 600 per-task records:

```
results/bench/report.txt        # headline table + McNemar / Wilcoxon
results/bench/table_main.md     # same table, markdown
results/bench/table_main.tex    # same table, LaTeX
results/bench/all_tasks.csv     # one row per (arm, seed, env, task)
results/bench/runs/<arm>/seed0/<env>/task<NN>.json
```

Regenerate the tables from the committed records without re-running anything:

```bash
cd src/experiment && python3 -m bench.report
```

## Demos

Four recorded runs, one matched FOLBP/PEFA pair per scene — each pair is the
**same task in the same environment**, so the only difference is the planner.
The panel under the viewport shows the live planner transcript, the step
counter, the running LLM-call count and elapsed time.

| scene | task | FOLBP | COHERENT / PEFA |
|---|---|---|---|
| `house_double_floor_lower` | kebab → grill, 11 steps | [house-folbp.mp4](results/demos/house-folbp.mp4) — **1 LLM call**, 5:27 | [house-pefa.mp4](results/demos/house-pefa.mp4) — 55 calls, 8:11 |
| `Merom_1_int` | apple → wicker basket, 9 steps | [merom-folbp.mp4](results/demos/merom-folbp.mp4) — **3 LLM calls**, 3:12 | [merom-pefa.mp4](results/demos/merom-pefa.mp4) — 45 calls, 4:55 |

Call counts are physical API calls recorded by `bench/llm_meter.py`, and the
times are the real elapsed wall clock shown in each header. The Merom FOLBP
clip is the **worst** of three deterministic takes (2, 2 and 3 calls); the other
two are in `results/videos/`.

All four runs are genuine head-to-heads on the same task, and both planners
reach the same goal in the same number of steps — the contrast is how many times
the LLM is consulted to get there. FOLBP demos are recorded with
`--executor_mode deterministic --verify z3 --repair symbolic --cdcl`; without
those flags `args.py` falls back to the PEFA-style LLM executor and films a
hybrid rather than FOLBP.

Clip length is not wall-clock: `tools/compose_video.py` accelerates robot motion
and holds near real time through the reasoning so the transcript stays readable.

Regenerate any of them with `tools/record_demo.sh <scene> <framework>`. Raw
captures (the ~150 MB `sim.mp4` plus per-run logs) stay under `results/videos/`
and are not committed.

## API keys

**No key is hardcoded anywhere in this repository.** Every entry point reads it
from the environment.

```bash
cp setup_env.example.sh setup_env.sh
$EDITOR setup_env.sh          # paste your Gemini key
source setup_env.sh
```

`setup_env.sh` is in `.gitignore` and must never be committed;
`setup_env.example.sh` is the committed template and must never contain a real
key. `--api_key` still works on the command line and overrides the environment.

## Running

```bash
source setup_env.sh
cd src/experiment

# one task
cd FOLBP && python3 main.py --env env0 --task 3

# every task of an env, one subprocess each, resumable
cd FOLBP && python3 run_all_tasks.py --env env0 --parallel 4 --resume

# the full ablation sweep — all 6 arms over a task set, then the table
python3 bench/sweep.py --env env0 --tasks 0-19 --resume
python3 bench/sweep.py --report-only
```

Arms are defined in one place, [`src/experiment/bench/arms.py`](src/experiment/bench/arms.py);
every FOLBP arm is the same pipeline under different flags
(`--verify` / `--repair` / `--cdcl` / `--max_oracle_plans`), never a forked copy
of the code.

The PEFA baseline is COHERENT's planner, unchanged apart from the LLM backend
and result instrumentation:

```bash
cd src/experiment/PEFA && python3 main.py --env env0 --task 2 --mode standalone
```

COHERENT also ships CRMS, DRMS and MCTS baselines. They are **not** included
here: this work compares against PEFA, which is COHERENT's strongest arm and the
one its paper leads with, and the other three were never ported to the Gemini
backend or run in this tree. Get them from
[upstream](https://github.com/MrKeee/COHERENT) if you want them. The model you
pick and the number of tasks you run determine your API bill.

## Relationship to COHERENT

This tree is a fork of [COHERENT](https://github.com/MrKeee/COHERENT)
([paper](https://arxiv.org/abs/2409.15146) · [video](https://youtu.be/dV1J-VXdEJA)).
What was changed, and why it does not compromise the baseline comparison:

* **LLM backend.** COHERENT used OpenAI GPT-4; both arms here use
  `gemini-2.5-flash` at `temperature=0`. The baseline was ported, not
  reimplemented — PEFA's prompts, its per-step oracle dialogue and its control
  flow are untouched.
* **Instrumentation.** `PEFA/main.py` gained a `record_task()` call and an LLM
  call counter so it emits the same per-task JSON record as FOLBP. No planning
  logic was changed.
* **Benchmark size.** Every `env*.json` is capped at **20 tasks (ids 0–19)**, so
  the benchmark is exactly the 100 tasks reported in the COHERENT paper.
  Upstream `env0.json` and `env1.json` each carried a 21st task that no reported
  number ever used; those were dropped so every env has the same shape. Tasks
  0–19 are byte-for-byte the upstream definitions.
* **Real-robot mode.** PEFA gained `--mode ws` (WebSocket to OmniGibson) and
  `--mode tb` (HTTP to physical devices) for the hardware demo, alongside
  `turtlebot/`, `crazyflie/` and `mpc.py`. The benchmark runs
  `--mode standalone`, which is the original code path.
* **ROS 1 → ROS 2.** See [ROS 2 notes](#ros-2-notes).

## Benchmark
![Figure](media/benchmark.png) 
we create a large-scale embodied benchmark tailored for heterogeneous multi-robot collaboration, including quadrotors, robotic dogs, and robotic arms. Built upon the [BEHAVIOR-1K](https://behavior.stanford.edu/behavior-1k), our benchmark covers 5 typical real-world scenes: 2 apartment scenes, 1 apartment with garden scene, 1 grocery store and 1 restaurant, with a wide range of interactive objects (both rigid and articulated) and various layouts (e.g., multi-room and multi-floor). The ground truth (GT) of each task represents the optimal number of steps for completion. Based on the minimum necessary number of robot types to perform each task, the benchmark is split into three categories: Mono-type Tasks, Dual-type Tasks and Trio-type Tasks.

## Install

```bash
conda create -n coherent python==3.10
conda activate coherent
pip install -r requirement.txt
```

`requirement.txt` covers the planner side. `torch` and `sentence_transformers`
are only needed by the MCTS baseline, and `scipy` is optional — `bench/report.py`
falls back to an exact permutation test without it. The simulator runs in its own
environment; see [Simulation Setup](#simulation-setup).

## Simulation Setup
Install NVIDIA's [Omniverse Isaac Sim](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/install_workstation.html). **Please make sure you have the 2022.2.0 version of Isaac Sim installed.** We have added controllers for the quadrotor and the robotic arm on the back of the robotic dog, so we are providing the compressed package of the Isaac Sim used [here](https://drive.google.com/file/d/1wmqztsn8vwgHB_fL4LMamfuiDY7nXrlT/view?usp=drive_link). You need to extract the downloaded files to the location ``~/.local/share/ov/pkg/isaac_sim-2022.2.0``.

We are using OmniGibson version v0.2.1, with modifications made on top of this version. Additionally, the ``Benchmark`` directory has been added. Consequently, we have uploaded the modified OmniGibson folder to the repository.

Download the heterogeneous robot asset files [here](https://drive.google.com/drive/folders/1CRX7mNndvNpty7dC37yHDOr25a0Xc-Ge?usp=drive_link) and move them to the `Benchmark` folder in `assets`.

Note: Before you run the `setup.sh`, you need to exit the conda environment first. This script file will create a new conda environment named `omnigibson`.

The upstream project targeted `Ubuntu 20.04` with `ROS 1 noetic`. **This tree has been ported to `ROS 2 Humble` on `Ubuntu 22.04`** — see [ROS 2 notes](#ros-2-notes) below for what changed and why the two processes run under different Python interpreters.

1、Set the COHERENT_PATH and source the ROS 2 environment script in your bashrc or zshrc.
```bash
export COHERENT_PATH="/the_path_to_clone_the_project/COHERENT"
source /the_path_to_clone_the_project/COHERENT/OmniGibson/Benchmark/ros_hademo_ws/ros2_env.sh
```
2、Build the ROS 2 interfaces (replaces `catkin_make`).
```bash
cd $COHERENT_PATH/OmniGibson/Benchmark/ros_hademo_ws
source ros2_env.sh
colcon build --symlink-install
```
3、Create the conda environment and download the dataset. 
```bash
cd OmniGibson
conda deactivate
./scripts/setup.sh 
python scripts/download_datasets.py
cp -r /the_path_to_clone_the_project/COHERENT/OmniGibson/omnigibson/assets/oven/insidq /the_path_to_clone_the_project/COHERENT/OmniGibson/omnigibson/data/og_dataset/objects/oven/
```
4、Activate the conda environment and run the following scripts:
```bash
cd Benchmark
conda activate omnigibson
sh run.sh
```
You can set the simulator's viewer camera's pose in `OmniGibson/Benchmark/sim.py`.

## ROS 2 notes

The `hademo` package is now an `ament_cmake` package built with `colcon`; there is no `roscore` and no `devel/` space. Topic names and message semantics are unchanged:

| | topic | type |
|---|---|---|
| LLM → simulator | `/actionTopic` | `hademo/msg/Action` |
| simulator → LLM | `/resultTopic` | `hademo/msg/Result` |

Both endpoints use `RELIABLE` + `TRANSIENT_LOCAL` QoS, the ROS 2 equivalent of a ROS 1 latched publisher — the startup handshake depends on a late joiner still receiving the last message.

`Func_and_Args.msg` was renamed to `FuncAndArgs.msg`: ROS 2 rejects underscores in interface type names.

**The two sides run under different Python interpreters, on purpose.** `rospy` was pure Python and imported anywhere; `rclpy` is a C extension bound to the exact CPython that ROS 2 Humble was built for (system `python3.10`). Isaac Sim 2022.2.0 embeds Python 3.7 and the `omnigibson` conda env is created to match it, so `sim.py` cannot import `rclpy` at all. Therefore:

* `action_publisher.py` (LLM side) runs under system `python3.10` and is an ordinary `rclpy` node.
* `sim.py` (Isaac Sim side) keeps running under the `omnigibson` env, and `action_subscriber.py` transparently launches itself as a **ROS 2 sidecar process** under `python3.10`, bridging over a local UNIX socket. `run.sh` does this for you.

The sidecar is the real ROS 2 node, so `ros2 topic echo /actionTopic`, `ros2 bag record` and `rqt` all work as normal. If you ever run the simulator under a Python 3.10 interpreter that does have `rclpy`, the same class uses an in-process node instead — no code change needed.

Useful overrides: `HADEMO_BACKEND` (`inprocess` / `sidecar`), `HADEMO_ROS2_PYTHON`, `HADEMO_ROS2_ENV`, `ROS_DOMAIN_ID`.

Note that `ros2_env.sh` deliberately pushes `/usr/bin` to the front of `PATH` and neutralises pyenv/conda for that shell — a shadowing interpreter makes `colcon build` fail with `No module named 'em'`.

## Acknowledgement
FOLBP is built directly on **COHERENT** (Liu et al., 2024), which provides the
benchmark, the simulator integration and the PEFA baseline this work is measured
against. COHERENT in turn adapts code from [llm-mcts](https://github.com/1989Ryan/llm-mcts), [Co-LLM-Agents](https://github.com/UMass-Foundation-Model/Co-LLM-Agents), [OmniGibson](https://github.com/StanfordVL/OmniGibson).

## Citation

If you use FOLBP, please cite both this work and the COHERENT baseline it is
built on and measured against.

**FOLBP (this work)** — currently under review; this entry becomes an
`@inproceedings` on acceptance.
```bibtex
@unpublished{gupta2026folbp,
  title  = {FOLBP: Certified Plan Repair and State-Aware Conflict Learning
            for LLM-Based Heterogeneous Multi-Robot Task Planning},
  author = {Gupta, Divyanshu and Mitra, Anik and Sinha, Arpita},
  note   = {Submitted to the 2027 IEEE International Conference on
            Robotics and Automation (ICRA)},
  year   = {2026}
}
```

**COHERENT** — provides the benchmark, the simulator integration and the PEFA
baseline. Accepted at ICRA 2025; the authors' own README cites the arXiv
preprint, reproduced here as they request:
```bibtex
@article{liu2024coherent,
  title={COHERENT: Collaboration of Heterogeneous Multi-Robot System with Large Language Models},
  author={Liu, Kehui and Tang, Zixin and Wang, Dong and Wang, Zhigang and Zhao, Bin and Li, Xuelong},
  journal={arXiv preprint arXiv:2409.15146},
  year={2024}
}
```

**BEHAVIOR-1K / OmniGibson** — this project is set up on
[**`OmniGibson`**](OmniGibson), built upon NVIDIA's
[Omniverse](https://www.nvidia.com/en-us/omniverse/) platform:
```bibtex
@inproceedings{
li2022behavior,
title={{BEHAVIOR}-1K: A Benchmark for Embodied {AI} with 1,000 Everyday Activities and Realistic Simulation},
author={Chengshu Li and Ruohan Zhang and Josiah Wong and Cem Gokmen and Sanjana Srivastava and Roberto Mart{\'\i}n-Mart{\'\i}n and Chen Wang and Gabrael Levine and Michael Lingelbach and Jiankai Sun and Mona Anvari and Minjune Hwang and Manasi Sharma and Arman Aydin and Dhruva Bansal and Samuel Hunter and Kyu-Young Kim and Alan Lou and Caleb R Matthews and Ivan Villa-Renteria and Jerry Huayang Tang and Claire Tang and Fei Xia and Silvio Savarese and Hyowon Gweon and Karen Liu and Jiajun Wu and Li Fei-Fei},
booktitle={6th Annual Conference on Robot Learning},
year={2022},
url={https://openreview.net/forum?id=_8DoIe8G3t}
}
```

## License

This tree carries **three** different sets of terms. See [`LICENSE`](LICENSE)
for the authoritative text.

| Code | Terms |
|---|---|
| Original FOLBP contributions — `src/experiment/FOLBP*`, `src/experiment/bench/`, `tools/`, `turtlebot/`, `crazyflie/`, `mpc*.py`, `results/bench/`, and the FOLBP modifications to upstream files | MIT |
| Upstream **COHERENT** — the benchmark task definitions, the PEFA baseline, `OmniGibson/Benchmark/` | **No license — all rights reserved by the COHERENT authors** |
| `OmniGibson/` (v0.2.1, Stanford Vision and Learning Group) | MIT, see [`OmniGibson/LICENSE`](OmniGibson/LICENSE) |

> **Upstream COHERENT is published without a license**, so its authors retain
> all rights and no redistribution permission has been granted. This fork exists
> under the GitHub Terms of Service forking permission, which applies *within
> GitHub only*. The MIT grant above covers the FOLBP contributions and nothing
> else. To use the COHERENT code outside GitHub, contact its authors.

The BEHAVIOR-1K dataset and the robot asset bundles are **not** in this
repository — they are downloaded separately under their own EULA.
