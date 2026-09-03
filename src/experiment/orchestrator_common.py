"""Shared complete-run orchestrator logic for the baseline methods.

Mirrors FOLBP/run_all_tasks.py: every task of an env is launched as its own
`main.py --env <env> --task <N>` subprocess, stdout+stderr are captured to
log/orchestrator_<env>/task<NN>.log, and after all tasks finish the per-task
logs are parsed and a summary report is written next to them.

Each method directory has a thin `run_all_tasks.py` that supplies:
  - the entry script and method-specific CLI flags,
  - a command builder (task_id -> argv),
  - optionally a custom log parser (default parses the shared
    `---------Env:X---Task:Y---` / success|failure / `steps: N` block that
    PEFA and FOLBP both print).
"""
import concurrent.futures as cf
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

# API keys are never hardcoded. Source setup_env.sh (see setup_env.example.sh)
# or export the variables yourself before running anything.


def add_common_args(parser, env_choices, default_timeout):
    parser.add_argument('--env', default='env0', choices=env_choices,
                        help='Environment name.')
    parser.add_argument('--tasks', default='all',
                        help='Comma-separated task ids, ranges (e.g. "0-7"), or "all".')
    parser.add_argument('--parallel', type=int, default=1,
                        help='Max concurrent task subprocesses. Default 1 (serial).')
    parser.add_argument('--timeout', type=int, default=default_timeout,
                        help=f'Per-task wall-clock timeout, seconds. Default {default_timeout}.')
    parser.add_argument('--resume', action='store_true',
                        help='Skip tasks whose per-task log already exists.')
    parser.add_argument('--api_key', default=None,
                        help='API key override (else taken from environment).')


def resolve_gemini_key(cli_key):
    key = (cli_key or os.environ.get('GEMINI_API_KEY')
           or os.environ.get('GOOGLE_API_KEY'))
    if not key:
        sys.exit('[orchestrator] no Gemini API key: pass --api_key, or '
                 '`source setup_env.sh` to export GEMINI_API_KEY / GOOGLE_API_KEY.')
    return key


def resolve_openai_key(cli_key):
    key = cli_key or os.environ.get('OPENAI_API_KEY')
    if not key:
        sys.exit('[orchestrator] no OpenAI API key: pass --api_key or set OPENAI_API_KEY')
    return key


def resolve_tasks(base_dir: Path, env: str, spec: str):
    with open(base_dir / 'env' / f'{env}.json') as f:
        n = len(json.load(f))
    if spec == 'all':
        return list(range(n))
    out = []
    for part in spec.split(','):
        part = part.strip()
        if '-' in part:
            lo, hi = part.split('-')
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return [t for t in out if 0 <= t < n]


def load_task_info(base_dir: Path, env: str):
    """task_id -> {'gt_steps': int|None, 'goal': str|None} from the env json."""
    with open(base_dir / 'env' / f'{env}.json') as f:
        data = json.load(f)
    info = {}
    for i, task in enumerate(data):
        gt = task.get('ground_truth_step_num')
        if isinstance(gt, (list, tuple)):
            gt = gt[0] if gt else None
        goal = task.get('goal_instruction')
        if isinstance(goal, (list, tuple)):
            goal = goal[0] if goal else None
        info[i] = {'gt_steps': gt, 'goal': goal}
    return info


def run_one(cmd, log_path: Path, timeout: int, cwd: Path, task_id: int) -> dict:
    t0 = time.time()
    with open(log_path, 'w') as f:
        f.write('[orchestrator] cmd: ' + ' '.join(str(c) for c in cmd) + '\n')
        f.flush()
        try:
            proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                                  timeout=timeout, cwd=str(cwd), text=True)
            return {'task_id': task_id, 'status': 'completed', 'rc': proc.returncode,
                    'elapsed_s': round(time.time() - t0, 1), 'log_path': str(log_path)}
        except subprocess.TimeoutExpired:
            return {'task_id': task_id, 'status': 'timeout', 'rc': None,
                    'elapsed_s': round(time.time() - t0, 1), 'log_path': str(log_path)}


# Regexes for the shared per-task result block printed by PEFA and FOLBP.
_BANNER_RE = re.compile(r'-+Env:\d+-+Task:(\d+)-+')
_STEPS_RE = re.compile(r'^steps:\s*(\d+)')
_ERROR_RE = re.compile(r'^An error occurred: (.+)$')
_EXC_RE = re.compile(r'^(?:[A-Za-z_][\w.]*\.)?[A-Za-z_]\w*(?:Error|Exception)\b:?.*$')


def parse_banner_log(log_path: Path) -> dict:
    """Parse the `---Env:X---Task:Y---` / success|failure / `steps: N` block."""
    success, steps = None, None
    failures = []
    in_result = False
    if not log_path.exists():
        return {'success': None, 'steps': None, 'failures': []}
    with open(log_path, errors='replace') as f:
        for line in f:
            line = line.rstrip('\n')
            if _BANNER_RE.search(line):
                in_result = True
                continue
            if in_result and success is None and line.strip() in ('success', 'failure'):
                success = (line.strip() == 'success')
                continue
            if in_result and steps is None:
                m = _STEPS_RE.match(line.strip())
                if m:
                    steps = int(m.group(1))
                    continue
            m = _ERROR_RE.match(line)
            if m:
                failures.append(('error', m.group(1)[:160]))
                continue
            if _EXC_RE.match(line) and 'Traceback' not in line:
                failures.append(('exception', line.strip()[:160]))
    return {'success': success, 'steps': steps, 'failures': failures}


def write_summary(summary_path: Path, method: str, env: str, rows: list):
    reached = [r for r in rows if r['success'] is not None]
    n_succ = sum(1 for r in reached if r['success'])
    steps_done = [r['steps'] for r in reached if r['steps'] is not None]
    gt_pairs = [(r['steps'], r['gt_steps']) for r in reached
                if r['success'] and r['steps'] is not None and r['gt_steps']]
    failure_kinds = Counter(k for r in rows for k, _ in r['failures'])
    failure_msgs = Counter(f'{k}: {d}' for r in rows for k, d in r['failures'])

    lines = []
    lines.append(f'{method} orchestrator summary — env={env}')
    lines.append(f'Tasks run: {len(rows)} ({len(reached)} reached a result, '
                 f'{len(rows) - len(reached)} timed out or crashed)')
    lines.append(f'Successes: {n_succ}/{len(reached) or 1} '
                 f'({100 * n_succ / max(1, len(reached)):.1f}%)')
    if steps_done:
        lines.append(f'Steps executed: avg={sum(steps_done)/len(steps_done):.1f}  '
                     f'min={min(steps_done)}  max={max(steps_done)}')
    if gt_pairs:
        eff = sum(gt / s for s, gt in gt_pairs if s) / len(gt_pairs)
        lines.append(f'Step efficiency on successes (gt/executed, 1.0 = optimal): {eff:.2f}')
    lines.append('')
    lines.append('=== Per task ===')
    for r in rows:
        ok = ('SUCCESS' if r['success']
              else ('TIMEOUT' if r['status'] == 'timeout'
                    else 'FAILURE' if r['success'] is False
                    else 'NO_FINAL'))
        lines.append(f'Task {r["task_id"]:02d} [{ok:8}] '
                     f'steps={r["steps"]}/{r["gt_steps"]} '
                     f'rc={r["rc"]} elapsed={r["elapsed_s"]}s')
        if r.get('goal'):
            lines.append(f'    goal: {str(r["goal"])[:140]}')
        for k, d in r['failures'][-3:]:
            lines.append(f'    - {k}: {str(d)[:140]}')
        lines.append(f'    log: {r["log_path"]}')
    if failure_kinds:
        lines.append('')
        lines.append('=== Aggregate failure kinds ===')
        for k, c in failure_kinds.most_common():
            lines.append(f'  {k}: {c}')
        lines.append('')
        lines.append('=== Top failure messages ===')
        for msg, c in failure_msgs.most_common(10):
            lines.append(f'  {c:3}  {msg[:150]}')
    summary_path.write_text('\n'.join(lines) + '\n')


def orchestrate(method: str, base_dir: Path, args, build_cmd, parse_log=parse_banner_log):
    """Run all requested tasks, then parse logs and write the summary."""
    tasks = resolve_tasks(base_dir, args.env, args.tasks)
    task_info = load_task_info(base_dir, args.env)
    out_dir = base_dir / 'log' / f'orchestrator_{args.env}'
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'[orchestrator] method={method}  env={args.env}  tasks={tasks}  '
          f'parallel={args.parallel}  timeout={args.timeout}s  out_dir={out_dir}')
    runs, pending = [], []
    for t in tasks:
        log_path = out_dir / f'task{t:02d}.log'
        if args.resume and log_path.exists():
            print(f'[orchestrator] skip task {t:02d} (resume; log exists)')
            runs.append({'task_id': t, 'status': 'resumed', 'rc': None,
                         'elapsed_s': 0.0, 'log_path': str(log_path)})
            continue
        pending.append((t, log_path))

    def _launch(t, p):
        print(f'[orchestrator] starting task {t:02d} -> {p}')
        r = run_one(build_cmd(t), p, args.timeout, base_dir, t)
        print(f'[orchestrator] finished task {r["task_id"]:02d} '
              f'status={r["status"]} rc={r["rc"]} elapsed={r["elapsed_s"]}s')
        return r

    if args.parallel > 1 and pending:
        with cf.ThreadPoolExecutor(max_workers=args.parallel) as ex:
            futures = [ex.submit(_launch, t, p) for t, p in pending]
            runs.extend(fut.result() for fut in cf.as_completed(futures))
    else:
        runs.extend(_launch(t, p) for t, p in pending)

    runs.sort(key=lambda r: r['task_id'])
    rows = []
    for r in runs:
        parsed = parse_log(Path(r['log_path']))
        info = task_info.get(r['task_id'], {})
        rows.append({**r, **parsed, 'gt_steps': info.get('gt_steps'),
                     'goal': info.get('goal')})

    summary_path = out_dir / 'summary.txt'
    write_summary(summary_path, method, args.env, rows)
    print(f'\n[orchestrator] summary written to {summary_path}\n')
    print(summary_path.read_text())
