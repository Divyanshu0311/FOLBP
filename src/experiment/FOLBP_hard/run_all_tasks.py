#!/usr/bin/env python3
"""Orchestrator — run every task of an env, store per-task logs, aggregate failures.

Each task is launched as its own `main.py --env <env> --task <N>` subprocess; stdout
(which includes every `[FOLBP][...]` line) is captured to log/orchestrator_<env>/task<NN>.log
so concurrent runs don't trample each other. After all tasks finish, the per-task logs
are parsed and a summary report is written to log/orchestrator_<env>/summary.txt.

Usage:
    python3 run_all_tasks.py --env env0
    python3 run_all_tasks.py --env env0 --tasks 2,6,10 --parallel 4
    python3 run_all_tasks.py --env env0 --resume        # skip tasks whose log already exists
"""
import argparse
import ast
import concurrent.futures as cf
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--env', default='env0', choices=['env0', 'env1', 'env2', 'env3', 'env4'])
    p.add_argument('--tasks', default='all',
                   help='Comma-separated task ids, ranges (e.g. "0-7"), or "all".')
    p.add_argument('--parallel', type=int, default=1,
                   help='Max concurrent task subprocesses. Default 1 (serial).')
    p.add_argument('--timeout', type=int, default=600,
                   help='Per-task wall-clock timeout, seconds. Default 600.')
    p.add_argument('--executor_mode', default='pefa_llm', choices=['pefa_llm', 'deterministic'])
    p.add_argument('--max_replan_attempts', type=int, default=5)
    p.add_argument('--resume', action='store_true',
                   help='Skip tasks whose per-task log already exists.')
    p.add_argument('--api_key', default=None,
                   help='Override Gemini API key (else inherited from environment / main.py default).')
    return p.parse_args()


def resolve_tasks(env: str, spec: str):
    with open(f'./env/{env}.json') as f:
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


def run_one(env: str, task_id: int, log_path: Path, args) -> dict:
    cmd = [
        sys.executable, 'main.py',
        '--env', env, '--task', str(task_id),
        '--executor_mode', args.executor_mode,
        '--max_replan_attempts', str(args.max_replan_attempts),
    ]
    sub_env = os.environ.copy()
    if args.api_key:
        sub_env['GEMINI_API_KEY'] = args.api_key
    t0 = time.time()
    with open(log_path, 'w') as f:
        try:
            proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                                  timeout=args.timeout, env=sub_env, text=True)
            return {'task_id': task_id, 'status': 'completed', 'rc': proc.returncode,
                    'elapsed_s': round(time.time() - t0, 1), 'log_path': str(log_path)}
        except subprocess.TimeoutExpired:
            return {'task_id': task_id, 'status': 'timeout', 'rc': None,
                    'elapsed_s': round(time.time() - t0, 1), 'log_path': str(log_path)}


def parse_log(log_path: Path) -> dict:
    success, steps, gt_steps, goal = None, None, None, None
    learned_clauses: list = []
    failures: list = []  # (kind, detail)
    oracle_calls = 0
    z3_unsat_count = 0
    cdcl_stats: dict = {}
    task_graph: dict = {}
    if not log_path.exists():
        return {'success': None, 'steps': None, 'failures': [],
                'learned_clauses': [], 'oracle_calls': 0,
                'z3_unsat_count': 0, 'cdcl_stats': {}, 'task_graph': {},
                'gt_steps': None, 'goal': None}
    with open(log_path, errors='replace') as f:
        for line in f:
            m = re.search(r'\[FOLBP\]\[TASK\] task_id=\d+ env_id=\d+ name=\S+ gt_steps=(\d+)', line)
            if m:
                gt_steps = int(m.group(1))
            m = re.search(r'\[FOLBP\]\[TASK\] goal: (.+)$', line)
            if m and goal is None:
                goal = m.group(1).strip()
            if '[FOLBP][ORACLE] requesting full plan' in line:
                oracle_calls += 1
            if '[FOLBP][Z3] UNSAT' in line:
                z3_unsat_count += 1
                m = re.search(r'missing=(\[.*?\])', line)
                if m:
                    failures.append(('z3_unsat', m.group(1)))
            if '[FOLBP][R] no remediation derivable' in line:
                m = re.search(r'no_remediation_for:(\[.*?\])', line)
                failures.append(('no_remediation', m.group(1) if m else line.strip()))
            if '[FOLBP][VAL] rejected' in line:
                m = re.search(r'reason=(\S+)', line)
                failures.append(('val_rejected', m.group(1) if m else line.strip()))
            if 'runtime failure on step' in line:
                m = re.search(r'runtime failure on step\[(\d+)\]: (.+)$', line)
                if m:
                    failures.append(('runtime', m.group(2)[:160]))
            m = re.search(r'\[FOLBP\]\[FINAL\] success=(\w+) steps=(\d+)', line)
            if m:
                success = (m.group(1) == 'True')
                steps = int(m.group(2))
            m = re.search(r'\[FOLBP\]\[FINAL\] cdcl: (\{.*\})', line)
            if m:
                try: cdcl_stats = ast.literal_eval(m.group(1))
                except Exception: pass
            m = re.search(r'\[FOLBP\]\[FINAL\] task_graph: (\{.*\})', line)
            if m:
                try: task_graph = ast.literal_eval(m.group(1))
                except Exception: pass
            m = re.search(r'\[FOLBP\]\[FINAL\] learned_clauses: (\[.*\])', line)
            if m:
                try: learned_clauses = ast.literal_eval(m.group(1))
                except Exception: pass
    return {'success': success, 'steps': steps, 'gt_steps': gt_steps, 'goal': goal,
            'failures': failures, 'learned_clauses': learned_clauses,
            'oracle_calls': oracle_calls, 'z3_unsat_count': z3_unsat_count,
            'cdcl_stats': cdcl_stats, 'task_graph': task_graph}


def write_summary(summary_path: Path, env: str, rows: list):
    completed = [r for r in rows if r['success'] is not None]
    n_succ = sum(1 for r in completed if r['success'])
    steps_done = [r['steps'] for r in completed if r['steps'] is not None]
    failure_kinds = Counter(k for r in rows for k, _ in r['failures'])
    no_remediation_atoms = Counter()
    for r in rows:
        for k, d in r['failures']:
            if k == 'no_remediation':
                no_remediation_atoms[d] += 1
    val_reasons = Counter(d for r in rows for k, d in r['failures'] if k == 'val_rejected')
    runtime_msgs = Counter(d.split(' Raw:')[0].strip(': ') for r in rows
                           for k, d in r['failures'] if k == 'runtime')
    # Aggregate learned clauses across all tasks.
    learned_global = Counter()
    for r in rows:
        for c in r['learned_clauses']:
            # tuple form: (agent_class, verb, target_class, reason[, precond])
            if isinstance(c, (list, tuple)) and len(c) >= 4:
                learned_global[(c[0], c[1], c[2], c[3])] += 1

    lines = []
    lines.append(f'FOLBP orchestrator summary — env={env}')
    lines.append(f'Tasks run: {len(rows)} ({len(completed)} reached FINAL, '
                 f'{len(rows) - len(completed)} timed out or crashed)')
    lines.append(f'Successes: {n_succ}/{len(completed) or 1} '
                 f'({100 * n_succ / max(1, len(completed)):.1f}%)')
    if steps_done:
        lines.append(f'Steps executed: avg={sum(steps_done)/len(steps_done):.1f}  '
                     f'min={min(steps_done)}  max={max(steps_done)}')
    lines.append('')
    lines.append('=== Per task ===')
    for r in rows:
        ok = ('SUCCESS' if r['success']
              else ('TIMEOUT' if r['status'] == 'timeout'
                    else 'FAILURE' if r['success'] is False
                    else 'NO_FINAL'))
        lines.append(f'Task {r["task_id"]:02d} [{ok:8}] '
                     f'steps={r["steps"]}/{r["gt_steps"]} '
                     f'oracle_calls={r["oracle_calls"]} '
                     f'z3_unsat={r["z3_unsat_count"]} '
                     f'clauses_learned={r["cdcl_stats"].get("clauses_learned", 0)} '
                     f'elapsed={r["elapsed_s"]}s')
        if r['goal']:
            lines.append(f'    goal: {r["goal"][:140]}')
        last = r['failures'][-3:] if r['failures'] else []
        for k, d in last:
            lines.append(f'    - {k}: {str(d)[:140]}')
        lines.append(f'    log: {r["log_path"]}')
    lines.append('')
    lines.append('=== Aggregate failure kinds ===')
    for k, c in failure_kinds.most_common():
        lines.append(f'  {k}: {c}')
    if no_remediation_atoms:
        lines.append('')
        lines.append('=== Top no_remediation UNSAT cores (replanner gaps) ===')
        for atom, c in no_remediation_atoms.most_common(10):
            lines.append(f'  {c:3}  {atom}')
    if val_reasons:
        lines.append('')
        lines.append('=== Top PlanValidator reject reasons ===')
        for reason, c in val_reasons.most_common(10):
            lines.append(f'  {c:3}  {reason}')
    if runtime_msgs:
        lines.append('')
        lines.append('=== Top executor runtime rejections ===')
        for msg, c in runtime_msgs.most_common(10):
            lines.append(f'  {c:3}  {msg[:140]}')
    if learned_global:
        lines.append('')
        lines.append('=== Most-learned CDCL clauses across env ===')
        for (a, v, t, why), c in learned_global.most_common(20):
            lines.append(f'  {c:3}  ({a}, [{v}], <{t}>) reason={why}')
    summary_path.write_text('\n'.join(lines) + '\n')


def main():
    args = parse_args()
    tasks = resolve_tasks(args.env, args.tasks)
    out_dir = Path(f'./log/orchestrator_{args.env}')
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'[orchestrator] env={args.env}  tasks={tasks}  parallel={args.parallel}  '
          f'timeout={args.timeout}s  out_dir={out_dir}')
    runs = []
    pending = []
    for t in tasks:
        log_path = out_dir / f'task{t:02d}.log'
        if args.resume and log_path.exists():
            print(f'[orchestrator] skip task {t:02d} (resume; log exists)')
            runs.append({'task_id': t, 'status': 'resumed', 'rc': None,
                         'elapsed_s': 0.0, 'log_path': str(log_path)})
            continue
        pending.append((t, log_path))

    if args.parallel > 1 and pending:
        with cf.ThreadPoolExecutor(max_workers=args.parallel) as ex:
            futures = {ex.submit(run_one, args.env, t, p, args): t for t, p in pending}
            for fut in cf.as_completed(futures):
                r = fut.result()
                print(f'[orchestrator] finished task {r["task_id"]:02d} '
                      f'status={r["status"]} rc={r["rc"]} elapsed={r["elapsed_s"]}s')
                runs.append(r)
    else:
        for t, p in pending:
            print(f'[orchestrator] starting task {t:02d} -> {p}')
            r = run_one(args.env, t, p, args)
            print(f'[orchestrator] finished task {t:02d} '
                  f'status={r["status"]} rc={r["rc"]} elapsed={r["elapsed_s"]}s')
            runs.append(r)

    # Parse and aggregate.
    runs.sort(key=lambda r: r['task_id'])
    rows = []
    for r in runs:
        parsed = parse_log(Path(r['log_path']))
        rows.append({**r, **parsed})

    summary_path = out_dir / 'summary.txt'
    write_summary(summary_path, args.env, rows)
    print(f'\n[orchestrator] summary written to {summary_path}\n')
    print(summary_path.read_text())


if __name__ == '__main__':
    main()
