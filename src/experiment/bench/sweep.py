#!/usr/bin/env python3
"""Run every arm over a task set, resumably, then print the comparison table.

    python3 bench/sweep.py --env env0 --tasks 1,2,3,4,5 --resume
    python3 bench/sweep.py --env env0 --tasks 0-7 --arms folbp,folbp-no-cdcl --parallel 4
    python3 bench/sweep.py --env env0 --report-only

The unit of work is (arm, seed, env, task). Each unit is a subprocess that writes
its own result JSON, so `--parallel` is safe and `--resume` is exact.

Execution order is task-major / arm-minor by design: every arm's measurement of a
given task happens within the same few minutes. Gemini's latency drifts over hours,
and running one arm to completion before starting the next would fold time-of-day
into the per-task latency comparison — which is half of what the paper claims.
"""
import argparse
import concurrent.futures as cf
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench.arms import ARMS, resolve_arms
from bench.record import (STATUS_COMPLETED, STATUS_CRASHED, STATUS_TIMEOUT,
                          log_path, read_record, record_path, write_stub_record)

DEFAULT_ROOT = Path(__file__).resolve().parents[3] / 'results' / 'bench'


def parse_args():
    p = argparse.ArgumentParser(
        description='Multi-arm benchmark sweep with resume and a final comparison table.',
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument('--env', default='env0',
                   help='Environment name, a comma-separated list, or "all" for env0..env4. '
                        'Records accumulate across invocations, so running envs separately '
                        'and running them together produce the same table.')
    p.add_argument('--tasks', default='all',
                   help='Task ids: "all", "1,2,3", "0-7", or a mix ("0-7,12,15").')
    p.add_argument('--arms', default='all',
                   help=f'Arm ids or "all". Known: {", ".join(a.id for a in ARMS)}')
    p.add_argument('--seeds', default='0',
                   help='Repeat indices, comma-separated. Use "0,1,2" for the paper numbers.')
    p.add_argument('--parallel', type=int, default=1, help='Concurrent subprocesses.')
    p.add_argument('--timeout', type=int, default=900, help='Per-task wall-clock timeout, s.')
    p.add_argument('--stagger', type=float, default=0.0,
                   help='Seconds between launches; helps stay under API rate limits.')
    p.add_argument('--out', default=str(DEFAULT_ROOT), help='Results root.')
    p.add_argument('--lm_id', default='gemini-2.5-flash')
    p.add_argument('--t', type=float, default=0.5,
                   help='Sampling temperature. 0.0 for experiments (default); the '
                        'planners ship 0.5 for interactive use.')
    p.add_argument('--api_key', default=None,
                   help='Gemini key; else $GEMINI_API_KEY / $GOOGLE_API_KEY.')
    p.add_argument('--order', default='task-major', choices=['task-major', 'arm-major'],
                   help='task-major (default) interleaves arms per task so latency is '
                        'comparable. arm-major is for debugging only.')

    p.add_argument('--resume', action='store_true',
                   help='Skip units that already have a completed record.')
    p.add_argument('--force', action='store_true', help='Re-run everything in the matrix.')
    p.add_argument('--skip-failed', action='store_true', dest='skip_failed',
                   help='With --resume, also skip units whose last run crashed or timed out '
                        '(default is to retry them).')
    p.add_argument('--strict-resume', action='store_true', dest='strict_resume',
                   help='With --resume, re-run records produced at a different git SHA.')

    p.add_argument('--dry-run', action='store_true', dest='dry_run',
                   help='Print the work matrix and exit.')
    p.add_argument('--report-only', action='store_true', dest='report_only',
                   help='Skip execution; rebuild the table from records on disk.')
    p.add_argument('--min-n', type=int, default=20, dest='min_n',
                   help='Minimum paired tasks before significance tests are reported. '
                        'Below this the table prints, the p-values do not.')
    return p.parse_args()


ALL_ENVS = ['env0', 'env1', 'env2', 'env3', 'env4']


def resolve_envs(spec: str):
    spec = spec.strip()
    if spec in ('all', '*'):
        return list(ALL_ENVS)
    return [e.strip() for e in spec.split(',') if e.strip()]


def resolve_tasks(base_dir: Path, env: str, spec: str):
    with open(base_dir / 'env' / f'{env}.json') as f:
        n = len(json.load(f))
    spec = spec.strip()
    if spec in ('all', '*'):
        return list(range(n))
    out = []
    for part in spec.split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            lo, hi = part.split('-', 1)
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    bad = [t for t in out if not 0 <= t < n]
    if bad:
        raise SystemExit(f'[bench] task id(s) out of range for {env} (0..{n - 1}): {bad}')
    return sorted(dict.fromkeys(out))


def resolve_key(cli_key):
    key = cli_key or os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY')
    if not key:
        raise SystemExit('[bench] no Gemini API key: pass --api_key, export GEMINI_API_KEY, '
                         'or `source setup_env.sh`')
    return key


def current_git_sha():
    try:
        return subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=Path(__file__).parent,
                              capture_output=True, text=True, timeout=10).stdout.strip() or None
    except Exception:
        return None


def should_skip(rec, args, head_sha, arm=None):
    """Resume policy. Returns (skip, why)."""
    if rec is None:
        return False, 'no record'
    status = rec.get('status')
    if status != STATUS_COMPLETED:
        if args.skip_failed:
            return True, f'previous run {status} (--skip-failed)'
        return False, f'previous run {status}; retrying'
    # A record produced under a different flag set answers a different question.
    # Reusing it would silently mix conditions inside one table cell.
    if arm is not None:
        recorded = (rec.get('manifest') or {}).get('flags') or {}
        for key, want in arm.expected_record_flags().items():
            if key in recorded and str(recorded[key]) != str(want):
                return False, (f'{key}={recorded[key]} in record, arm now wants '
                               f'{key}={want}; re-running')
    # Same argument for the sampling configuration, which lives in the manifest rather
    # than in flags. Without this, pointing a t=0.5 sweep at a tree built at t=0.0
    # silently reuses every record and reports the old temperature's numbers as the new
    # one's — the failure mode is invisible in the output.
    man = rec.get('manifest') or {}
    if man.get('temperature') is not None and float(man['temperature']) != float(args.t):
        return False, f'record at t={man["temperature"]}, sweep wants t={args.t}; re-running'
    if man.get('model') and man['model'] != args.lm_id:
        return False, f'record from {man["model"]}, sweep wants {args.lm_id}; re-running'
    if args.strict_resume and head_sha:
        rec_sha = (rec.get('manifest') or {}).get('git_sha')
        if rec_sha and rec_sha != head_sha:
            return False, f'record from {rec_sha[:8]} != HEAD (--strict-resume)'
    return True, 'completed'


def run_unit(unit, args, key, root: Path):
    arm, seed, task_id = unit['arm'], unit['seed'], unit['task_id']
    env = unit['env']
    rec_path = record_path(root, arm.id, seed, env, task_id)
    lg_path = log_path(root, arm.id, seed, env, task_id)
    lg_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = arm.build_cmd(sys.executable, env, task_id, seed, rec_path,
                        args.lm_id, args.t, key)
    child_env = os.environ.copy()
    child_env['GEMINI_API_KEY'] = key
    child_env['GOOGLE_API_KEY'] = key

    t0 = time.time()
    status, detail = STATUS_COMPLETED, ''
    try:
        with open(lg_path, 'w') as f:
            proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                                  cwd=arm.base_dir, timeout=args.timeout,
                                  env=child_env, text=True)
        if proc.returncode != 0:
            status = STATUS_CRASHED
            detail = f'exit code {proc.returncode}; see {lg_path}'
    except subprocess.TimeoutExpired:
        status = STATUS_TIMEOUT
        detail = f'exceeded {args.timeout}s; see {lg_path}'
    elapsed = time.time() - t0

    # The child writes its own record, including on a handled crash — it is the only
    # process that still has the LLM counters. Only write a stub when it left nothing.
    if status != STATUS_COMPLETED and read_record(rec_path) is None:
        write_stub_record(rec_path, arm=arm.id, seed=seed, env=env, task_id=task_id,
                          status=status, wall_clock_s=elapsed, detail=detail,
                          model=args.lm_id, temperature=args.t, flags=dict(arm.flags))

    rec = read_record(rec_path)
    return {'arm': arm.id, 'seed': seed, 'env': env, 'task_id': task_id, 'status': status,
            'elapsed_s': round(elapsed, 1),
            'success': bool(rec.get('success')) if rec else False}


def build_matrix(arms, seeds, tasks_by_env, args, root, head_sha):
    units, skipped = [], []
    combos = []
    for env, tasks in tasks_by_env.items():
        if args.order == 'task-major':
            combos += [(s, env, t, a) for s in seeds for t in tasks for a in arms]
        else:
            combos += [(s, env, t, a) for s in seeds for a in arms for t in tasks]
    for seed, env, task_id, arm in combos:
        unit = {'arm': arm, 'seed': seed, 'env': env, 'task_id': task_id}
        if args.force or not args.resume:
            units.append(unit)
            continue
        rec = read_record(record_path(root, arm.id, seed, env, task_id))
        skip, why = should_skip(rec, args, head_sha, arm)
        (skipped if skip else units).append({**unit, 'why': why})
    return units, skipped


def report(root, envs, arms, seeds, tasks_by_env, min_n):
    """Per-env tables, then the pooled table when more than one env is in play."""
    from bench.report import render
    for e in envs:
        render(root, env=e, arms=arms, seeds=seeds, tasks=tasks_by_env[e], min_n=min_n)
    if len(envs) > 1:
        print('\n' + '=' * 91)
        print('POOLED across ' + ', '.join(envs))
        render(root, env=None, arms=arms, seeds=seeds, tasks=None, min_n=min_n)


def main():
    args = parse_args()
    # Absolute, because `--record_path` is handed to a child that runs with
    # cwd=arm.base_dir (src/experiment/<arm dir>) rather than the parent's cwd. A
    # relative --out would resolve differently in the two processes: the child would
    # write its record somewhere else entirely, and the parent -- reading the path it
    # thinks it asked for -- would find nothing and report every run as a failure.
    root = Path(args.out).resolve()
    arms = resolve_arms(args.arms)
    seeds = [int(s) for s in args.seeds.split(',') if s.strip() != '']
    envs = resolve_envs(args.env)
    # Task counts differ per env (22/21/20/20/20), so "all" resolves per env.
    tasks_by_env = {e: resolve_tasks(arms[0].base_dir, e, args.tasks) for e in envs}
    head_sha = current_git_sha()

    if args.report_only:
        report(root, envs, arms, seeds, tasks_by_env, args.min_n)
        return

    units, skipped = build_matrix(arms, seeds, tasks_by_env, args, root, head_sha)

    for e in envs:
        print(f'[bench] env={e}  tasks={tasks_by_env[e]}')
    print(f'[bench] arms={[a.id for a in arms]}  seeds={seeds}')
    print(f'[bench] order={args.order}  parallel={args.parallel}  timeout={args.timeout}s')
    print(f'[bench] out={root}')
    print(f'[bench] {len(units)} unit(s) to run, {len(skipped)} skipped by --resume')
    if args.dry_run:
        for u in units:
            print(f'    RUN  {u["arm"].id:20} seed{u["seed"]} {u["env"]} task{u["task_id"]:02d}')
        for u in skipped[:10]:
            print(f'    skip {u["arm"].id:20} seed{u["seed"]} {u["env"]} '
                  f'task{u["task_id"]:02d}  ({u["why"]})')
        if len(skipped) > 10:
            print(f'    ... and {len(skipped) - 10} more skipped')
        return
    if not units:
        print('[bench] nothing to run; rebuilding table from existing records')
    else:
        key = resolve_key(args.api_key)
        done = 0
        t_start = time.time()

        def _launch(unit):
            if args.stagger:
                time.sleep(args.stagger)
            return run_unit(unit, args, key, root)

        results = []
        if args.parallel > 1:
            with cf.ThreadPoolExecutor(max_workers=args.parallel) as ex:
                futures = {ex.submit(_launch, u): u for u in units}
                for fut in cf.as_completed(futures):
                    r = fut.result()
                    done += 1
                    results.append(r)
                    print(f'[bench] ({done}/{len(units)}) {r["arm"]:20} seed{r["seed"]} '
                          f'{r["env"]} task{r["task_id"]:02d}  {r["status"]:9} '
                          f'success={r["success"]}  {r["elapsed_s"]}s', flush=True)
        else:
            for u in units:
                print(f'[bench] ({done + 1}/{len(units)}) starting {u["arm"].id:20} '
                      f'seed{u["seed"]} {u["env"]} task{u["task_id"]:02d}', flush=True)
                r = _launch(u)
                done += 1
                results.append(r)
                print(f'[bench] ({done}/{len(units)}) {r["arm"]:20} seed{r["seed"]} '
                      f'{r["env"]} task{r["task_id"]:02d}  {r["status"]:9} '
                      f'success={r["success"]}  {r["elapsed_s"]}s', flush=True)

        bad = [r for r in results if r['status'] != STATUS_COMPLETED]
        print(f'\n[bench] {len(results)} run(s) in {time.time() - t_start:.0f}s; '
              f'{len(bad)} did not complete')
        for r in bad:
            print(f'    {r["status"]:9} {r["arm"]:20} seed{r["seed"]} {r["env"]} '
                  f'task{r["task_id"]:02d}')

    report(root, envs, arms, seeds, tasks_by_env, args.min_n)


if __name__ == '__main__':
    main()
