"""Flatten per-task records into one tidy table.

Records on disk are the only input, so the CSV can always be rebuilt without
re-running anything — that is what makes `--resume` and `--report-only` honest.

    python3 bench/aggregate.py                 # writes results/bench/all_tasks.csv
    python3 bench/aggregate.py --env env0
"""
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench.arms import BY_ID
from bench.record import as_scalar_int, iter_records

COLUMNS = [
    'arm', 'arm_label', 'seed', 'env', 'task_id', 'status', 'success',
    'gt_steps', 'executed_steps', 'exec_over_gt', 'wall_clock_s',
    'total_llm_calls', 'oracle_calls', 'repair_calls', 'executor_calls', 'judge_calls',
    'prompt_tokens', 'completion_tokens', 'total_tokens', 'llm_latency_s',
    'verify_mode', 'z3_checks', 'z3_unsat_events', 'shadow_unsat_events',
    'repair_mode', 'repair_attempts', 'repair_successes', 'escalations',
    'cdcl_enabled', 'clauses_learned', 'outer_iters', 'oracle_plans',
    'termination_reason', 'deadlocked', 'validator_rejections', 'forbidden_rejections',
    'hard_bans',
    'goal', 'git_sha', 'model', 'temperature', 'timestamp',
]


def flatten(rec) -> dict:
    llm = rec.get('llm') or {}
    verify = rec.get('verify') or {}
    repair = rec.get('repair') or {}
    cdcl = rec.get('cdcl') or {}
    # Absent on every record written before the hard-constraint ablation; the columns
    # then render empty rather than breaking the CSV.
    dl = rec.get('deadlock') or {}
    man = rec.get('manifest') or {}
    arm_id = rec.get('arm', '')
    # Defensive: records written before gt_steps was normalized may still hold [4].
    gt = as_scalar_int(rec.get('gt_steps'))
    ex = as_scalar_int(rec.get('executed_steps'))
    ratio = ''
    if rec.get('success') and gt and ex is not None:
        ratio = round(ex / gt, 4)
    return {
        'arm': arm_id,
        'arm_label': BY_ID[arm_id].label.strip() if arm_id in BY_ID else arm_id,
        'seed': rec.get('seed'),
        'env': rec.get('env'),
        'task_id': rec.get('task_id'),
        'status': rec.get('status'),
        'success': bool(rec.get('success')),
        'gt_steps': gt,
        'executed_steps': ex,
        'exec_over_gt': ratio,
        'wall_clock_s': rec.get('wall_clock_s'),
        'total_llm_calls': llm.get('total_calls'),
        'oracle_calls': llm.get('oracle_calls'),
        'repair_calls': llm.get('repair_calls'),
        'executor_calls': llm.get('executor_calls'),
        'judge_calls': llm.get('judge_calls'),
        'prompt_tokens': llm.get('prompt_tokens'),
        'completion_tokens': llm.get('completion_tokens'),
        'total_tokens': llm.get('total_tokens'),
        'llm_latency_s': llm.get('llm_latency_s'),
        'verify_mode': verify.get('mode'),
        'z3_checks': verify.get('z3_checks'),
        'z3_unsat_events': verify.get('z3_unsat_events'),
        'shadow_unsat_events': verify.get('shadow_unsat_events'),
        'repair_mode': repair.get('mode'),
        'repair_attempts': repair.get('attempts'),
        'repair_successes': repair.get('successes'),
        'escalations': repair.get('escalations'),
        'cdcl_enabled': cdcl.get('enabled'),
        'clauses_learned': cdcl.get('clauses_learned'),
        'outer_iters': rec.get('outer_iters'),
        'oracle_plans': rec.get('oracle_plans'),
        'termination_reason': dl.get('termination_reason'),
        'deadlocked': dl.get('deadlocked'),
        'validator_rejections': dl.get('validator_rejections'),
        'forbidden_rejections': dl.get('forbidden_rejections'),
        'hard_bans': dl.get('hard_bans'),
        'goal': (rec.get('goal') or '')[:200],
        'git_sha': man.get('git_sha'),
        'model': man.get('model'),
        'temperature': man.get('temperature'),
        'timestamp': man.get('timestamp'),
    }


def collect(root: Path, env=None, arms=None, seeds=None, tasks=None):
    """Load every record under `root`, optionally filtered to a slice of the matrix."""
    rows = []
    for _, rec in iter_records(root):
        if env is not None and rec.get('env') != env:
            continue
        if arms is not None and rec.get('arm') not in arms:
            continue
        if seeds is not None and rec.get('seed') not in seeds:
            continue
        if tasks is not None and rec.get('task_id') not in tasks:
            continue
        rows.append(flatten(rec))
    rows.sort(key=lambda r: (str(r['env']), r['task_id'] if r['task_id'] is not None else -1,
                             str(r['arm']), r['seed'] if r['seed'] is not None else -1))
    return rows


def write_csv(rows, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in COLUMNS})
    return out_path


def main():
    p = argparse.ArgumentParser(description='Flatten benchmark records into all_tasks.csv')
    default_root = Path(__file__).resolve().parents[3] / 'results' / 'bench'
    p.add_argument('--out', default=str(default_root))
    p.add_argument('--env', default=None)
    args = p.parse_args()
    root = Path(args.out).resolve()
    rows = collect(root, env=args.env)
    path = write_csv(rows, root / 'all_tasks.csv')
    print(f'[bench] {len(rows)} record(s) -> {path}')


if __name__ == '__main__':
    main()
