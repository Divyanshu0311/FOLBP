#!/usr/bin/env python3
"""Complete run for PEFA — run every task of an env, store per-task logs, aggregate results.

Each task is launched as its own `main.py --env <env> --task <N> --mode standalone`
subprocess; stdout is captured to log/orchestrator_<env>/task<NN>.log and a summary
report is written to log/orchestrator_<env>/summary.txt.

Usage:
    python3 run_all_tasks.py --env env0
    python3 run_all_tasks.py --env env0 --tasks 2,4,9-11 --parallel 4
    python3 run_all_tasks.py --env env0 --resume        # skip tasks whose log already exists
"""
import argparse
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR.parent))
from orchestrator_common import add_common_args, orchestrate, resolve_gemini_key


def main():
    p = argparse.ArgumentParser(description='PEFA complete run (all tasks of an env)')
    add_common_args(p, env_choices=['env0', 'env1', 'env2', 'env3', 'env4',
                                    'env_test', 'env_merom', 'env_tb'],
                    default_timeout=1800)
    p.add_argument('--lm_id', default='gemini-2.5-flash', help='Gemini model name.')
    p.add_argument('--t', type=float, default=0.5, help='Sampling temperature.')
    args = p.parse_args()
    key = resolve_gemini_key(args.api_key)

    def build_cmd(task_id):
        return [sys.executable, 'main.py',
                '--env', args.env, '--task', str(task_id),
                '--mode', 'standalone',
                '--lm_id', args.lm_id, '--t', str(args.t),
                '--api_key', key]

    orchestrate('PEFA', BASE_DIR, args, build_cmd)


if __name__ == '__main__':
    main()
