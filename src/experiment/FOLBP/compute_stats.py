"""Compute statistics from FOLBP orchestrator summaries.

Reads log/orchestrator_env{N}/summary.txt for each env and produces:
  - per-env stats: success rate, GT/exec mean/median/std/min/max, oracle_calls,
    z3_unsat, clauses_learned, elapsed
  - aggregate stats across all envs
  - excludes FAILURE tasks from executed-steps stats (failed tasks still
    counted in success rate + GT-step stats)
"""
import csv
import math
import os
import re
import statistics
from collections import OrderedDict

LOG_DIR = os.path.join(os.path.dirname(__file__), "log")

TASK_RE = re.compile(
    r"Task\s+(\d+)\s+\[(SUCCESS|FAILURE)\s*\]\s+"
    r"steps=(\d+)/(\d+)\s+oracle_calls=(\d+)\s+z3_unsat=(\d+)\s+"
    r"clauses_learned=(\d+)\s+elapsed=([\d.]+)s"
)


def parse_summary(path):
    tasks = []
    with open(path) as f:
        for line in f:
            m = TASK_RE.search(line)
            if not m:
                continue
            tasks.append({
                "task_id": int(m.group(1)),
                "success": m.group(2) == "SUCCESS",
                "steps": int(m.group(3)),
                "gt_steps": int(m.group(4)),
                "oracle_calls": int(m.group(5)),
                "z3_unsat": int(m.group(6)),
                "clauses_learned": int(m.group(7)),
                "elapsed": float(m.group(8)),
            })
    return tasks


def stats(xs):
    if not xs:
        return {"n": 0, "mean": 0, "median": 0, "std": 0, "min": 0, "max": 0, "sum": 0}
    return {
        "n": len(xs),
        "mean": statistics.mean(xs),
        "median": statistics.median(xs),
        "std": statistics.stdev(xs) if len(xs) > 1 else 0.0,
        "min": min(xs),
        "max": max(xs),
        "sum": sum(xs),
    }


def fmt(s, prec=2):
    return f"mean={s['mean']:.{prec}f} med={s['median']:.{prec}f} std={s['std']:.{prec}f} min={s['min']} max={s['max']}"


def report(env_name, tasks):
    n = len(tasks)
    succ = [t for t in tasks if t["success"]]
    fail = [t for t in tasks if not t["success"]]
    succ_rate = len(succ) / n if n else 0.0

    gt_all = stats([t["gt_steps"] for t in tasks])
    gt_succ = stats([t["gt_steps"] for t in succ])
    exec_succ = stats([t["steps"] for t in succ])
    delta = stats([t["steps"] - t["gt_steps"] for t in succ])
    oracle = stats([t["oracle_calls"] for t in tasks])
    z3u = stats([t["z3_unsat"] for t in tasks])
    cl = stats([t["clauses_learned"] for t in tasks])
    el = stats([t["elapsed"] for t in tasks])

    print(f"=== {env_name} ===")
    print(f"  tasks={n}  succ={len(succ)}  fail={len(fail)}  success_rate={succ_rate:.3f}")
    print(f"  GT steps (all):       {fmt(gt_all)}")
    print(f"  GT steps (succ only): {fmt(gt_succ)}")
    print(f"  exec steps (succ):    {fmt(exec_succ)}")
    print(f"  delta exec-GT (succ): {fmt(delta)}")
    print(f"  oracle_calls:         {fmt(oracle)}")
    print(f"  z3_unsat:             {fmt(z3u)}")
    print(f"  clauses_learned:      {fmt(cl)}")
    print(f"  elapsed (s):          {fmt(el, prec=3)}")
    print(f"  failed task_ids: {[t['task_id'] for t in fail]}")
    return {
        "env": env_name, "n": n, "succ": len(succ), "fail": len(fail),
        "rate": succ_rate, "gt_all": gt_all, "gt_succ": gt_succ,
        "exec_succ": exec_succ, "delta": delta, "oracle": oracle,
        "z3u": z3u, "cl": cl, "el": el,
        "tasks": tasks,
    }


def main():
    all_tasks = []
    rows = []
    for i in range(5):
        path = os.path.join(LOG_DIR, f"orchestrator_env{i}", "summary.txt")
        if not os.path.exists(path):
            continue
        tasks = parse_summary(path)
        for t in tasks:
            t["env"] = f"env{i}"
        rows.append(report(f"env{i}", tasks))
        all_tasks.extend(tasks)
        print()

    # Aggregate (pooled — not weighted mean of means)
    print("=== aggregate (pooled across all envs) ===")
    n = len(all_tasks)
    succ = [t for t in all_tasks if t["success"]]
    fail = [t for t in all_tasks if not t["success"]]
    print(f"  tasks={n}  succ={len(succ)}  fail={len(fail)}  success_rate={len(succ)/n:.3f}")
    print(f"  GT steps (all):       {fmt(stats([t['gt_steps'] for t in all_tasks]))}")
    print(f"  GT steps (succ only): {fmt(stats([t['gt_steps'] for t in succ]))}")
    print(f"  exec steps (succ):    {fmt(stats([t['steps'] for t in succ]))}")
    print(f"  delta exec-GT (succ): {fmt(stats([t['steps'] - t['gt_steps'] for t in succ]))}")
    print(f"  oracle_calls:         {fmt(stats([t['oracle_calls'] for t in all_tasks]))}")
    print(f"  z3_unsat:             {fmt(stats([t['z3_unsat'] for t in all_tasks]))}")
    print(f"  clauses_learned:      {fmt(stats([t['clauses_learned'] for t in all_tasks]))}")
    print(f"  elapsed (s):          {fmt(stats([t['elapsed'] for t in all_tasks]), prec=3)}")

    # plan-length ratio (exec / GT) for successes
    ratios = [t["steps"] / t["gt_steps"] for t in succ if t["gt_steps"] > 0]
    if ratios:
        rs = stats(ratios)
        print(f"  plan-length ratio (exec/GT, succ): mean={rs['mean']:.3f} med={rs['median']:.3f} std={rs['std']:.3f} min={rs['min']:.3f} max={rs['max']:.3f}")

    # optimal-plan rate: succeeded AND exec == GT
    optimal = sum(1 for t in succ if t["steps"] == t["gt_steps"])
    print(f"  optimal plans (exec==GT among succ): {optimal}/{len(succ)} = {optimal/len(succ):.3f}")

    # write summary csv
    out_csv = os.path.join(LOG_DIR, "stats_summary.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "env", "tasks", "succ", "fail", "success_rate",
            "gt_all_mean", "gt_all_std",
            "gt_succ_mean", "exec_succ_mean", "exec_succ_std",
            "delta_mean", "delta_std",
            "oracle_calls_mean", "z3_unsat_mean",
            "clauses_learned_mean", "elapsed_mean_s",
        ])
        for r in rows:
            w.writerow([
                r["env"], r["n"], r["succ"], r["fail"], f"{r['rate']:.3f}",
                f"{r['gt_all']['mean']:.2f}", f"{r['gt_all']['std']:.2f}",
                f"{r['gt_succ']['mean']:.2f}",
                f"{r['exec_succ']['mean']:.2f}", f"{r['exec_succ']['std']:.2f}",
                f"{r['delta']['mean']:+.2f}", f"{r['delta']['std']:.2f}",
                f"{r['oracle']['mean']:.2f}", f"{r['z3u']['mean']:.2f}",
                f"{r['cl']['mean']:.2f}", f"{r['el']['mean']:.3f}",
            ])
    print(f"\n  wrote: {out_csv}")


if __name__ == "__main__":
    main()
