"""Compute metrics from FOLBP env*.txt logs.

For each env:
- Parse all (task_id, gt_steps, success, steps) tuples
- For tasks that appear multiple times (re-runs), keep the LAST occurrence
- Report: total tasks, succeeded, failed, success rate,
          avg GT steps (all tasks), avg success steps (succeeded only)
"""
import csv
import os
import re
from collections import OrderedDict

LOG_DIR = os.path.join(os.path.dirname(__file__), "log")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "log")

TASK_RE = re.compile(r"\[FOLBP\]\[TASK\] task_id=(\d+) env_id=\d+ name=\S+ gt_steps=(\d+)")
FINAL_RE = re.compile(r"\[FOLBP\]\[FINAL\] success=(True|False) steps=(\d+)")


def parse_env(path):
    """Walk lines; pair each TASK header with its next FINAL line.
    Tasks without a FINAL (interrupted runs) are dropped.
    Re-runs of the same task_id overwrite earlier entries (keep last).
    """
    tasks = OrderedDict()
    pending = None
    with open(path) as f:
        for line in f:
            m = TASK_RE.search(line)
            if m:
                pending = (int(m.group(1)), int(m.group(2)))
                continue
            m = FINAL_RE.search(line)
            if m and pending is not None:
                tid, gt = pending
                success = m.group(1) == "True"
                steps = int(m.group(2))
                tasks[tid] = {"gt_steps": gt, "success": success, "steps": steps}
                pending = None
    return tasks


def report(env_name, tasks):
    n = len(tasks)
    successes = [t for t in tasks.values() if t["success"]]
    failures = [t for t in tasks.values() if not t["success"]]

    avg_gt = sum(t["gt_steps"] for t in tasks.values()) / n if n else 0.0
    avg_succ_steps = (
        sum(t["steps"] for t in successes) / len(successes) if successes else 0.0
    )
    avg_succ_gt = (
        sum(t["gt_steps"] for t in successes) / len(successes) if successes else 0.0
    )
    success_rate = len(successes) / n if n else 0.0

    print(f"=== {env_name} ===")
    print(f"  tasks={n}  success={len(successes)}  fail={len(failures)}  success_rate={success_rate:.3f}")
    print(f"  avg GT steps (all tasks):        {avg_gt:.2f}")
    print(f"  avg GT steps (successes only):   {avg_succ_gt:.2f}")
    print(f"  avg executed steps (successes):  {avg_succ_steps:.2f}")
    delta = avg_succ_steps - avg_succ_gt
    print(f"  delta (executed - GT, successes): {delta:+.2f}")
    failed_ids = [tid for tid, t in tasks.items() if not t["success"]]
    print(f"  failed task_ids: {failed_ids}")
    return {
        "env": env_name,
        "n": n,
        "succ": len(successes),
        "fail": len(failures),
        "rate": success_rate,
        "avg_gt": avg_gt,
        "avg_succ_gt": avg_succ_gt,
        "avg_succ_steps": avg_succ_steps,
    }


def write_per_task_csv(env_name, tasks, out_path):
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["env", "task_id", "gt_steps", "success", "executed_steps", "delta_vs_gt"])
        for tid, t in tasks.items():
            delta = t["steps"] - t["gt_steps"] if t["success"] else ""
            w.writerow([env_name, tid, t["gt_steps"], t["success"], t["steps"], delta])


def main():
    overall = []
    per_task_rows = []
    for fn in sorted(os.listdir(LOG_DIR)):
        if not (fn.startswith("env") and fn.endswith(".txt")):
            continue
        path = os.path.join(LOG_DIR, fn)
        env_name = fn[:-4]
        tasks = parse_env(path)
        overall.append(report(env_name, tasks))
        for tid, t in tasks.items():
            per_task_rows.append((env_name, tid, t))
        print()

    print("=== aggregate over envs ===")
    tot_n = sum(r["n"] for r in overall)
    tot_s = sum(r["succ"] for r in overall)
    tot_f = sum(r["fail"] for r in overall)
    # weighted means
    if tot_n:
        avg_gt = sum(r["avg_gt"] * r["n"] for r in overall) / tot_n
    else:
        avg_gt = 0.0
    if tot_s:
        avg_succ_gt = sum(r["avg_succ_gt"] * r["succ"] for r in overall) / tot_s
        avg_succ_steps = sum(r["avg_succ_steps"] * r["succ"] for r in overall) / tot_s
    else:
        avg_succ_gt = avg_succ_steps = 0.0
    print(f"  total tasks={tot_n}  succ={tot_s}  fail={tot_f}  success_rate={(tot_s/tot_n if tot_n else 0):.3f}")
    print(f"  avg GT steps (all):              {avg_gt:.2f}")
    print(f"  avg GT steps (successes only):   {avg_succ_gt:.2f}")
    print(f"  avg executed steps (successes):  {avg_succ_steps:.2f}")
    print(f"  delta (executed - GT, successes): {avg_succ_steps - avg_succ_gt:+.2f}")

    # write summary and per-task csvs
    summary_path = os.path.join(RESULTS_DIR, "metrics_summary.csv")
    with open(summary_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["env", "tasks", "succ", "fail", "success_rate",
                    "avg_gt_all", "avg_gt_succ", "avg_exec_succ", "delta_exec_minus_gt"])
        for r in overall:
            w.writerow([r["env"], r["n"], r["succ"], r["fail"], f"{r['rate']:.3f}",
                        f"{r['avg_gt']:.2f}", f"{r['avg_succ_gt']:.2f}",
                        f"{r['avg_succ_steps']:.2f}",
                        f"{r['avg_succ_steps'] - r['avg_succ_gt']:+.2f}"])
        w.writerow(["ALL", tot_n, tot_s, tot_f, f"{(tot_s/tot_n if tot_n else 0):.3f}",
                    f"{avg_gt:.2f}", f"{avg_succ_gt:.2f}", f"{avg_succ_steps:.2f}",
                    f"{avg_succ_steps - avg_succ_gt:+.2f}"])
    print(f"\n  wrote: {summary_path}")

    per_task_path = os.path.join(RESULTS_DIR, "metrics_per_task.csv")
    with open(per_task_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["env", "task_id", "gt_steps", "success", "executed_steps", "delta_vs_gt"])
        for env_name, tid, t in per_task_rows:
            delta = t["steps"] - t["gt_steps"] if t["success"] else ""
            w.writerow([env_name, tid, t["gt_steps"], t["success"], t["steps"], delta])
    print(f"  wrote: {per_task_path}")


if __name__ == "__main__":
    main()
