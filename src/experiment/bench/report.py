"""The comparison table — console, Markdown, and LaTeX, plus paired statistics.

Built entirely from the records on disk, so it is regenerable at any time:

    python3 bench/report.py --env env0
    python3 bench/sweep.py --env env0 --report-only

Two behaviours worth knowing about:

* **Underpowered tests are suppressed, not printed.** Below `--min-n` paired tasks
  the descriptive table still renders and the p-values do not. A McNemar p-value on
  five tasks is noise with a decimal point, and one in a draft undermines every
  other number in it.
* **Ragged coverage is explicit.** Arms that ran different task sets are never
  silently averaged together: per-arm n is shown, and paired tests run over the
  intersection with its size stated.
* **Multi-seed runs render as mean +/- std, single-seed runs do not.** The std is
  taken over *seed-level* summaries (each seed's own success rate, its own mean
  latency), not over tasks, so it answers "how much does the number move if you run
  the benchmark again". With one seed there is no spread to report and the table is
  byte-identical to what it was before this existed.
"""
import argparse
import statistics
import sys
from collections import Counter
from math import comb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench.aggregate import collect, write_csv
from bench.arms import ARMS, BY_ID, REFERENCE_ARM, resolve_arms

# The benchmark is 20 tasks per environment. env0 ships 22 and env1 ships 21; the
# 3 extras are excluded from the tables so each env contributes equally to the
# pooled result (100 tasks, 20 x 5).
CANONICAL_TASKS_PER_ENV = 20

try:
    from scipy.stats import wilcoxon as _scipy_wilcoxon
except Exception:                                    # scipy is optional
    _scipy_wilcoxon = None


# ------------------------------------------------------------------ statistics

def mcnemar_exact(b: int, c: int):
    """Two-sided exact (binomial) McNemar on discordant pairs.

    b = reference won only, c = other won only. Exact rather than chi-square
    because the discordant counts on a near-saturated benchmark are small, which
    is exactly where the chi-square approximation misleads.
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def wilcoxon(pairs):
    """Wilcoxon signed-rank + matched-pairs rank-biserial effect size."""
    diffs = [a - b for a, b in pairs if a is not None and b is not None]
    nonzero = [d for d in diffs if d != 0]
    if len(nonzero) < 6 or _scipy_wilcoxon is None:
        return None
    try:
        stat, p = _scipy_wilcoxon(nonzero)
    except Exception:
        return None
    n = len(nonzero)
    total = n * (n + 1) / 2
    ranks = {}
    for rank, (_, idx) in enumerate(sorted(((abs(d), i) for i, d in enumerate(nonzero))), 1):
        ranks[idx] = rank
    pos = sum(r for i, r in ranks.items() if nonzero[i] > 0)
    rbc = 2 * pos / total - 1 if total else 0.0
    return {'stat': stat, 'p': p, 'n': n, 'rank_biserial': round(rbc, 3)}


def holm(pvals):
    """Holm-Bonferroni adjusted p-values, order preserved."""
    indexed = sorted(enumerate(pvals), key=lambda kv: kv[1])
    m = len(pvals)
    out = [None] * m
    running = 0.0
    for rank, (idx, p) in enumerate(indexed):
        adj = min(1.0, (m - rank) * p)
        running = max(running, adj)
        out[idx] = running
    return out


# ------------------------------------------------------------------- helpers

def _mean(xs):
    # float() because statistics.mean preserves int for exact means, which would
    # make one column render as "2" and its neighbour as "2.4".
    xs = [x for x in xs if isinstance(x, (int, float)) and not isinstance(x, bool)]
    return float(statistics.mean(xs)) if xs else None


def _fmt(v, prec=2, dash='—'):
    if v is None or v == '':
        return dash
    if isinstance(v, float):
        return f'{v:.{prec}f}'
    return str(v)


# Metrics carried through the per-seed spread. Every one is a scalar summary of a
# whole seed, so taking a std over them is meaningful; counts like `succ` are not
# here because they do not average across seeds of unequal size.
SPREAD_METRICS = (
    'success_pct', 'exec_over_gt', 'llm_calls', 'latency', 'tokens', 'unsat',
    'shadow_unsat', 'repair_attempts', 'repair_successes', 'clauses',
    'deadlock_pct', 'forbidden_rejections', 'hard_bans', 'oracle_plans', 'outer_iters',
)


def summarize(rows):
    """Per-arm descriptive summary over whatever rows were supplied."""
    done = [r for r in rows if r['status'] == 'completed']
    succ = [r for r in done if r['success']]
    # `deadlocked` is absent from every record written before the hard-constraint
    # ablation, so `is True` rather than truthiness: None must not count as False in a
    # denominator that also has no numerator.
    deadlocked = [r for r in done if r['deadlocked'] is True]
    instrumented = [r for r in done if r['termination_reason']]
    return {
        'n': len(done),
        'n_attempted': len(rows),
        'n_incomplete': len(rows) - len(done),
        'succ': len(succ),
        'success_rate': (len(succ) / len(done)) if done else None,
        'success_pct': (100 * len(succ) / len(done)) if done else None,
        'n_instrumented': len(instrumented),
        'deadlocked': len(deadlocked),
        'deadlock_pct': (100 * len(deadlocked) / len(instrumented)) if instrumented else None,
        'forbidden_rejections': _mean([r['forbidden_rejections'] for r in done]),
        'validator_rejections': _mean([r['validator_rejections'] for r in done]),
        'hard_bans': _mean([r['hard_bans'] for r in done]),
        'oracle_plans': _mean([r['oracle_plans'] for r in done]),
        'outer_iters': _mean([r['outer_iters'] for r in done]),
        'termination': Counter(r['termination_reason'] for r in instrumented),
        'exec_over_gt': _mean([r['exec_over_gt'] for r in succ]),
        'llm_calls': _mean([r['total_llm_calls'] for r in done]),
        'latency': _mean([r['wall_clock_s'] for r in done]),
        'tokens': _mean([r['total_tokens'] for r in done]),
        'unsat': _mean([r['z3_unsat_events'] for r in done]),
        'shadow_unsat': _mean([r['shadow_unsat_events'] for r in done]),
        'shadow_tasks': sum(1 for r in done if (r['shadow_unsat_events'] or 0) > 0),
        'repair_attempts': _mean([r['repair_attempts'] for r in done]),
        'repair_successes': _mean([r['repair_successes'] for r in done]),
        'clauses': _mean([r['clauses_learned'] for r in done]),
        'rows': done,
    }


def seed_spread(rows):
    """(seeds, per-seed summaries, {metric: (mean, std)}).

    The std is over seeds, so it is a run-to-run spread rather than a task-to-task one.
    It is None whenever fewer than two seeds produced a usable value — the caller uses
    that to fall back to the single-seed rendering instead of printing a bogus +/- 0.
    """
    seeds = sorted({r['seed'] for r in rows
                    if r['status'] == 'completed' and r['seed'] is not None})
    per_seed = {sd: summarize([r for r in rows if r['seed'] == sd]) for sd in seeds}
    spread = {}
    for key in SPREAD_METRICS:
        vals = [per_seed[sd][key] for sd in seeds if per_seed[sd].get(key) is not None]
        if not vals:
            spread[key] = (None, None)
        else:
            spread[key] = (float(statistics.mean(vals)),
                           float(statistics.stdev(vals)) if len(vals) >= 2 else None)
    return seeds, per_seed, spread


def _cell(s, key, prec=2, dash='—'):
    """One table cell: the pooled value, or mean+/-std once two or more seeds exist."""
    mean, sd = s['spread'].get(key, (None, None))
    if sd is None:
        return _fmt(s.get(key), prec, dash)
    return f'{mean:.{prec}f}±{sd:.{prec}f}'


def _tok_cell(s):
    """Tokens per task in thousands. Raw counts run to six figures, which would set the
    column width for the whole table; 'k' keeps it beside LLM/task where it belongs.

    Reported because the call ratio and the token ratio differ by roughly 2x — one long
    proposal against many short calls — and quoting only calls overstates the saving.
    """
    mean, sd = (s.get('spread') or {}).get('tokens', (None, None))
    if sd is None:
        v = s.get('tokens')
        return '—' if v is None else f'{v / 1000:.1f}k'
    return f'{mean / 1000:.1f}±{sd / 1000:.1f}k'


def _succ_cell(s):
    mean, sd = s['spread'].get('success_pct', (None, None))
    if sd is None:
        return f'{s["succ"]}/{s["n"]}' if s['n'] else '0/0'
    return f'{mean:.1f}±{sd:.1f}'


def by_key(rows):
    """(env, seed, task_id) -> row, for pairing across arms.

    `env` is part of the key because task ids restart at 0 in every environment —
    without it env0/t00, env1/t00, ... collapse onto one another and a pooled
    103-task comparison silently shrinks to the size of the largest single env.
    """
    return {(r['env'], r['seed'], r['task_id']): r
            for r in rows if r['status'] == 'completed'}


# --------------------------------------------------------------------- render

def render(root: Path, env=None, arms=None, seeds=None, tasks=None, min_n=20,
           tasks_per_env=CANONICAL_TASKS_PER_ENV):
    arms = arms or list(ARMS)
    arm_ids = [a.id for a in arms]
    rows = collect(root, env=env, arms=set(arm_ids),
                   seeds=set(seeds) if seeds else None,
                   tasks=set(tasks) if tasks else None)
    # The canonical benchmark is the first N tasks of each env. env0 and env1 carry
    # 2 and 1 extra tasks respectively; excluding them here keeps every env equally
    # weighted in the pooled table. Records for the extras stay on disk and in
    # all_tasks.csv — this filters the report, it does not discard data.
    excluded = []
    if tasks_per_env:
        excluded = sorted({(r['env'], r['task_id']) for r in rows
                           if r['task_id'] is not None and r['task_id'] >= tasks_per_env})
        rows = [r for r in rows if r['task_id'] is not None and r['task_id'] < tasks_per_env]
    csv_path = write_csv(collect(root), root / 'all_tasks.csv')

    per_arm = {}
    for a in arms:
        arm_rows = [r for r in rows if r['arm'] == a.id]
        summary = summarize(arm_rows)
        summary['seeds'], summary['per_seed'], summary['spread'] = seed_spread(arm_rows)
        per_arm[a.id] = summary
    # One arm having two seeds is enough to switch the whole table to mean+/-std: mixing
    # bare numbers and +/- values in one column would read as "this arm has zero spread".
    seeds_seen = sorted({sd for s in per_arm.values() for sd in s['seeds']})
    multi = len(seeds_seen) > 1
    # Deadlock columns only exist for records written after the ablation landed. Without
    # this gate, re-rendering the pre-existing single-seed tree would grow empty columns.
    has_deadlock = any(s['n_instrumented'] for s in per_arm.values())
    lines = []

    header = f'env={env or "all"}'
    if tasks:
        header += f' · tasks={{{",".join(str(t) for t in tasks)}}}'
    if seeds:
        header += f' · seeds={{{",".join(str(s) for s in seeds)}}}'
    total = sum(s['n_attempted'] for s in per_arm.values())
    n_tasks = len({(r['env'], r['task_id']) for r in rows})
    lines.append('')
    lines.append(f'{header} · {len(arms)} arms · {n_tasks} tasks · {total} records')
    if multi:
        lines.append(f'values are mean±std over {len(seeds_seen)} seeds '
                     f'({",".join(str(x) for x in seeds_seen)}); std is the run-to-run '
                     'spread of each seed\'s own summary, not a confidence interval')
    if excluded:
        lines.append(f'excluded {len(excluded)} task(s) beyond {tasks_per_env}/env: '
                     + ', '.join(f'{e} t{t:02d}' for e, t in excluded))
    lines.append('')

    # Every column but `succ` is a per-task mean over the completed runs. The widths
    # widen only in multi-seed mode, so the single-seed table stays byte-identical.
    w = dict(succ=11, ex=11, llm=12, tok=13, lat=12, uns=11, rep=9, cl=11) if multi else \
        dict(succ=8, ex=8, llm=9, tok=9, lat=9, uns=7, rep=9, cl=8)
    cols = (f'{"arm":26} {"succ":>{w["succ"]}} {"exec/GT":>{w["ex"]}} '
            f'{"LLM/task":>{w["llm"]}} {"tok/task":>{w["tok"]}} {"latency":>{w["lat"]}} '
            f'{"UNSAT":>{w["uns"]}} {"repairs":>{w["rep"]}} {"clauses":>{w["cl"]}}')
    lines.append(cols)
    lines.append('─' * len(cols))
    for a in arms:
        s = per_arm[a.id]
        if not s['n_attempted']:
            lines.append(f'{a.label:26} {"— not run —":>{w["succ"]}}')
            continue
        succ = _succ_cell(s)
        # Shadow-mode UNSAT is bracketed: detected on purpose, deliberately not acted on.
        unsat = (f'[{_cell(s, "shadow_unsat")}]' if a.id == 'folbp-naive'
                 else _cell(s, 'unsat'))
        # Repairs stay a bare successes/attempts pair even in multi-seed mode: two
        # +/- values in one cell is unreadable, and table_by_seed.md carries the detail.
        repairs = ('—' if s['repair_attempts'] in (None, 0)
                   else f'{_fmt(s["repair_successes"], 1)}/{_fmt(s["repair_attempts"], 1)}')
        lines.append(
            f'{a.label:26} {succ:>{w["succ"]}} {_cell(s, "exec_over_gt"):>{w["ex"]}} '
            f'{_cell(s, "llm_calls", 1):>{w["llm"]}} '
            f'{_tok_cell(s):>{w["tok"]}} '
            f'{_cell(s, "latency", 1) + "s":>{w["lat"]}} '
            f'{unsat:>{w["uns"]}} {repairs:>{w["rep"]}} '
            f'{_cell(s, "clauses", 1):>{w["cl"]}}')
        if s['n_incomplete']:
            lines.append(f'{"":26} ({s["n_incomplete"]} run(s) did not complete)')

    # ---- soundness headline from the shadow-verified arm
    naive = per_arm.get('folbp-naive')
    if naive and naive['n']:
        hit = naive['shadow_tasks']
        pct = 100 * hit / naive['n']
        lines.append('')
        lines.append(f'soundness: the naive one-shot planner would have dispatched a '
                     f'physically infeasible action on {hit}/{naive["n"]} tasks ({pct:.0f}%)')

    # ---- the hard-constraint deadlock ablation
    if has_deadlock:
        lines.append('')
        lines.extend(_deadlock_block(arms, per_arm))

    # ---- paired statistics against the reference arm
    ref = REFERENCE_ARM if REFERENCE_ARM.id in per_arm else None
    if ref:
        lines.append('')
        lines.append(f'paired vs {ref.label.strip()}:')
        ref_by = by_key(per_arm[ref.id]['rows'])
        comparisons = []
        for a in arms:
            if a.id == ref.id:
                continue
            other = by_key(per_arm[a.id]['rows'])
            shared = sorted(set(ref_by) & set(other))
            if not shared:
                lines.append(f'  {a.label.strip():24} no overlapping tasks')
                continue
            if len(shared) < min_n:
                lines.append(f'  {a.label.strip():24} n={len(shared)} — SUPPRESSED '
                             f'(< --min-n {min_n}; descriptive only)')
                continue
            b = sum(1 for k in shared if ref_by[k]['success'] and not other[k]['success'])
            c = sum(1 for k in shared if not ref_by[k]['success'] and other[k]['success'])
            p_mc = mcnemar_exact(b, c)
            calls = wilcoxon([(ref_by[k]['total_llm_calls'], other[k]['total_llm_calls'])
                              for k in shared])
            lat = wilcoxon([(ref_by[k]['wall_clock_s'], other[k]['wall_clock_s'])
                            for k in shared])
            comparisons.append((a, len(shared), b, c, p_mc, calls, lat))

        raw_p = [c[4] for c in comparisons]
        adj_p = holm(raw_p) if raw_p else []
        for (a, n, b, c, p_mc, calls, lat), p_adj in zip(comparisons, adj_p):
            lines.append(f'  {a.label.strip():24} n={n}')
            lines.append(f'      McNemar  discordant b={b} c={c}  p={p_mc:.4f}  '
                         f'p_holm={p_adj:.4f}')
            for name, w in (('LLM calls', calls), ('latency', lat)):
                if w is None:
                    lines.append(f'      Wilcoxon {name:10} n too small / scipy unavailable')
                else:
                    lines.append(f'      Wilcoxon {name:10} p={w["p"]:.4g}  '
                                 f'rank-biserial={w["rank_biserial"]:+.3f}  n={w["n"]}')

    # ---- provenance: rows built from different code versions are flagged, not hidden
    shas = {}
    for r in rows:
        if r['status'] == 'completed' and r['git_sha']:
            shas.setdefault(r['git_sha'][:8], 0)
            shas[r['git_sha'][:8]] += 1
    if len(shas) > 1:
        lines.append('')
        lines.append('WARNING: records span multiple code versions — '
                     + ', '.join(f'{sha}: {n}' for sha, n in sorted(shas.items())))
        lines.append('         re-run with --force, or with --resume --strict-resume, '
                     'before quoting these numbers.')

    text = '\n'.join(lines)
    print(text)

    md = _markdown(arms, per_arm, header, multi, seeds_seen)
    (root / 'table_main.md').write_text(md)
    (root / 'table_main.tex').write_text(_latex(arms, per_arm, multi))
    (root / 'report.txt').write_text(text + '\n')
    written = [str(root / 'table_main.md'), str(root / 'table_main.tex'),
               str(root / 'report.txt')]
    # The two extra artifacts are written only when they have something to say, so a
    # single-seed tree with pre-ablation records keeps exactly the files it had.
    if multi:
        (root / 'table_by_seed.md').write_text(_by_seed_markdown(arms, per_arm, seeds_seen))
        written.append(str(root / 'table_by_seed.md'))
    if has_deadlock:
        (root / 'table_deadlock.md').write_text(_deadlock_markdown(arms, per_arm))
        written.append(str(root / 'table_deadlock.md'))
    print(f'\n[bench] wrote {csv_path}')
    print('[bench] wrote ' + ', '.join(written))
    return text


def _markdown(arms, per_arm, header, multi=False, seeds_seen=()):
    succ_head = 'success (%)' if multi else 'success'
    out = [f'# Benchmark — {header}', '',
           f'| arm | {succ_head} | exec/GT | LLM calls/task | tokens/task | latency (s) | UNSAT | repairs | clauses |',
           '|---|---|---|---|---|---|---|---|---|']
    for a in arms:
        s = per_arm[a.id]
        if not s['n_attempted']:
            out.append(f'| {a.label.strip()} | — not run — | | | | | | | |')
            continue
        unsat = (f'[{_cell(s, "shadow_unsat")}]' if a.id == 'folbp-naive'
                 else _cell(s, 'unsat'))
        repairs = ('—' if s['repair_attempts'] in (None, 0)
                   else f'{_fmt(s["repair_successes"], 1)}/{_fmt(s["repair_attempts"], 1)}')
        out.append(f'| {a.label.strip()} | {_succ_cell(s)} | '
                   f'{_cell(s, "exec_over_gt")} | {_cell(s, "llm_calls", 1)} | '
                   f'{_tok_cell(s)} | '
                   f'{_cell(s, "latency", 1)} | {unsat} | {repairs} | '
                   f'{_cell(s, "clauses", 1)} |')
    out += ['', '`[n]` = shadow-mode UNSAT: detected by Z3, deliberately not acted on.']
    if multi:
        out.append(f'`a±b` = mean ± std over {len(seeds_seen)} seeds '
                   f'({", ".join(str(x) for x in seeds_seen)}). The std is the spread of '
                   'the per-seed summaries — how far the number moves when the benchmark '
                   'is run again — not a confidence interval.')
    return '\n'.join(out) + '\n'


def _by_seed_markdown(arms, per_arm, seeds_seen):
    """Every arm x seed row, so the aggregate above never hides a divergent seed."""
    out = ['# Benchmark — per seed', '',
           'The rows `table_main.md` averages. Present so a large ± can be traced to the '
           'seed that produced it.', '',
           '| arm | seed | n | success | exec/GT | LLM calls/task | tokens/task | latency (s) | UNSAT | clauses |',
           '|---|---|---|---|---|---|---|---|---|']
    for a in arms:
        arm_summary = per_arm[a.id]
        for sd in seeds_seen:
            s = arm_summary['per_seed'].get(sd)
            if not s or not s['n']:
                out.append(f'| {a.label.strip()} | {sd} | 0 | — not run — | | | | | |')
                continue
            out.append(f'| {a.label.strip()} | {sd} | {s["n"]} | '
                       f'{s["succ"]}/{s["n"]} ({_fmt(s["success_pct"], 1)}%) | '
                       f'{_fmt(s["exec_over_gt"])} | {_fmt(s["llm_calls"], 1)} | '
                       f'{_tok_cell(s)} | '
                       f'{_fmt(s["latency"], 1)} | {_fmt(s["unsat"])} | '
                       f'{_fmt(s["clauses"], 1)} |')
    return '\n'.join(out) + '\n'


# --------------------------------------------------------- deadlock ablation
DEADLOCK_CAPTION = (
    'deadlock = the run ended with no progress in 3 consecutive Oracle re-plans while '
    'its own learned bans were rejecting every plan it produced (record field '
    'deadlock.deadlocked). A stall with no such rejection is an ordinary failure and is '
    'not counted. "blocked plans" is how many plans the Plan Validator threw out for '
    'violating a learned ban.'
)


def _deadlock_rows(arms, per_arm):
    """(arm, summary) for arms that actually carry deadlock instrumentation."""
    return [(a, per_arm[a.id]) for a in arms if per_arm[a.id]['n_instrumented']]


def _deadlock_block(arms, per_arm):
    lines = ['deadlock ablation — learned constraints treated as HARD vs. soft:', '']
    cols = (f'{"arm":26} {"succ":>9} {"deadlock":>10} {"oracle plans":>13} '
            f'{"outer iters":>12} {"hard bans":>10} {"blocked plans":>14}')
    lines.append(cols)
    lines.append('─' * len(cols))
    for a, s in _deadlock_rows(arms, per_arm):
        dl = f'{s["deadlocked"]}/{s["n_instrumented"]}'
        lines.append(
            f'{a.label:26} {_succ_cell(s):>9} {dl:>10} '
            f'{_cell(s, "oracle_plans", 1):>13} {_cell(s, "outer_iters", 1):>12} '
            f'{_cell(s, "hard_bans", 2):>10} '
            f'{_cell(s, "forbidden_rejections", 2):>14}')
    lines.append('')
    lines.append(DEADLOCK_CAPTION)
    hard = per_arm.get('folbp-hard')
    ref = per_arm.get(REFERENCE_ARM.id)
    if hard and ref and hard['n_instrumented'] and ref['n_instrumented']:
        lines.append('')
        lines.append(
            f'headline: promoting every learned clause to a hard constraint deadlocks '
            f'{hard["deadlocked"]}/{hard["n_instrumented"]} tasks '
            f'({_fmt(hard["deadlock_pct"], 0)}%) and costs '
            f'{_fmt((ref["success_pct"] or 0) - (hard["success_pct"] or 0), 1)} points of '
            f'success; the same pipeline keeping state-dependent failures soft deadlocks '
            f'{ref["deadlocked"]}/{ref["n_instrumented"]}.')
    return lines


def _deadlock_markdown(arms, per_arm):
    out = ['# Deadlock ablation — hard vs. soft learned constraints', '',
           '| arm | success | deadlocked | oracle plans | outer iters | hard bans | blocked plans |',
           '|---|---|---|---|---|---|---|']
    for a, s in _deadlock_rows(arms, per_arm):
        out.append(f'| {a.label.strip()} | {_succ_cell(s)} | '
                   f'{s["deadlocked"]}/{s["n_instrumented"]} '
                   f'({_fmt(s["deadlock_pct"], 0)}%) | '
                   f'{_cell(s, "oracle_plans", 1)} | {_cell(s, "outer_iters", 1)} | '
                   f'{_cell(s, "hard_bans", 2)} | '
                   f'{_cell(s, "forbidden_rejections", 2)} |')
    out += ['', DEADLOCK_CAPTION, '', '## How each run ended', '',
            '| arm | ' + ' | '.join(_termination_keys(arms, per_arm)) + ' |',
            '|---|' + '---|' * len(_termination_keys(arms, per_arm))]
    for a, s in _deadlock_rows(arms, per_arm):
        out.append(f'| {a.label.strip()} | '
                   + ' | '.join(str(s['termination'].get(k, 0))
                                for k in _termination_keys(arms, per_arm)) + ' |')
    return '\n'.join(out) + '\n'


def _termination_keys(arms, per_arm):
    keys = set()
    for _, s in _deadlock_rows(arms, per_arm):
        keys |= set(s['termination'])
    # goal_reached first, then the failure modes alphabetically — the ordering the table
    # is read in, rather than whatever order the Counter happened to fill.
    ordered = [k for k in ('goal_reached',) if k in keys]
    return ordered + sorted(keys - set(ordered))


def _tex(cell: str) -> str:
    """Console '96.0±1.4' -> LaTeX '96.0 $\\pm$ 1.4'."""
    return cell.replace('±', r' $\pm$ ')


def _latex(arms, per_arm, multi=False):
    out = [r'\begin{tabular}{lrrrrrr}', r'\toprule',
           r'Method & Success & Exec/GT & LLM calls & Tokens & Latency (s) & Repairs \\',
           r'\midrule']
    for a in arms:
        s = per_arm[a.id]
        label = a.label.strip().replace('-', r'$-$') if a.label.startswith(' ') else a.label
        if not s['n_attempted']:
            continue
        repairs = ('--' if s['repair_attempts'] in (None, 0)
                   else f'{_fmt(s["repair_successes"], 1)}/{_fmt(s["repair_attempts"], 1)}')
        out.append(f'{label} & {_tex(_succ_cell(s))} & {_tex(_cell(s, "exec_over_gt"))} & '
                   f'{_tex(_cell(s, "llm_calls", 1))} & {_tex(_tok_cell(s))} & '
                   f'{_tex(_cell(s, "latency", 1))} & {repairs} \\\\')
    out += [r'\bottomrule', r'\end{tabular}']
    return '\n'.join(out) + '\n'


def main():
    p = argparse.ArgumentParser(description='Render the benchmark comparison table.')
    default_root = Path(__file__).resolve().parents[3] / 'results' / 'bench'
    p.add_argument('--out', default=str(default_root))
    p.add_argument('--env', default=None)
    p.add_argument('--arms', default='all')
    p.add_argument('--seeds', default=None)
    p.add_argument('--tasks', default=None)
    p.add_argument('--min-n', type=int, default=20, dest='min_n')
    p.add_argument('--tasks-per-env', type=int, default=CANONICAL_TASKS_PER_ENV,
                   dest='tasks_per_env',
                   help='Cap each env to its first N tasks (default '
                        f'{CANONICAL_TASKS_PER_ENV}). Pass 0 to include every task.')
    args = p.parse_args()
    seeds = [int(s) for s in args.seeds.split(',')] if args.seeds else None
    tasks = [int(t) for t in args.tasks.split(',')] if args.tasks else None
    render(Path(args.out).resolve(), env=args.env, arms=resolve_arms(args.arms),
           seeds=seeds, tasks=tasks, min_n=args.min_n,
           tasks_per_env=args.tasks_per_env)


if __name__ == '__main__':
    main()
