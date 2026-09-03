"""Arm registry — the single definition of what each benchmark condition runs.

`sweep.py`, `aggregate.py`, and `report.py` all read from here, so adding, dropping,
or re-labelling an arm is a one-line edit that cannot drift between the runner and
the table.

The FOLBP arms are combinations of orthogonal flags over the *same* pipeline
(`--verify` / `--repair` / `--cdcl` / `--max_oracle_plans`), never forked copies of
the code. That is what lets the ablation table claim the arms differ only where
stated.

One deliberate exception: `folbp-hard` runs out of `src/experiment/FOLBP_hard/`,
a fork of `FOLBP/`. Its ablation axis is the *soft/hard* split inside
`constraint_injector.HARD_REASONS`, which is a module-level constant consulted on
the learning path rather than a runtime option, so there is no flag to turn. The
fork is held to a one-file diff:

    diff -r --exclude=log --exclude=__pycache__ FOLBP FOLBP_hard

must print `constraint_injector.py` and nothing else — that check is what lets this
arm make the same "differ only where stated" claim as the flag-based ones.

Note on `seed`: neither codebase can pin Gemini's sampler, so `--seed` is a *repeat
index*, not a reproducibility seed. It exists to label independent samples of a
stochastic system for the paired statistics — it does not make a run replayable.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List

EXPERIMENT_DIR = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Arm:
    id: str
    label: str            # table row label
    dir_name: str         # planner directory under src/experiment/
    flags: Dict[str, str] = field(default_factory=dict)
    note: str = ''
    is_reference: bool = False   # the full method every ablation is measured against

    @property
    def base_dir(self) -> Path:
        return EXPERIMENT_DIR / self.dir_name

    def expected_record_flags(self) -> Dict[str, str]:
        """The semantic flag dict a planner writes into its record for this arm.

        `main.py` records `{'verify': 'z3', 'cdcl': True, ...}` rather than the CLI
        spellings, so `--resume` needs this translation to tell whether an existing
        record was produced under the configuration currently being asked for.
        """
        out: Dict[str, str] = {}
        for flag, value in self.flags.items():
            if flag == '--cdcl':
                out['cdcl'] = True
            elif flag == '--no-cdcl':
                out['cdcl'] = False
            elif flag == '--nudge':
                out['nudge'] = True
            elif flag == '--no-nudge':
                out['nudge'] = False
            elif flag == '--max_oracle_plans':
                out['max_oracle_plans'] = int(value)
            else:
                out[flag.lstrip('-')] = value
        return out

    def build_cmd(self, python: str, env: str, task_id: int, seed: int,
                  record_path: Path, lm_id: str, temperature: float,
                  api_key: str) -> List[str]:
        cmd = [python, 'main.py',
               '--env', env, '--task', str(task_id),
               '--arm', self.id, '--seed', str(seed),
               '--record_path', str(record_path),
               '--lm_id', lm_id, '--t', str(temperature)]
        if api_key:
            cmd += ['--api_key', api_key]
        if self.dir_name == 'PEFA':
            cmd += ['--mode', 'standalone']
        for flag, value in self.flags.items():
            cmd.append(flag)
            if value is not None:      # None means a bare switch like --no-cdcl
                cmd.append(str(value))
        return cmd


ARMS: List[Arm] = [
    Arm(
        id='folbp',
        label='FOLBP (full, ours)',
        dir_name='FOLBP',
        flags={'--verify': 'z3', '--repair': 'symbolic', '--cdcl': None,
               '--executor_mode': 'deterministic'},
        note='Z3 verification + symbolic UNSAT-core repair + CDCL learning.',
        is_reference=True,
    ),
    Arm(
        id='folbp-prompt-only',
        label='  - verification',
        dir_name='FOLBP',
        flags={'--verify': 'none', '--repair': 'none', '--no-cdcl': None,
               '--executor_mode': 'deterministic'},
        note='Same prompt and the same outer re-plan loop, no symbolic layer. '
             'Runtime executor rejections still trigger a re-plan, so this is the '
             'strong "good prompt + environment feedback" baseline.',
    ),
    Arm(
        id='folbp-llm-repair',
        label='  repair: LLM re-prompt',
        dir_name='FOLBP',
        flags={'--verify': 'z3', '--repair': 'llm', '--cdcl': None,
               '--executor_mode': 'deterministic'},
        note='Z3 supplies the same diagnosis; the fix comes from re-prompting the '
             'Oracle instead of replanner.py. Same attempt budget as symbolic.',
    ),
    Arm(
        id='folbp-no-cdcl',
        label='  - CDCL',
        dir_name='FOLBP',
        flags={'--verify': 'z3', '--repair': 'symbolic', '--no-cdcl': None,
               '--executor_mode': 'deterministic'},
        note='Verification and symbolic repair intact, conflict learning off.',
    ),
    Arm(
        id='folbp-nudge',
        label='  - CDCL, generic nudge',
        dir_name='FOLBP',
        flags={'--verify': 'z3', '--repair': 'symbolic', '--no-cdcl': None,
               '--nudge': None, '--executor_mode': 'deterministic'},
        note='Floor control for the CDCL ablation. Identical to - CDCL except that a '
             'fixed content-free string ("your previous plan was rejected, produce a '
             'different one") is injected on re-plan. - CDCL leaves the prompt '
             'byte-identical across re-plans, so the proposer returns the same plan and '
             'the outer loop is a no-op; this arm separates the value of the *content* '
             'of the feedback from the value of perturbing the prompt at all.',
    ),
    Arm(
        id='folbp-hard',
        label='  constraints: hard',
        dir_name='FOLBP_hard',
        flags={'--verify': 'z3', '--repair': 'symbolic', '--cdcl': None,
               '--executor_mode': 'deterministic'},
        note='Flags byte-identical to the full method — the difference is the fork: '
             'every learned clause becomes a HARD ban in the Plan Validator instead of '
             'a state-dependent hint. One state-dependent failure is thereby '
             'generalized into "this agent can never [verb] this class", so no later '
             'plan can reach the goal and the loop stalls at no_progress. This is the '
             'deadlock the soft/hard split in FOLBP exists to avoid.',
    ),
    Arm(
        id='folbp-naive',
        label='  naive one-shot',
        dir_name='FOLBP',
        flags={'--verify': 'shadow', '--repair': 'none', '--no-cdcl': None,
               '--max_oracle_plans': '1', '--executor_mode': 'deterministic'},
        note='Plan once, execute regardless, never re-plan. Z3 runs in shadow mode: '
             'every UNSAT is logged but not acted on, which is the soundness number.',
    ),
    Arm(
        id='pefa',
        label='PEFA (baseline)',
        dir_name='PEFA',
        flags={},
        note='Stock PEFA: per-step oracle dialogue, no symbolic layer.',
    ),
]

BY_ID: Dict[str, Arm] = {a.id: a for a in ARMS}
REFERENCE_ARM = next(a for a in ARMS if a.is_reference)


def resolve_arms(spec: str) -> List[Arm]:
    """'all' or a comma-separated list of arm ids, preserving registry order."""
    if spec.strip() in ('all', '*'):
        return list(ARMS)
    wanted = [s.strip() for s in spec.split(',') if s.strip()]
    unknown = [w for w in wanted if w not in BY_ID]
    if unknown:
        raise SystemExit(f'[bench] unknown arm(s): {", ".join(unknown)}\n'
                         f'        known: {", ".join(BY_ID)}')
    return [a for a in ARMS if a.id in wanted]
