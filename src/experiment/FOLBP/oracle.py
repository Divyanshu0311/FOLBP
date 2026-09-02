"""(1) Oracle LLM — generates the full plan in a single LLM call.

Loads full_plan_prompt.txt, fills in observations, task goal, and any constraints
that the Constraint Injector (D) has appended, calls Gemini once, returns a list
of {"agent": "<class name>(id)", "action": "[verb] <obj>(id)"} dicts.
"""
import json
import re
from typing import Callable, Dict, List


def _render_atoms(missing_preconds) -> str:
    """('close', 23, 26) -> 'close(23, 26)' — the same ground atoms the Replanner sees."""
    if not missing_preconds:
        return '-'
    rendered = []
    for atom in missing_preconds:
        if not atom:
            continue
        name, args = atom[0], atom[1:]
        rendered.append(f'{name}({", ".join(str(a) for a in args)})' if args else str(name))
    return ', '.join(rendered) or '-'


class OracleLLM:
    def __init__(self, args, generator: Callable, sampling_params: Dict):
        self.args = args
        self.generator = generator
        self.sampling_params = sampling_params
        with open('./prompt/full_plan_prompt.txt', 'r') as f:
            self.template = f.read()
        self.repair_template = None  # loaded lazily; only the --repair llm arm needs it

    def generate_plan(self, agent_observations: str, task_goal: str,
                      num_agents: int, injected_constraints: str = '',
                      last_attempt: str = '') -> List[Dict[str, str]]:
        prompt = self.template
        prompt = prompt.replace('#TASK_GOAL#', task_goal)
        prompt = prompt.replace('#AGENT_OBSERVATIONS#', agent_observations)
        prompt = prompt.replace('#NUMBER_AGENTS#', str(num_agents))
        prompt = prompt.replace('#INJECTED_CONSTRAINTS#', injected_constraints or '')
        prompt = prompt.replace('#LAST_ATTEMPT#', last_attempt or '')

        outputs, _ = self.generator([{'role': 'user', 'content': prompt}], self.sampling_params)
        raw = outputs[0] if outputs else '[]'
        return self._parse(raw)

    def repair_plan(self, agent_observations: str, task_goal: str, num_agents: int,
                    injected_constraints: str, plan: List[Dict[str, str]],
                    result) -> List[Dict[str, str]]:
        """(R, --repair llm) Re-prompt the Oracle with the Z3 diagnosis instead of
        deriving the fix symbolically.

        The LLM is handed exactly what replanner.py is handed — the failing step index
        and the ground atoms in the UNSAT core, with the same semantics spelled out —
        so the arms differ only in the repair mechanism, not in the diagnosis quality.
        """
        if self.repair_template is None:
            with open('./prompt/repair_prompt.txt', 'r') as f:
                self.repair_template = f.read()

        idx = result.failed_step_index
        failed_step = plan[idx] if 0 <= idx < len(plan) else {}
        prompt = self.repair_template
        prompt = prompt.replace('#FAILED_STEP_INDEX#', str(idx))
        prompt = prompt.replace('#FAILED_STEP#', json.dumps(failed_step))
        prompt = prompt.replace('#MISSING_PRECONDS#', _render_atoms(result.missing_preconds))
        prompt = prompt.replace('#UNSAT_CORE#', ', '.join(str(c) for c in result.unsat_core_labels) or '-')
        prompt = prompt.replace('#CURRENT_PLAN#', json.dumps(plan, indent=2))
        prompt = prompt.replace('#TASK_GOAL#', task_goal)
        prompt = prompt.replace('#AGENT_OBSERVATIONS#', agent_observations)
        prompt = prompt.replace('#NUMBER_AGENTS#', str(num_agents))
        prompt = prompt.replace('#INJECTED_CONSTRAINTS#', injected_constraints or '')

        outputs, _ = self.generator([{'role': 'user', 'content': prompt}],
                                    self.sampling_params, 'repair')
        raw = outputs[0] if outputs else '[]'
        return self._parse(raw)

    def _parse(self, raw: str) -> List[Dict[str, str]]:
        # Strip code fences and locate the outer JSON array.
        text = re.sub(r'```(?:json)?\s*', '', raw).replace('```', '').strip()
        start = text.find('[')
        end = text.rfind(']')
        if start == -1 or end == -1 or end <= start:
            return []
        try:
            arr = json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            return []
        plan = []
        for item in arr:
            if isinstance(item, dict) and 'agent' in item and 'action' in item:
                plan.append(self._normalize_step(item))
        return [s for s in plan if s]

    @staticmethod
    def _normalize_step(item: Dict[str, str]) -> Dict[str, str]:
        """Coerce Gemini's looser shapes into canonical <class>(id) / [verb] <obj>(id).

        Gemini frequently drops the brackets the prompt asks for, e.g. it returns
        "robot arm(50)" / "grab <silver coin>(48)" instead of "<robot arm>(50)" /
        "[grab] <silver coin>(48)". Without normalization the downstream Z3 parser
        labels every step malformed and the arena loops indefinitely.
        """
        agent = str(item.get('agent', '')).strip()
        action = str(item.get('action', '')).strip()

        # agent: ensure <class>(id).
        if '<' not in agent or '>' not in agent:
            m = re.match(r'^([^()<>]+?)\s*\((\d+)\)\s*$', agent)
            if m:
                agent = f'<{m.group(1).strip()}>({m.group(2)})'

        # action: ensure [verb] prefix.
        if not action.startswith('['):
            m = re.match(r'^([a-zA-Z_]+)\b\s*(.*)$', action)
            if m:
                verb, rest = m.group(1), m.group(2).strip()
                action = f'[{verb}] {rest}' if rest else f'[{verb}]'

        # object: ensure each (id) is preceded by <class>; if Gemini wrote bare
        # `silver coin(48)`, wrap to `<silver coin>(48)`. The negative lookahead
        # keeps the putinto/puton connectors ("into"/"on") out of the class name.
        action = re.sub(
            r'(?<![<>\w])(?!into\s|on\s)([a-zA-Z][a-zA-Z _]*?)\s*\((\d+)\)',
            lambda m: f'<{m.group(1).strip()}>({m.group(2)})',
            action,
        )

        return {'agent': agent, 'action': action}
