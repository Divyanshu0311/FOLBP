"""(4) Plan Validator — capability dict (robot x action), PEFA-style.

Walks the plan; for each step verifies the verb is in the agent's capability vocabulary.
A capability is sourced two ways:
  1. A static per-robot verb set (the "capability dict" of the diagram).
  2. Injected forbidden patterns from box D (Constraint Injection) — learned clauses
     that say e.g. "robot dog cannot grab object X without movetowards X first".

This is the lightweight sibling of the Z3 reasoner: Z3 checks state preconditions,
PlanValidator checks the verb-vs-robot capability mapping plus injected constraints.
"""
import re
from dataclasses import dataclass, field
from typing import List, Tuple


CAPABILITY = {
    'quadrotor':  {'takeoff_from', 'movetowards', 'land_on'},
    'drone':      {'takeoff_from', 'movetowards', 'land_on'},
    'robot dog':  {'movetowards', 'open', 'close', 'grab', 'putinto', 'puton'},
    'robot_dog':  {'movetowards', 'open', 'close', 'grab', 'putinto', 'puton'},
    'robot arm':  {'open', 'close', 'grab', 'putinto', 'puton'},
    'robot_arm':  {'open', 'close', 'grab', 'putinto', 'puton'},
}


@dataclass
class ValidationResult:
    valid: bool
    failed_step_index: int = -1
    reason: str = ''
    injected_violation: str = ''
    failed_action: str = ''


@dataclass
class PlanValidator:
    """Tracks injected constraints from box D; each is a forbidden (agent_cls, verb, target_class) triple."""
    forbidden: List[Tuple[str, str, str]] = field(default_factory=list)

    def inject_forbidden(self, agent_cls: str, verb: str, target_cls: str):
        triple = (agent_cls, verb, target_cls)
        if triple not in self.forbidden:
            self.forbidden.append(triple)

    def validate(self, plan):
        for idx, step in enumerate(plan):
            agent_cls, _ = _parse_agent(step.get('agent', ''))
            verb, target_cls = _parse_action(step.get('action', ''))
            if not agent_cls or not verb:
                return ValidationResult(False, idx, 'malformed_step', failed_action=step.get('action', ''))
            allowed = CAPABILITY.get(agent_cls)
            if allowed is None:
                return ValidationResult(False, idx, f'unknown_robot_class:{agent_cls}',
                                        failed_action=step.get('action', ''))
            if verb not in allowed:
                return ValidationResult(False, idx, f'{agent_cls}_cannot_{verb}',
                                        failed_action=step.get('action', ''))
            if (agent_cls, verb, target_cls) in self.forbidden:
                return ValidationResult(False, idx, 'forbidden_by_learned_constraint',
                                        injected_violation=f'{agent_cls}/{verb}/{target_cls}',
                                        failed_action=step.get('action', ''))
        return ValidationResult(True)


def _parse_agent(s: str):
    m = re.match(r'<([^>]+)>\((\d+)\)', s.strip())
    if not m:
        return None, None
    return m.group(1).strip(), int(m.group(2))


def _parse_action(s: str):
    verb_m = re.search(r'\[(\w+)\]', s)
    obj_m = re.search(r'<([^>]+)>', s)
    return (verb_m.group(1) if verb_m else None,
            obj_m.group(1).strip() if obj_m else '')
