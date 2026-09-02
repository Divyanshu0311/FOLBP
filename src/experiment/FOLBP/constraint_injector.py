"""(D) Constraint Injection — pushes learned clauses into TWO sinks:

  1. The Oracle prompt — appended as natural-language constraints under the
     #INJECTED_CONSTRAINTS# placeholder, so the next full plan avoids them.
  2. The Plan Validator's forbidden list — so any plan that re-uses the
     learned pattern is rejected before reaching the Executor.

Hard vs soft handling:
  - CAPABILITY_VIOLATION: the verb is genuinely not in the agent's capability
    dict (e.g. robot arm cannot movetowards). Push a hard ban into the validator.
  - RUNTIME_REJECTION / PRECOND_MISSING / PARSE_FAILURE: the combination *itself*
    is fine — only the state at that moment was wrong (agent not in the right
    room, target hidden, agent already at target, missing prereq, ...). Emit a
    state-dependent natural-language hint to the Oracle (no hard ban) so it can
    re-plan with the precondition satisfied.

NOTE on RUNTIME_REJECTION specifically: runtime failures from the per-step PEFA
executor LLM are almost always context-dependent ("I can't see X from this room",
"I am already CLOSE to X"). A hard ban over-generalizes these into "this agent
can never [verb] this target_class", which deadlocks future plans on the same
goal. Keep them soft.
"""
from typing import Iterable, List

from cdcl_engine import LearnedClause
from plan_validator import PlanValidator


HARD_REASONS = {'capability_violation'}


PRECOND_HINT = {
    'not_holding': 'the agent must have empty hands beforehand (drop held items first)',
    'close':       'the agent must be CLOSE to the target — schedule [movetowards] first',
    'hold':        'the agent must already be holding the object — schedule [grab] first',
    'state_open':  'the target must be in OPEN state — schedule [open] first',
    'state_closed': 'the target must be in CLOSED state — schedule [close] first',
    'state_flying': 'the agent must be FLYING — schedule [takeoff_from] first',
    'state_land':  'the agent must be LAND(ed) — schedule [land_on] first',
    'reachable_room': 'the target room must be reachable — schedule [open] <door> before crossing',
    'same_room':   'the agent must be in the same room as the target — schedule [movetowards] <target_room> first (open any closed door on the path)',
    'movetowards_reachable': 'the target is not directly reachable — open any closed door, then movetowards the target_room before movetowards the object',
    # Static-property mismatches: no insertion can repair these. The Oracle must
    # re-route through a different target or a different agent.
    'container':   ('the target does NOT have the CONTAINERS property in this scene — '
                    'it cannot be [open]ed and you cannot [putinto] it. Pick a different '
                    'container (must have CONTAINERS), or use [puton] on a flat surface.'),
    'landable':    ('the target surface does NOT have the LANDABLE property — quadrotor '
                    'cannot [land_on] it. Pick a different surface tagged LANDABLE '
                    '(typically dining tables, kitchen tables, high tables, or floors).'),
    'grabable':    ('the target object does NOT have the GRABABLE property — robots '
                    'cannot [grab] it. Pick a different object, or check the agent '
                    'observations to find a similar object that is tagged GRABABLE.'),
    'ground_reachable': ('the target is out of a ground robot\'s reach — it is a HIGH '
                    'surface, sits ON a high surface, or is a floor/agent. A robot dog '
                    'can NEVER [movetowards], [grab], [puton] or [putinto] it, in any '
                    'state or order. To DELIVER an object to a high surface: the dog '
                    '[putinto]s it into the quadrotor\'s <basket> while the basket is '
                    'landed within reach, the quadrotor [land_on]s the high surface, '
                    'and the robot arm mounted there [grab]s it from the basket and '
                    'places it. To FETCH from a high surface, reverse the same ferry. '
                    'For floors, [movetowards] the room itself.'),
    'quad_movetarget': ('while FLYING the quadrotor can only [movetowards] LANDABLE '
                    'surfaces (tables, floors) or adjacent open rooms — never loose '
                    'objects. To interact with an object, [land_on] a LANDABLE surface '
                    'near it, or let a ground robot handle it.'),
    'same_surface': ('the robot arm is FIXED and reaches ONLY objects on its own '
                    'surface. FIRST check whether a DIFFERENT robot arm is already '
                    'mounted on the surface this step needs — scenes usually have one '
                    'arm per surface, and picking the right arm is the whole fix. If no '
                    'arm is in place, the object must be brought to the acting arm\'s '
                    'surface BEFORE it acts: the quadrotor [land_on]s that surface with '
                    'the basket, or a robot dog [puton]s the object there.'),
    'goal_unreached': ('this plan executes but does NOT accomplish the stated goal — '
                    'most often the wrong object id is targeted (two objects share a '
                    'class name, e.g. two beds). Re-read the goal and use EXACTLY the '
                    'ids named in it.'),
}


def _verb_specific_hint(c) -> str:
    """Verb-aware override for cases where the generic PRECOND_HINT misleads."""
    if c.failed_precond == 'not_holding' and c.verb in ('open', 'close'):
        # The classic open-door-while-holding ordering bug. Generic "drop held items
        # first" is *worse* advice than fixing the plan order: dropping means walking
        # back later. Tell the Oracle to fix the plan order instead.
        target_phrase = f'<{c.target_class}>' if c.target_class else 'the door/container'
        return (f'a {c.agent_class} cannot [{c.verb}] {target_phrase} while holding an object. '
                f'In the plan, schedule [{c.verb}] {target_phrase} BEFORE the [grab] of any '
                'object the agent will be carrying through it. Do NOT drop and re-grab — '
                'reorder grab to happen after the door/container is open.')
    if c.failed_precond in ('antipattern_open_while_holding', 'antipattern_closed_door_while_holding'):
        return (f'a {c.agent_class} is currently holding an object AND a door between the '
                f'agent and a <{c.target_class}> is closed. The agent cannot open a door '
                'with its hands full. Reorder the plan: [open] all doors on the path BEFORE '
                'the [grab] of the object you carry through. Do NOT drop and re-grab.')
    if c.failed_precond == 'antipattern_robot_arm_wrong_room':
        base = ('robot arms are FIXED — they cannot move between rooms. The chosen arm '
                'is in a different room from the target. Either pick the robot arm that '
                'is already in the target\'s room, or have the quadrotor land its basket '
                'on the chosen arm\'s surface so the target comes to the arm. Check the '
                'agent observations to confirm each arm\'s current room before reassigning.')
        if c.context:
            base += f' Context from validator: {c.context}'
        return base
    return ''


class ConstraintInjector:
    def inject(self, clauses: Iterable[LearnedClause], validator: PlanValidator) -> str:
        clauses = list(clauses)
        for c in clauses:
            if c.reason in HARD_REASONS and c.agent_class and c.verb and c.target_class:
                validator.inject_forbidden(c.agent_class, c.verb, c.target_class)
        return self._format_for_oracle(clauses)

    def _format_for_oracle(self, clauses: List[LearnedClause]) -> str:
        if not clauses:
            return ''
        hard = [c for c in clauses if c.reason in HARD_REASONS]
        # Verified structural defects get their own imperative section: the soft
        # header's "combination is allowed if the precondition is satisfied" is
        # actively wrong for them — the verifier proved the pattern cannot work as
        # planned, and an Oracle told the combination is allowed keeps producing it.
        STATIC_PRECONDS = ('ground_reachable', 'same_surface', 'quad_movetarget',
                           'goal_unreached')
        rejected = [c for c in clauses
                    if c.reason not in HARD_REASONS and c.failed_precond in STATIC_PRECONDS]
        soft = [c for c in clauses
                if c.reason not in HARD_REASONS and c.failed_precond not in STATIC_PRECONDS]
        lines: List[str] = []
        if hard:
            lines.append('Hard constraints (DO NOT use these patterns — they are not feasible in this scene):')
            for c in hard:
                tgt = f' on objects of class {c.target_class}' if c.target_class else ''
                lines.append(f'- avoid [{c.verb}] by <{c.agent_class}>{tgt} (cause: {c.reason})')
        if rejected:
            lines.append('VERIFIED PLAN DEFECTS — the verifier REJECTED your previous plan for these. '
                         'The step cannot work as you planned it; produce a DIFFERENT plan that follows '
                         'the stated route:')
            for c in rejected:
                if c.agent_class and c.verb and c.target_class:
                    prefix = f'[{c.verb}] by <{c.agent_class}> on a <{c.target_class}>'
                else:
                    prefix = f'[{c.verb}]' if c.verb else 'the failed step'
                hint = PRECOND_HINT.get(c.failed_precond, '')
                if c.context:
                    hint += f' [{c.context}]'
                lines.append(f'- {prefix}: {hint}.')
        if soft:
            lines.append('State-dependent hints from prior failures (combination is allowed if the precondition is satisfied):')
            for c in soft:
                if c.agent_class and c.verb and c.target_class:
                    prefix = f'when planning [{c.verb}] by <{c.agent_class}> on a <{c.target_class}>'
                else:
                    prefix = f'when planning [{c.verb}]' if c.verb else 'when planning the failed step'
                # 1. Verb-specific override (e.g. open-while-holding antipattern).
                hint = _verb_specific_hint(c)
                # 2. Otherwise the canned precondition message.
                if not hint:
                    if c.reason == 'runtime_rejection':
                        hint = ('the per-step executor refused this last time — most likely the agent '
                                'was not in the right room, the target was hidden, or the agent was '
                                'already at the target. Re-order so this step runs only after the '
                                'agent is verifiably in the same room as the target, or skip if the '
                                'precondition is already satisfied')
                    else:
                        hint = PRECOND_HINT.get(c.failed_precond, f'ensure the {c.failed_precond} precondition holds')
                lines.append(f'- {prefix}: {hint}.')
        return '\n'.join(lines)
