"""(5) Robot Executor — grounds and emits actions to the environment.

Two modes (chosen by --executor_mode):

  pefa_llm (default):
      1 LLM call per step per agent. The validated plan step is passed in as
      the "instruction"; the per-robot prompt asks the agent to pick the best
      available action from its current state-conditioned vocabulary.

  deterministic:
      No LLM. The validated plan step's action string is emitted verbatim,
      provided it is in the agent's currently-available action set.

Returns the grounded action string + a per-step info dict for the arena.

If the executor LLM refuses an action because the precondition is already
satisfied ("I am already CLOSE to X", "current state already meets the
requirements"), the result is flagged as `skip=True` rather than
`runtime_failure=True`. The arena treats skip as a successful no-op and
advances to the next plan step without learning a CDCL clause.
"""
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


# Phrases the PEFA per-robot prompt uses to signal "the precondition is already
# satisfied, so this action would be a no-op." Case-insensitive substring match.
_ALREADY_PHRASES = (
    'already close',
    'already at',
    'already in',
    'already on the',
    'already holding',
    'already landed',
    'already flying',
    'already open',
    'already closed',
    'current state already meets',
    'no further action is needed',
    'no other actions need to be performed',
    'no action is needed',
    'state already meets the requirements',
)


def _is_already_satisfied(message: str) -> bool:
    """Return True iff the executor LLM refused because the precondition is met."""
    if not message:
        return False
    low = message.lower()
    return any(phrase in low for phrase in _ALREADY_PHRASES)


def _normalize_action(action: str) -> str:
    """Canonical form for comparing a planned action against the enumerated set.

    Collapses whitespace and case so that a spacing artifact between the Oracle's
    JSON and the f-strings in get_available_plans() cannot masquerade as an
    infeasible action.
    """
    return re.sub(r'\s+', ' ', (action or '').strip()).casefold()


def _match_available(action: str, available):
    """Return the enumerated action string equal to `action`, or None.

    Matching returns the *enumerated* spelling, not the planner's, so whatever is
    emitted to the environment is byte-identical to what the agent advertised.
    """
    target = _normalize_action(action)
    if not target:
        return None
    for candidate in available:
        if _normalize_action(candidate) == target:
            return candidate
    return None


def _noop_reason(agent, action: str):
    """Symbolic 'this step is already satisfied' test for deterministic mode.

    In pefa_llm mode a redundant step is caught by phrase-matching the executor
    LLM's refusal. With no executor LLM there is no message to match, so the same
    judgement is made from the agent's own features — which is the honest way round
    anyway: a certified plan should not need a language model to notice a no-op.
    """
    verb_m = re.search(r'\[(\w+)\]', action or '')
    if not verb_m:
        return None
    verb = verb_m.group(1)
    ids = [int(x) for x in re.findall(r'\((\d+)\)', action or '')]
    if not ids:
        return None
    target_id = ids[0]
    id2node = getattr(agent, 'id2node', {}) or {}
    node = id2node.get(target_id)
    states = (node or {}).get('states', []) or []

    if verb == 'movetowards':
        if any(n['id'] == target_id for n in (agent.reachable_objects or [])):
            return f'already close to ({target_id})'
        room = getattr(agent, 'current_room', None)
        if room is not None and room.get('id') == target_id:
            return f'already in room ({target_id})'
        # Quadrotor: the surface it is currently ABOVE is offered as [land_on], never
        # as [movetowards] (get_available_plans removes it from the list), so a
        # movetowards toward it is a satisfied precondition, not a failure.
        surf = getattr(agent, 'landable_surfaces', None)
        if surf is not None and surf.get('id') == target_id:
            return f'already above ({target_id})'
    elif verb == 'open':
        if 'OPEN' in states or 'OPEN_FOREVER' in states:
            return f'({target_id}) already open'
    elif verb == 'close':
        if 'CLOSED' in states:
            return f'({target_id}) already closed'
    elif verb == 'grab':
        held = getattr(agent, 'grabbed_objects', None)
        if held is not None and held.get('id') == target_id:
            return f'already holding ({target_id})'
    elif verb == 'takeoff_from':
        if 'FLYING' in (agent.agent_node.get('states') or []):
            return 'already flying'
    elif verb == 'land_on':
        surf = getattr(agent, 'on_surfaces', None)
        if surf is not None and surf.get('id') == target_id:
            return f'already landed on ({target_id})'
    return None


@dataclass
class ExecutionResult:
    action: Optional[str]
    message: str
    info: Dict[str, Any]
    runtime_failure: bool = False
    skip: bool = False  # set when the action would be a no-op (precondition already met)


class RobotExecutor:
    def __init__(self, args, agents, get_text_obs):
        self.args = args
        self.agents = agents
        self.get_text_obs = get_text_obs  # callable(obs, agent_idx) -> str
        self.mode = args.executor_mode
        self.prompt_paths = {
            'quadrotor': args.quadrotor_prompt_path,
            'drone': args.quadrotor_prompt_path,
            'robot dog': args.robot_dog_prompt_path,
            'robot_dog': args.robot_dog_prompt_path,
            'robot arm': args.robot_arm_prompt_path,
            'robot_arm': args.robot_arm_prompt_path,
        }

    def execute_step(self, obs, step, env) -> ExecutionResult:
        target_agent, agent_idx = self._lookup_agent(step)
        if target_agent is None:
            return ExecutionResult(None, f'unknown_agent:{step.get("agent")}', {}, runtime_failure=True)

        class_name = target_agent.agent_node['class_name']
        real_id = target_agent.agent_node['id']
        per_agent_obs = obs[agent_idx]

        if self.mode == 'deterministic':
            # available_plans() also refreshes the agent's features, which _noop_reason reads.
            available = target_agent.available_plans(per_agent_obs)
            action = step.get('action', '')
            matched = _match_available(action, available)
            if matched is not None:
                return ExecutionResult(matched, f'deterministic emit: {matched}',
                                       {'available': available})
            noop = _noop_reason(target_agent, action)
            if noop:
                return ExecutionResult(None, f'precondition already satisfied: {noop}',
                                       {'available': available}, skip=True)
            return ExecutionResult(None, f'deterministic_action_not_available:{action}',
                                   {'available': available}, runtime_failure=True)

        # pefa_llm mode
        chat_agent_info = {
            'class_name': class_name,
            'id': real_id,
            'observation': self.get_text_obs(obs, agent_idx),
            'instruction': step.get('action', ''),
            'prompt_path': self.prompt_paths.get(class_name),
        }
        plan, message, info = target_agent.get_action(per_agent_obs, chat_agent_info, env.task_goal)
        if plan is None:
            # Distinguish "precondition already met" (skip) from genuine refusal (failure).
            if _is_already_satisfied(message):
                return ExecutionResult(action=None, message=message, info=info, skip=True)
            return ExecutionResult(action=None, message=message, info=info, runtime_failure=True)
        return ExecutionResult(action=plan, message=message, info=info)

    def _lookup_agent(self, step) -> Tuple[Optional[object], Optional[int]]:
        m = re.match(r'<([^>]+)>\((\d+)\)', step.get('agent', '').strip())
        if not m:
            return None, None
        target_id = int(m.group(2))
        for i, a in enumerate(self.agents):
            if a.agent_node['id'] == target_id:
                return a, i
        return None, None
