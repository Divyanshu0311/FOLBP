"""(3) Z3 SMT Reasoner — checks plan preconditions, returns SAT/UNSAT + UNSAT-core.

For each plan step we encode the preconditions as Z3 Bools tracked with named assertions.
If a step is reachable from the forward-simulated predicate state, the conjunction is SAT.
Otherwise Z3 returns the unsat_core() — the named precondition labels that failed, which
the Replanner (R) uses to insert a remediation step.
"""
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from typing import Optional

from z3 import And, Bool, Not, Solver, sat, unsat

from fol_generator import FOLGenerator, Predicates


PRECOND_RULES = {
    # robot dog — adds movetowards_reachable: same_room iff target is non-room, else open-door path.
    # ground_reachable on the object verbs too: `close` can only be established by a
    # [movetowards], and movetowards toward a HIGH_HEIGHT / ON_HIGH_SURFACE target is
    # never offered — so a dog acting on such a target is infeasible unless a CLOSE
    # edge already exists (the 3-ary form honors that).
    ('robot dog', 'grab'):       [('same_room', 'A', 'O1'), ('close', 'A', 'O1'), ('ground_reachable', 'A', 'O1'), ('grabable', 'O1'), ('not_holding', 'A')],
    ('robot_dog', 'grab'):       [('same_room', 'A', 'O1'), ('close', 'A', 'O1'), ('ground_reachable', 'A', 'O1'), ('grabable', 'O1'), ('not_holding', 'A')],
    ('robot dog', 'open'):       [('same_room', 'A', 'O1'), ('close', 'A', 'O1'), ('ground_reachable', 'A', 'O1'), ('state_closed', 'O1'), ('not_holding', 'A')],
    ('robot_dog', 'open'):       [('same_room', 'A', 'O1'), ('close', 'A', 'O1'), ('ground_reachable', 'A', 'O1'), ('state_closed', 'O1'), ('not_holding', 'A')],
    ('robot dog', 'close'):      [('same_room', 'A', 'O1'), ('close', 'A', 'O1'), ('ground_reachable', 'A', 'O1'), ('state_open', 'O1'), ('not_holding', 'A')],
    ('robot_dog', 'close'):      [('same_room', 'A', 'O1'), ('close', 'A', 'O1'), ('ground_reachable', 'A', 'O1'), ('state_open', 'O1'), ('not_holding', 'A')],
    ('robot dog', 'puton'):      [('hold', 'A', 'O1'), ('same_room', 'A', 'O2'), ('close', 'A', 'O2'), ('ground_reachable', 'A', 'O2'), ('surface', 'O2')],
    ('robot_dog', 'puton'):      [('hold', 'A', 'O1'), ('same_room', 'A', 'O2'), ('close', 'A', 'O2'), ('ground_reachable', 'A', 'O2'), ('surface', 'O2')],
    ('robot dog', 'putinto'):    [('hold', 'A', 'O1'), ('same_room', 'A', 'O2'), ('close', 'A', 'O2'), ('ground_reachable', 'A', 'O2'), ('container', 'O2'), ('state_open', 'O2')],
    ('robot_dog', 'putinto'):    [('hold', 'A', 'O1'), ('same_room', 'A', 'O2'), ('close', 'A', 'O2'), ('ground_reachable', 'A', 'O2'), ('container', 'O2'), ('state_open', 'O2')],
    # ground_reachable mirrors LLM_agent.unreached_objects: the environment never
    # offers a ground robot [movetowards] toward HIGH_HEIGHT / ON_HIGH_SURFACE nodes,
    # Floor nodes, or other agents — same audit as `above` on land_on (see below):
    # omitting it let Z3 certify approaches the executor then refused.
    ('robot dog', 'movetowards'):[('movetowards_reachable', 'A', 'O1'), ('ground_reachable', 'O1')],
    ('robot_dog', 'movetowards'):[('movetowards_reachable', 'A', 'O1'), ('ground_reachable', 'O1')],
    # robot arm — operates only on same-surface objects. same_surface (not same_room)
    # is what LLM.get_available_plans actually enforces: the arm is FIXED and its
    # vocabulary is built from on_same_surface_objects, so a target merely in the same
    # room (e.g. a basket flying past with the quadrotor) is not actionable.
    ('robot arm', 'grab'):       [('same_surface', 'A', 'O1'), ('grabable', 'O1'), ('not_holding', 'A')],
    ('robot_arm', 'grab'):       [('same_surface', 'A', 'O1'), ('grabable', 'O1'), ('not_holding', 'A')],
    ('robot arm', 'open'):       [('same_surface', 'A', 'O1'), ('container', 'O1'), ('state_closed', 'O1'), ('not_holding', 'A')],
    ('robot_arm', 'open'):       [('same_surface', 'A', 'O1'), ('container', 'O1'), ('state_closed', 'O1'), ('not_holding', 'A')],
    ('robot arm', 'close'):      [('same_surface', 'A', 'O1'), ('container', 'O1'), ('state_open', 'O1'), ('not_holding', 'A')],
    ('robot_arm', 'close'):      [('same_surface', 'A', 'O1'), ('container', 'O1'), ('state_open', 'O1'), ('not_holding', 'A')],
    ('robot arm', 'puton'):      [('hold', 'A', 'O1'), ('same_surface', 'A', 'O2'), ('surface', 'O2')],
    ('robot_arm', 'puton'):      [('hold', 'A', 'O1'), ('same_surface', 'A', 'O2'), ('surface', 'O2')],
    ('robot arm', 'putinto'):    [('hold', 'A', 'O1'), ('same_surface', 'A', 'O2'), ('container', 'O2'), ('state_open', 'O2')],
    ('robot_arm', 'putinto'):    [('hold', 'A', 'O1'), ('same_surface', 'A', 'O2'), ('container', 'O2'), ('state_open', 'O2')],
    # quadrotor — flies freely between rooms when FLYING; only land_on needs LANDABLE.
    # land_on additionally requires above(A, O1): the environment only offers
    # [land_on] X while the quadrotor holds an ABOVE edge to X (see
    # LLM.get_available_plans), so omitting it lets Z3 certify a landing from
    # across the apartment that the executor then refuses. `above` is produced by
    # fol_generator and maintained through forward simulation by takeoff_from /
    # movetowards / land_on, so this is a missing rule, not a missing predicate.
    ('quadrotor', 'takeoff_from'):[('on', 'A', 'O1'), ('state_land', 'A')],
    ('drone',     'takeoff_from'):[('on', 'A', 'O1'), ('state_land', 'A')],
    ('quadrotor', 'land_on'):     [('state_flying', 'A'), ('landable', 'O1'), ('above', 'A', 'O1')],
    ('drone',     'land_on'):     [('state_flying', 'A'), ('landable', 'O1'), ('above', 'A', 'O1')],
    # quad_movetarget: while FLYING the environment offers [movetowards] only toward
    # LANDABLE surfaces and open adjacent rooms (LLM.get_available_plans quadrotor
    # branch) — anything else is refused at dispatch.
    ('quadrotor', 'movetowards'): [('state_flying', 'A'), ('quad_movetarget', 'A', 'O1')],
    ('drone',     'movetowards'): [('state_flying', 'A'), ('quad_movetarget', 'A', 'O1')],
}


def same_surface_holds(binary, agent_id: int, target: int) -> bool:
    """True iff `target` is on the surface the (fixed) agent stands on.

    Mirrors the on_same_surfaces walk in LLM_agent._refresh_features: resolve the
    agent's `on` surface, then follow the target's carrier chain — with(carrier),
    hold(holder), then on/inside parents — until the surface, the agent itself, or a
    dead end. Bounded like the environment's own 3-iteration closure.

    Module-level and shared: the Z3 reasoner decides feasibility with it and the
    Replanner picks reassignment candidates with it. If those two ever disagreed, the
    repairer would propose steps the verifier rejects — the exact failure this whole
    guard family exists to prevent.
    """
    surface = next((b[2] for b in binary if b[0] == 'on' and b[1] == agent_id), None)
    if surface is None:
        return False
    cur, seen = target, set()
    for _ in range(8):
        if cur == surface or cur == agent_id:
            return True
        if cur in seen:
            return False
        seen.add(cur)
        carrier = next((b[1] for b in binary if b[0] == 'with' and b[2] == cur), None)
        if carrier is not None:
            cur = carrier
            continue
        holder = next((b[1] for b in binary if b[0] == 'hold' and b[2] == cur), None)
        if holder is not None:
            cur = holder
            continue
        parents = sorted(b[2] for b in binary if b[0] in ('inside', 'on') and b[1] == cur)
        if not parents:
            return False
        cur = parents[0]
    return False


@dataclass
class CheckResult:
    sat: bool
    failed_step_index: int = -1
    unsat_core_labels: List[str] = None
    missing_preconds: List[Tuple[str, ...]] = None
    state_at_failure: Optional[Predicates] = None  # forward-simulated state at the failing step

    def __post_init__(self):
        if self.unsat_core_labels is None:
            self.unsat_core_labels = []
        if self.missing_preconds is None:
            self.missing_preconds = []


class Z3SMTReasoner:
    """Forward-simulate the plan; for each step, encode preconditions as named Bools.
    Returns the index of the first step whose preconditions can't be satisfied and
    the UNSAT core of which atoms failed."""

    def __init__(self, fol: FOLGenerator):
        self.fol = fol

    def check(self, predicates: Predicates, plan: List[Dict[str, str]],
              goal: Optional[Dict] = None) -> CheckResult:
        state = self._snapshot(predicates)

        for idx, step in enumerate(plan):
            agent_cls, agent_id = self._parse_agent(step.get('agent', ''))
            verb, args = self._parse_action(step.get('action', ''))
            if agent_cls is None or verb is None:
                return CheckResult(sat=False, failed_step_index=idx,
                                   unsat_core_labels=[f'malformed_step_{idx}'],
                                   missing_preconds=[('malformed', step.get('action', ''))])

            rules = PRECOND_RULES.get((agent_cls, verb), [])
            if not rules:
                # no rule = unconstrained verb (we still require it to be in the vocabulary;
                # plan validator (4) catches vocabulary violations).
                self._apply_effects(state, agent_cls, agent_id, verb, args, predicates)
                continue

            binding = {'A': agent_id}
            if len(args) >= 1:
                binding['O1'] = args[0]
            if len(args) >= 2:
                binding['O2'] = args[1]

            solver = Solver()
            label_atoms: Dict[str, Tuple[str, ...]] = {}
            label_holds: Dict[str, bool] = {}
            label_vars = []
            for j, atom in enumerate(rules):
                label = f's{idx}_p{j}_{atom[0]}'
                ground = self._ground_atom(atom, binding)
                holds = self._holds(state, ground)
                var = Bool(f'{label}_var')
                solver.assert_and_track(var if holds else Not(var), label)
                label_atoms[label] = ground
                label_holds[label] = holds
                label_vars.append(var)
            solver.add(And(*label_vars))

            if solver.check() == unsat:
                # Report every false atom, not just the solver's minimal core: a
                # static-property miss (ground_reachable, same_surface, ...) must not
                # hide behind a co-failing repairable one, or the replanner burns its
                # budget on insertions that cannot make the step feasible. Static
                # atoms are ordered first so downstream consumers see the real cause.
                _static = ('ground_reachable', 'same_surface', 'quad_movetarget')
                core_labels = sorted(
                    (lbl for lbl, h in label_holds.items() if not h),
                    key=lambda lbl: (label_atoms[lbl][0] not in _static, lbl))
                missing = [label_atoms[lbl] for lbl in core_labels]
                snap = Predicates(nodes=predicates.nodes,
                                  unary=set(state['unary']),
                                  binary=set(state['binary']))
                return CheckResult(sat=False, failed_step_index=idx,
                                   unsat_core_labels=core_labels,
                                   missing_preconds=missing,
                                   state_at_failure=snap)
            self._apply_effects(state, agent_cls, agent_id, verb, args, predicates)

        # Goal entailment — preconditions alone certify only that every step CAN run,
        # not that the plan achieves anything. A plan that grounds the wrong object
        # (puton socks on bed 27 when the goal names bed 28) passes every precondition,
        # executes fully, and burns the step budget. Check the simulated final state
        # against the env's own goal atoms (rel_<cls>(from)_<cls>(to)) before dispatch.
        if goal and plan:
            unmet = []
            for goal_desc in goal:
                parts = goal_desc.split('_')
                if len(parts) < 3:
                    continue
                ids = re.findall(r'\((\d+)\)', goal_desc)
                if len(ids) < 2:
                    continue
                rel, from_id, to_id = parts[0], int(ids[0]), int(ids[1])
                if (rel, from_id, to_id) not in state['binary']:
                    unmet.append((goal_desc, from_id))
            if unmet:
                # Blame the last step that manipulates the goal object, else the last step.
                culprit = len(plan) - 1
                for gd, fid in unmet:
                    for i in range(len(plan) - 1, -1, -1):
                        if f'({fid})' in plan[i].get('action', ''):
                            culprit = i
                            break
                    else:
                        continue
                    break
                snap = Predicates(nodes=predicates.nodes,
                                  unary=set(state['unary']),
                                  binary=set(state['binary']))
                return CheckResult(sat=False, failed_step_index=culprit,
                                   unsat_core_labels=['goal_unreached'],
                                   missing_preconds=[('goal_unreached', gd) for gd, _ in unmet],
                                   state_at_failure=snap)

        return CheckResult(sat=True)

    def _snapshot(self, predicates: Predicates) -> Dict[str, set]:
        # 'nodes' is a read-only reference (never mutated by _apply_effects); it is
        # carried so _holds can consult category/class the way the environment's own
        # availability filter does.
        return {'unary': set(predicates.unary), 'binary': set(predicates.binary),
                'nodes': predicates.nodes}

    def _holds(self, state: Dict[str, set], atom: Tuple) -> bool:
        if atom[0] == 'not_holding':
            aid = atom[1]
            return not any(b for b in state['binary'] if b[0] == 'hold' and b[1] == aid)
        if atom[0] == 'same_room':
            return self._same_room(state, atom[1], atom[2])
        if atom[0] == 'movetowards_reachable':
            # Target is a room: reachability via open-door chain. Target is an object: must be same room.
            aid, target = atom[1], atom[2]
            if ('room', target) in state['unary']:
                return self._reachable_room(state, aid, target)
            return self._same_room(state, aid, target)
        if atom[0] == 'reachable_room':
            return self._reachable_room(state, atom[1], atom[2])
        if atom[0] == 'ground_reachable':
            # Mirrors the unreached_objects filter in LLM_agent._refresh_features:
            # rooms are fine (movetowards_reachable handles doors); everything the
            # environment strips from the enumeration is unreachable for a ground robot.
            # 3-ary form (agent, target): an existing CLOSE edge overrides the
            # property test — the robot is already there.
            if len(atom) == 3:
                if ('close', atom[1], atom[2]) in state['binary']:
                    return True
                atom = (atom[0], atom[2])
            target = atom[1]
            if ('room', target) in state['unary']:
                return True
            if (('high_height', target) in state['unary']
                    or ('on_high_surface', target) in state['unary']):
                return False
            category = (state.get('nodes') or {}).get(target, {}).get('category')
            return category not in ('Floor', 'Agents')
        if atom[0] == 'same_surface':
            return self._same_surface(state, atom[1], atom[2])
        if atom[0] == 'quad_movetarget':
            aid, target = atom[1], atom[2]
            if ('room', target) in state['unary']:
                return self._reachable_room(state, aid, target)
            return ('landable', target) in state['unary']
        if len(atom) == 2:
            return (atom[0], atom[1]) in state['unary']
        if len(atom) == 3:
            return (atom[0], atom[1], atom[2]) in state['binary']
        return False

    def _room_of(self, state: Dict[str, set], oid: int):
        """Resolve which room contains `oid` right now.

        At every step of the walk, priority order is:
          1. Node IS a room.
          2. Node is carried via `with(carrier, node)` — recurse on the carrier
             (so a basket WITH a quadrotor follows the quadrotor's current room,
             ignoring the basket's stale `on/inside` parent).
          3. Node is held via `hold(holder, node)` — recurse on the holder
             (so an object follows whoever grabbed it).
          4. Node is an agent with `at(node, room)` — return that room.
          5. Walk a single `inside`/`on` parent. Parents are sorted by id so the
             choice is deterministic when multiple exist.

        Re-applying (2) and (3) at every iteration is the critical difference from
        the previous version, which only checked them for the initial `oid` and
        therefore wrongly resolved e.g. "meat ball in basket on the kitchen table
        with the quadrotor in kitchen" via the basket's stale `on(livingroom floor)`
        edge.
        """
        seen: set = set()
        cur = oid
        for _ in range(20):
            if cur in seen:
                return None
            seen.add(cur)
            # 1. cur itself is a room.
            if ('room', cur) in state['unary']:
                return cur
            # 2. WITH carrier (basket WITH quadrotor, etc.).
            carrier = next((b[1] for b in state['binary']
                            if b[0] == 'with' and b[2] == cur), None)
            if carrier is not None:
                cur = carrier
                continue
            # 3. HOLD carrier (object follows holder).
            holder = next((b[1] for b in state['binary']
                           if b[0] == 'hold' and b[2] == cur), None)
            if holder is not None:
                cur = holder
                continue
            # 4. Agent at-room.
            at = next((b[2] for b in state['binary']
                       if b[0] == 'at' and b[1] == cur), None)
            if at is not None:
                cur = at
                continue
            # 5. Walk the on/inside chain, deterministically.
            parents = sorted(b[2] for b in state['binary']
                             if b[0] in ('inside', 'on') and b[1] == cur)
            if not parents:
                return None
            cur = parents[0]
        return None

    def _same_room(self, state: Dict[str, set], a: int, b: int) -> bool:
        ra = self._room_of(state, a)
        if ra is None:
            return False
        # Doors live on the boundary of two rooms (via leading_to); treat them as
        # "in" either connected room.
        door_rooms = [bb[2] for bb in state['binary']
                      if bb[0] == 'leading_to' and bb[1] == b]
        if door_rooms:
            return ra in door_rooms
        rb = self._room_of(state, b)
        return rb is not None and rb == ra

    def _same_surface(self, state: Dict[str, set], agent_id: int, target: int) -> bool:
        return same_surface_holds(state['binary'], agent_id, target)

    def _reachable_room(self, state: Dict[str, set], agent_id: int, target_room: int) -> bool:
        a_room = self._room_of(state, agent_id)
        if a_room is None:
            return False
        if a_room == target_room:
            return True
        # Find a door that connects a_room and target_room AND is open.
        for db in state['binary']:
            if db[0] == 'leading_to' and db[2] == target_room:
                door = db[1]
                if ('state_open', door) not in state['unary']:
                    continue
                for db2 in state['binary']:
                    if db2[0] == 'leading_to' and db2[1] == door and db2[2] == a_room:
                        return True
        return False

    def _ground_atom(self, atom: Tuple, binding: Dict[str, int]) -> Tuple:
        return tuple([atom[0]] + [binding.get(s, s) for s in atom[1:]])

    def _apply_effects(self, state, agent_cls, agent_id, verb, args, predicates):
        """Minimal forward model — only effects needed by downstream precondition checks."""
        u, b = state['unary'], state['binary']
        if verb == 'movetowards' and agent_cls in ('robot dog', 'robot_dog'):
            if args:
                target = args[0]
                # Clear stale close edges before reassigning.
                b -= {x for x in b if x[0] == 'close' and x[1] == agent_id}
                if ('room', target) in u:
                    # Move into the target room.
                    b -= {x for x in b if x[0] == 'at' and x[1] == agent_id}
                    b.add(('at', agent_id, target))
                else:
                    # Same-room object: become close + co-locate via target's room.
                    b.add(('close', agent_id, target))
                    target_room = self._room_of(state, target)
                    if target_room is not None:
                        b -= {x for x in b if x[0] == 'at' and x[1] == agent_id}
                        b.add(('at', agent_id, target_room))
        elif verb == 'movetowards' and agent_cls in ('quadrotor', 'drone'):
            if args:
                target = args[0]
                # Quadrotor's `above` edge gets reassigned; basket follows via the WITH
                # binding (the `_room_of` walker consults `with`, no need to mutate edges).
                b -= {x for x in b if x[0] == 'above' and x[1] == agent_id}
                if ('room', target) in u:
                    b -= {x for x in b if x[0] == 'at' and x[1] == agent_id}
                    b.add(('at', agent_id, target))
                    # get_env_info retargets the ABOVE edge to the room's floor on a
                    # room move, which is what makes an immediate [land_on] <floor>
                    # available. Without this the replanner inserts a movetowards
                    # toward that floor, which the environment then refuses as the
                    # current above-target.
                    nodes = state.get('nodes') or {}
                    floor = next((x[1] for x in b
                                  if x[0] == 'inside' and x[2] == target
                                  and nodes.get(x[1], {}).get('category') == 'Floor'), None)
                    if floor is not None:
                        b.add(('above', agent_id, floor))
                else:
                    b.add(('above', agent_id, target))
                    target_room = self._room_of(state, target)
                    if target_room is not None:
                        b -= {x for x in b if x[0] == 'at' and x[1] == agent_id}
                        b.add(('at', agent_id, target_room))
        elif verb == 'grab' and agent_cls in ('robot dog', 'robot_dog', 'robot arm', 'robot_arm'):
            if args:
                b -= {x for x in b if x[0] == 'close' and x[1] == agent_id and x[2] == args[0]}
                b -= {x for x in b if x[0] in ('on', 'inside') and x[1] == args[0]}
                b.add(('hold', agent_id, args[0]))
        elif verb == 'puton' and len(args) >= 2:
            b -= {x for x in b if x[0] == 'hold' and x[1] == agent_id and x[2] == args[0]}
            b.add(('on', args[0], args[1]))
            self._sync_height(state, args[0], args[1])
        elif verb == 'putinto' and len(args) >= 2:
            b -= {x for x in b if x[0] == 'hold' and x[1] == agent_id and x[2] == args[0]}
            b.add(('inside', args[0], args[1]))
            self._sync_height(state, args[0], args[1])
        elif verb == 'open' and args:
            u.discard(('state_closed', args[0]))
            u.add(('state_open', args[0]))
        elif verb == 'close' and args:
            u.discard(('state_open', args[0]))
            u.add(('state_closed', args[0]))
        elif verb == 'takeoff_from' and args:
            u.discard(('state_land', agent_id))
            u.add(('state_flying', agent_id))
            for x in list(b):
                if x[0] == 'on' and x[1] == agent_id and x[2] == args[0]:
                    b.discard(x)
                    b.add(('above', agent_id, args[0]))
        elif verb == 'land_on' and args:
            u.discard(('state_flying', agent_id))
            u.add(('state_land', agent_id))
            for x in list(b):
                if x[0] == 'above' and x[1] == agent_id:
                    b.discard(x)
            b.add(('on', agent_id, args[0]))
            # Whatever the quadrotor carries (basket, and its contents) inherits the
            # landing surface's height, exactly as get_env_info.step edits properties.
            for carried in [x[2] for x in b if x[0] == 'with' and x[1] == agent_id]:
                self._sync_height(state, carried, args[0])
                for inner in [x[1] for x in b if x[0] == 'inside' and x[2] == carried]:
                    self._sync_height(state, inner, args[0])
            target_room = self._room_of(state, args[0])
            if target_room is not None:
                b -= {x for x in b if x[0] == 'at' and x[1] == agent_id}
                b.add(('at', agent_id, target_room))

    def _sync_height(self, state, obj_id: int, dest_id: int):
        """Keep on_high_surface(obj) consistent with where the object just moved."""
        u = state['unary']
        if ('high_height', dest_id) in u or ('on_high_surface', dest_id) in u:
            u.add(('on_high_surface', obj_id))
        else:
            u.discard(('on_high_surface', obj_id))

    def _parse_agent(self, agent_str: str):
        m = re.match(r'<([^>]+)>\((\d+)\)', agent_str.strip())
        if not m:
            return None, None
        return m.group(1).strip(), int(m.group(2))

    def _parse_action(self, action_str: str):
        verb_m = re.search(r'\[(\w+)\]', action_str)
        if not verb_m:
            return None, []
        ids = [int(x) for x in re.findall(r'\((\d+)\)', action_str)]
        return verb_m.group(1), ids
