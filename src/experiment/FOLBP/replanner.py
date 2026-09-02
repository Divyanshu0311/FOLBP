"""(R) UNSAT-Core Replanner — inserts a prerequisite step before the failing step.

Takes the UNSAT core from box (3) Z3 and synthesizes the simplest remediation:
  - missing close(robot, obj)        -> insert [movetowards] <obj>(id) by the same robot
  - missing same_room(A, target)     -> insert [movetowards] <target_room>(rid) by the same robot
  - missing reachable_room(A, R)     -> insert [open] <door>(id) on the door leading to R
                                        (by a free-handed robot dog)
  - missing movetowards_reachable    -> dispatches to same_room or reachable_room above
  - missing state_open(door)         -> insert [open] <door>(id) by the closest free-handed robot dog
  - missing hold(robot, obj)         -> insert [grab] <obj>(id) by the same robot (after movetowards)
  - missing state_flying(quadrotor)  -> insert [takeoff_from] <current_surface>(id)
  - missing not_holding(robot)       -> insert [movetowards] <surface>(id) + [puton] <held>(id)
                                        on <surface>(id) by the same robot (drop held item before
                                        the failing step)

ANTIPATTERN: when not_holding fails on an [open] or [close] step AND the agent is currently
holding an object that the goal still cares about, the drop-and-re-grab repair is wasteful
and prone to executor drift. The replanner refuses to repair (returns `inserted=None` with
reason `antipattern:open_while_holding`), forcing the arena to escalate to a full Oracle
re-plan. The Constraint Injector then emits a targeted hint telling the Oracle to schedule
[open] BEFORE the [grab] that picks up the carry-through object.

The remaining suffix of the plan is preserved verbatim (per the diagram: "insert prereq; keep rest of plan").
"""
import copy
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from fol_generator import Predicates
from plan_validator import CAPABILITY
from z3_reasoner import CheckResult, same_surface_holds


# Verbs whose agent may be reassigned by the Replanner. [movetowards] is excluded on
# purpose: dropping or retargeting a motion step restructures the plan, which is the
# Oracle's job.
_MANIP_VERBS = ('grab', 'puton', 'putinto', 'open', 'close')


def _is_holder(state: Predicates, agent_id: int) -> bool:
    return any(b for b in state.binary if b[0] == 'hold' and b[1] == agent_id)


@dataclass
class RemediationResult:
    plan: List[Dict[str, str]]
    inserted: Optional[Dict[str, str]] = None      # primary insert (for the log)
    inserted_chain: List[Dict[str, str]] = None    # full sequence when one remediation needs >1 prereq
    reason: str = ''

    def __post_init__(self):
        if self.inserted_chain is None:
            self.inserted_chain = [self.inserted] if self.inserted else []


class UnsatCoreReplanner:
    def replan(self, plan: List[Dict[str, str]], result: CheckResult,
               predicates: Predicates, agents: List[Dict[str, Any]]) -> RemediationResult:
        new_plan = copy.deepcopy(plan)
        if result.failed_step_index < 0 or result.failed_step_index >= len(new_plan):
            return RemediationResult(new_plan, None, reason='no_failed_index')

        # Prefer the forward-simulated state at the failing step (so hold(...)
        # introduced earlier in the plan is visible to the remediation rules).
        # Fall back to the initial scene predicates if Z3 didn't ship a snapshot.
        state = result.state_at_failure if result.state_at_failure is not None else predicates

        failed = new_plan[result.failed_step_index]
        agent_cls, agent_id = self._parse_agent(failed.get('agent', ''))
        _, args = self._parse_action(failed.get('action', ''))

        # same_surface with a quadrotor-carried target IS symbolically repairable:
        # ferry the basket to the arm's surface (takeoff -> movetowards -> land_on)
        # before the arm's step — the same shape as the not_holding drop chain.
        ss = next((a for a in result.missing_preconds if a[0] == 'same_surface'), None)
        if ss is not None and len(ss) >= 3:
            chain = self._ferry_chain(state, arm_id=ss[1], target_id=ss[2])
            if chain:
                for j, step in enumerate(chain):
                    new_plan.insert(result.failed_step_index + j, step)
                return RemediationResult(
                    new_plan, chain[-1], inserted_chain=chain,
                    reason=(f'insert quadrotor ferry ({len(chain)} step(s)) to satisfy '
                            f'same_surface({ss[1]},{ss[2]})'),
                )

        # Agent reassignment. The step is infeasible for THIS agent, but another agent
        # is already within reach of everything the step touches — the scene commonly
        # mounts a second robot arm on exactly the surface the goal names. Rewriting the
        # assignment is a one-edit repair the Oracle otherwise has to rediscover by
        # resampling. Only manipulation verbs are reassigned: [movetowards] is a motion
        # step whose removal would change the plan's structure, which is the Oracle's
        # job, not the Replanner's. Z3 re-checks the whole plan afterwards, so a wrong
        # pick cannot yield an unsound plan — it just fails again inside the budget.
        reach_miss = next((a for a in result.missing_preconds
                           if a[0] in ('same_surface', 'ground_reachable')), None)
        failed_verb = self._verb_of(failed.get('action', ''))
        if (reach_miss is not None and failed_verb in _MANIP_VERBS
                and args and agent_id is not None):
            cand = self._reassign_candidate(state, agents, failed_verb, args, agent_id)
            if cand is not None:
                new_id, new_cls = cand
                new_plan[result.failed_step_index] = {
                    'agent': f'<{new_cls}>({new_id})',
                    'action': failed.get('action', ''),
                }
                return RemediationResult(
                    new_plan, new_plan[result.failed_step_index],
                    reason=(f'reassign [{failed_verb}] from agent {agent_id} to '
                            f'<{new_cls}>({new_id}), already on the same surface as {args}'),
                )

        # Static-property / goal misses: no insertion can repair these — the step is
        # infeasible for this agent (or the plan achieves the wrong goal) in every
        # ordering. Escalate immediately so the Oracle gets the targeted hint instead
        # of burning the attempt budget on movetowards insertions that cannot help.
        def _is_static(a):
            if a[0] in ('ground_reachable', 'same_surface', 'goal_unreached'):
                return True
            if a[0] == 'quad_movetarget':
                # A room target is repairable (open the door on the path); only a
                # non-LANDABLE object target is a structural defect.
                return not (len(a) >= 3 and state.nodes.get(a[2], {}).get('category') == 'Rooms')
            return False
        static_miss = next((a for a in result.missing_preconds if _is_static(a)), None)
        if static_miss is not None:
            return RemediationResult(new_plan, None,
                                     reason=f'static_unrepairable:{static_miss}')

        # Antipattern checks — escalate so the Oracle reorders rather than wasting
        # budget on drop-and-re-grab repairs.
        verb = self._verb_of(failed.get('action', ''))
        held_id = self._held_by(state, agent_id) if agent_id is not None else None
        # (a) Open-while-holding. Try to repair *in place* by lifting the earlier [grab]
        #     of the held object to AFTER the failing [open]/[close]. No new Oracle
        #     call, no budget burn, no drop-and-re-grab.
        if (verb in ('open', 'close')
                and held_id is not None
                and any(a[0] == 'not_holding' for a in result.missing_preconds)):
            reordered, moved = self._reorder_grab_after_open(
                new_plan, result.failed_step_index, agent_id, held_id,
            )
            if reordered is not None:
                return RemediationResult(
                    reordered, moved, inserted_chain=[moved],
                    reason=(f'reorder:open_while_holding moved [grab] <{self._class_of(state, held_id)}>'
                            f'({held_id}) to after [{verb}] step {result.failed_step_index}'),
                )
            # No earlier [grab] in this plan to lift — fall through to the escalate path.
            held_cls = self._class_of(state, held_id)
            target_cls = self._class_of(state, args[0]) if args else ''
            return RemediationResult(
                new_plan, None,
                reason=(f'antipattern:open_while_holding agent={agent_id} verb={verb} '
                        f'target=<{target_cls}>({args[0] if args else "?"}) '
                        f'held=<{held_cls}>({held_id}) (no liftable [grab] in plan)'),
            )
        # (c) Robot arm in the wrong room. Arms are FIXED — they cannot [movetowards].
        #     If a same_room/movetowards_reachable precondition fails for an arm, no
        #     insertion can repair it (a movetowards by the arm is itself a capability
        #     violation). Escalate to the Oracle with per-arm location info so it can
        #     pick the arm that's already in the target's room — or use a different
        #     agent (robot dog / quadrotor ferry).
        if (agent_cls in ('robot arm', 'robot_arm')
                and any(a[0] in ('same_room', 'movetowards_reachable')
                        for a in result.missing_preconds)):
            arm_locs = []
            for a in agents:
                node = a.get('agent_node') or {}
                if node.get('class_name') in ('robot arm', 'robot_arm'):
                    rid = self._room_of_node(state, node['id'])
                    rcls = self._class_of(state, rid) if rid is not None else '?'
                    arm_locs.append(f'<robot arm>({node["id"]}) is in <{rcls}>({rid})')
            target_id = next(a[2] for a in result.missing_preconds
                             if a[0] in ('same_room', 'movetowards_reachable'))
            tcls = self._class_of(state, target_id)
            troom = self._room_of_node(state, target_id)
            troom_cls = self._class_of(state, troom) if troom is not None else '?'
            return RemediationResult(
                new_plan, None,
                reason=(f'antipattern:robot_arm_wrong_room agent={agent_id} '
                        f'target=<{tcls}>({target_id}) in <{troom_cls}>({troom}); '
                        f'arm_locations=[{"; ".join(arm_locs)}]'),
            )
        # (b) Cross-room-while-holding-through-closed-door: same root cause. Agent is
        #     carrying something AND the only opener is busy holding it AND the path
        #     requires opening a closed door. Escalate.
        if (verb == 'movetowards'
                and held_id is not None
                and any(a[0] in ('same_room', 'movetowards_reachable', 'reachable_room')
                        for a in result.missing_preconds)):
            target_id = next(a[2] for a in result.missing_preconds
                             if a[0] in ('same_room', 'movetowards_reachable', 'reachable_room'))
            a_room = self._room_of_node(state, agent_id)
            if state.nodes.get(target_id, {}).get('category') == 'Rooms':
                target_room = target_id
            else:
                door_rooms = [b[2] for b in state.binary
                              if b[0] == 'leading_to' and b[1] == target_id]
                target_room = (next((r for r in door_rooms if r != a_room), door_rooms[0])
                               if door_rooms else self._room_of_node(state, target_id))
            if target_room is not None and a_room != target_room:
                _, next_door, door_open = self._next_room_on_path(state, a_room, target_room)
                if next_door is not None and not door_open:
                    # Is there any other free-handed dog who could open it for us? If not, escalate.
                    opener = self._free_handed_robot_dog(state, agents)
                    if opener is None:
                        held_cls = self._class_of(state, held_id)
                        return RemediationResult(
                            new_plan, None,
                            reason=(f'antipattern:closed_door_while_holding agent={agent_id} '
                                    f'verb={verb} door={next_door} '
                                    f'held=<{held_cls}>({held_id})'),
                        )

        for atom in result.missing_preconds:
            name = atom[0]
            insert = None
            reason = ''
            if name == 'close' and agent_id is not None and len(atom) >= 3:
                target_id = atom[2]
                target_cls = self._class_of(state, target_id)
                if target_cls:
                    insert = {
                        'agent': failed['agent'],
                        'action': f'[movetowards] <{target_cls}>({target_id})',
                    }
                    reason = f'insert movetowards to satisfy close({agent_id},{target_id})'
            elif name == 'state_open' and len(atom) >= 2:
                door_id = atom[1]
                door_cls = self._class_of(state, door_id)
                free_dog = self._free_handed_robot_dog(state, agents)
                if free_dog is not None and door_cls:
                    dog_cls = self._class_of(state, free_dog)
                    insert = {
                        'agent': f'<{dog_cls}>({free_dog})',
                        'action': f'[open] <{door_cls}>({door_id})',
                    }
                    reason = f'insert open({door_id}) by robot dog'
            elif name == 'hold' and agent_id is not None and len(atom) >= 3:
                obj_id = atom[2]
                obj_cls = self._class_of(state, obj_id)
                if obj_cls:
                    insert = {
                        'agent': failed['agent'],
                        'action': f'[grab] <{obj_cls}>({obj_id})',
                    }
                    reason = f'insert grab({obj_id}) to satisfy hold'
            elif name == 'above' and agent_id is not None and len(atom) >= 3:
                # The quadrotor must be ABOVE a surface before it may land on it.
                # [movetowards] <surface> is what establishes that edge.
                target_id = atom[2]
                target_cls = self._class_of(state, target_id)
                if target_cls:
                    insert = {
                        'agent': failed['agent'],
                        'action': f'[movetowards] <{target_cls}>({target_id})',
                    }
                    reason = f'insert movetowards to satisfy above({agent_id},{target_id})'
            elif name == 'on' and verb == 'takeoff_from' and agent_id is not None:
                actual = self._on_what(state, agent_id)
                if actual is not None and args and actual != args[0]:
                    actual_cls = self._class_of(state, actual)
                    if actual_cls:
                        new_plan[result.failed_step_index] = {
                            'agent': failed['agent'],
                            'action': f'[takeoff_from] <{actual_cls}>({actual})',
                        }
                        return RemediationResult(
                            new_plan, new_plan[result.failed_step_index],
                            reason=(f'rewrite takeoff_from target {args[0]} -> {actual} '
                                    f'(agent is on {actual} at this point in the plan)'),
                        )
            elif name == 'state_flying' and agent_id is not None:
                surface = self._on_what(state, agent_id)
                if surface:
                    surf_cls = self._class_of(state, surface)
                    insert = {
                        'agent': failed['agent'],
                        'action': f'[takeoff_from] <{surf_cls}>({surface})',
                    }
                    reason = f'insert takeoff_from({surface}) to satisfy state_flying'
            elif name == 'state_land' and agent_id is not None:
                # Need to land before takeoff_from — but if state_land was missing,
                # the quadrotor is already flying; let the outer loop full-replan.
                pass
            elif name in ('same_room', 'movetowards_reachable', 'reachable_room', 'quad_movetarget') and agent_id is not None and len(atom) >= 3:
                target_id = atom[2]
                a_room = self._room_of_node(state, agent_id)
                if state.nodes.get(target_id, {}).get('category') == 'Rooms':
                    target_room = target_id
                else:
                    # If target is a door, pick the door's "far side" room (the one the
                    # agent is NOT currently in). Else walk on/inside chain.
                    door_rooms = [b[2] for b in state.binary
                                  if b[0] == 'leading_to' and b[1] == target_id]
                    if door_rooms:
                        target_room = next((r for r in door_rooms if r != a_room), door_rooms[0])
                    else:
                        target_room = self._room_of_node(state, target_id)
                if target_room is None:
                    continue
                if a_room == target_room:
                    continue  # Already in target room; a later atom (close/...) will repair.
                # BFS over door-connected rooms to find the next hop on the shortest path.
                next_room, next_door, door_open = self._next_room_on_path(state, a_room, target_room)
                if next_room is None:
                    continue  # No door-connected path.
                if next_door is not None and not door_open:
                    door_cls = self._class_of(state, next_door)
                    opener = self._free_handed_robot_dog(state, agents)
                    if opener is not None:
                        opener_cls = self._class_of(state, opener)
                        insert = {
                            'agent': f'<{opener_cls}>({opener})',
                            'action': f'[open] <{door_cls}>({next_door})',
                        }
                        reason = f'insert open(door {next_door}) to traverse toward room {target_room}'
                else:
                    next_room_cls = self._class_of(state, next_room)
                    insert = {
                        'agent': failed['agent'],
                        'action': f'[movetowards] <{next_room_cls}>({next_room})',
                    }
                    reason = f'insert movetowards <{next_room_cls}>({next_room}) toward target room {target_room}'
            elif name == 'not_holding' and agent_id is not None:
                held_id = self._held_by(state, agent_id)
                surface_id = self._nearby_low_surface(state, agent_id, exclude=held_id)
                if held_id is not None and surface_id is not None:
                    held_cls = self._class_of(state, held_id)
                    surf_cls = self._class_of(state, surface_id)
                    chain = [
                        {'agent': failed['agent'],
                         'action': f'[movetowards] <{surf_cls}>({surface_id})'},
                        {'agent': failed['agent'],
                         'action': f'[puton] <{held_cls}>({held_id}) on <{surf_cls}>({surface_id})'},
                    ]
                    for j, step in enumerate(chain):
                        new_plan.insert(result.failed_step_index + j, step)
                    return RemediationResult(
                        new_plan, chain[-1], inserted_chain=chain,
                        reason=f'insert drop({held_id})-on({surface_id}) to satisfy not_holding({agent_id})',
                    )

            if insert is not None:
                new_plan.insert(result.failed_step_index, insert)
                return RemediationResult(new_plan, insert, reason=reason)

        return RemediationResult(new_plan, None,
                                 reason=f'no_remediation_for:{result.unsat_core_labels}')

    def _reassign_candidate(self, state, agents, verb: str, targets, exclude_id: int):
        """An agent other than `exclude_id` that could perform `verb` on `targets` now.

        Conservative on purpose. A candidate must (a) have the verb in the same
        capability dict the Plan Validator enforces, so a reassignment can never
        produce a step the validator would reject, and (b) be on the same surface as
        EVERY object the step touches — for [puton]/[putinto] that is both the carried
        object and the destination, since an agent that does not already hold the
        object must be able to [grab] it too. Ground agents additionally may not take
        over a high target. Returns (id, class) or None.
        """
        for a in agents:
            node = a.get('agent_node') or {}
            aid, cls = node.get('id'), node.get('class_name')
            if aid is None or cls is None or aid == exclude_id:
                continue
            if verb not in CAPABILITY.get(cls, ()):
                continue
            ground = cls in ('robot dog', 'robot_dog')
            ok = True
            for t in targets:
                if not same_surface_holds(state.binary, aid, t):
                    ok = False
                    break
                if ground and (state.has_unary('high_height', t)
                               or state.has_unary('on_high_surface', t)):
                    ok = False
                    break
            if ok:
                return aid, cls
        return None

    def _ferry_chain(self, state, arm_id: int, target_id: int):
        """Deterministic prelude that brings a quadrotor-carried container onto the
        (fixed) arm's surface: [takeoff_from] if landed, [movetowards] the surface,
        [land_on] it. None when the target is not riding a quadrotor, the arm has no
        surface, or the surface is not LANDABLE."""
        carrier = next((b[1] for b in state.binary
                        if b[0] in ('with', 'hold') and b[2] == target_id), None)
        if carrier is None:
            return None
        carrier_node = state.nodes.get(carrier, {})
        if carrier_node.get('class_name') not in ('quadrotor', 'drone'):
            return None
        surface = next((b[2] for b in state.binary
                        if b[0] == 'on' and b[1] == arm_id), None)
        if surface is None or not state.has_unary('landable', surface):
            return None
        carrier_cls = carrier_node.get('class_name')
        surface_cls = self._class_of(state, surface)
        chain = []
        if not state.has_unary('state_flying', carrier):
            on_now = self._on_what(state, carrier)
            if on_now is None:
                return None
            chain.append({'agent': f'<{carrier_cls}>({carrier})',
                          'action': f'[takeoff_from] <{self._class_of(state, on_now)}>({on_now})'})
        chain.append({'agent': f'<{carrier_cls}>({carrier})',
                      'action': f'[movetowards] <{surface_cls}>({surface})'})
        chain.append({'agent': f'<{carrier_cls}>({carrier})',
                      'action': f'[land_on] <{surface_cls}>({surface})'})
        return chain

    def _parse_agent(self, s: str) -> Tuple[Optional[str], Optional[int]]:
        m = re.match(r'<([^>]+)>\((\d+)\)', s.strip())
        return (m.group(1).strip(), int(m.group(2))) if m else (None, None)

    def _parse_action(self, s: str):
        verb = re.search(r'\[(\w+)\]', s)
        ids = [int(x) for x in re.findall(r'\((\d+)\)', s)]
        return (verb.group(1) if verb else None), ids

    def _class_of(self, predicates: Predicates, oid: int) -> str:
        node = predicates.nodes.get(oid)
        return node['class_name'] if node else ''

    def _free_handed_robot_dog(self, predicates: Predicates, agents: List[Dict[str, Any]]) -> Optional[int]:
        for a in agents:
            node = a.get('agent_node') or {}
            if node.get('class_name') in ('robot dog', 'robot_dog'):
                holding = any(b for b in predicates.binary
                              if b[0] == 'hold' and b[1] == node['id'])
                if not holding:
                    return node['id']
        return None

    def _on_what(self, predicates: Predicates, agent_id: int) -> Optional[int]:
        for b in predicates.binary:
            if b[0] == 'on' and b[1] == agent_id:
                return b[2]
        return None

    def _held_by(self, predicates: Predicates, agent_id: int) -> Optional[int]:
        for b in predicates.binary:
            if b[0] == 'hold' and b[1] == agent_id:
                return b[2]
        return None

    def _reorder_grab_after_open(self, plan: List[Dict[str, str]], failed_idx: int,
                                 agent_id: int, held_id: int):
        """Lift the most recent [grab] <held_id>(id) by agent_id from earlier in the
        plan to immediately after the failing [open]/[close] step. Returns
        (new_plan, moved_step) on success, (None, None) if no liftable grab is found."""
        grab_idx = None
        for j in range(failed_idx - 1, -1, -1):
            step = plan[j]
            sa, said = self._parse_agent(step.get('agent', ''))
            verb, sargs = self._parse_action(step.get('action', ''))
            if (said == agent_id and verb == 'grab' and sargs and sargs[0] == held_id):
                grab_idx = j
                break
        if grab_idx is None:
            return None, None
        new_plan = copy.deepcopy(plan)
        moved = new_plan.pop(grab_idx)
        # After pop the failing step is at failed_idx - 1; insert grab at failed_idx
        # so it lands right after the now-shifted [open]/[close].
        new_plan.insert(failed_idx, moved)
        return new_plan, moved

    def _verb_of(self, action: str) -> str:
        m = re.search(r'\[(\w+)\]', action)
        return m.group(1) if m else ''

    def _room_of_node(self, state: Predicates, oid: int) -> Optional[int]:
        """Resolve `oid`'s current room from the predicate snapshot.
        Mirrors `z3_reasoner._room_of`: at EVERY step prefer carriers (`with`,
        `hold`) and `at` before falling back to the `inside`/`on` chain — so a
        basket WITH a moved quadrotor follows the quadrotor's current room
        instead of its stale physical-anchor edge."""
        seen: set = set()
        cur = oid
        for _ in range(20):
            if cur in seen:
                return None
            seen.add(cur)
            if state.nodes.get(cur, {}).get('category') == 'Rooms':
                return cur
            carrier = next((b[1] for b in state.binary
                            if b[0] == 'with' and b[2] == cur), None)
            if carrier is not None:
                cur = carrier
                continue
            holder = next((b[1] for b in state.binary
                           if b[0] == 'hold' and b[2] == cur), None)
            if holder is not None:
                cur = holder
                continue
            at = next((b[2] for b in state.binary
                       if b[0] == 'at' and b[1] == cur), None)
            if at is not None:
                cur = at
                continue
            parents = sorted(b[2] for b in state.binary
                             if b[0] in ('inside', 'on') and b[1] == cur)
            if not parents:
                return None
            cur = parents[0]
        return None

    def _next_room_on_path(self, state: Predicates, a_room: Optional[int],
                           target_room: Optional[int]):
        """BFS over door-connected rooms from a_room to target_room.
        Returns (next_room, door_connecting_a_room_to_next_room, door_is_open).
        Doors are considered passable whether or not they are open; if closed,
        the caller is expected to insert an [open] step before the movetowards."""
        if a_room is None or target_room is None or a_room == target_room:
            return None, None, False
        # Build the door-room adjacency: each door has two `leading_to` edges, one per side.
        door_rooms: Dict[int, list] = {}
        for b in state.binary:
            if b[0] == 'leading_to':
                door_rooms.setdefault(b[1], []).append(b[2])
        adj: Dict[int, list] = {}
        for door, rooms in door_rooms.items():
            for r1 in rooms:
                for r2 in rooms:
                    if r1 != r2:
                        adj.setdefault(r1, []).append((r2, door))
        # BFS, recording the first-hop (room, door) so we can return it.
        from collections import deque
        q = deque([(a_room, None, None)])  # (current, first_hop_room, first_hop_door)
        seen = {a_room}
        while q:
            cur, first_room, first_door = q.popleft()
            for nxt, door in adj.get(cur, []):
                if nxt in seen:
                    continue
                fh_room = first_room if first_room is not None else nxt
                fh_door = first_door if first_door is not None else door
                if nxt == target_room:
                    return fh_room, fh_door, state.has_unary('state_open', fh_door)
                seen.add(nxt)
                q.append((nxt, fh_room, fh_door))
        return None, None, False

    def _nearby_low_surface(self, predicates: Predicates, agent_id: int,
                            exclude: Optional[int] = None) -> Optional[int]:
        """Pick a low-height non-floor SURFACES node to drop a held item on.
        Prefer surfaces the agent is already CLOSE to; else surfaces in the same room
        (transitively, via on(surface, floor) or inside(surface, room)); else any
        suitable surface in the scene."""
        def acceptable(sid: int) -> bool:
            if sid == exclude or sid == agent_id:
                return False
            if not predicates.has_unary('surface', sid):
                return False
            if predicates.has_unary('high_height', sid):
                return False
            node = predicates.nodes.get(sid, {})
            if node.get('category') == 'Floor':
                return False
            return True

        room_id = next((b[2] for b in predicates.binary
                        if b[0] == 'at' and b[1] == agent_id), None)

        # Build a reachable-room-set: room_id itself + every node directly
        # inside that room (floor, etc.) — surfaces sit ON those.
        room_set = set()
        if room_id is not None:
            room_set.add(room_id)
            for b in predicates.binary:
                if b[0] == 'inside' and b[2] == room_id:
                    room_set.add(b[1])

        close_pool, room_pool, global_pool = [], [], []
        for sid in predicates.nodes:
            if not acceptable(sid):
                continue
            global_pool.append(sid)
            in_room = any(
                b for b in predicates.binary
                if b[1] == sid and b[0] in ('inside', 'on') and b[2] in room_set
            )
            if in_room:
                room_pool.append(sid)
            if any(b for b in predicates.binary
                   if b[0] == 'close' and b[1] == agent_id and b[2] == sid):
                close_pool.append(sid)

        # Prefer LOW_HEIGHT, then anything that fits, in order: close -> same room -> global.
        for pool in (close_pool, room_pool, global_pool):
            for sid in pool:
                if predicates.has_unary('low_height', sid):
                    return sid
            if pool:
                return pool[0]
        return None
