"""ArenaMP — wires the FOLBP architecture exactly as drawn.

Forward (badges 1-6):
    obs -> Oracle.generate_plan -> FOL.generate -> Z3.check -> PlanValidator.validate
    -> Executor.execute_step -> env.step -> next obs.

Repair (R):
    Z3 UNSAT -> Replanner.replan (insert prereq) -> re-validate.
    If still UNSAT after `--max_replan_attempts` tries, escalate to a full Oracle re-plan.

Learning (A-D):
    Each emitted action becomes a TaskNode. On any failure (Z3 UNSAT after replan
    cap, PlanValidator reject, runtime executor failure) -> ConflictAnalyzer.analyze
    -> CDCLEngine.learn -> ConstraintInjector.inject — feeding the next Oracle call
    and the PlanValidator's forbidden list.
"""
import copy
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List

import backoff
from google import genai
from google.genai import types

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bench.eventlog import EVENTS
from bench.llm_meter import METER

from cdcl_engine import CDCLEngine
from conflict_analyzer import ConflictAnalyzer, FailureClass
from constraint_injector import ConstraintInjector
from executor import RobotExecutor
from fol_generator import FOLGenerator
from oracle import OracleLLM
from plan_validator import PlanValidator
from replanner import UnsatCoreReplanner
from task_graph import TaskGraph
from z3_reasoner import Z3SMTReasoner


class ArenaMP:
    def __init__(self, env_fn, agents, args):
        self.env_fn = env_fn
        self.agents = agents
        self.args = args
        self.num_agents = len(agents)
        self.debug = args.debug
        self.env = env_fn()
        self.record_dir = f'./log/{args.env}.txt'

        # Execution bridge. 'standalone' (the benchmark code path) runs the symbolic
        # planner alone; 'ws' forwards every executed action to OmniGibson and only
        # advances the scene graph once the simulator confirms it. Same wire format
        # PEFA uses -- {"agent", "action"} out, {"success", "info"} back.
        self.mode = getattr(args, 'mode', 'standalone')
        self.ws = None
        self.name_mapping = {}
        if self.mode == 'ws':
            ws_url = getattr(args, 'ws_url', None)
            if not ws_url:
                raise ValueError('--mode=ws requires --ws_url (e.g. ws://127.0.0.1:8765)')
            import websocket
            self.ws = websocket.create_connection(ws_url)
            print(f'[mode=ws] Connected to OmniGibson at {ws_url}', flush=True)
            with open(f'./env/{args.env}.json') as f:
                env_data = json.load(f)
            task_ids = getattr(args, 'task', [0])
            self.name_mapping = env_data[task_ids[0]].get('name_mapping', {})

        self.client = self._build_gemini_client()
        self.sampling_params = {
            'max_output_tokens': args.max_tokens,
            'temperature': args.t,
            'candidate_count': args.n,
        }

        @backoff.on_exception(backoff.expo, Exception, max_tries=4)
        def _generate(prompt_messages, sp, role='oracle'):
            text = '\n'.join(m['content'] for m in prompt_messages if m.get('role') in ('user', 'system'))
            with METER.timed(role) as call:
                response = self.client.models.generate_content(
                    model=args.lm_id,
                    contents=text,
                    config=types.GenerateContentConfig(
                        temperature=sp.get('temperature', 0),
                        candidate_count=sp.get('candidate_count', 1),
                    ),
                )
                call.observe(response)
            return [response.text or ''], 0.0

        self.generator = _generate

        # FOLBP modules — one per diagram box.
        self.oracle = OracleLLM(args, self.generator, self.sampling_params)
        self.fol = FOLGenerator()
        self.z3 = Z3SMTReasoner(self.fol)
        self.validator = PlanValidator()
        self.replanner = UnsatCoreReplanner()
        self.executor = RobotExecutor(args, agents, self._agent_obs2text)

        # Ablation axes (see args.py). The arms of the benchmark are combinations of
        # these three switches plus max_oracle_plans — never forks of this file.
        self.verify_mode = getattr(args, 'verify', 'z3')
        self.repair_mode = getattr(args, 'repair', 'symbolic')
        self.max_oracle_plans = getattr(args, 'max_oracle_plans', 0) or 0

        # Instrumentation for the per-task record.
        self.stats = {
            'z3_checks': 0,
            'z3_unsat_events': 0,
            'shadow_unsat_events': 0,
            'unsat_cores': [],
            'repair_attempts': 0,
            'repair_successes': 0,
            'escalations': 0,
            'oracle_plans': 0,
            'outer_iters': 0,
            'failures': [],
            # Deadlock instrumentation. `forbidden_rejections` is the one that
            # distinguishes a hard-constraint deadlock from any other stall: it counts
            # plans the validator threw out because of a ban this run itself learned.
            'validator_rejections': 0,
            'forbidden_rejections': 0,
            'hard_bans': 0,
            'forbidden_triples': [],
            'termination_reason': '',
            'deadlocked': False,
        }

        # CDCL learning layer (A-D) — persists across replans and tasks.
        self.cdcl_enabled = getattr(args, 'enable_cdcl', True)
        self.task_graph = TaskGraph()
        self.conflict_analyzer = ConflictAnalyzer()
        self.cdcl = CDCLEngine()
        self.injector = ConstraintInjector()
        self.cdcl.bind_scene_classes({n['id']: n['class_name'] for n in self.env.graph['nodes']})
        self.injected_text = ''

        self.task_goal = None
        # Set by _record_failure_and_inject; consumed and cleared by the next Oracle call.
        self.last_rejected = None

        self._log('INIT', f'FOLBP arena ready: agents={self.num_agents} env={args.env} '
                          f'executor_mode={args.executor_mode} max_replan_attempts={args.max_replan_attempts} '
                          f'cdcl_enabled={self.cdcl_enabled} lm_id={args.lm_id}')
        self._log('INIT', f'arm={getattr(args, "arm", "folbp")} seed={getattr(args, "seed", 0)} '
                          f'verify={self.verify_mode} repair={self.repair_mode} '
                          f'max_oracle_plans={self.max_oracle_plans or "unlimited"} t={args.t}')

    def _build_gemini_client(self):
        api_key = self.args.api_key or os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY')
        return genai.Client(api_key=api_key) if api_key else genai.Client()

    def write_log(self, msg: str):
        with open(self.record_dir, 'a') as f:
            f.write(str(msg) + '\n')

    def _log(self, tag: str, msg: str):
        """Print to stdout AND persist to the per-env log file with a [FOLBP][TAG] prefix.

        Also the single funnel every pipeline stage already goes through, so one
        emit here feeds the demo-video compositor the whole run for free. Banner
        rules carry no information, so they are not emitted.
        """
        line = f'[FOLBP][{tag}] {msg}'
        print(line, flush=True)
        self.write_log(line)
        if msg and not set(str(msg).strip()) <= {'='}:
            EVENTS.emit(tag, body=msg)

    def _banner(self, tag: str, title: str):
        bar = '=' * 60
        self._log(tag, bar)
        self._log(tag, title)
        self._log(tag, bar)

    # ------------------------------------------------------------------ obs

    def _agent_obs2text(self, obs, agent_id):
        observation = obs[agent_id]
        id2node = {n['id']: n for n in observation['nodes']}
        agent_class = id2node[int(self.env.id_name_dict[agent_id][1])]['class_name']
        with_quad = None
        text = ''
        for node in observation['nodes']:
            if node['category'] == 'Agents' and self.env.id_name_dict[agent_id][1] == node['id']:
                text += f"I am <{node['class_name']}>({node['id']}). "
                if node.get('states'):
                    text += 'Now my state is: ' + ', '.join(node['states']) + '. '
                for e in observation.get('edges', []):
                    if e['from_id'] == node['id']:
                        text += f"I am {e['relation_type']} the <{id2node[e['to_id']]['class_name']}>({e['to_id']}). "
                    if e['relation_type'] == 'WITH':
                        with_quad = e['to_id']
                text += '\n'
        for node in observation['nodes']:
            if node['category'] == 'Rooms' and node['id'] == observation.get('agent_in_room_id'):
                text += f"Now I am in the <{node['class_name']}>({node['id']}). In this room, I can see:\n"
        for node in observation['nodes']:
            if node['id'] != self.env.id_name_dict[agent_id][1] and node['category'] != 'Rooms':
                text += f"<{node['class_name']}>({node['id']}). "
                if node.get('properties'):
                    text += 'Its properties are: ' + ', '.join(node['properties']) + '. '
                if node.get('states'):
                    text += 'Now its state is: ' + ', '.join(node['states']) + '.\n'
                else:
                    text += '\n'
        text += 'These objects have a certain position relationship with each other:\n'
        for node in observation['nodes']:
            if node['id'] != self.env.id_name_dict[agent_id][1] and node['category'] != 'Rooms':
                for e in observation.get('edges', []):
                    if e['from_id'] == node['id']:
                        text += (f"The <{node['class_name']}>({node['id']}) is "
                                 f"{e['relation_type']} the <{id2node[e['to_id']]['class_name']}>({e['to_id']}).\n")
        for e in observation.get('edges', []):
            if e['relation_type'] == 'HOLD' and agent_class not in ('quadrotor', 'drone'):
                text += f"I am holding a <{id2node[e['to_id']]['class_name']}>({e['to_id']}) in my hand.\n"
        return text

    def _multi_agent_obs_text(self, obs) -> str:
        return '\n'.join(self._agent_obs2text(obs, i) for i in range(self.num_agents))

    # ----------------------------------------------------------- main loop

    def run(self):
        self.task_goal = copy.deepcopy(self.env.task_goal)
        success = False
        steps = 0
        saved_info: List[Dict[str, Any]] = []
        max_steps = 2 * self.env.ground_truth_step_num
        outer_iters = 0
        no_progress_iters = 0
        no_progress_cap = 3  # bail out if 3 Oracle re-plans in a row execute zero steps
        # Why the loop stopped. Set at every exit; the bridge-timeout path sets it early
        # because it does not break directly -- it forces no_progress_iters to the cap and
        # lets the next iteration exit, which would otherwise misreport as 'no_progress'.
        termination_reason = ''

        self._banner('TASK', f'task_id={self.env.task_id} env_id={self.env.env_id} '
                             f'name={self.env.task_name} gt_steps={self.env.ground_truth_step_num}')
        self._log('TASK', f'goal: {self.env.goal_instruction}')
        EVENTS.emit('TASK', title=self.env.task_name,
                    body=self.env.goal_instruction,
                    framework='folbp', task_id=self.env.task_id,
                    env_id=self.env.env_id, gt_steps=self.env.ground_truth_step_num,
                    mode=self.mode)
        self._log('TASK', f'agents: {[(a.agent_node["class_name"], a.agent_node["id"]) for a in self.agents]}')

        while True:
            outer_iters += 1
            self._banner('ITER', f'outer_iter={outer_iters} env_steps={self.env.steps}/{max_steps}')
            if self.env.steps > max_steps or outer_iters > max_steps * 2:
                self._log('ITER', f'Step budget exceeded ({self.env.steps} > {max_steps}); failing task.')
                termination_reason = termination_reason or 'step_budget'
                break
            if no_progress_iters >= no_progress_cap:
                self._log('ITER', f'No progress in {no_progress_cap} consecutive Oracle re-plans; failing task.')
                termination_reason = termination_reason or 'no_progress'
                break

            obs = self.env.get_observations()
            obs_text = self._multi_agent_obs_text(obs)

            # (1) Oracle LLM — full plan in 1 call.
            if self.max_oracle_plans and self.stats['oracle_plans'] >= self.max_oracle_plans:
                self._log('ITER', f'oracle plan cap reached ({self.max_oracle_plans}); '
                                  're-planning not permitted — failing task.')
                termination_reason = termination_reason or 'oracle_plan_cap'
                break
            self._log('ORACLE', f'requesting full plan (injected_constraints_chars={len(self.injected_text)})')
            last_attempt = self._render_last_attempt()
            self.last_rejected = None   # consumed; a stale block would misdirect later replans
            if last_attempt:
                self._log('ORACLE', 'replanning with the rejected plan shown to the Oracle')
            plan = self.oracle.generate_plan(
                obs_text, self.env.goal_instruction, self.env.num_agent, self.injected_text,
                last_attempt,
            )
            self.stats['oracle_plans'] += 1
            self._log('ORACLE', f'received plan with {len(plan)} step(s)')
            for i, s in enumerate(plan):
                self._log('ORACLE', f'  plan[{i}]: agent={s.get("agent")} action={s.get("action")}')
            # Structured emit so the compositor can render the plan as one block
            # rather than reassembling it from the per-step log lines above. The
            # whole plan goes on screen at once, uncondensed -- that one-call plan
            # is the thing that distinguishes FOLBP from PEFA's step-at-a-time loop.
            EVENTS.emit('ORACLE_PLAN',
                        title=f'Oracle plan — {len(plan)} step(s)',
                        plan=[{'agent': st.get('agent'), 'action': st.get('action')}
                              for st in plan],
                        oracle_plans=self.stats['oracle_plans'],
                        injected_chars=len(self.injected_text))
            if not plan:
                self._log('ORACLE', 'Empty plan; escalating to next outer iteration.')
                no_progress_iters += 1
                if self._cdcl_full_replan_failure(reason='empty_plan_from_oracle'):
                    continue
                termination_reason = termination_reason or 'empty_plan'
                break

            # (2) FOL Generator — typed predicates from current scene.
            predicates = self.fol.generate(self.env.graph)
            self._log('FOL', f'predicates: unary={len(predicates.unary)} binary={len(predicates.binary)} '
                             f'nodes={len(predicates.nodes)}')

            # (3) + (R) Z3 SMT Reasoner — what happens on UNSAT is the ablation axis.
            if self.verify_mode == 'none':
                self._log('Z3', 'verification disabled (--verify none); dispatching plan unchecked')
                result = None
            else:
                plan, result, rem = self._verify_and_repair(plan, predicates, obs_text)

            if result is not None and not result.sat:
                if self.verify_mode == 'shadow':
                    # Detect, log, and deliberately do NOT act. This counter is the
                    # soundness number: how often an unverified planner would have
                    # dispatched a physically infeasible action to a robot.
                    self.stats['shadow_unsat_events'] += 1
                    self.stats['failures'].append({
                        'kind': 'shadow_unsat',
                        'step_index': result.failed_step_index,
                        'detail': str(result.unsat_core_labels)[:300],
                    })
                    self._log('SHADOW', f'UNSAT @ step {result.failed_step_index} '
                                        f'core={result.unsat_core_labels} — executing anyway '
                                        '(--verify shadow)')
                else:
                    learned = self._escalate_unsat(plan, result, rem)
                    # A rejection that learned a NEW clause is progress — the next
                    # Oracle call plans against strictly more knowledge. Only a
                    # re-plan that yields neither env steps nor new clauses (the
                    # Oracle resubmitting a known-bad pattern; dedup learns nothing)
                    # counts toward the stall cap, which the outer_iters bound still
                    # limits absolutely.
                    no_progress_iters = 0 if learned else no_progress_iters + 1
                    continue  # full re-plan

            # (4) Plan Validator — capability check + injected forbiddens.
            v = self.validator.validate(plan)
            if not v.valid:
                self._log('VAL', f'rejected step {v.failed_step_index}: reason={v.reason} '
                                 f'action={v.failed_action} '
                                 f'injected_violation={v.injected_violation or "-"}')
                self.stats['validator_rejections'] += 1
                if v.reason == 'forbidden_by_learned_constraint':
                    # Rejected by a ban this run learned, not by the static capability
                    # dict. Repeated occurrences with no step executed in between are
                    # exactly the deadlock signature.
                    self.stats['forbidden_rejections'] += 1
                    self.stats['failures'].append({
                        'kind': 'forbidden_by_learned_constraint',
                        'step_index': v.failed_step_index,
                        'detail': f'{v.injected_violation} on {v.failed_action}'[:300],
                    })
                self._record_failure_and_inject(plan, v.failed_step_index,
                                                f'capability:{v.reason}')
                no_progress_iters += 1
                continue  # full re-plan
            self._log('VAL', f'accepted — all {len(plan)} step(s) within capability + injected constraints')

            # (5) Robot Executor — ground each step and (6) step the environment.
            self._log('EXEC', f'mode={self.args.executor_mode}; executing {len(plan)} step(s)')
            runtime_failed = False
            executed_any = False
            for i, step in enumerate(plan):
                self._log('EXEC', f'step[{i}] {step["agent"]} -> {step["action"]}')
                exec_res = self.executor.execute_step(obs, step, self.env)
                if exec_res.skip:
                    self._log('EXEC', f'step[{i}] noop (precondition already satisfied); '
                                      f'message: {exec_res.message[:140]}')
                    # Count this as progress so no_progress_iters resets — the world is in
                    # the expected post-state already.
                    executed_any = True
                    no_progress_iters = 0
                    continue
                if exec_res.runtime_failure or exec_res.action is None:
                    self._log('EXEC', f'runtime failure on step[{i}]: {exec_res.message[:160]}')
                    self._record_failure_and_inject(plan, i, f'runtime:{exec_res.message[:80]}')
                    runtime_failed = True
                    break

                agent_id, verb, targets = self._extract_step_ids(step)
                class_name = self._lookup_agent_class(agent_id)

                # Dispatch to the simulator BEFORE the symbolic graph moves, so the
                # scene graph never claims a transition the robot did not make. In
                # standalone mode this is a no-op that always succeeds. A physical
                # rejection takes the same route as an executor runtime failure --
                # into the CDCL layer, where it becomes a learned clause.
                ok, bridge_info, timed_out = self._dispatch_with_retry(
                    class_name, agent_id, exec_res.action)
                if not ok:
                    self._log('EXEC', f'bridge {"timed out on" if timed_out else "rejected"} '
                                      f'step[{i}]: {bridge_info}')
                    self.stats['failures'].append({
                        'kind': 'bridge_timeout' if timed_out else 'bridge_rejected',
                        'step_index': i,
                        'detail': str(bridge_info)[:300],
                    })
                    if timed_out:
                        # Not physical feedback -- see _dispatch_with_retry. Stop
                        # the task rather than re-plan around a phantom constraint.
                        self._log('EXEC', 'simulator never answered; ending the task '
                                          'without learning a constraint from it')
                        termination_reason = termination_reason or 'bridge_lost'
                        runtime_failed = True
                        no_progress_iters = no_progress_cap
                        break
                    self._record_failure_and_inject(
                        plan, i, f'bridge:{str(bridge_info)[:80]}')
                    runtime_failed = True
                    break

                # Stamp into Task Graph (A).
                node_id = self.task_graph.add_step(agent_id, verb, tuple(targets))
                self._log('A', f'task_graph node={node_id} verb={verb} agent={agent_id} targets={targets}')

                done, task_results, _, unsatisfied, steps = self.env.step(
                    class_name, agent_id, exec_res.action, self.task_goal,
                )
                self.task_graph.mark_executed(node_id)
                executed_any = True
                no_progress_iters = 0
                obs = self.env.get_observations()
                self._log('ENV', f'step={steps} executed=[{verb}]<{class_name}>({agent_id}) '
                                 f'-> action={exec_res.action} done={done} '
                                 f'remaining_goals={list(unsatisfied.keys()) if unsatisfied else []}')
                saved_info.append({
                    'task_id': self.env.task_id,
                    'env_id': self.env.env_id,
                    'step': steps,
                    'agent': step['agent'],
                    'action': exec_res.action,
                    'done': done,
                })
                if done:
                    success = True
                    break
                if self.env.steps > max_steps:
                    break

            if success:
                termination_reason = 'goal_reached'
                break
            if not executed_any:
                # A runtime failure that executed nothing is not progress. This must be
                # counted before the `runtime_failed` continue below, or a plan the
                # executor keeps rejecting spins until the outer-iteration cap
                # (2 * max_steps Oracle calls) instead of tripping no_progress_cap
                # after 3 attempts.
                no_progress_iters += 1
            if runtime_failed:
                continue
            self._log('ITER', 'plan exhausted without goal completion; re-planning with current state')
            self.last_rejected = None

        self.stats['outer_iters'] = outer_iters
        self.stats['termination_reason'] = termination_reason or 'loop_exit'
        self.stats['hard_bans'] = len(self.validator.forbidden)
        self.stats['forbidden_triples'] = [list(t) for t in self.validator.forbidden]
        # A deadlock is a stall the planner inflicted on itself: it ran out of re-plans
        # while its own learned bans were rejecting every plan it produced. A stall with
        # no forbidden rejection is an ordinary failure and must not be counted here.
        self.stats['deadlocked'] = bool(
            self.stats['termination_reason'] == 'no_progress'
            and self.stats['forbidden_rejections'] > 0)
        self.stats['clauses_learned'] = len(self.cdcl.clauses)
        self.stats['learned_clauses'] = [
            (c.agent_class, c.verb, c.target_class, c.reason) for c in self.cdcl.clauses]

        self._banner('FINAL', f'success={success} steps={steps}')
        self._log('FINAL', f'termination_reason={self.stats["termination_reason"]} '
                           f'deadlocked={self.stats["deadlocked"]} '
                           f'validator_rejections={self.stats["validator_rejections"]} '
                           f'forbidden_rejections={self.stats["forbidden_rejections"]} '
                           f'hard_bans={self.stats["hard_bans"]}')
        self._log('FINAL', f'cdcl: {self.cdcl.stats()}')
        self._log('FINAL', f'task_graph: {self.task_graph.summary()}')
        self._log('FINAL', f'learned_clauses: {[(c.agent_class, c.verb, c.target_class, c.reason) for c in self.cdcl.clauses]}')
        return success, steps, saved_info

    # -------------------------------------------------- verification + repair (R)

    def _verify_and_repair(self, plan, predicates, obs_text):
        """Z3-check the plan, then repair per --repair until SAT or the budget runs out.

        Returns (plan, CheckResult, last_RemediationResult|None). The attempt budget is
        `--max_replan_attempts` for every repair strategy, so the symbolic and LLM arms
        get exactly the same number of chances — the comparison is about what each
        strategy costs per chance, not about who was allowed more of them.
        """
        attempt = 0
        rem = None
        # Goal entailment only in full-verify mode: shadow mode's UNSAT counter is the
        # soundness number for the naive arm and must keep its original semantics.
        goal = self.task_goal if self.verify_mode == 'z3' else None
        while True:
            result = self.z3.check(predicates, plan, goal=goal)
            self.stats['z3_checks'] += 1
            if result.sat:
                self._log('Z3', f'SAT — all {len(plan)} step(s) preconditions satisfied')
                if attempt:
                    self.stats['repair_successes'] += 1
                return plan, result, rem
            self.stats['z3_unsat_events'] += 1
            self.stats['unsat_cores'].append({
                'step_index': result.failed_step_index,
                'core': [str(c) for c in result.unsat_core_labels],
                'missing': [str(m) for m in result.missing_preconds],
            })
            self._log('Z3', f'UNSAT @ step {result.failed_step_index} '
                            f'core={result.unsat_core_labels} missing={result.missing_preconds}')

            if self.verify_mode == 'shadow' or self.repair_mode == 'none':
                why = 'shadow mode' if self.verify_mode == 'shadow' else '--repair none'
                self._log('R', f'repair disabled ({why}); returning UNSAT verdict.')
                return plan, result, rem

            if attempt >= self.args.max_replan_attempts:
                self._log('R', f'repair cap reached ({attempt}/{self.args.max_replan_attempts}); '
                               'escalating to full re-plan.')
                self.stats['escalations'] += 1
                return plan, result, rem

            attempt += 1
            self.stats['repair_attempts'] += 1
            self._log('R', f'attempt {attempt}/{self.args.max_replan_attempts} '
                           f'(strategy={self.repair_mode})')

            if self.repair_mode == 'llm':
                repaired = self.oracle.repair_plan(
                    obs_text, self.env.goal_instruction, self.env.num_agent,
                    self.injected_text, plan, result,
                )
                if not repaired:
                    self._log('R', 'LLM repair returned no parseable plan; escalating.')
                    self.stats['escalations'] += 1
                    return plan, result, rem
                self._log('R', f'LLM returned a repaired plan of {len(repaired)} step(s)')
                plan = repaired
                continue

            # symbolic — derive the remediation from the UNSAT core, no LLM call.
            rem = self.replanner.replan(plan, result, predicates,
                                        [{'agent_node': a.agent_node} for a in self.agents])
            if rem.inserted is None:
                self._log('R', f'no remediation derivable ({rem.reason}); escalating.')
                self.stats['escalations'] += 1
                return plan, result, rem
            if len(rem.inserted_chain) > 1:
                self._log('R', f'inserted chain of {len(rem.inserted_chain)} step(s) '
                               f'(reason={rem.reason}):')
                for s in rem.inserted_chain:
                    self._log('R', f'    + {s}')
            else:
                self._log('R', f'inserted {rem.inserted} (reason={rem.reason})')
            plan = rem.plan
            self._log('R', f'new plan length: {len(plan)} step(s)')

    def _escalate_unsat(self, plan, result, rem):
        """Record an unrepairable UNSAT and feed the CDCL layer before the full re-plan."""
        first_missing = (result.missing_preconds[0][0]
                         if result.missing_preconds else '')
        # A static-property or goal miss names the actual cause; prefer it over
        # whatever other atom happened to land first in the UNSAT core.
        for _atom in result.missing_preconds:
            if _atom[0] in ('goal_unreached', 'ground_reachable', 'same_surface',
                            'quad_movetarget'):
                first_missing = _atom[0]
                break
        # If the Replanner refused with an antipattern reason on its final attempt,
        # surface that as the failed_precond so the injector emits a targeted hint.
        # The remainder of the antipattern reason carries per-task context (e.g.
        # which arms are in which rooms) — pass it through as `context`.
        reason = f'z3_unsat:{result.unsat_core_labels}'
        precond_key = first_missing
        context = ''
        if first_missing == 'goal_unreached':
            unmet = [str(a[1]) for a in result.missing_preconds if a[0] == 'goal_unreached']
            context = 'unmet goal(s): ' + ', '.join(unmet)
        elif first_missing in ('same_surface', 'ground_reachable', 'quad_movetarget'):
            context = self._static_miss_context(first_missing, result)
        if rem is not None and rem.inserted is None and 'antipattern:' in rem.reason:
            import re as _re
            m = _re.search(r'antipattern:(\w+)\s*(.*)', rem.reason)
            if m:
                precond_key = f'antipattern_{m.group(1)}'
                reason = f'antipattern:{m.group(1)}'
                context = m.group(2).strip()
        self.stats['failures'].append({
            'kind': 'z3_unsat_escalated',
            'step_index': result.failed_step_index,
            'detail': reason[:300],
        })
        return self._record_failure_and_inject(plan, result.failed_step_index, reason,
                                               failed_precond=precond_key, context=context)

    def _render_last_attempt(self) -> str:
        """The concrete rejected plan, with the offending step marked.

        The abstract clause hints say what rule was broken; they do not say which of
        the Oracle's own steps broke it, and they stop changing once they dedup. A
        weaker planner given an unchanged prompt returns an unchanged plan. Showing the
        rejected attempt itself is the only part of the prompt guaranteed to differ
        between re-plans.
        """
        lr = self.last_rejected
        if not lr or not lr.get('plan'):
            return ''
        plan, idx = lr['plan'], lr['failed_idx']
        lines = ['', 'YOUR PREVIOUS PLAN WAS REJECTED before execution. Do not submit it again.',
                 '', 'Rejected plan:']
        for i, st in enumerate(plan):
            mark = '   <-- REJECTED HERE' if i == idx else ''
            lines.append(f'  {i}. {st.get("agent", "?")} {st.get("action", "?")}{mark}')
        why, advice = self._explain_rejection(
            lr.get('reason', ''), plan[idx] if 0 <= idx < len(plan) else {},
            lr.get('precond', ''), lr.get('targets') or [])
        lines += ['', f'Why step {idx} was rejected: {why}', '', advice]
        return '\n'.join(lines)

    # Advice that fits a structural defect — the agent genuinely cannot do the work.
    _REROUTE = ('Write a NEW plan for the same goal that does not contain this defect. '
                'Re-ordering the same steps will not fix it — the step itself is not '
                'possible for the agent it was given to, so the work must be routed '
                'differently (a different agent, or the object moved first).')
    # Advice that fits a naming/ordering slip — the plan shape was right.
    _LOCAL_FIX = ('Keep the rest of the plan as it is and fix only this step. The overall '
                  'approach was sound — do NOT switch to a different agent and do NOT add a '
                  'quadrotor ferry, which would make the plan far longer than necessary.')

    def _explain_rejection(self, reason: str, step, precond: str, targets):
        """(why, what-to-do) for the rejected step, in the Oracle's own vocabulary.

        The split matters more than the wording. A HIGH target is a structural defect —
        no ordering helps and the work has to be re-routed. A FLOOR target is a naming
        slip: ground robots change room by naming the ROOM, and the surrounding plan is
        usually fine. Collapsing the two sent the Oracle off building 13-step ferries
        for tasks a robot dog could finish in six.
        """
        agent = step.get('agent', 'the agent')
        action = step.get('action', 'this action')
        nodes = {n['id']: n for n in self.env.graph['nodes']}

        def name(i):
            n = nodes.get(i, {})
            return f"<{n.get('class_name', '?')}>({i})"

        if 'ground_reachable' in (precond or '') or 'ground_reachable' in (reason or ''):
            tid = targets[0] if targets else None
            node = nodes.get(tid, {}) if tid is not None else {}
            if node.get('category') == 'Floor':
                room = next((e['to_id'] for e in self.env.graph['edges']
                             if e['from_id'] == tid and e['relation_type'] == 'INSIDE'), None)
                room_s = name(room) if room is not None else 'the room'
                return (f'{name(tid)} is a FLOOR. Floors are never valid [movetowards] targets — '
                        f'a ground robot changes room by naming the room itself. Use '
                        f'[movetowards] {room_s} instead of {name(tid)}.', self._LOCAL_FIX)
            return (f'{agent} is a ground robot and {action} targets something on a HIGH '
                    'surface it can never touch. No ordering makes it reachable.', self._REROUTE)

        table = {
            'same_surface':    (f'{agent} is a FIXED arm; {action} targets something that is not on '
                                "that arm's own surface, so it is out of reach.", self._REROUTE),
            'quad_movetarget': (f'{action} sends the quadrotor to something that is not a LANDABLE '
                                'surface and not an adjacent open room.', self._LOCAL_FIX),
            'goal_unreached':  ('the plan runs to completion but the goal state is still not '
                                'satisfied — check the exact object ids named in the goal.',
                                self._LOCAL_FIX),
        }
        for key, pair in table.items():
            if key in (precond or '') or key in (reason or ''):
                return pair
        return (f'the preconditions of {action} cannot hold at that point in the plan ({reason}).',
                self._LOCAL_FIX)

    def _static_miss_context(self, precond: str, result) -> str:
        """Concrete scene facts for a static-property miss, so the Oracle's hint names
        the actual route (which arm is on which surface, where the target sits) the
        way the robot_arm_wrong_room antipattern already does. Facts only — the plan
        stays the Oracle's job."""
        snap = result.state_at_failure
        if snap is None:
            return ''
        nodes = snap.nodes or {}

        def name(i):
            n = nodes.get(i, {})
            return f"<{n.get('class_name', '?')}>({i})"

        atom = next((a for a in result.missing_preconds if a[0] == precond), None)
        if atom is None:
            return ''
        try:
            if precond == 'same_surface':
                aid, tid = atom[1], atom[2]
                surface = next((b[2] for b in snap.binary
                                if b[0] == 'on' and b[1] == aid), None)
                carrier = next((b[1] for b in snap.binary
                                if b[0] in ('with', 'hold') and b[2] == tid), None)
                parent = next((b[2] for b in snap.binary
                               if b[0] in ('on', 'inside') and b[1] == tid), None)
                where = (f'currently with {name(carrier)}' if carrier is not None
                         else f'currently on/in {name(parent)}' if parent is not None else 'elsewhere')
                facts = [f'{name(aid)} is mounted on {name(surface)}; {name(tid)} is {where}']
                facts.append(self._who_can_reach(snap, tid, exclude=aid))
                return '. '.join(f for f in facts if f)
            if precond == 'ground_reachable':
                tid = atom[-1]
                # Cargo case: the target rides the quadrotor (basket). The issue is
                # plan ORDER, not the surface — say so explicitly.
                carrier = next((b[1] for b in snap.binary
                                if b[0] in ('with', 'hold') and b[2] == tid), None)
                if carrier is not None and (snap.nodes.get(carrier, {}).get('class_name')
                                            in ('quadrotor', 'drone')):
                    perch = next((b[2] for b in snap.binary
                                  if b[0] == 'on' and b[1] == carrier), None)
                    where = f', which at this point in the plan is on {name(perch)}' if perch is not None else ''
                    return (f'{name(tid)} rides {name(carrier)}{where}. The dog can reach '
                            f'{name(tid)} ONLY while the quadrotor is landed on a LOW '
                            f'surface or floor — schedule every dog [putinto]/[grab] on '
                            f'{name(tid)} BEFORE the quadrotor lifts it to a high surface')
                facts = [f'{name(tid)} is unreachable for ground robots']
                # For an object ON a high surface, the actionable facts live on the
                # parent surface: which arm is mounted there, whether it is LANDABLE.
                surf = tid
                for _ in range(4):
                    if any(b[0] == 'on' and b[1] == i and b[2] == surf
                           for b in snap.binary
                           for i in [b[1]]
                           if nodes.get(i, {}).get('class_name') in ('robot arm', 'robot_arm')):
                        break
                    parent = next((b[2] for b in snap.binary
                                   if b[0] in ('on', 'inside') and b[1] == surf), None)
                    if parent is None:
                        break
                    surf = parent
                if surf != tid:
                    facts.append(f'it sits on {name(surf)}')
                arms = [i for i, n in nodes.items()
                        if n.get('category') == 'Agents'
                        and n.get('class_name') in ('robot arm', 'robot_arm')
                        and any(b[0] == 'on' and b[1] == i and b[2] == surf
                                for b in snap.binary)]
                if arms:
                    facts.append(', '.join(name(a) for a in arms)
                                 + f' is mounted on {name(surf)}')
                if snap.has_unary('landable', surf):
                    facts.append(f'{name(surf)} is LANDABLE for the quadrotor')
                return '; '.join(facts)
            if precond == 'quad_movetarget':
                return f'{atom[2] and name(atom[2])} is not a LANDABLE surface or open room'
        except Exception:
            return ''
        return ''

    def _who_can_reach(self, snap, target_id: int, exclude: int = None) -> str:
        """Name the agents already within reach of `target_id`, else say where every
        fixed arm is parked.

        Scenes routinely mount one robot arm per surface, so "this arm cannot reach it"
        is only half the answer — the Oracle needs to know which arm CAN. Uses the same
        same_surface_holds() the verifier decides with, so the agent named here is one
        Z3 will actually accept.
        """
        from z3_reasoner import same_surface_holds
        nodes = snap.nodes or {}

        def name(i):
            n = nodes.get(i, {})
            return f"<{n.get('class_name', '?')}>({i})"

        # Only manipulators can "reach" anything — a quadrotor has no gripper. And
        # same_surface_holds is a FIXED-arm test: a ground robot shares the room floor
        # with every object on it, including ones on high furniture it can never touch,
        # so ground candidates need the same height guard the verifier applies.
        high = (snap.has_unary('high_height', target_id)
                or snap.has_unary('on_high_surface', target_id))
        reachers = []
        for i, n in nodes.items():
            if i == exclude or n.get('category') != 'Agents':
                continue
            cls = n.get('class_name')
            if cls in ('robot arm', 'robot_arm'):
                if same_surface_holds(snap.binary, i, target_id):
                    reachers.append(i)
            elif cls in ('robot dog', 'robot_dog'):
                if not high and same_surface_holds(snap.binary, i, target_id):
                    reachers.append(i)
        agents = [i for i, n in nodes.items() if n.get('category') == 'Agents']
        if reachers:
            return ('within reach of ' + ', '.join(name(i) for i in reachers)
                    + ' in the current state')
        arms = [i for i in agents
                if nodes.get(i, {}).get('class_name') in ('robot arm', 'robot_arm')]
        if not arms:
            return ''
        locs = []
        for i in arms:
            surf = next((b[2] for b in snap.binary if b[0] == 'on' and b[1] == i), None)
            locs.append(f'{name(i)} is on {name(surf)}')
        return ('no agent can reach it in the current state; arms in this scene: '
                + '; '.join(locs))

    # ----------------------------------------------------------- execution bridge

    def _dispatch(self, class_name, agent_id, action):
        """Send one grounded action to the simulator. Returns (ok, info).

        Standalone mode always succeeds -- there is nothing to reject. In ws mode
        the symbolic names the planner speaks ("<apple>(30)") are rewritten to the
        simulator's prim names ("apple_agveuv_0") using the env json's
        name_mapping, exactly as PEFA does, so both planners drive the scene
        through the identical wire format.

        (ok=False, info) means the SIMULATOR rejected the action -- real physical
        feedback, and the caller feeds it to the CDCL layer. A transport failure
        is not that, and raises instead.
        """
        if self.ws is None:
            return True, '', False

        og_action = action
        for key, og_name in self.name_mapping.items():
            # "apple(30)" -> "<apple>(30)", the form that appears in an action string.
            head, _, tail = key.rpartition('(')
            symbolic = f'<{head}>({tail}'
            if symbolic in og_action:
                og_action = og_action.replace(symbolic, og_name)
        og_agent = self.name_mapping.get(f'{class_name}({agent_id})', class_name)

        msg = json.dumps({'agent': og_agent, 'action': og_action})
        self._log('WS', f'send {msg}')
        EVENTS.emit('WS_SEND', title=f'{og_agent}', body=og_action)
        try:
            self.ws.send(msg)
            raw = self.ws.recv()
        except Exception as e:
            # A dead socket is not physical feedback. Returning False here would
            # route it into the CDCL layer, which would then "learn" that a
            # perfectly feasible action is impossible and carry that bogus clause
            # for the rest of the task. There is also nothing to retry against, so
            # fail the run loudly instead.
            self._log('WS', f'transport error: {e} — aborting, the bridge is gone')
            EVENTS.emit('WS_RECV', title='transport error', body=str(e), success=False)
            raise RuntimeError(f'ws_transport_error: {e}')

        self._log('WS', f'recv {raw}')
        try:
            result = json.loads(raw)
        except Exception:
            # Same reasoning: a malformed reply says nothing about the world.
            self._log('WS', f'unparseable reply from the bridge: {str(raw)[:200]}')
            EVENTS.emit('WS_RECV', title='bad reply', body=str(raw)[:200], success=False)
            raise RuntimeError(f'bad_json_from_server: {str(raw)[:200]}')

        ok = bool(result.get('success', False))
        timed_out = bool(result.get('timed_out', False))
        EVENTS.emit('WS_RECV', title='ok' if ok else 'rejected',
                    body=str(result.get('info', '')), success=ok,
                    timed_out=timed_out)
        return ok, result.get('info', ''), timed_out

    def _dispatch_with_retry(self, class_name, agent_id, action):
        """Dispatch, retrying a simulator TIMEOUT but not a simulator REJECTION.

        The two mean different things and must not be conflated. A rejection is
        the simulator saying "that action does not work here" -- real physical
        feedback, and exactly what the CDCL layer should learn from. A timeout is
        the simulator never answering at all, which on this benchmark means a
        controller has hung: `franka_move_with_waypoints` returns
        done=False/waypoint_ind=0 forever when the arm's IK cannot converge on a
        pose near the edge of its reach, so whether it happens depends on where
        the drone happened to land.

        Learning a clause from that would teach the planner that a perfectly
        feasible action is impossible, and the ban would persist for the rest of
        the task. So retry it, and only give up -- without learning -- if it keeps
        hanging.
        """
        attempts = int(getattr(self.args, 'bridge_retries', 1)) + 1
        for i in range(attempts):
            ok, info, timed_out = self._dispatch(class_name, agent_id, action)
            if ok or not timed_out:
                return ok, info, timed_out
            if i + 1 < attempts:
                self._log('WS', f'simulator did not answer ({info}); '
                                f'retrying {i + 2}/{attempts} — a hung controller '
                                f'is not evidence the action is infeasible')
        return False, info, True

    # ----------------------------------------------------------- helpers

    def _extract_step_ids(self, step):
        import re
        m = re.match(r'<([^>]+)>\((\d+)\)', step.get('agent', '').strip())
        agent_id = int(m.group(2)) if m else -1
        verb_m = re.search(r'\[(\w+)\]', step.get('action', ''))
        verb = verb_m.group(1) if verb_m else ''
        targets = [int(x) for x in re.findall(r'\((\d+)\)', step.get('action', ''))]
        return agent_id, verb, targets

    def _lookup_agent_class(self, agent_id: int) -> str:
        for a in self.agents:
            if a.agent_node['id'] == agent_id:
                return a.agent_node['class_name']
        return ''

    def _record_failure_and_inject(self, plan, failed_idx: int, reason: str,
                                   failed_precond: str = '', context: str = ''):
        if failed_idx < 0 or failed_idx >= len(plan):
            return 0
        if not self.cdcl_enabled:
            self._log('CDCL', 'disabled; skipping learning step')
            return 0
        # Remember what was rejected, so the next full re-plan can be told. Without
        # this the Oracle is asked to plan from scratch with only the accumulated
        # abstract clauses, which stop growing once they dedup — the prompt then
        # becomes byte-identical between iterations and the same plan comes back.
        agent_id, verb, targets = self._extract_step_ids(plan[failed_idx])
        self.last_rejected = {'plan': list(plan), 'failed_idx': failed_idx, 'reason': reason,
                              'precond': failed_precond, 'targets': list(targets)}
        nid = self.task_graph.add_step(agent_id, verb, tuple(targets))
        self.task_graph.mark_failed(nid, reason)
        self._log('A', f'task_graph FAILED node={nid} verb={verb} agent={agent_id} targets={targets} reason={reason}')
        if 'forbidden_by_learned_constraint' in reason:
            # The PlanValidator rejected this step because of a clause we already learned
            # on a previous iteration. Don't re-record a duplicate clause; the Oracle just
            # needs to keep re-planning until it produces a plan that avoids the existing
            # ban.
            self._log('B', 'skipping clause learning: secondary effect of an existing forbidden triple')
            return 0
        report = self.conflict_analyzer.analyze(self.task_graph, nid, reason)
        self._log('B', f'conflict report: class={report.failure_class.value} '
                       f'failed_node={report.failed_node_id} implicated={report.implicated_node_ids} '
                       f'failed_precond={failed_precond or "-"} context={context or "-"}')
        new_clauses = self.cdcl.learn(self.task_graph, report,
                                      failed_precond=failed_precond, context=context)
        backtrack = self.cdcl.backtrack_level(self.task_graph, report)
        self._log('C', f'cdcl learned {len(new_clauses)} new clause(s) '
                       f'(total={len(self.cdcl.clauses)}); non_chrono_backtrack_level={backtrack}')
        for c in new_clauses:
            self._log('C', f'  + clause: agent={c.agent_class} verb={c.verb} '
                           f'target={c.target_class} reason={c.reason} precond={c.failed_precond or "-"}')
        if new_clauses:
            self.injected_text = self.injector.inject(self.cdcl.clauses, self.validator)
            self._log('D', f'injected {len(self.cdcl.clauses)} clause(s) into Oracle prompt + Plan Validator')
            self._log('D', f'forbidden_triples={self.validator.forbidden}')
            self._log('D', f'oracle_prefix:\n{self.injected_text}')
        return len(new_clauses)

    def _cdcl_full_replan_failure(self, reason: str) -> bool:
        # On a fully-failed Oracle round (empty plan etc.), still feed the loop.
        if not self.cdcl_enabled:
            return False
        return True
