"""FOLBP — First-Order Logic + Backtrack Planner. Entry point.

Forward flow (badges 1-6):
    Task Goal + Scene Graph -> Oracle LLM -> FOL Generator -> Z3 SMT Reasoner
    -> Plan Validator -> Robot Executor -> Environment.
Repair (R): Z3-UNSAT -> UNSAT-Core Replanner -> re-validate, escalate to full re-plan after 1 try.
Learning (A-D): failures feed Task Graph -> Conflict Analyzer -> CDCL Engine -> Constraint Injection.
"""
import json
import sys
import time
import traceback
from pathlib import Path

from args import get_args
from get_env_info import Get_env_info
from LLM_agent import LLM_agent
from LLM_oracle import ArenaMP

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bench.eventlog import EVENTS
from bench.llm_meter import METER
from bench.record import write_task_record

args = get_args()
# No-op unless --event_log / $COHERENT_EVENT_LOG is set.
EVENTS.open(args.event_log)


def write_log(msg, file_name=None):
    file_name = file_name or f'./log/{args.env}.log'
    with open(file_name, 'a') as f:
        f.write(str(msg) + '\n')


def record_task(args, arena, task_id, env_id, task_name, goal, gt_steps,
                success, steps, wall_clock_s, status='completed', detail=''):
    """Write the per-task benchmark record, if --record_path was given."""
    if not args.record_path:
        return
    st = getattr(arena, 'stats', {}) or {}
    failures = list(st.get('failures', []))
    if detail:
        failures.append({'kind': status, 'detail': detail[:2000]})
    write_task_record(
        Path(args.record_path),
        arm=args.arm, seed=args.seed, env=args.env, task_id=task_id,
        env_id=env_id, task_name=task_name, goal=goal,
        success=success, gt_steps=gt_steps, executed_steps=steps,
        wall_clock_s=wall_clock_s, llm=METER.snapshot(),
        model=args.lm_id, temperature=args.t,
        flags={
            'verify': args.verify,
            'repair': args.repair,
            'cdcl': args.enable_cdcl,
            'nudge': args.enable_nudge,
            'max_oracle_plans': args.max_oracle_plans,
            'max_replan_attempts': args.max_replan_attempts,
            'executor_mode': args.executor_mode,
        },
        verify={
            'mode': args.verify,
            'z3_checks': st.get('z3_checks', 0),
            'z3_unsat_events': st.get('z3_unsat_events', 0),
            'shadow_unsat_events': st.get('shadow_unsat_events', 0),
            'unsat_cores': st.get('unsat_cores', []),
        },
        repair={
            'mode': args.repair,
            'attempts': st.get('repair_attempts', 0),
            'successes': st.get('repair_successes', 0),
            'escalations': st.get('escalations', 0),
        },
        cdcl={
            'enabled': args.enable_cdcl,
            'clauses_learned': st.get('clauses_learned', 0),
            'clauses': st.get('learned_clauses', []),
        },
        deadlock={
            'termination_reason': st.get('termination_reason', ''),
            'deadlocked': bool(st.get('deadlocked')),
            'validator_rejections': st.get('validator_rejections', 0),
            'forbidden_rejections': st.get('forbidden_rejections', 0),
            'hard_bans': st.get('hard_bans', 0),
            'forbidden_triples': st.get('forbidden_triples', []),
        },
        outer_iters=st.get('outer_iters'),
        failures=failures,
        status=status,
        extra={'oracle_plans': st.get('oracle_plans', 0)},
    )


if __name__ == '__main__':
    with open(f'./env/{args.env}.json') as f:
        data = json.load(f)

    steps_list, results = [], []
    success_tasks, failed_tasks = [], []

    for task_id in args.task:
        d = data[task_id]
        env_id = d['env_id']
        task_name = d['task_name']
        graph = d['init_graph']
        task_goal = d['task_goal']
        goal_instruction = d['goal_instruction']
        ground_truth_step_num = d['ground_truth_step_num']

        agent = [[n['class_name'], n['id']] for n in graph['nodes'] if n['category'] == 'Agents']
        agent_nodes = []
        for a in agent:
            agent_nodes += [n for n in graph['nodes'] if n['id'] == a[1]]

        dict_list = [
            {'agent_id': i, 'args': args, 'agent_node': agent_nodes[i], 'init_graph': graph}
            for i, _ in enumerate(agent)
        ]

        def env_fn():
            return Get_env_info(
                task_id=task_id, env_id=env_id, task_name=task_name,
                graph=graph, task_goal=task_goal, goal_instruction=goal_instruction,
                ground_truth_step_num=ground_truth_step_num,
                agent=agent, num_agent=len(agent),
            )

        agents = [LLM_agent(**d) for d in dict_list]
        arena = ArenaMP(env_fn, agents, args)

        steps = 0
        METER.reset()
        t0 = time.time()
        try:
            success, steps, _ = arena.run()
            print(success)
        except Exception as e:
            print(f'An error occurred: {e}')
            traceback.print_exc()
            write_log(f'An error occurred: {e}')
            write_log(traceback.format_exc())
            success = False
            # Record the crash from inside the child, where the LLM counters still
            # exist — the sweep parent can only write a contentless stub.
            record_task(args, arena, task_id, env_id, task_name, goal_instruction,
                        ground_truth_step_num, success=False, steps=steps,
                        wall_clock_s=time.time() - t0, status='crashed',
                        detail=traceback.format_exc())
            raise
        record_task(args, arena, task_id, env_id, task_name, goal_instruction,
                    ground_truth_step_num, success=success, steps=steps,
                    wall_clock_s=time.time() - t0)

        EVENTS.emit('RESULT', title='success' if success else 'failure',
                    body=f'{steps} steps executed (ground truth {ground_truth_step_num})',
                    success=bool(success), steps=steps,
                    gt_steps=ground_truth_step_num,
                    llm=METER.snapshot().get('total_calls'),
                    wall_clock_s=round(time.time() - t0, 1))

        print(f'---------Env:{env_id}---Task:{task_id}-------------------------')
        print('success' if success else 'failure')
        print('steps:', steps)
        print('-------------------------------------')

        steps_list.append(steps)
        results.append(1 if success else 0)
        (success_tasks if success else failed_tasks).append(task_id)

    avg_steps = sum(steps_list) / len(steps_list) if steps_list else None
    success_rate = sum(results) / len(results) if results else 0
    write_log(f'average steps: {avg_steps}')
    write_log(f'success rate: {success_rate}')
    write_log(f'successful tasks: {success_tasks or None}')
    write_log(f'failed tasks: {failed_tasks or None}')
    print('average steps:', avg_steps)
    print('success rate:', success_rate)
    print('successful tasks:', success_tasks or None)
    print('failed tasks:', failed_tasks or None)

    EVENTS.close()
