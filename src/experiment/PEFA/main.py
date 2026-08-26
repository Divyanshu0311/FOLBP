import json
import sys
import time
from pathlib import Path
from LLM_agent import LLM_agent
from args import get_args
from LLM_oracle import ArenaMP
from get_env_info import Get_env_info
import traceback
import argparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bench.eventlog import EVENTS
from bench.llm_meter import METER
from bench.record import write_task_record


def record_task(args, task_id, env_id, task_name, goal, gt_steps,
                success, steps, wall_clock_s, status='completed', detail=''):
    """Write the per-task benchmark record, if --record_path was given."""
    if not getattr(args, 'record_path', None):
        return
    write_task_record(
        Path(args.record_path),
        arm=args.arm, seed=args.seed, env=args.env, task_id=task_id,
        env_id=env_id, task_name=task_name, goal=goal,
        success=success, gt_steps=gt_steps, executed_steps=steps,
        wall_clock_s=wall_clock_s, llm=METER.snapshot(),
        model=args.lm_id, temperature=args.t,
        flags={'mode': args.mode},
        failures=[{'kind': status, 'detail': detail[:2000]}] if detail else [],
        status=status,
    )

args = get_args()
# No-op unless --event_log / $COHERENT_EVENT_LOG is set.
EVENTS.open(args.event_log)

def write_log_to_file(log_message, file_name=f'./log/log_test.txt'):
        with open(file_name, 'a') as file:  
            file.write(log_message + '\n')  


if __name__ == '__main__':

    
    with open(f'./env/{args.env}.json') as file:
        data = json.load(file)
    '''
    The task uesed in our paper is as follows:
    Env0: [2, 4, 9, 10, 11, 15, 16, 20]
    Env1: [1, 3, 7, 8, 11, 10, 16, 20]
    Env2: [3, 5, 6, 7, 10, 11, 16 ,17]
    Env3: [2, 4, 6, 7, 10, 16, 17, 19]
    Env4: [0, 1, 7, 10, 12, 17, 18, 19]
    '''
    tasklist = args.task
    # S = [[] for _ in range(len(tasklist))]
    # L = [[] for _ in range(len(tasklist))] 
    steps_list, tasks_result = [], []
    success_tasks, failed_tasks = [], []

    for task_id in tasklist:
        env_id = data[task_id]["env_id"]
        task_name = data[task_id]["task_name"]
        graph = data[task_id]["init_graph"]
        task_goal = data[task_id]["task_goal"]
        goal_instruction = data[task_id]["goal_instruction"]
        ground_truth_step_num = data[task_id]["ground_truth_step_num"]

        test_results = {}
        agent = []
        for node in data[task_id]["init_graph"]["nodes"]:
            if node["category"] == "Agents":
                agent.append([node["class_name"],node["id"]])

        agent_nodes = []
        for a in agent:
            agent_node = [node for node in graph['nodes'] if node['id'] == a[1]]
            agent_nodes += agent_node

        dict_list = []
        for i, agent_name in enumerate(agent):
            dict_item = {
                'agent_id': i,
                'args': args,
                'agent_node': agent_nodes[i],
                'init_graph': graph,
            }
            dict_list.append(dict_item)


        def env_fn():
            return Get_env_info(
                                task_id = task_id,
                                env_id = env_id,
                                task_name = task_name,
                                graph = graph,
                                task_goal = task_goal,
                                goal_instruction = goal_instruction,
                                ground_truth_step_num = ground_truth_step_num,
                                agent = agent,
                                num_agent = len(agent)
                                )

        def LLM_agent_fn(args_llm):

            return LLM_agent(**args_llm)

        agents = [LLM_agent_fn(dict_list[i]) for i in range(len(dict_list))]
        # print(len(agents))
        # input('s')
        arena = ArenaMP(env_fn, agents, args)

        steps = 0
        # import ipdb ;ipdb.set_trace()

        METER.reset()
        t0 = time.time()
        try:
            success, steps, saved_info = arena.run()
            print(success)
        except Exception as e:

            print(f"An error occurred: {e}")
            traceback.print_exc()
            error_info = traceback.format_exc()
            write_log_to_file(f"An error occurred: {e}")
            write_log_to_file(error_info+'\n\n')
            success = False
            # Record the crash from inside the child, where the LLM counters
            # still exist — the sweep parent can only write a contentless stub.
            record_task(args, task_id, env_id, task_name, goal_instruction,
                        ground_truth_step_num, success=False, steps=steps,
                        wall_clock_s=time.time() - t0, status='crashed',
                        detail=error_info)
            raise Exception
            # input('STOP')
        record_task(args, task_id, env_id, task_name, goal_instruction,
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
        tasks_result.append(1 if success else 0)
        if success:
            success_tasks.append(task_id)
        else:
            failed_tasks.append(task_id)

    write_log_to_file(f'average steps: {sum(steps_list)/len(steps_list) if len(steps_list) > 0 else None}')
    write_log_to_file(f'successful tasks: {success_tasks if len(success_tasks) > 0 else None}')
    write_log_to_file(f'failed tasks: {failed_tasks if len(failed_tasks) > 0 else None}')
    print('average steps:', sum(steps_list)/len(steps_list) if len(steps_list) > 0 else None)
    print('successful tasks:', success_tasks if len(success_tasks) > 0 else None )
    print('failed tasks:', failed_tasks if len(failed_tasks) > 0 else None)    

    EVENTS.close()
