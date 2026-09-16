import argparse
import os

# import torch
import pdb
import yaml
from typing import Dict


def get_args():
    parser = argparse.ArgumentParser(description='LLM-based Heterogeneous Multi-Robot System Setting')

    parser.add_argument('--env', type=str, default='env0',
                    choices=['env0', 'env1', 'env2', 'env3', 'env4', 'env_test', 'env_merom',
                             'env_house', 'env_tb'],
                    help='Select a simulation environment. env0-env4 are the COHERENT benchmark '
                         'environments; env_tb is the real-robot demo (robot dog on TurtleBot + drone).')
    parser.add_argument('--mode', type=str, default='standalone',
                    choices=['standalone', 'ws', 'tb'],
                    help="Execution mode: 'standalone' runs only the symbolic planner (no simulator). "
                         "'ws' forwards each executed action to OmniGibson via WebSocket (requires --ws_url). "
                         "'tb' forwards each executed action to physical/virtual devices via HTTP "
                         "(uses --tb_urls if provided, else --tb_url).")
    parser.add_argument('--ws_url', type=str, default=os.environ.get('COHERENT_WS_URL', 'ws://127.0.0.1:8765'),
                    help='WebSocket URL of OmniGibson bridge. Required when --mode=ws. '
                         'Override with $COHERENT_WS_URL.')
    parser.add_argument('--tb_url', type=str, default=os.environ.get('COHERENT_TB_URL', 'http://127.0.0.1:8080'),
                    help='HTTP URL of a single TurtleBot web_client (e.g. http://<bot-ip>:8080). '
                         'Override with $COHERENT_TB_URL. Used when --tb_urls is not set.')
    parser.add_argument('--tb_urls', type=str, default=None,
                    help='Path to a JSON file mapping agent class names (lowercase, e.g. '
                         '"robot dog") to device targets, for multi-device setups. A '
                         'target is an http:// bridge URL (POST /execute) or the literal '
                         '"manual" (print the action and wait for the operator to press '
                         'Enter). When set, overrides --tb_url; PEFA routes each action '
                         'by class_name.')
    parser.add_argument('--tb_timeout_s', type=float, default=180.0,
                    help='Per-action HTTP timeout for the TurtleBot bridge.')
    parser.add_argument('--task', type=int, default=[0], nargs='+',
                    help='Specifies the ID(s) of the task(s) to run (0-19 for the benchmark envs).')
    parser.add_argument('--source', default='gemini', choices=['huggingface', 'gemini'], 
                    help='gemini API or load huggingface models')
    parser.add_argument('--lm_id', default='gemini-2.5-flash',
                    help='name for gemini model or huggingface model name/path')
    parser.add_argument('--debug', action='store_true', default=False,
                    help='debugging mode')
    parser.add_argument('--oracle_prompt_path', default="./prompt/oracle_prompt.txt" ,
                    help='path of oracle_prompt')
    parser.add_argument('--quadrotor_prompt_path', default="./prompt/quadrotor_prompt.txt",
                    help='path of quadrotor_prompt')
    parser.add_argument('--robot_dog_prompt_path', default="./prompt/robot_dog_prompt.txt",
                    help='path of robot_dog_prompt')
    parser.add_argument('--robot_arm_prompt_path', default="./prompt/robot_arm_prompt.txt",
                    help='path of robot_arm_prompt')
    parser.add_argument('--judge_prompt_path', default="./prompt/judge_prompt.txt",
                    help='path of judge_prompt')
    parser.add_argument('--api_key', default=None,
                    help='Gemini API key. Defaults to $GEMINI_API_KEY / $GOOGLE_API_KEY.')
    parser.add_argument('--arm', default='pefa',
                    help='Arm label recorded in the result JSON.')
    parser.add_argument('--seed', type=int, default=0,
                    help='Repeat index; recorded for paired statistics.')
    parser.add_argument('--record_path', default=None,
                    help='Where to write the per-task result JSON. Omit to skip recording.')
    parser.add_argument('--event_log', default=None,
                    help='Write a stage-event JSONL stream here, for the demo '
                         'video compositor. Defaults to $COHERENT_EVENT_LOG; '
                         'omit both to disable.')
    parser.add_argument('--organization', default='', 
                    help='not used for gemini, kept for compatibility')
    parser.add_argument("--t", default=0.5, type=float)

    parser.add_argument("--top_p", default=1.0, type=float)

    parser.add_argument("--max_tokens", default=512, type=int)

    parser.add_argument("--n", default=1, type=int)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()