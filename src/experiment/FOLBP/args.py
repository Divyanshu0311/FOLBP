import argparse
import os


def get_args():
    parser = argparse.ArgumentParser(
        description='FOLBP — First-Order Logic + Backtrack Planner (heterogeneous multi-robot)'
    )

    parser.add_argument('--env', type=str, default='env0',
                        choices=['env0', 'env1', 'env2', 'env3', 'env4',
                                 'env_merom', 'env_house', 'env_tb'],
                        help='Simulation environment. env0-env4 are the COHERENT '
                             'benchmark environments; env_merom and env_house are '
                             'the two OmniGibson demo scenes, used with --mode ws; '
                             'env_tb is the real-robot demo scene (TurtleBot3 with '
                             'an OpenMANIPULATOR-X arm, plus a basket-carrying drone).')
    parser.add_argument('--task', type=int, default=[2], nargs='+',
                        help='Task ID(s) to run.')

    # ------------------------------------------------------------- execution bridge
    # Mirrors PEFA/args.py. 'standalone' is the benchmark code path and the default;
    # nothing in results/bench/ is affected by these existing.
    parser.add_argument('--mode', type=str, default='standalone',
                        choices=['standalone', 'ws', 'tb'],
                        help="Execution mode: 'standalone' runs only the symbolic "
                             "planner (no simulator). 'ws' forwards each executed "
                             "action to OmniGibson via WebSocket (requires --ws_url). "
                             "'tb' forwards each executed action to physical devices "
                             "over HTTP (uses --tb_urls if provided, else --tb_url).")
    parser.add_argument('--ws_url', type=str,
                        default=os.environ.get('COHERENT_WS_URL', 'ws://127.0.0.1:8765'),
                        help='WebSocket URL of the OmniGibson bridge. Used when '
                             '--mode=ws. Override with $COHERENT_WS_URL.')
    parser.add_argument('--tb_url', type=str,
                        default=os.environ.get('COHERENT_TB_URL', 'http://127.0.0.1:8080'),
                        help='HTTP URL of a single device bridge (web_client.py), e.g. '
                             'http://<bot-ip>:8080. Used when --tb_urls is not set. '
                             'Override with $COHERENT_TB_URL.')
    parser.add_argument('--tb_urls', type=str, default=None,
                        help='Path to a JSON file mapping agent class names (lowercase, '
                             'e.g. "robot dog", "drone") to device targets, for '
                             'multi-device setups. A target is an http:// bridge URL '
                             '(POST /execute), a ws:// bridge URL (one JSON frame each '
                             'way), or the literal "manual" (print the action and wait '
                             'for the operator to press Enter). When set, overrides '
                             '--tb_url; each action is routed by the acting agent class.')
    parser.add_argument('--tb_timeout_s', type=float, default=180.0,
                        help='Per-action HTTP timeout for the device bridge.')

    parser.add_argument('--source', default='gemini', choices=['gemini'],
                        help='LLM backend (only gemini supported).')
    parser.add_argument('--lm_id', default='gemini-2.5-flash',
                        help='Gemini model id.')
    parser.add_argument('--api_key', default=None,
                        help='Gemini API key. Defaults to $GEMINI_API_KEY / $GOOGLE_API_KEY.')

    parser.add_argument('--oracle_prompt_path', default="./prompt/oracle_prompt.txt")
    parser.add_argument('--quadrotor_prompt_path', default="./prompt/quadrotor_prompt.txt")
    parser.add_argument('--robot_dog_prompt_path', default="./prompt/robot_dog_prompt.txt")
    parser.add_argument('--robot_arm_prompt_path', default="./prompt/robot_arm_prompt.txt")

    parser.add_argument('--executor_mode', default='pefa_llm',
                        choices=['pefa_llm', 'deterministic'],
                        help='pefa_llm: 1 LLM call per step per agent (PEFA-style). '
                             'deterministic: execute the validated plan verbatim, no LLM in executor.')
    parser.add_argument('--max_replan_attempts', type=int, default=5,
                        help='How many times to invoke the UNSAT-Core Replanner before escalating to a full Oracle re-plan. '
                             'A not_holding cascade (drop -> walk back -> open -> walk -> grab -> walk -> puton) '
                             'can take 5-6 R passes, so default=5.')

    # ---------------------------------------------------------------- ablation axes
    # Orthogonal switches over the one pipeline — the benchmark arms are combinations
    # of these, never forked copies of the code (see src/experiment/bench/arms.py).
    parser.add_argument('--verify', default='z3', choices=['z3', 'none', 'shadow'],
                        help="z3: check preconditions and act on UNSAT (default). "
                             "none: skip the Z3 reasoner entirely. "
                             "shadow: run Z3 and log every UNSAT, but execute the plan "
                             "unmodified — measures what an unverified planner would "
                             "have dispatched to a robot.")
    parser.add_argument('--repair', default='symbolic', choices=['symbolic', 'llm', 'none'],
                        help="symbolic: repair from the UNSAT core with replanner.py (default). "
                             "llm: feed the unsat-core atom back to the Oracle as a re-prompt. "
                             "none: escalate to a full re-plan on the first UNSAT.")
    parser.add_argument('--cdcl', dest='enable_cdcl', action=argparse.BooleanOptionalAction,
                        default=True,
                        help='Enable the CDCL conflict-driven learning layer (A-D). '
                             'Use --no-cdcl to ablate it.')
    parser.add_argument('--nudge', dest='enable_nudge', action=argparse.BooleanOptionalAction,
                        default=False,
                        help='Content-free re-plan feedback, for the floor control of the '
                             'CDCL ablation. With --no-cdcl the Oracle is re-prompted with a '
                             'byte-identical prompt and returns the same plan, so the outer '
                             'loop contributes nothing; --nudge injects a fixed string saying '
                             'the last plan was rejected, carrying no core, no failed index '
                             'and no plan echo. Isolates the value of the *content* of the '
                             'feedback from the value of perturbing the prompt at all.')

    parser.add_argument('--max_oracle_plans', type=int, default=0,
                        help='Cap on full Oracle plan generations per task. 0 = unlimited '
                             '(default). 1 = strict one-shot: plan once, execute, never '
                             're-plan — used by the naive arm.')

    # ------------------------------------------------------------- benchmark harness
    parser.add_argument('--arm', default='folbp',
                        help='Arm label recorded in the result JSON.')
    parser.add_argument('--seed', type=int, default=0,
                        help='Repeat index; recorded for paired statistics.')
    parser.add_argument('--record_path', default=None,
                        help='Where to write the per-task result JSON. Omit to skip recording.')

    parser.add_argument('--bridge_retries', type=int, default=1,
                        help='How many times to re-send an action the simulator '
                             'never answered. A timeout means a controller hung, '
                             'not that the action is infeasible, so it is retried '
                             'rather than fed to the CDCL layer.')
    parser.add_argument('--event_log', default=None,
                        help='Write a stage-event JSONL stream here, for the demo '
                             'video compositor. Defaults to $COHERENT_EVENT_LOG; '
                             'omit both to disable.')

    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--t', default=0.5, type=float, help='Temperature.')
    parser.add_argument('--top_p', default=1.0, type=float)
    parser.add_argument('--max_tokens', default=2048, type=int)
    parser.add_argument('--n', default=1, type=int, help='Candidate count.')

    return parser.parse_args()


if __name__ == '__main__':
    get_args()
