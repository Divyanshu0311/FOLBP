import copy
import numpy as np
from tqdm import tqdm
import time
import json
from google import genai
from google.genai import types
import backoff

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bench.eventlog import EVENTS
from bench.llm_meter import METER
import traceback
import re as _re

# @ray.remote
class ArenaMP(object):
    def __init__(self, environment_fn, agent_fn, args, run_predefined_actions=False):
        self.env_fn = environment_fn
        self.agents = agent_fn
        self.args = args
        self.num_agents = len(agent_fn)
        self.task_goal = None
        self.record_dir = f'./log/{args.env}.txt'
        self.debug = args.debug
        print("Init Env")
        self.env = environment_fn()
        self.run_predefined_actions = run_predefined_actions

        self.oracle_prompt_path = args.oracle_prompt_path
        self.quadrotor_prompt_path = args.quadrotor_prompt_path
        self.robot_dog_prompt_path = args.robot_dog_prompt_path
        self.robot_arm_prompt_path = args.robot_arm_prompt_path

        self.dialogue_history = ""
        self.total_dialogue_history = []
        self.step_action_log = []  # tracks (step, agent, action) for post-run summary
        self.chat = True
        self.source = args.source
        self.lm_id = args.lm_id
        self.device = None
        self.sampling_parameters = None
        self.total_cost = 0

        self.last_done = False
        self.last_task_results = None
        self.last_satisfied = None
        self.last_unsatisfied = None
        self.costdict = {}

        # External execution bridges. Exactly one of (ws, tb) is active depending on mode.
        # 'standalone' runs only the symbolic planner, no physical/simulated execution.
        self.ws = None
        self.name_mapping = {}
        self.tb_url = None
        self.tb_session = None
        self.tb_targets = {}
        self.tb_default_target = None
        self.tb_timeout_s = float(getattr(args, 'tb_timeout_s', 180.0))
        # Execution timing. manual_wait_s is the operator's own reaction time and is
        # tracked apart from bridge_s so a run with hand-confirmed steps can still
        # report a device-only execution figure.
        self.bridge_s = 0.0
        self.manual_wait_s = 0.0
        self.step_timings = []
        self.mode = getattr(args, 'mode', 'standalone')

        if self.mode == 'ws':
            ws_url = getattr(args, 'ws_url', None)
            if not ws_url:
                raise ValueError("--mode=ws requires --ws_url to be set (e.g. ws://host:port)")
            import websocket
            self.ws = websocket.create_connection(ws_url)
            print(f"[mode=ws] Connected to OmniGibson at {ws_url}")
            env_name = getattr(args, 'env', None)
            if env_name:
                with open(f'./env/{env_name}.json') as f:
                    env_data = json.load(f)
                task_ids = getattr(args, 'task', [0])
                self.name_mapping = env_data[task_ids[0]].get('name_mapping', {})
        elif self.mode == 'tb':
            import requests
            self.tb_session = requests.Session()
            # Build a class_name -> URL map. Two sources: a JSON file (preferred for
            # multi-device) or the single --tb_url fallback used when the map is empty
            # / has no entry for the requested class.
            # A target is routed per acting agent class and may be an http:// bridge
            # URL (POST /execute) or the literal "manual" — print the action and wait
            # for an operator to press Enter. 'manual' covers a device with no bridge
            # yet; the run blocks until a human confirms the action happened.
            tb_urls_path = getattr(args, 'tb_urls', None)
            if tb_urls_path:
                with open(tb_urls_path) as f:
                    raw = json.load(f)
                # Skip metadata keys (anything starting with '_', e.g. "_comment").
                self.tb_targets = {
                    k.strip().lower(): self._classify_target(v)
                    for k, v in raw.items()
                    if not k.startswith('_') and isinstance(v, str)
                }
                print(f"[mode=tb] Device map: "
                      f"{ {k: f'{t[0]}:{t[1]}' for k, t in self.tb_targets.items()} }")
            tb_url = getattr(args, 'tb_url', None)
            if tb_url and not self.tb_targets:
                self.tb_default_target = self._classify_target(tb_url)
            if not self.tb_targets and not self.tb_default_target:
                raise ValueError("--mode=tb requires --tb_url or --tb_urls")

            # Probe each HTTP endpoint; warn but don't bail — bots may come up later.
            # manual targets need no probe.
            probe_targets = list(self.tb_targets.items())
            if self.tb_default_target and not self.tb_targets:
                probe_targets.append(('default', self.tb_default_target))
            for label, (kind, target) in probe_targets:
                if kind != 'http':
                    print(f"[mode=tb] {label}: {kind} -> {target}")
                    continue
                try:
                    resp = self.tb_session.get(f"{target}/health", timeout=3.0)
                    print(f"[mode=tb] /health {label} {target}: {resp.json()}")
                except Exception as e:
                    print(f"[mode=tb] WARNING /health probe failed for {label} {target}: {e}")
        else:
            print("[mode=standalone] Running symbolic planner only (no execution bridge)")

        # Gemini client + sampling params
        self.client = None
        if "gemini" in str(self.source).lower():
            api_key = getattr(args, "api_key", None)
            self.client = genai.Client(api_key=api_key) if api_key else genai.Client()
            if self.chat:
                # Use names compatible with generation_config
                self.sampling_params = {
                    "max_output_tokens": getattr(args, "max_tokens", 512),
                    "temperature": getattr(args, "t", 0.0),
                    "candidate_count": getattr(args, "n", 1)
                }
        else:
            # If you want to support other sources, extend here.
            raise ValueError("Only 'gemini' source supported by this wrapper.")

        # generator function bound to self
        @backoff.on_exception(backoff.expo, Exception)
        def _generate(prompt_messages, sampling_params):
            """
            prompt_messages: list of {"role": "...", "content": "..."}
            sampling_params: dict with keys max_output_tokens, temperature, candidate_count
            returns: (outputs:list[str], usage_est:float, prompt_tokens_est:int, completion_tokens_est:int)
            """
            print(">>> GEMINI REQUEST")
            gemini_prompt = ""
            for msg in prompt_messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                gemini_prompt += f"{role.upper()}: {content}\n"

            try:
                with METER.timed('oracle') as _call:
                    response = self.client.models.generate_content(
                        model=self.lm_id,
                        contents=gemini_prompt,
                        config=types.GenerateContentConfig(
                            # max_output_tokens=sampling_params.get("max_output_tokens", 2048),
                            temperature=sampling_params.get("temperature", 0),
                            candidate_count=sampling_params.get("candidate_count", 1),
                        )
                    )
                    _call.observe(response)
            except Exception as e:
                if self.debug:
                    print("genai.generate_content error:", e)
                raise

            outputs = []

            if hasattr(response, "candidates") and response.candidates:
                for c in response.candidates:
                    if hasattr(c, "text") and c.text:
                        outputs.append(c.text)
                    elif hasattr(c, "content") and hasattr(c.content, "parts"):
                        text = "".join(
                            p.text for p in c.content.parts if hasattr(p, "text")
                        )
                        outputs.append(text)
                    else:
                        outputs.append(str(c))
            else:
                if hasattr(response, "text") and response.text:
                    outputs.append(response.text)
                else:
                    outputs.append(str(response))

            # heuristic token / cost estimate
            prompt_tokens = int(len(gemini_prompt.split()) * 1.3)
            completion_tokens = int(sum(len(o.split()) for o in outputs) * 1.3)
            # conservative price estimate (adjust to real pricing)
            if "flash" in self.lm_id:
                usage = prompt_tokens * 0.075 / 1_000_000 + completion_tokens * 0.3 / 1_000_000
            elif "pro" in self.lm_id:
                usage = prompt_tokens * 1.25 / 1_000_000 + completion_tokens * 5.0 / 1_000_000
            else:
                usage = (prompt_tokens + completion_tokens) * 0.0000001

            if self.debug:
                with open("./chat_raw.json", "a") as f:
                    f.write(json.dumps({"prompt": gemini_prompt, "response": outputs}, indent=2, ensure_ascii=False))
                    f.write("\n")

            return outputs, usage, prompt_tokens, completion_tokens

        self.generator = _generate

    def get_actions(self, obs, chat_agent_info):
        # return the first matching agent's action result
        for idx, agent in enumerate(self.agents):
            if agent.agent_node["id"] == chat_agent_info["id"]:
                return agent.get_action(obs[idx], chat_agent_info, self.env.task_goal)
        # if not found, return None tuple
        return None, None, None

    def agent_obs2text(self, observation, agent_id):
        text = ""
        observation = observation[agent_id]

        id2node = {node["id"]: node for node in observation["nodes"]}
        agent_class = id2node[int(self.env.id_name_dict[agent_id][1])]["class_name"]
        with_quadrotor_id = None
        for node in observation["nodes"]:
            if node["category"] == "Agents" and self.env.id_name_dict[agent_id][1] == node["id"]:
                text += "I am <" + node["class_name"] + ">(" + str(node["id"]) + "). "
                if len(node.get("states", [])) != 0:
                    states = ", ".join(node["states"])
                    text += "Now my state is: " + states + ". "
                for edge in observation.get("edges", []):
                    if edge["from_id"] == node["id"]:
                        text += "I am " + edge["relation_type"] + " the <" + id2node[edge["to_id"]]["class_name"] + ">(" + str(edge["to_id"]) + "). "
                    if edge["relation_type"] == "WITH":
                        with_quadrotor_id = edge["to_id"]
                text += "\n"

        for node in observation["nodes"]:
            if node["category"] == "Rooms" and node["id"] == observation.get("agent_in_room_id"):
                text += "Now I am in the <" + node["class_name"] + ">(" + str(node["id"]) + "). In this room, I can see : \n"
        for node in observation["nodes"]:
            if node["id"] != self.env.id_name_dict[agent_id][1] and node["category"] != "Rooms":
                text += "<" + node["class_name"] + ">(" + str(node["id"]) + "). "
                if len(node.get("properties", [])) != 0:
                    properties = ", ".join(node["properties"])
                    text += "Its properties are: " + properties + ". "
                if len(node.get("states", [])) != 0:
                    states = ", ".join(node["states"])
                    text += "Now its state is: " + states + ". \n"
                else:
                    text += "\n"
        text += "These objects have a certain position relationship with each other: \n"
        for node in observation["nodes"]:
            if node["id"] != self.env.id_name_dict[agent_id][1] and node["category"] != "Rooms":
                for edge in observation.get("edges", []):
                    if edge["from_id"] == node["id"]:
                        if edge["from_id"] == with_quadrotor_id and agent_class in ("quadrotor", "drone"):
                            text += "The <" + node["class_name"] + ">(" + str(node["id"]) + ") is with me LAND " + edge["relation_type"] + " the <" + id2node[edge["to_id"]]["class_name"] + ">(" + str(edge["to_id"]) + "). \n"
                        elif edge["relation_type"] == "LEADING TO":
                            text += "The <" + node["class_name"] + ">(" + str(node["id"]) + ") is " + edge["relation_type"] + " the <" + id2node[edge["to_id"]]["class_name"] + ">(" + str(edge["to_id"]) + "). \n"
                        else:
                            text += "The <" + node["class_name"] + ">(" + str(node["id"]) + ") is " + edge["relation_type"] + " the <" + id2node[edge["to_id"]]["class_name"] + ">(" + str(edge["to_id"]) + "). \n"
        for edge in observation.get("edges", []):
            if edge["relation_type"] == "WITH" and agent_class in ("quadrotor", "drone"):
                in_basket = False
                text += "I have a <" + id2node[edge["to_id"]]["class_name"] + ">(" + str(edge["to_id"]) + ") with me. "
                for edges in observation.get("edges", []):
                    if edges["to_id"] == edge["to_id"] and edges["relation_type"] == "INSIDE":
                        text += "<" + id2node[edges["from_id"]]["class_name"] + ">(" + str(edges["from_id"]) + ") is in my <" + id2node[edge["to_id"]]["class_name"] + ">(" + str(edge["to_id"]) + "). \n"
                        in_basket = True
                if not in_basket:
                    text += "But nothing is in my <" + id2node[edge["to_id"]]["class_name"] + ">(" + str(edge["to_id"]) + "). \n"
            if edge["relation_type"] == "HOLD" and agent_class not in ("quadrotor", "drone"):
                text += "I am holding a <" + id2node[edge["to_id"]]["class_name"] + ">(" + str(edge["to_id"]) + ") in my hand. \n"
        return text

    def write_log_to_file(self, log_message, file_name=None):
        file_name = self.record_dir
        with open(file_name, "a") as file:
            file.write(log_message + "\n")

    def step(self):
        if self.env.steps == 0:
            pass

        obs = self.env.get_observations()
        id_name_dict = self.env.id_name_dict

        obs2text = ""
        for i in range(self.num_agents):
            obs2text += self.agent_obs2text(obs, i) + "\n"

        # prepare oracle prompt
        oracle_prompt = self.oracle_prompt_path
        with open(oracle_prompt, "r") as f:
            oracle_prompt = f.read()
        oracle_prompt = oracle_prompt.replace("#AGENT_OBSERVATIONS#", obs2text)
        oracle_prompt = oracle_prompt.replace("#TASK_GOAL#", self.env.goal_instruction)
        oracle_prompt = oracle_prompt.replace("#NUMBER_AGENTS#", str(self.env.num_agent))
        oracle_prompt = oracle_prompt.replace("#DIALOGUE_HISTORY#", self.dialogue_history)
        if self.debug:
            print(self.dialogue_history)

        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        print(f"@@@@@@@@@@@@@@@@@@@@@@@@ Task_ID: {self.env.task_id} @@@@@@@@@@@")
        print(f"$$$$$$$$$$$$$$$$$$$$$$$ Step:{self.env.steps} $$$$$$$$$$$$$$$$$$$$$$$")
        print(self.env.goal_instruction)
        EVENTS.emit('STEP', title=f'step {self.env.steps}', step=self.env.steps)
        _t_oracle = time.time()

        # call oracle
        chat_prompt = [{"role": "user", "content": oracle_prompt}]
        outputs, usage, in_tokens, out_tokens = self.generator(chat_prompt, self.sampling_params)
        self.total_cost += usage
        message = outputs[0] if outputs else ""

        self.write_log_to_file(f"@@@@@@@@@@@@@@@@@@@@@@@ Task_ID: {self.env.task_id} @@@@@@@@@@@")
        self.write_log_to_file(f"$$$$$$$$$$$$$$$$$$$$$$$ Step:{self.env.steps} $$$$$$$$$$$$$$$$$$$$$$$")
        self.write_log_to_file(f"*******************************************************************************************\n                               TASK_GOAL: {self.env.goal_instruction}\n                               ")
        self.write_log_to_file("OBSERVATIONS: \n" + obs2text)
        self.write_log_to_file("Oracle: " + message)

        self.total_dialogue_history.append("Oracle: " + message)
        EVENTS.emit('ORACLE', body=message, step=self.env.steps,
                    note=f'{time.time() - _t_oracle:.1f}s')

        # extract instruction in strict format (transform/normalize)
        extract_instruction_prompt = (
            message
            + "\n"
            + 'Extract from the above paragraph the content of the format "Hello <class name>(id): message.". Then output the contents of this section. Be careful not to output any superfluous content, exactly in the format given. If the above paragraph is not exactly formatted as "Hello <class name>(id): #message#.", output similar content in this format. As an example, the output might read: "Hello <robot dog>(0): please movetowards the <door>(1), and then open the <door>(1)". If this format does not appear in the preceding text, please summarize the above content into this format for output. To emphasize once again, the names of all objects and agent robots must be enclosed in <>, and the (id) must not be omitted. Class name missing <> and (id) should be completed with these elements. Please strictly follow this format in the output content.'
        )
        chat_prompt = [{"role": "user", "content": extract_instruction_prompt}]
        outputs, usage, in_tokens2, out_tokens2 = self.generator(chat_prompt, self.sampling_params)
        self.total_cost += usage
        message = outputs[0] if outputs else ""
        self.subgoal = message

        self.write_log_to_file("Oracle (normalized): " + message)
        EVENTS.emit('ORACLE_NORM', body=message, step=self.env.steps)

        if self.debug:
            print(f"message_oracle_prompt:\n{oracle_prompt}")
            print("\n")
            print(f"message_oracle_outputs:\n{message}")

        # prepare defaults in case parsing fails
        agent_action = None
        agent_message = None
        id_list = [None]

        try:
            start_class_name = message.find("<") + 1
            end_class_name = message.find(">")
            start_id = message.find("(") + 1
            end_id = message.find(")")

            class_name = message[start_class_name:end_class_name]
            real_id = int(message[start_id:end_id])
            id_list = [key for key, value in id_name_dict.items() if value[1] == real_id]
            if len(id_list) == 0:
                raise ValueError(f"Could not map real_id {real_id} to internal agent index")

            agent_idx = id_list[0]
            agent_obs = self.agent_obs2text(obs, agent_idx)

            # decide prompt path based on class_name
            # "drone" is treated as an alias of "quadrotor" — same flying-robot prompt.
            if class_name in ("quadrotor", "drone"):
                prompt_path = self.quadrotor_prompt_path
            elif class_name in ("robot dog", "robot_dog"):
                prompt_path = self.robot_dog_prompt_path
            elif class_name in ("robot arm", "robot_arm"):
                prompt_path = self.robot_arm_prompt_path
            else:
                prompt_path = None

            chat_agent_info = {
                "class_name": class_name,
                "id": real_id,
                "observation": agent_obs,
                "instruction": message,
                "prompt_path": prompt_path,
            }

            agent_action, agent_message, agent_info = self.get_actions(obs, chat_agent_info)
            # log LLM action list if available
            if agent_info and "LLM" in agent_info and "action_list" in agent_info["LLM"]:
                self.write_log_to_file(str(agent_info["LLM"]["action_list"]))

            self.write_log_to_file(f"<{class_name}>({real_id}): " + str(agent_message))

            # cost bookkeeping (agent_info may supply its own cost estimate)
            if agent_info and "LLM" in agent_info:
                self.costdict = self.update_dict(f"<{class_name}>({real_id})", agent_info["LLM"].get("cost", 0), self.costdict)

            self.write_log_to_file(f"COST1:{self.total_cost}!!!!!")
            self.write_log_to_file(str(self.costdict))
            self.write_log_to_file(f"COST2:{sum(self.costdict.values())}!!!!!")
            self.write_log_to_file(f'Total_cost: {self.total_cost + sum(self.costdict.values())}')
            self.write_log_to_file("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ ")

            self.total_dialogue_history.append(f"<{class_name}>({real_id}): " + str(agent_message))
            numbered_list = [f"[{i+1}]、{item}" for i, item in enumerate(self.total_dialogue_history)]
            self.dialogue_history = "\n".join(numbered_list[-10:])
        except Exception as e:
            print(f"An error occurred while parsing oracle output or getting agent action: {e}")
            traceback.print_exc()
            error_info = traceback.format_exc()
            self.write_log_to_file(f"An error occurred: {e}")
            self.write_log_to_file(error_info + "\n\n")
            agent_action = None
            agent_message = "all robot agents: In the last step, the oracle's reasoning was incorrect, and no instructions were given to any of the robot agents, therefore none of the robot agents performed any actions. Please reassess the information in the environment and give a correct instruction strictly following the template 'Hello <class name>(id): #message#.'"
            self.total_dialogue_history.append(agent_message)
            numbered_list = [f"[{i+1}]、{item}" for i, item in enumerate(self.total_dialogue_history)]
            self.dialogue_history = "\n".join(numbered_list[-10:])

        # Execute or skip action
        if agent_action is None:
            done = self.last_done
            task_results = self.last_task_results
            satisfied = self.last_satisfied
            unsatisfied = self.last_unsatisfied
            # NOTE: do NOT increment self.env.steps for null actions —
            # only real executed actions should count toward the budget.
        else:
            try:
                # Default to success when no execution bridge is configured
                # (standalone mode just runs the symbolic planner).
                sim_success = True
                bridge_label = None  # set when an external bridge runs

                # ---- WebSocket bridge to OmniGibson ----
                if self.ws:
                    bridge_label = "WS"
                    og_action = agent_action
                    for pefa_key, og_name in self.name_mapping.items():
                        pefa_in_action = f"<{pefa_key.rsplit('(', 1)[0]}>({pefa_key.rsplit('(', 1)[1]}"
                        if pefa_in_action in og_action:
                            og_action = og_action.replace(pefa_in_action, og_name)
                    og_agent = self.name_mapping.get(f"{class_name}({real_id})", class_name)
                    msg = json.dumps({"agent": og_agent, "action": og_action})
                    print(f"[WS SEND] {msg}")
                    EVENTS.emit('WS_SEND', title=og_agent, body=og_action)
                    self.ws.send(msg)
                    result_raw = self.ws.recv()
                    print(f"[WS RECV] {result_raw}")
                    self.write_log_to_file(f"[WS SEND] {msg}")
                    self.write_log_to_file(f"[WS RECV] {result_raw}")
                    try:
                        result_obj = json.loads(result_raw)
                    except Exception:
                        result_obj = {"success": False, "info": f"bad_json_from_server: {result_raw[:200]}"}
                    sim_success = result_obj.get("success", False)
                    EVENTS.emit('WS_RECV', title='ok' if sim_success else 'rejected',
                                body=str(result_obj.get('info', '')),
                                success=bool(sim_success))

                # ---- Device bridge: HTTP to web_client.py, or a manual confirm ----
                elif self.tb_targets or self.tb_default_target:
                    bridge_label = "TB"
                    # Resolve which device owns this action. The map is keyed by
                    # class_name (lowercased). Fall back to --tb_url only when the map
                    # is empty (single-device mode).
                    target = self.tb_targets.get(class_name.strip().lower())
                    if target is None and not self.tb_targets:
                        target = self.tb_default_target
                    agent_label = f"{class_name}({real_id})"
                    _t_step = time.time()
                    kind = target[0] if target else None
                    if target is None:
                        result_obj = {
                            "success": False,
                            "info": f"no device target registered for agent class {class_name!r}",
                        }
                    elif kind == 'manual':
                        result_obj = self._manual_confirm(agent_label, agent_action)
                    else:
                        target_url = target[1]
                        payload = {"agent": agent_label, "action": agent_action}
                        print(f"[TB SEND -> {target_url}] {payload}")
                        self.write_log_to_file(f"[TB SEND -> {target_url}] {json.dumps(payload)}")
                        try:
                            resp = self.tb_session.post(
                                f"{target_url}/execute",
                                json=payload,
                                timeout=self.tb_timeout_s,
                            )
                            result_obj = resp.json()
                        except Exception as e:
                            result_obj = {"success": False, "info": f"http_error: {e}"}
                    _dt = time.time() - _t_step
                    if kind == 'manual':
                        self.manual_wait_s += _dt
                    else:
                        self.bridge_s += _dt
                    self.step_timings.append({
                        "agent": agent_label, "action": agent_action,
                        "transport": kind or "unrouted", "seconds": round(_dt, 2),
                    })
                    print(f"[TB RECV] {result_obj}  ({_dt:.1f}s, {kind or 'unrouted'})")
                    self.write_log_to_file(
                        f"[TB RECV] {json.dumps(result_obj)} ({_dt:.2f}s, {kind or 'unrouted'})")
                    sim_success = bool(result_obj.get("success", False))

                # If a bridge ran and reported failure, log it and let the LLM retry.
                if bridge_label and not sim_success:
                    err = (f"[{bridge_label}] Bridge reported FAILURE for action "
                           f"'{agent_action}': {result_obj.get('info')}")
                    print(err)
                    self.write_log_to_file(err)
                    self.total_dialogue_history.append(
                        f"<{class_name}>({real_id}): The execution bridge rejected "
                        f"'{agent_action}' (reason: {result_obj.get('info')}). "
                        f"I must choose a different action."
                    )

                # Only update the graph state if the action succeeded
                if sim_success:
                    done, task_results, satisfied, unsatisfied, steps = self.env.step(class_name, real_id, agent_action, self.task_goal)
                    self.last_done = done
                    self.last_task_results = task_results
                    self.last_satisfied = satisfied
                    self.last_unsatisfied = unsatisfied
                    self.step_action_log.append({
                        "step": self.env.steps,
                        "agent": f"<{class_name}>({real_id})",
                        "action": agent_action,
                    })
                    EVENTS.emit(
                        'ENV',
                        body=('goal reached' if done else
                              'still to satisfy:  ' + str(list(unsatisfied.keys())
                                                          if unsatisfied else [])),
                        step=self.env.steps, done=bool(done))
                else:
                    done = self.last_done
                    task_results = self.last_task_results
                    satisfied = self.last_satisfied
                    unsatisfied = self.last_unsatisfied
            except Exception as e:
                print("Exception occurs when performing action: ", agent_action)
                traceback.print_exc()
                raise

        self.write_log_to_file(f"\nDIALOGUE_HISTORY:\n{self.dialogue_history}")
        steps = self.env.steps
        return (done, task_results, satisfied, unsatisfied, id_list, agent_action, agent_message, steps)

    @staticmethod
    def _classify_target(value):
        """'http://h:p' -> ('http', url); 'manual' -> ('manual', 'operator')."""
        v = value.strip()
        low = v.lower()
        if low in ('manual', 'human', 'operator'):
            return ('manual', 'operator')
        if low.startswith(('http://', 'https://')):
            return ('http', v.rstrip('/'))
        raise ValueError(f"unrecognised device target {value!r} — expected an "
                         f"http:// URL or the literal \"manual\"")

    def _manual_confirm(self, agent_label, action):
        """Print the action and block until the operator reports the outcome.

        Enter means it happened, so the planner advances. 'f <reason>' is a genuine
        failure report and reaches the retry path exactly as a bridge rejection
        would, which is the only way a hand-driven device can push back. 'a' aborts.
        """
        self.write_log_to_file(f"[TB MANUAL] {agent_label} {action}")
        EVENTS.emit('WS_SEND', title=agent_label, body=action)
        banner = ('\n' + '=' * 68 + '\n'
                  f'  MANUAL STEP — {agent_label}\n'
                  f'  {action}\n' + '=' * 68 + '\n'
                  '  [Enter] done   |   f <reason> failed   |   a abort\n> ')
        try:
            reply = input(banner).strip()
        except EOFError:
            # No operator attached (piped stdin, nohup). Blocking forever or silently
            # passing would both be worse than saying so.
            raise RuntimeError('manual transport needs an interactive terminal '
                               '(stdin is closed) — run it in a real tty')
        except KeyboardInterrupt:
            raise RuntimeError('manual transport: aborted by operator')

        low = reply.lower()
        if low in ('a', 'abort', 'q', 'quit'):
            raise RuntimeError('manual transport: aborted by operator')
        if low.startswith('f'):
            reason = reply[1:].strip(' :') or 'operator reported failure'
            return {"success": False, "info": f"manual: {reason}"}
        return {"success": True, "info": "manual: operator confirmed"}

    def run(self):
        self.task_goal = copy.deepcopy(self.env.task_goal)
        EVENTS.emit('TASK', title=self.env.task_name,
                    body=self.env.goal_instruction,
                    framework='pefa', task_id=self.env.task_id,
                    env_id=self.env.env_id,
                    gt_steps=self.env.ground_truth_step_num, mode=self.mode)
        saved_info = []
        total_iterations = 0  # counts ALL iterations (including null actions)

        success = False
        while True:
            total_iterations += 1
            done, task_results, satisfied, unsatisfied, id_list, agent_action, agent_message, steps = self.step()
            saved_info.append(
                {
                    "task_id": self.env.task_id,
                    "env_id": self.env.env_id,
                    "task_name": self.env.task_name,
                    "gt_steps": self.env.ground_truth_step_num,
                    "task_goal": self.task_goal,
                    "goal_instruction": self.env.goal_instruction,
                    "step": steps,
                    "subgoal": getattr(self, "subgoal", None),
                    "agent_id": id_list[0] if id_list else None,
                    "action": agent_action,
                    "agent_message": agent_message,
                    "satisfied": satisfied,
                    "unsatisfied": unsatisfied,
                    "env_graph": self.env.graph,
                }
            )

            success = done

            max_setp = 2 * self.env.ground_truth_step_num
            if self.env.steps > max_setp or total_iterations > max_setp * 2:
                print("---------------------------")
                print("The task failed, exceeding 2 times the number of GT steps")
                print(f"Whether steps in gt*2+1 are successful:{done}")
                print(f" setps: {steps}")
                print("---------------------------")
                self.write_log_to_file(
                    """---------------------------
                                       The task failed, exceeding 2 times the number of GT steps
                                       Whether steps in gt*2+1 are successful:{}
                                       setps: {}
                                       ---------------------------""".format(done, steps)
                )
                success = False
                break

            if success:
                self.write_log_to_file(
                    """-------------------------------------
                                            success!
                                            setps: {}
                                            --------------------------------""".format(
                        steps
                    )
                )
                break

        # Log step-by-step action summary
        self.write_log_to_file("\n=== Step-by-Step Action Log ===")
        print("\n=== Step-by-Step Action Log ===")
        for entry in self.step_action_log:
            line = f"  Step {entry['step']:>2}: {entry['agent']} -> {entry['action']}"
            self.write_log_to_file(line)
            print(line)
        self.write_log_to_file(f"  Total steps executed: {len(self.step_action_log)}")
        print(f"  Total steps executed: {len(self.step_action_log)}")

        # Execution timing. Device time and operator time are reported apart: a run
        # with hand-confirmed steps has human reaction time in its wall clock, and
        # quoting that as execution time would be meaningless.
        if self.step_timings:
            self.write_log_to_file("\n=== Execution Timing ===")
            print("\n=== Execution Timing ===")
            for e in self.step_timings:
                line = (f"  {e['seconds']:>7.2f}s  {e['transport']:<7} "
                        f"{e['agent']} -> {e['action']}")
                self.write_log_to_file(line)
                print(line)
            total = self.bridge_s + self.manual_wait_s
            summary = (f"  device {self.bridge_s:.2f}s | operator {self.manual_wait_s:.2f}s "
                       f"| total {total:.2f}s")
            self.write_log_to_file(summary)
            print(summary)

        if saved_info:
            saved_info[steps - 1]["is_finished"] = success
        return success, steps, saved_info

    def update_dict(self, key, value, my_dict):
        my_dict[key] = value
        return my_dict