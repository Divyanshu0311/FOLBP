"""Gemini client + per-robot action enumeration.

Owns two responsibilities:
  - generate(prompt) -> text (used by Oracle, Executor, Replanner)
  - get_available_plans(robot, state) -> action list (used by Plan Validator and Executor)

The action enumeration is the "capability dict (robot x action)" of the diagram:
quadrotor / robot_dog / robot_arm each get a different vocabulary, gated on state.
"""
import copy
import os
import re
import sys
from pathlib import Path

import backoff
from google import genai
from google.genai import types

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bench.llm_meter import METER


class LLM:
    def __init__(self, source, lm_id, args):
        self.args = args
        self.source = source
        self.lm_id = lm_id
        self.debug = args.debug
        self.record_dir = f'./log/{args.env}.txt'
        self.total_cost = 0.0

        api_key = args.api_key or os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY')
        self.client = genai.Client(api_key=api_key) if api_key else genai.Client()
        self.sampling_params = {
            'max_output_tokens': args.max_tokens,
            'temperature': args.t,
            'candidate_count': args.n,
        }

        @backoff.on_exception(backoff.expo, Exception, max_tries=4)
        def _generate(prompt_messages, sampling_params, role='executor'):
            text = '\n'.join(m['content'] for m in prompt_messages if m.get('role') in ('user', 'system'))
            with METER.timed(role) as call:
                response = self.client.models.generate_content(
                    model=self.lm_id,
                    contents=text,
                    config=types.GenerateContentConfig(
                        temperature=sampling_params.get('temperature', 0),
                        candidate_count=sampling_params.get('candidate_count', 1),
                    ),
                )
                call.observe(response)
            out = response.text or ''
            prompt_tokens = int(len(text.split()) * 1.3)
            completion_tokens = int(len(out.split()) * 1.3)
            if 'flash' in self.lm_id:
                usage = prompt_tokens * 0.075 / 1e6 + completion_tokens * 0.3 / 1e6
            elif 'pro' in self.lm_id:
                usage = prompt_tokens * 1.25 / 1e6 + completion_tokens * 5.0 / 1e6
            else:
                usage = (prompt_tokens + completion_tokens) * 1e-7
            return [out], usage

        self.generator = _generate

    def write_log(self, msg):
        with open(self.record_dir, 'a') as f:
            f.write(msg + '\n')

    def parse_answer(self, available_actions, text):
        """Loose match of LLM output back to one of the enumerated actions."""
        norm = text.replace('_', ' ').replace('takeoff from', 'takeoff_from').replace('land on', 'land_on')
        for action in available_actions:
            if action in norm:
                return action
        tokens = norm.split(' ')
        for i, action in enumerate(available_actions):
            opt = chr(ord('A') + i)
            if f'option {opt}' in norm or f'{opt}.' in tokens or f'{opt},' in tokens \
                    or f'Option {opt}' in norm or f'({opt})' in norm:
                return action
        low = norm.lower()
        for action in available_actions:
            m = re.match(r'\[(\w+)\]\s*<([^>]+)>\((\d+)\)', action)
            if not m:
                continue
            verb, cls, aid = m.group(1), m.group(2).lower(), m.group(3)
            verb_root = verb.split('_')[0]
            has_verb = (verb in low) or (verb.replace('_', ' ') in low) or (verb_root in low)
            has_target = (f'({aid})' in low) or (f' {aid}' in low) or (cls in low)
            if has_verb and has_target:
                return action
        return None

    def get_available_plans(self, agent_node, next_rooms, all_landable_surfaces,
                            landable_surfaces, on_surfaces, grabbed_objects,
                            reached_objects, unreached_objects, on_same_surface_objects):
        """The "capability dict (robot x action)" — branches on agent class and state."""
        available = []

        if agent_node['class_name'] in ('quadrotor', 'drone'):
            other_landable = []
            if 'FLYING' in agent_node['states']:
                if landable_surfaces is not None:
                    available.append(f"[land_on] <{landable_surfaces['class_name']}>({landable_surfaces['id']})")
                    all_landable_surfaces.remove(landable_surfaces)
                    other_landable = copy.deepcopy(all_landable_surfaces)
                for surface in other_landable:
                    available.append(f"[movetowards] <{surface['class_name']}>({surface['id']})")
                for next_room in next_rooms:
                    if 'OPEN' in next_room[1]['states'] or 'OPEN_FOREVER' in next_room[1]['states']:
                        available.append(f"[movetowards] <{next_room[0]['class_name']}>({next_room[0]['id']})")
            if 'LAND' in agent_node['states'] and on_surfaces is not None:
                available.append(f"[takeoff_from] <{on_surfaces['class_name']}>({on_surfaces['id']})")

        elif agent_node['class_name'] in ('robot dog', 'robot_dog'):
            for reached in reached_objects:
                if grabbed_objects is None:
                    if ('CONTAINERS' in reached['properties'] and 'CLOSED' in reached['states']) \
                            or (reached['class_name'] == 'door' and 'CLOSED' in reached['states']):
                        available.append(f"[open] <{reached['class_name']}>({reached['id']})")
                    if ('CONTAINERS' in reached['properties'] and 'OPEN' in reached['states']) \
                            or (reached['class_name'] == 'door' and 'OPEN' in reached['states']):
                        available.append(f"[close] <{reached['class_name']}>({reached['id']})")
                    if 'GRABABLE' in reached['properties']:
                        available.append(f"[grab] <{reached['class_name']}>({reached['id']})")
                else:
                    if 'CONTAINERS' in reached['properties'] and (
                            'OPEN' in reached['states'] or 'OPEN_FOREVER' in reached['states']):
                        available.append(
                            f"[putinto] <{grabbed_objects['class_name']}>({grabbed_objects['id']}) "
                            f"into <{reached['class_name']}>({reached['id']})"
                        )
                    if 'SURFACES' in reached['properties']:
                        available.append(
                            f"[puton] <{grabbed_objects['class_name']}>({grabbed_objects['id']}) "
                            f"on <{reached['class_name']}>({reached['id']})"
                        )
            for unreached in unreached_objects:
                available.append(f"[movetowards] <{unreached['class_name']}>({unreached['id']})")
            for next_room in next_rooms:
                if 'OPEN' in next_room[1]['states'] or 'OPEN_FOREVER' in next_room[1]['states']:
                    available.append(f"[movetowards] <{next_room[0]['class_name']}>({next_room[0]['id']})")

        elif agent_node['class_name'] in ('robot arm', 'robot_arm'):
            if grabbed_objects is not None and on_surfaces is not None:
                available.append(
                    f"[puton] <{grabbed_objects['class_name']}>({grabbed_objects['id']}) "
                    f"on <{on_surfaces['class_name']}>({on_surfaces['id']})"
                )
            for same in on_same_surface_objects:
                if grabbed_objects is None:
                    if 'CONTAINERS' in same['properties'] and 'OPEN' in same['states']:
                        available.append(f"[close] <{same['class_name']}>({same['id']})")
                    if 'CONTAINERS' in same['properties'] and 'CLOSED' in same['states']:
                        available.append(f"[open] <{same['class_name']}>({same['id']})")
                    if 'GRABABLE' in same['properties']:
                        available.append(f"[grab] <{same['class_name']}>({same['id']})")
                else:
                    if 'CONTAINERS' in same['properties'] and (
                            'OPEN' in same['states'] or 'OPEN_FOREVER' in same['states']):
                        available.append(
                            f"[putinto] <{grabbed_objects['class_name']}>({grabbed_objects['id']}) "
                            f"into <{same['class_name']}>({same['id']})"
                        )
                    if 'SURFACES' in same['properties']:
                        available.append(
                            f"[puton] <{grabbed_objects['class_name']}>({grabbed_objects['id']}) "
                            f"on <{same['class_name']}>({same['id']})"
                        )

        rendered = ''.join(f"{chr(ord('A') + i)}. {p}\n" for i, p in enumerate(available))
        return rendered, len(available), available

    def run(self, agent_node, chat_agent_info, current_room, next_rooms,
            all_landable_surfaces, landable_surfaces, on_surfaces,
            grabbed_objects, reached_objects, unreached_objects, on_same_surface_objects):
        """PEFA-style: 1 LLM call to pick the next action from the available set."""
        info = {
            'num_available_actions': None, 'plan': None,
            'action_list': None, 'cost': self.total_cost,
        }
        with open(chat_agent_info['prompt_path'], 'r') as f:
            agent_prompt = f.read()

        rendered, num, plans_list = self.get_available_plans(
            agent_node, next_rooms, all_landable_surfaces, landable_surfaces, on_surfaces,
            grabbed_objects, reached_objects, unreached_objects, on_same_surface_objects,
        )
        agent_prompt = agent_prompt.replace('#OBSERVATION#', chat_agent_info['observation'])
        agent_prompt = agent_prompt.replace('#ACTIONLIST#', rendered)
        agent_prompt = agent_prompt.replace('#INSTRUCTION#', chat_agent_info['instruction'])

        outputs, usage = self.generator(
            [{'role': 'user', 'content': agent_prompt}], self.sampling_params,
        )
        self.total_cost += usage
        output = outputs[0]
        self.write_log(output)

        upper = output.upper()
        yes_can = 'YES I CAN' in upper
        sorry = 'SORRY I CANNOT' in upper
        if yes_can and sorry:
            yes_can = upper.rfind('YES I CAN') > upper.rfind('SORRY I CANNOT')
            sorry = not yes_can

        message = f'No action selected. Raw: {output[:200]}'
        plan = None
        if yes_can:
            follow_up = [
                {'role': 'user', 'content': agent_prompt},
                {'role': 'assistant', 'content': output},
                {'role': 'user', 'content': 'Answer with only one best next action in the list of available actions. So the answer is'},
            ]
            outputs, usage = self.generator(follow_up, self.sampling_params)
            self.total_cost += usage
            plan = self.parse_answer(plans_list, outputs[0])
            if plan:
                message = f'The action I finally decided to perform is {plan}.'
        if plan is None and num > 0:
            plan = self.parse_answer(plans_list, output)
            if plan:
                message = f'The action I finally decided to perform is {plan}.'

        info.update({
            'num_available_actions': num, 'plan': plan,
            'action_list': plans_list, 'cost': self.total_cost,
            'outputs': message,
        })
        return message, info
