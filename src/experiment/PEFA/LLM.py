
import copy
import os
import sys
from pathlib import Path
from google import genai
from google.genai import types

import json
import time
import backoff

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bench.eventlog import EVENTS
from bench.llm_meter import METER


class LLM:
	def __init__(self, source, lm_id, args):

		self.args = args
		self.debug = args.debug
		self.source = args.source
		self.lm_id = args.lm_id
		self.chat = True
		self.total_cost = 0
		self.device = None
		self.record_dir = f'./log/{args.env}.txt'

		if self.source == 'gemini':

			api_key = args.api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")

			self.client = genai.Client(api_key=api_key) if api_key else genai.Client()
			if self.chat:
				self.sampling_params = {
					"max_output_tokens": args.max_tokens,
                    "temperature": args.t,
                    # "top_p": 1.0,
                    "candidate_count": args.n
				}

		def lm_engine(source, lm_id, device):
			
			@backoff.on_exception(backoff.expo, Exception)
			def _generate(prompt, sampling_params, role='executor'):
				usage = 0
				if source == 'gemini':
					try:
						if self.chat:
							# Convert OpenAI-style messages to Gemini format
							gemini_prompt = ""
							for msg in prompt:
								if msg["role"] == "user":
									gemini_prompt += msg["content"] + "\n"
							
							# Generate multiple candidates
							generated_samples = []
							for _ in range(sampling_params.get('candidate_count', 1)):
								with METER.timed(role) as _call:
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
								generated_samples.append(response.text)
								print("\n========== GEMINI RAW OUTPUT (PEFA) ==========")
								print(response.text)
								print("==============================================\n", flush=True)

							if self.debug:
								with open(f"./chat_raw.json", 'a') as f:
									f.write(json.dumps({"prompt": gemini_prompt, "response": generated_samples}, indent=4))
									f.write('\n')
							
							# Gemini pricing estimation
							prompt_tokens = len(gemini_prompt.split()) * 1.3
							completion_tokens = sum(len(s.split()) for s in generated_samples) * 1.3
							if 'flash' in self.lm_id:
								usage = prompt_tokens * 0.075 / 1000000 + completion_tokens * 0.3 / 1000000
							elif 'pro' in self.lm_id:
								usage = prompt_tokens * 1.25 / 1000000 + completion_tokens * 5.0 / 1000000
						# mean_log_probs = [np.mean(response['choices'][i]['logprobs']['token_logprobs']) for i in
						# 				  range(sampling_params['n'])]
						else:
							raise ValueError(f"{lm_id} not available!")
					except Exception as e:
						print(e)
						raise e
				
				else:
					raise ValueError("invalid source")
				return generated_samples, usage

			return _generate

		self.generator = lm_engine(self.source, self.lm_id, self.device)

	def parse_answer(self, available_actions, text):
		import re as _re

		# Normalize the LLM text: underscores -> spaces, then restore a few bracketed verbs
		norm = text.replace("_", " ")
		norm = norm.replace("takeoff from", "takeoff_from")
		norm = norm.replace("land on", "land_on")

		# 1) exact substring match of "[verb] <class>(id)" form
		for action in available_actions:
			if action in norm:
				return action
		self.write_log_to_file('\nThe first action parsing failed!!!')

		# 2) option letter match (A., B), (C), etc.)
		tokens = norm.split(' ')
		for i, action in enumerate(available_actions):
			option = chr(ord('A') + i)
			if (f"option {option}" in norm or f"{option}." in tokens or f"{option}," in tokens
					or f"Option {option}" in norm or f"({option})" in norm):
				return action
		self.write_log_to_file('\nThe second action parsing failed!!!')

		# 3) loose match: extract verb + id from each available action, then look for
		#    (verb AND id) co-occurring in the LLM output. Handles hallucinations like
		#    "land livingroom floor" when the real action is "[land_on] <livingroom floor>(5)".
		low = norm.lower()
		for action in available_actions:
			m = _re.match(r'\[(\w+)\]\s*<([^>]+)>\((\d+)\)', action)
			if not m:
				continue
			verb, cls, aid = m.group(1), m.group(2).lower(), m.group(3)
			verb_spaced = verb.replace('_', ' ')
			# require the object id OR the class name, plus the verb root
			verb_root = verb.split('_')[0]  # "land" from "land_on", "takeoff" from "takeoff_from"
			has_verb = (verb in low) or (verb_spaced in low) or (verb_root in low)
			has_target = (f"({aid})" in low) or (f" {aid}" in low) or (cls in low)
			if has_verb and has_target:
				self.write_log_to_file(f'\n[Loose parse] matched "{action}" in output')
				return action

		print("WARNING! No available action parsed!!! Output plan NONE!\n")
		return None

	def get_available_plans(self, agent_node, next_rooms, all_landable_surfaces, landable_surfaces, on_surfaces, 
						 grabbed_objects, reached_objects, unreached_objecs, on_same_surface_objects
						 ):
		"""
		'quadrotor':
		[land_on] <surface>
		[movetowards] <surface>/<next_room>
		[takeoff_from] <surface>

		'robot dog':
		[open] <container>/<door>
		[close] <container>/<door>
		[grab] <object>
		[putinto] <object> into <container>
		[puton] <object> on <surface>
		[movetowards] <object>

		'robot arm':
		[open] <container>
		[close] <container>
		[grab] <object>
		[putinto] <object> into <container>
		[puton] <object> on <surface>

		"""
		available_plans = []
		if agent_node["class_name"] in ("quadrotor", "drone"):
			other_landable_surfaces = []
			if "FLYING" in agent_node["states"]:
				if landable_surfaces is not None:
					available_plans.append(f"[land_on] <{landable_surfaces['class_name']}>({landable_surfaces['id']})")
					all_landable_surfaces.remove(landable_surfaces)
					other_landable_surfaces = copy.deepcopy(all_landable_surfaces)
				if len(other_landable_surfaces) != 0:
					for surface in other_landable_surfaces :
						available_plans.append(f"[movetowards] <{surface['class_name']}>({surface['id']})")
				for next_room in next_rooms:
					if 'OPEN' in next_room[1]['states'] or "OPEN_FOREVER" in next_room[1]['states']:
						available_plans.append(f"[movetowards] <{next_room[0]['class_name']}>({next_room[0]['id']})")

			if "LAND" in agent_node["states"]:
				if on_surfaces is not None:
					available_plans.append(f"[takeoff_from] <{on_surfaces['class_name']}>({on_surfaces['id']})")

		if agent_node["class_name"] == "robot dog" or agent_node["class_name"] == "robot_dog":
			# if grabbed_objects is not None:
			# 	available_plans.append(f"[puton] <{grabbed_objects['class_name']}>({grabbed_objects['id']}) on <{on_surfaces['class_name']}>({on_surfaces['id']})")
			# The robotic dog is not allowed to put things on the floor. If it needs to open the door and has something in its hand, it needs to find a low surface to put things first
			if len(reached_objects) != 0:
				for reached_object in reached_objects:
					if grabbed_objects is None:
						if 'CONTAINERS' in reached_object['properties'] and 'CLOSED' in reached_object['states'] or \
							reached_object['class_name'] == 'door' and 'CLOSED' in reached_object['states']:
							available_plans.append(f"[open] <{reached_object['class_name']}>({reached_object['id']})")
						if 'CONTAINERS' in reached_object['properties'] and 'OPEN' in reached_object['states'] or \
							reached_object['class_name'] == 'door' and 'OPEN' in reached_object['states']:
							available_plans.append(f"[close] <{reached_object['class_name']}>({reached_object['id']})")
						if 'GRABABLE' in reached_object['properties']:
							available_plans.append(f"[grab] <{reached_object['class_name']}>({reached_object['id']})")
					if grabbed_objects is not None:
						if 'CONTAINERS' in reached_object['properties'] and ('OPEN' in reached_object['states'] or "OPEN_FOREVER" in reached_object['states']):
							available_plans.append(f"[putinto] <{grabbed_objects['class_name']}>({grabbed_objects['id']}) into <{reached_object['class_name']}>({reached_object['id']})")
						if 'SURFACES' in reached_object['properties']:
							available_plans.append(f"[puton] <{grabbed_objects['class_name']}>({grabbed_objects['id']}) on <{reached_object['class_name']}>({reached_object['id']})")
			
			if len(unreached_objecs) != 0:
				for unreached_object in unreached_objecs:
					available_plans.append(f"[movetowards] <{unreached_object['class_name']}>({unreached_object['id']})")
			for next_room in next_rooms:
					if 'OPEN' in next_room[1]['states'] or "OPEN_FOREVER" in next_room[1]['states']:
						available_plans.append(f"[movetowards] <{next_room[0]['class_name']}>({next_room[0]['id']})")


		if agent_node['class_name'] == 'robot arm' or agent_node['class_name'] == 'robot_arm':
			if grabbed_objects is not None:
				available_plans.append(f"[puton] <{grabbed_objects['class_name']}>({grabbed_objects['id']}) on <{on_surfaces['class_name']}>({on_surfaces['id']})")
			for on_same_surface_object in on_same_surface_objects:
				if grabbed_objects is None:
					if 'CONTAINERS' in on_same_surface_object['properties'] and 'OPEN' in on_same_surface_object['states']:
						available_plans.append(f"[close] <{on_same_surface_object['class_name']}>({on_same_surface_object['id']})")
					if 'CONTAINERS' in on_same_surface_object['properties'] and 'CLOSED' in on_same_surface_object['states']:
						available_plans.append(f"[open] <{on_same_surface_object['class_name']}>({on_same_surface_object['id']})")
					if 'GRABABLE' in on_same_surface_object['properties']:
						available_plans.append(f"[grab] <{on_same_surface_object['class_name']}>({on_same_surface_object['id']})")

				if grabbed_objects is not None:
					
					if 'CONTAINERS' in on_same_surface_object['properties'] and ('OPEN' in on_same_surface_object['states'] or "OPEN_FOREVER" in on_same_surface_object['states']):
						available_plans.append(f"[putinto] <{grabbed_objects['class_name']}>({grabbed_objects['id']}) into <{on_same_surface_object['class_name']}>({on_same_surface_object['id']})")
					if 'SURFACES' in on_same_surface_object['properties']:
						available_plans.append(f"[puton] <{grabbed_objects['class_name']}>({grabbed_objects['id']}) on <{on_same_surface_object['class_name']}>({on_same_surface_object['id']})")

		plans = ""
		for i, plan in enumerate(available_plans):
			plans += f"{chr(ord('A') + i)}. {plan}\n"
		print(agent_node["class_name"],agent_node['id'])
		print(available_plans)
		EVENTS.emit('ACTIONS', title=f"<{agent_node['class_name']}>({agent_node['id']})",
			            body="; ".join(available_plans), n=len(available_plans))
		return plans, len(available_plans), available_plans

		
	def run(self, agent_node, chat_agent_info,current_room, next_rooms, all_landable_surfaces,landable_surfaces, on_surfaces, grabbed_objects, reached_objects,unreached_objecs, on_same_surface_objects):
		info = {"num_available_actions": None,
			"prompts": None,
			"outputs": None,
			"plan": None,
			"action_list": None,
			"cost":self.total_cost, 
			f"<{agent_node['class_name']}>({agent_node['id']}) total_cost": self.total_cost}

		prompt_path = chat_agent_info['prompt_path']
		with open(prompt_path, 'r') as f:
			agent_prompt = f.read()

		available_plans, num, available_plans_list = self.get_available_plans(agent_node, next_rooms, all_landable_surfaces,landable_surfaces, on_surfaces, grabbed_objects, reached_objects,unreached_objecs, on_same_surface_objects,
																		 )
		
		agent_prompt = agent_prompt.replace('#OBSERVATION#', chat_agent_info['observation'])
		agent_prompt = agent_prompt.replace('#ACTIONLIST#', available_plans)
		agent_prompt = agent_prompt.replace('#INSTRUCTION#', chat_agent_info['instruction'])
		
		if self.debug:
			print(f"cot_prompt:\n{agent_prompt}")
		chat_prompt = [{"role": "user", "content": agent_prompt}]
		_t_exec = time.time()
		outputs, usage = self.generator(chat_prompt, self.sampling_params)
		output = outputs[0]

		self.write_log_to_file(output+'\n111111111')
		EVENTS.emit('EXECUTOR', title=f"<{agent_node['class_name']}>({agent_node['id']})",
			            body=output, note=f'{time.time() - _t_exec:.1f}s')
		self.total_cost += usage
		info['cot_outputs'] = outputs

		if self.debug:
			print(f"cot_output:\n{output}")
			print(f"total cost: {self.total_cost}")
		# Detect "YES I CAN" / "SORRY I CANNOT" anywhere in the output (LLMs often
		# put their chain-of-thought first and the verdict later). Previously we
		# only looked at the first sentence, which caused the refinement branch
		# to be skipped for the quadrotor on long outputs.
		upper_out = output.upper()
		yes_can = ("YES I CAN" in upper_out)
		sorry_cannot = ("SORRY I CANNOT" in upper_out)
		# If both appear (rare), prefer SORRY I CANNOT only when it appears AFTER
		# the last YES I CAN — otherwise trust YES I CAN.
		if yes_can and sorry_cannot:
			if upper_out.rfind("SORRY I CANNOT") > upper_out.rfind("YES I CAN"):
				yes_can = False
			else:
				sorry_cannot = False
		message = f"No action selected. LLM output: {output[:200]}"

		if yes_can:
			chat_prompt = [{"role": "user", "content": agent_prompt},
							{"role": "assistant", "content": output},
							{"role": "user", "content": "Answer with only one best next action in the list of available actions. So the answer is"}]

			outputs, usage = self.generator(chat_prompt, self.sampling_params)
			output = outputs[0]
			self.total_cost += usage
			self.write_log_to_file(output+'\n2222222222222')
			if "SORRY I CANNOT" not in output.upper():

				if self.debug:
					print(f"cot_output:\n{output}")
					print(f"total cost: {self.total_cost}")

				plan = self.parse_answer(available_plans_list, output)
				if plan is None:
					plan_str = 'no plan'
				else:
					plan_str = plan
				print(plan)
				if self.debug:
					print(f"plan: {plan}\n")
				EVENTS.emit('EXECUTOR_PICK',
					            title=f"<{agent_node['class_name']}>({agent_node['id']})",
					            body=plan_str if plan else '', n=num)
				info.update({"num_available_actions": num,
						"prompts": chat_prompt,
						# "outputs": outputs,
						"plan": plan,
						"action_list": available_plans_list,
						f"<{agent_node['class_name']}>({agent_node['id']}) total_cost": self.total_cost})
				message = f" The action I finally decided to perform is {plan_str}. "

				prompt_path = self.args.judge_prompt_path
				with open(prompt_path, 'r') as f:
					prompt = f.read()
				prompt = prompt.replace('#INSTRUCTION#', chat_agent_info['instruction'])
				prompt = prompt.replace('#PLAN#', plan_str)
				prompt = prompt.replace('#AGENT#', f"<{agent_node['class_name']}>")
				prompt = [{"role": "user", "content": prompt}]
				_t_judge = time.time()
				outputs, usage = self.generator(prompt, self.sampling_params, 'judge')
				output = outputs[0]
				self.total_cost += usage
				message += output
				EVENTS.emit('JUDGE', body=output,
					            note=f'{time.time() - _t_judge:.1f}s')
				self.write_log_to_file(output+'\n333333333333333333')
				info.update({"outputs": message})


		if sorry_cannot:
			message = f"Sorry, the current actions I can perform cannot complete this instrcution. LLM reasoning: {output} My current actionlist is: {available_plans}"
			self.write_log_to_file(message+'\n4444444444444')

		# Fallback: if plan is still None but LLM output contains a valid action,
		# try to parse it directly (handles cases where LLM doesn't start with
		# "Yes I can" or "Sorry I cannot" exactly)
		if info.get("plan") is None and num > 0:
			fallback_plan = self.parse_answer(available_plans_list, output)
			if fallback_plan is not None:
				print(f"[Fallback parse succeeded] {fallback_plan}")
				self.write_log_to_file(f"[Fallback parse] {fallback_plan}")
				info.update({"plan": fallback_plan, "action_list": available_plans_list,
							"num_available_actions": num})
				message = f" The action I finally decided to perform is {fallback_plan}. "

		info['cost'] = self.total_cost
		self.write_log_to_file(f"total cost: {self.total_cost}")
		info.update({"outputs": message})
		return message, info

	def write_log_to_file(self,log_message, file_name=None):
		file_name = self.record_dir
		with open(file_name, 'a') as file:  
			file.write(log_message + '\n')  