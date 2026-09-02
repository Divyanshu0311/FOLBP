"""Environment wrapper: partial-observation graph + deterministic step() transitions.

Mirrors the PEFA env interface so the COHERENT env JSONs work unchanged.
This is box (6) Environment in the FOLBP diagram.
"""
import copy
import re


class Get_env_info:
    def __init__(self, task_id=None, env_id=None, task_name=None, graph=None,
                 task_goal=None, goal_instruction=None, ground_truth_step_num=None,
                 agent=None, num_agent=None):
        self.task_id = task_id
        self.env_id = env_id
        self.task_name = task_name
        self.graph = graph
        self.task_goal = task_goal
        self.goal_instruction = goal_instruction[0]
        self.ground_truth_step_num = ground_truth_step_num[0]
        self.agent = agent
        self.num_agent = num_agent
        self.id2node = {n['id']: n for n in graph['nodes']}
        self.id_to_name = {n['id']: n['class_name'] for n in graph['nodes']}
        self.id_name_dict = {i: agent_name for i, agent_name in enumerate(agent)}
        self.steps = 0

    def get_observations(self):
        return {aid: self.get_observation(aid) for aid in self.id_name_dict}

    def get_observation(self, agent_id):
        obs, agent_in_room = self._get_visible_node(agent_id)
        return {**obs, 'agent_in_room_id': agent_in_room}

    def _get_visible_node(self, agent_id):
        id2node = {n['id']: n for n in self.graph['nodes']}
        rooms_ids = [n['id'] for n in self.graph['nodes'] if n['category'] == 'Rooms']
        real_agent_id = self.id_name_dict[agent_id][1]

        inside_of_what, what_is_inside = {}, {}
        grabbed_ids, with_quadrotor_id, inside_quadrotor_basket_id = [], [], []
        quadrotor_in_room_id = None

        for edge in self.graph['edges']:
            if edge['relation_type'] == 'INSIDE':
                what_is_inside.setdefault(edge['to_id'], []).append(edge['from_id'])
                inside_of_what[edge['from_id']] = edge['to_id']
            elif edge['relation_type'] == 'HOLD' and edge['from_id'] == real_agent_id:
                grabbed_ids.append(edge['to_id'])
            elif edge['relation_type'] == 'WITH':
                with_quadrotor_id.append(edge['to_id'])
                qid = edge['from_id']
                for ee in self.graph['edges']:
                    if ee['relation_type'] == 'INSIDE' and ee['from_id'] == qid:
                        quadrotor_in_room_id = ee['to_id']

        for edge in self.graph['edges']:
            if edge['relation_type'] == 'INSIDE' and edge['to_id'] in with_quadrotor_id:
                inside_quadrotor_basket_id.append(edge['from_id'])

        room_id = inside_of_what[real_agent_id]
        agents_in_same_room = [
            e['from_id'] for e in self.graph['edges']
            if id2node[e['from_id']]['category'] == 'Agents'
            and e['to_id'] == room_id and e['relation_type'] == 'INSIDE'
        ]

        # Fixpoint: pull objects on/inside surfaces and containers that are already in the room.
        new = True
        temp_edges = copy.deepcopy(self.graph['edges'])
        while new:
            new = False
            for edge in temp_edges:
                if edge['relation_type'] in ('ON', 'INSIDE') and edge['to_id'] in what_is_inside[room_id]:
                    what_is_inside[room_id].append(edge['from_id'])
                    what_is_inside[room_id] = list(set(what_is_inside[room_id]))
                    temp_edges.remove(edge)
                    new = True
                    break

        doors_ids = [e['from_id'] for e in self.graph['edges']
                     if e['relation_type'] == 'LEADING TO' and e['to_id'] == room_id]

        object_in_room_ids = list(what_is_inside[room_id])
        curr = list(object_in_room_ids)
        while curr:
            inside = []
            for cid in curr:
                inside += what_is_inside.get(cid, [])
            object_in_room_ids += list(inside)
            curr = list(inside)

        # Hide objects that are inside a CLOSED non-room container.
        hidden = [
            k for k, parent in inside_of_what.items()
            if parent not in rooms_ids
            and 'OPEN' not in id2node[parent]['states']
            and 'OPEN_FOREVER' not in id2node[parent]['states']
        ]

        all_room_ids = [n['id'] for n in self.graph['nodes'] if n['category'] == 'Rooms']
        observable = (
            [oid for oid in object_in_room_ids if oid not in hidden]
            + [room_id] + doors_ids + [real_agent_id]
            + grabbed_ids + agents_in_same_room + all_room_ids
        )
        if room_id == quadrotor_in_room_id:
            observable += with_quadrotor_id + inside_quadrotor_basket_id
        observable = list(set(observable))

        partial = {
            'edges': [e for e in self.graph['edges']
                      if e['from_id'] in observable and e['to_id'] in observable],
            'nodes': [id2node[i] for i in observable],
        }
        return partial, room_id

    def step(self, class_name, agent_id, agent_action, goal):
        """Apply a single grounded action to the scene graph. Returns (done, task_results, satisfied, unsatisfied, steps)."""
        action = re.findall(r'\[(.*?)\]', agent_action)[0]
        first_name = re.findall(r'<(.*?)>', agent_action)[0]
        first_id = int(re.findall(r'\((.*?)\)', agent_action)[0])
        if action in ('putinto', 'puton'):
            second_id = int(re.findall(r'\((.*?)\)', agent_action)[1])

        room2floor = {}
        for node in self.graph['nodes']:
            if node['category'] == 'Rooms':
                for edge in self.graph['edges']:
                    if (edge['relation_type'] == 'INSIDE'
                            and self.id2node[edge['from_id']]['category'] == 'Floor'
                            and edge['to_id'] == node['id']):
                        room2floor[node['id']] = edge['from_id']

        if class_name in ('quadrotor', 'drone'):
            with_quadrotor = next(
                (e['to_id'] for e in self.graph['edges']
                 if e['from_id'] == agent_id and e['relation_type'] == 'WITH'),
                None,
            )
            if action == 'land_on':
                self.id2node[agent_id]['states'] = ['LAND']
                for edge in self.graph['edges']:
                    if edge['from_id'] == agent_id and edge['to_id'] == first_id and edge['relation_type'] == 'ABOVE':
                        edge['relation_type'] = 'ON'
                self.graph['edges'].append({'from_id': with_quadrotor, 'to_id': first_id, 'relation_type': 'ON'})
                if 'HIGH_HEIGHT' in self.id2node[first_id]['properties']:
                    self.id2node[with_quadrotor]['properties'] = list(
                        set(self.id2node[with_quadrotor]['properties'] + ['ON_HIGH_SURFACE'])
                    )
                    for edge in self.graph['edges']:
                        if edge['to_id'] == with_quadrotor and edge['relation_type'] == 'INSIDE':
                            self.id2node[edge['from_id']]['properties'] = list(
                                set(self.id2node[edge['from_id']]['properties'] + ['ON_HIGH_SURFACE'])
                            )
                if 'LOW_HEIGHT' in self.id2node[first_id]['properties'] and \
                        'ON_HIGH_SURFACE' in self.id2node[with_quadrotor]['properties']:
                    self.id2node[with_quadrotor]['properties'].remove('ON_HIGH_SURFACE')
                    for edge in self.graph['edges']:
                        if edge['to_id'] == with_quadrotor and edge['relation_type'] == 'INSIDE':
                            if 'ON_HIGH_SURFACE' in self.id2node[edge['from_id']]['properties']:
                                self.id2node[edge['from_id']]['properties'].remove('ON_HIGH_SURFACE')

            elif action == 'takeoff_from':
                self.id2node[agent_id]['states'] = ['FLYING']
                for edge in self.graph['edges']:
                    if edge['from_id'] == agent_id and edge['to_id'] == first_id and edge['relation_type'] == 'ON':
                        edge['relation_type'] = 'ABOVE'
                self.graph['edges'] = [e for e in self.graph['edges']
                                       if not (e['from_id'] == with_quadrotor and e['to_id'] == first_id and e['relation_type'] == 'ON')]
                self.graph['edges'] = [e for e in self.graph['edges']
                                       if not (e['to_id'] == with_quadrotor and e['relation_type'] == 'CLOSE')]
                self.id2node[with_quadrotor]['properties'] = list(
                    set(self.id2node[with_quadrotor]['properties'] + ['ON_HIGH_SURFACE'])
                )
                for edge in self.graph['edges']:
                    if edge['to_id'] == with_quadrotor and edge['relation_type'] == 'INSIDE':
                        self.graph['edges'] = [e for e in self.graph['edges']
                                               if not (e['to_id'] == edge['from_id'] and e['relation_type'] == 'CLOSE')]
                        self.id2node[edge['from_id']]['properties'] = list(
                            set(self.id2node[edge['from_id']]['properties'] + ['ON_HIGH_SURFACE'])
                        )

            elif action == 'movetowards':
                for edge in self.graph['edges']:
                    if edge['from_id'] == agent_id and edge['relation_type'] == 'ABOVE':
                        edge['to_id'] = first_id if self.id2node[first_id]['category'] != 'Rooms' else room2floor[first_id]
                if self.id2node[first_id]['category'] == 'Rooms':
                    for edge in self.graph['edges']:
                        if edge['from_id'] == agent_id and edge['relation_type'] == 'INSIDE':
                            edge['to_id'] = first_id

        elif class_name in ('robot dog', 'robot_dog'):
            if action == 'open':
                if first_name == 'door' or 'CONTAINERS' in self.id2node[first_id]['properties']:
                    self.id2node[first_id]['states'] = ['OPEN']
            elif action == 'close':
                if first_name == 'door' or 'CONTAINERS' in self.id2node[first_id]['properties']:
                    self.id2node[first_id]['states'] = ['CLOSED']
            elif action == 'grab':
                for edge in self.graph['edges']:
                    if edge['from_id'] == agent_id and edge['to_id'] == first_id and edge['relation_type'] == 'CLOSE':
                        edge['relation_type'] = 'HOLD'
                for edge in self.graph['edges']:
                    if edge['from_id'] == first_id and edge['relation_type'] in ('INSIDE', 'ON'):
                        self.graph['edges'].remove(edge)
                        break
            elif action == 'putinto':
                for edge in self.graph['edges']:
                    if edge['from_id'] == agent_id and edge['to_id'] == first_id and edge['relation_type'] == 'HOLD':
                        edge['relation_type'] = 'CLOSE'
                        break
                self.graph['edges'].append({'from_id': first_id, 'to_id': second_id, 'relation_type': 'INSIDE'})
            elif action == 'puton':
                for edge in self.graph['edges']:
                    if edge['from_id'] == agent_id and edge['to_id'] == first_id and edge['relation_type'] == 'HOLD':
                        edge['relation_type'] = 'CLOSE'
                        break
                self.graph['edges'].append({'from_id': first_id, 'to_id': second_id, 'relation_type': 'ON'})
            elif action == 'movetowards':
                if self.id2node[first_id]['category'] == 'Rooms':
                    for edge in self.graph['edges']:
                        if edge['from_id'] == agent_id and edge['relation_type'] == 'INSIDE':
                            edge['to_id'] = first_id
                        if edge['from_id'] == agent_id and edge['relation_type'] == 'ON':
                            edge['to_id'] = room2floor[first_id]
                    self.graph['edges'] = [e for e in self.graph['edges']
                                           if not (e['from_id'] == agent_id and e['relation_type'] == 'CLOSE')]
                else:
                    self.graph['edges'] = [e for e in self.graph['edges']
                                           if not (e['from_id'] == agent_id and e['relation_type'] == 'CLOSE')]
                    self.graph['edges'].append({'from_id': agent_id, 'to_id': first_id, 'relation_type': 'CLOSE'})
                    if ('CONTAINERS' in self.id2node[first_id]['properties']
                            and ('OPEN' in self.id2node[first_id]['states']
                                 or 'OPEN_FOREVER' in self.id2node[first_id]['states'])):
                        extras = []
                        for edge in self.graph['edges']:
                            if edge['to_id'] == first_id and edge['relation_type'] == 'INSIDE':
                                extras.append({'from_id': agent_id, 'to_id': edge['from_id'], 'relation_type': 'CLOSE'})
                        self.graph['edges'].extend(extras)
                    extras = []
                    for edge in self.graph['edges']:
                        if (edge['from_id'] == first_id
                                and edge['relation_type'] in ('ON', 'INSIDE')
                                and self.id2node[edge['to_id']]['category'] != 'Floor'):
                            extras.append({'from_id': agent_id, 'to_id': edge['to_id'], 'relation_type': 'CLOSE'})
                    self.graph['edges'].extend(extras)
                    if 'SURFACES' in self.id2node[first_id]['properties'] and self.id2node[first_id]['category'] != 'Floor':
                        extras = []
                        for edge in self.graph['edges']:
                            if edge['to_id'] == first_id and edge['relation_type'] == 'ON':
                                extras.append({'from_id': agent_id, 'to_id': edge['from_id'], 'relation_type': 'CLOSE'})
                        self.graph['edges'].extend(extras)

        elif class_name in ('robot arm', 'robot_arm'):
            if action == 'open':
                if 'CONTAINERS' in self.id2node[first_id]['properties']:
                    self.id2node[first_id]['states'] = ['OPEN']
            elif action == 'close':
                if 'CONTAINERS' in self.id2node[first_id]['properties']:
                    self.id2node[first_id]['states'] = ['CLOSED']
            elif action == 'grab':
                for edge in self.graph['edges']:
                    if edge['from_id'] == first_id and edge['relation_type'] in ('INSIDE', 'ON'):
                        self.graph['edges'].remove(edge)
                        break
                self.graph['edges'].append({'from_id': agent_id, 'to_id': first_id, 'relation_type': 'HOLD'})
            elif action == 'putinto':
                for edge in self.graph['edges']:
                    if edge['from_id'] == agent_id and edge['to_id'] == first_id and edge['relation_type'] == 'HOLD':
                        self.graph['edges'].remove(edge)
                        break
                self.id2node[first_id]['properties'] = list(
                    set(self.id2node[first_id]['properties'] + ['ON_HIGH_SURFACE'])
                )
                self.graph['edges'].append({'from_id': first_id, 'to_id': second_id, 'relation_type': 'INSIDE'})
            elif action == 'puton':
                for edge in self.graph['edges']:
                    if edge['from_id'] == agent_id and edge['to_id'] == first_id and edge['relation_type'] == 'HOLD':
                        self.graph['edges'].remove(edge)
                        break
                self.id2node[first_id]['properties'] = list(
                    set(self.id2node[first_id]['properties'] + ['ON_HIGH_SURFACE'])
                )
                self.graph['edges'].append({'from_id': first_id, 'to_id': second_id, 'relation_type': 'ON'})

        # Sync node mutations back into the canonical node list.
        for node in self.graph['nodes']:
            stored = self.id2node.get(node['id'])
            if stored is not None:
                node['properties'] = stored['properties']
                node['states'] = stored['states']

        # Goal check.
        unsatisfied = copy.deepcopy(goal)
        satisfied, task_results, done = [], [], False
        for goal_descs, nums in goal.items():
            parts = goal_descs.split('_')
            from_id = int(re.findall(r'\((.*?)\)', parts[1])[0])
            to_id = int(re.findall(r'\((.*?)\)', parts[2])[0])
            want = nums[0]
            got = 0
            for edge in self.graph['edges']:
                if (edge['from_id'] == from_id and edge['to_id'] == to_id
                        and edge['relation_type'] == parts[0].upper()):
                    got += 1
                    satisfied.append(edge)
            task_results.append({goal_descs: got})
            if got == want:
                del unsatisfied[goal_descs]
                if not unsatisfied:
                    done = True
                    break
            else:
                done = False
                break

        self.steps += 1
        return done, task_results, satisfied, unsatisfied, self.steps
