"""Per-agent observation aggregator + action choice via LLM (box 5 helper).

Computes the PEFA-style observation features (reachable_objects, on_surfaces,
landable_surfaces, next_rooms, ...) that the executor's per-step LLM call
needs. In deterministic executor mode this still runs because the Plan
Validator and Executor both reuse get_available_plans() output.
"""
import copy

from LLM import LLM


class LLM_agent:
    def __init__(self, agent_id, args, agent_node, init_graph):
        self.agent_id = agent_id
        self.args = args
        self.agent_node = agent_node
        self.init_graph = init_graph
        self.init_id2node = {n['id']: n for n in init_graph['nodes']}
        self.LLM = LLM(args.source, args.lm_id, args)

        self.current_room = None
        self.grabbed_objects = None
        self.on_surfaces = None
        self.landable_surfaces = None
        self.all_landable_surfaces = []
        self.reachable_objects = []
        self.unreached_objects = []
        self.on_same_surfaces = []
        self.next_rooms = []
        self.doors = []
        self.id2node = {}
        self.chat_agent_info = None
        self.steps = 0

    def _refresh_features(self, obs):
        # Keep agent_node in sync with current observation (FLYING/LAND state, etc.)
        for n in obs['nodes']:
            if n['id'] == self.agent_node['id']:
                self.agent_node = n
                break

        self.id2node = {n['id']: n for n in obs['nodes']}
        self.grabbed_objects = None
        self.reachable_objects = []
        self.landable_surfaces = None
        self.on_surfaces = None
        self.all_landable_surfaces = [n for n in obs['nodes'] if 'LANDABLE' in n['properties']]
        self.on_same_surfaces = []
        on_same_ids = []

        for e in obs['edges']:
            x, r, y = e['from_id'], e['relation_type'], e['to_id']
            if x != self.agent_node['id']:
                continue
            if r == 'INSIDE':
                self.current_room = self.id2node[y]
            elif r == 'ON':
                self.on_surfaces = self.id2node[y]
                if self.agent_node['class_name'] in ('robot arm', 'robot_arm'):
                    for _ in range(3):
                        for edge in obs['edges']:
                            if (edge['from_id'] != x
                                    and (edge['to_id'] == y or edge['to_id'] in on_same_ids)
                                    and edge['relation_type'] in ('ON', 'INSIDE')):
                                on_same_ids.append(edge['from_id'])
                                same_node = self.id2node[edge['from_id']]
                                if 'SURFACES' in same_node['properties'] or 'CONTAINERS' in same_node['properties']:
                                    for ee in obs['edges']:
                                        if ee['to_id'] == edge['from_id'] and ee['relation_type'] in ('INSIDE', 'ON'):
                                            on_same_ids.append(ee['from_id'])
                                on_same_ids = list(set(on_same_ids))
                    self.on_same_surfaces = [self.id2node[i] for i in on_same_ids]
            elif r == 'HOLD':
                self.grabbed_objects = self.id2node[y]
            elif r == 'CLOSE':
                self.reachable_objects.append(self.id2node[y])
            elif r == 'ABOVE' and 'LANDABLE' in self.id2node[y]['properties']:
                self.landable_surfaces = self.id2node[y]

        self.unreached_objects = copy.deepcopy(obs['nodes'])
        for node in obs['nodes']:
            if node == self.grabbed_objects or node in self.reachable_objects:
                self.unreached_objects.remove(node)
            elif (node['category'] in ('Rooms', 'Agents', 'Floor')
                  or 'HIGH_HEIGHT' in node['properties']
                  or 'ON_HIGH_SURFACE' in node['properties']):
                self.unreached_objects.remove(node)

        self.doors = [n for n in obs['nodes'] if n['class_name'] == 'door']
        self.next_rooms = []
        for door in self.doors:
            for edge in self.init_graph['edges']:
                if (edge['relation_type'] == 'LEADING TO'
                        and edge['from_id'] == door['id']
                        and edge['to_id'] != self.current_room['id']):
                    self.next_rooms.append([self.init_id2node[edge['to_id']], door])

    def available_plans(self, obs):
        """Enumerate currently valid action strings for this agent given obs.
        Used by the Plan Validator (4) without firing the executor LLM."""
        self._refresh_features(obs)
        _, _, plans_list = self.LLM.get_available_plans(
            self.agent_node, self.next_rooms, copy.deepcopy(self.all_landable_surfaces),
            self.landable_surfaces, self.on_surfaces, self.grabbed_objects,
            self.reachable_objects, self.unreached_objects, self.on_same_surfaces,
        )
        return plans_list

    def get_action(self, observation, chat_agent_info, goal):
        """PEFA-style: 1 LLM call to pick an action from the current available set."""
        self.chat_agent_info = chat_agent_info
        self._refresh_features(observation)

        info = {
            'obs': {
                'agent_class': self.agent_node['class_name'],
                'agent_id': self.agent_node['id'],
                'grabbed_objects': self.grabbed_objects,
                'reachable_objects': self.reachable_objects,
                'on_surfaces': self.on_surfaces,
                'landable_surfaces': self.landable_surfaces,
                'doors': self.doors,
                'next_rooms': self.next_rooms,
                'current_room': self.current_room['class_name'] if self.current_room else None,
            },
            'graph': observation,
        }
        message, a_info = self.LLM.run(
            self.agent_node, self.chat_agent_info, self.current_room, self.next_rooms,
            copy.deepcopy(self.all_landable_surfaces), self.landable_surfaces,
            self.on_surfaces, self.grabbed_objects, self.reachable_objects,
            self.unreached_objects, self.on_same_surfaces,
        )
        plan = a_info.get('plan')
        a_info['steps'] = self.steps
        info['LLM'] = a_info
        return plan, message, info
