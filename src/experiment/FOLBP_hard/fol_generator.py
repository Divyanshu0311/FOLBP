"""(2) FOL Generator — Scene -> typed predicates. Deterministic, no LLM.

Emits ground predicates over typed objects:
  agent(id), object(id), room(id), surface(id), container(id), door(id)
  at(agent, room)            -- agent is INSIDE that room
  on(obj, surface)           -- obj is ON surface
  inside(obj, container)     -- obj is INSIDE container
  hold(agent, obj)           -- agent HOLDs obj (robot dog / arm)
  with(quadrotor, basket)    -- quadrotor carries basket
  state_flying(agent)        -- quadrotor FLYING
  state_land(agent)          -- quadrotor LAND
  state_open(obj)            -- door/container OPEN
  state_closed(obj)
  close(agent, obj)          -- robot dog CLOSE to obj
  above(agent, surface)      -- quadrotor ABOVE surface
  landable(surface)          -- has LANDABLE property
  high_height(surface)       -- has HIGH_HEIGHT property
  on_high_surface(obj)
  grabable(obj)
  leading_to(door, room)
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Set, Tuple


@dataclass
class Predicates:
    """A bag of ground predicates over the scene."""
    nodes: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    unary: Set[Tuple[str, int]] = field(default_factory=set)
    binary: Set[Tuple[str, int, int]] = field(default_factory=set)

    def add_unary(self, name: str, x: int):
        self.unary.add((name, x))

    def add_binary(self, name: str, x: int, y: int):
        self.binary.add((name, x, y))

    def has_unary(self, name: str, x: int) -> bool:
        return (name, x) in self.unary

    def has_binary(self, name: str, x: int, y: int) -> bool:
        return (name, x, y) in self.binary


class FOLGenerator:
    def generate(self, graph: Dict[str, Any]) -> Predicates:
        p = Predicates()
        p.nodes = {n['id']: n for n in graph['nodes']}

        for n in graph['nodes']:
            nid = n['id']
            cat = n['category']
            props, states = n.get('properties', []), n.get('states', [])
            if cat == 'Agents':
                p.add_unary('agent', nid)
            elif cat == 'Rooms':
                p.add_unary('room', nid)
            else:
                p.add_unary('object', nid)
            if 'SURFACES' in props:
                p.add_unary('surface', nid)
            if 'CONTAINERS' in props:
                p.add_unary('container', nid)
            if n['class_name'] == 'door':
                p.add_unary('door', nid)
            if 'LANDABLE' in props:
                p.add_unary('landable', nid)
            if 'HIGH_HEIGHT' in props:
                p.add_unary('high_height', nid)
            if 'LOW_HEIGHT' in props:
                p.add_unary('low_height', nid)
            if 'ON_HIGH_SURFACE' in props:
                p.add_unary('on_high_surface', nid)
            if 'GRABABLE' in props:
                p.add_unary('grabable', nid)
            if 'OPEN' in states or 'OPEN_FOREVER' in states:
                p.add_unary('state_open', nid)
            if 'CLOSED' in states:
                p.add_unary('state_closed', nid)
            if 'FLYING' in states:
                p.add_unary('state_flying', nid)
            if 'LAND' in states:
                p.add_unary('state_land', nid)

        for e in graph['edges']:
            x, r, y = e['from_id'], e['relation_type'], e['to_id']
            if r == 'INSIDE':
                if p.has_unary('agent', x):
                    p.add_binary('at', x, y)
                else:
                    p.add_binary('inside', x, y)
            elif r == 'ON':
                p.add_binary('on', x, y)
            elif r == 'HOLD':
                p.add_binary('hold', x, y)
            elif r == 'CLOSE':
                p.add_binary('close', x, y)
            elif r == 'ABOVE':
                p.add_binary('above', x, y)
            elif r == 'WITH':
                p.add_binary('with', x, y)
            elif r == 'LEADING TO':
                p.add_binary('leading_to', x, y)
        return p

    def class_of(self, p: Predicates, oid: int) -> str:
        n = p.nodes.get(oid)
        return n['class_name'] if n else ''
