"""(A) Task Graph — DAG of subtasks with temporal & resource edges.

A task node is a (agent_id, verb, target_ids) tuple, created whenever the executor
emits an action. Temporal edges connect consecutive steps from the same agent;
resource edges connect actions that touch the same target object across different
agents (e.g. robot dog [grab] and robot arm [grab] on the same id share a resource).
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Set, Tuple


class NodeStatus(Enum):
    PENDING = 'pending'
    EXECUTED = 'executed'
    FAILED = 'failed'


class EdgeKind(Enum):
    TEMPORAL = 'temporal'
    RESOURCE = 'resource'


@dataclass
class TaskNode:
    id: int
    agent_id: int
    verb: str
    targets: Tuple[int, ...]
    status: NodeStatus = NodeStatus.PENDING
    failure_reason: str = ''


@dataclass
class TaskEdge:
    src: int
    dst: int
    kind: EdgeKind


@dataclass
class TaskGraph:
    nodes: Dict[int, TaskNode] = field(default_factory=dict)
    edges: List[TaskEdge] = field(default_factory=list)
    _next_id: int = 0
    _last_node_per_agent: Dict[int, int] = field(default_factory=dict)
    _last_node_per_target: Dict[int, int] = field(default_factory=dict)

    def add_step(self, agent_id: int, verb: str, targets: Tuple[int, ...]) -> int:
        node = TaskNode(id=self._next_id, agent_id=agent_id, verb=verb, targets=tuple(targets))
        self.nodes[node.id] = node
        nid = node.id
        self._next_id += 1
        prev = self._last_node_per_agent.get(agent_id)
        if prev is not None:
            self.edges.append(TaskEdge(prev, nid, EdgeKind.TEMPORAL))
        self._last_node_per_agent[agent_id] = nid
        for t in targets:
            prev_t = self._last_node_per_target.get(t)
            if prev_t is not None and self.nodes[prev_t].agent_id != agent_id:
                self.edges.append(TaskEdge(prev_t, nid, EdgeKind.RESOURCE))
            self._last_node_per_target[t] = nid
        return nid

    def mark_executed(self, nid: int):
        if nid in self.nodes:
            self.nodes[nid].status = NodeStatus.EXECUTED

    def mark_failed(self, nid: int, reason: str):
        if nid in self.nodes:
            self.nodes[nid].status = NodeStatus.FAILED
            self.nodes[nid].failure_reason = reason

    def predecessors(self, nid: int) -> List[TaskNode]:
        return [self.nodes[e.src] for e in self.edges if e.dst == nid and e.src in self.nodes]

    def ancestors(self, nid: int) -> Set[int]:
        seen, stack = set(), [nid]
        while stack:
            cur = stack.pop()
            for e in self.edges:
                if e.dst == cur and e.src not in seen:
                    seen.add(e.src)
                    stack.append(e.src)
        return seen

    def summary(self) -> Dict[str, int]:
        return {
            'nodes': len(self.nodes),
            'edges': len(self.edges),
            'failed': sum(1 for n in self.nodes.values() if n.status == NodeStatus.FAILED),
            'executed': sum(1 for n in self.nodes.values() if n.status == NodeStatus.EXECUTED),
        }
