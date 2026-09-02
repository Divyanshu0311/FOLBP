"""(C) CDCL Engine — learns generalized clauses from conflicts; non-chrono backtrack.

A learned clause is a forbidden (agent_class, verb, target_class) triple — read as
"this combination tends to fail in this scene; the planner should avoid it next time".

Non-chronological backtrack: when the i-th step fails for reason R, we return the
earliest ancestor index that touches the failed target. The arena uses that index
to decide how many steps of the executed plan to roll back conceptually (we don't
undo env state — just rewind the cursor for the next plan).

Persistence: the engine state survives across replans and tasks within a run.
"""
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from conflict_analyzer import ConflictReport, FailureClass
from task_graph import TaskGraph


@dataclass(frozen=True)
class LearnedClause:
    agent_class: str
    verb: str
    target_class: str
    reason: str
    failed_precond: str = ''  # name of the Z3 atom that was missing (e.g. 'not_holding', 'close')
    context: str = ''         # free-text context from the Replanner (e.g. per-arm room locations)


@dataclass
class CDCLEngine:
    clauses: List[LearnedClause] = field(default_factory=list)
    failure_counts: Counter = field(default_factory=Counter)
    _id_to_class: Dict[int, str] = field(default_factory=dict)

    def learn(self, task_graph: TaskGraph, report: ConflictReport,
              failed_precond: str = '', context: str = '') -> List[LearnedClause]:
        new_clauses = []
        if report.failed_node_id not in task_graph.nodes:
            return new_clauses
        failed = task_graph.nodes[report.failed_node_id]
        target_class = self._class_of(failed.targets[0]) if failed.targets else ''
        agent_class = self._class_of(failed.agent_id)
        clause = LearnedClause(
            agent_class=agent_class,
            verb=failed.verb,
            target_class=target_class,
            reason=report.failure_class.value,
            failed_precond=failed_precond,
            context=context,
        )
        if clause not in self.clauses:
            self.clauses.append(clause)
            new_clauses.append(clause)
        self.failure_counts[(agent_class, failed.verb, target_class)] += 1
        return new_clauses

    def backtrack_level(self, task_graph: TaskGraph, report: ConflictReport) -> int:
        """Earliest ancestor index touching the failed target — that's the non-chrono jump."""
        if not report.implicated_node_ids:
            return max(0, report.failed_node_id - 1)
        return min(report.implicated_node_ids)

    def stats(self) -> Dict[str, int]:
        return {
            'clauses_learned': len(self.clauses),
            'distinct_patterns_failed': len(self.failure_counts),
        }

    def _class_of(self, scene_id: int) -> str:
        return self._id_to_class.get(scene_id, '')

    def bind_scene_classes(self, id_to_class: Dict[int, str]):
        self._id_to_class = dict(id_to_class)
