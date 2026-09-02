"""(B) Conflict Analyzer — classifies failures and builds an implication graph.

Failure classes:
  - PRECOND_MISSING      Z3 returned UNSAT (missing close/hold/state_open/...)
  - CAPABILITY_VIOLATION Plan Validator rejected the verb for that robot
  - RUNTIME_REJECTION    Executor (or the bridge) refused the action
  - PARSE_FAILURE        Oracle output was malformed

The implication graph picks the ancestors of the failed task node in the Task Graph
that touch the same target objects — those are the candidates the CDCL Engine learns
generalized clauses about.
"""
from dataclasses import dataclass
from enum import Enum
from typing import List

from task_graph import TaskGraph


class FailureClass(Enum):
    PRECOND_MISSING = 'precond_missing'
    CAPABILITY_VIOLATION = 'capability_violation'
    RUNTIME_REJECTION = 'runtime_rejection'
    PARSE_FAILURE = 'parse_failure'
    UNKNOWN = 'unknown'


@dataclass
class ConflictReport:
    failure_class: FailureClass
    failed_node_id: int
    implicated_node_ids: List[int]
    reason: str


def classify(reason: str) -> FailureClass:
    r = (reason or '').lower()
    if 'malformed' in r or 'parse' in r:
        return FailureClass.PARSE_FAILURE
    if 'antipattern' in r:
        # Ordering antipatterns are state-dependent — same bucket as PRECOND_MISSING.
        # Soft hint only; never a hard ban.
        return FailureClass.PRECOND_MISSING
    if 'capability' in r or 'cannot_' in r or 'forbidden' in r:
        return FailureClass.CAPABILITY_VIOLATION
    if 'runtime' in r or 'bridge' in r or 'http_error' in r:
        return FailureClass.RUNTIME_REJECTION
    if 'precond' in r or 'unsat' in r or 'missing' in r or 'close(' in r or 'hold(' in r:
        return FailureClass.PRECOND_MISSING
    return FailureClass.UNKNOWN


class ConflictAnalyzer:
    def analyze(self, task_graph: TaskGraph, failed_node_id: int, reason: str) -> ConflictReport:
        klass = classify(reason)
        if failed_node_id not in task_graph.nodes:
            return ConflictReport(klass, failed_node_id, [], reason)
        failed = task_graph.nodes[failed_node_id]
        ancestors = task_graph.ancestors(failed_node_id)
        implicated = [
            nid for nid in ancestors
            if any(t in failed.targets for t in task_graph.nodes[nid].targets)
        ]
        return ConflictReport(klass, failed_node_id, implicated, reason)
