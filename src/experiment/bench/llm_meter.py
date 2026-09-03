"""Process-scoped LLM accounting — calls, tokens, and latency, broken out by role.

Every arm runs one task per subprocess, so a module-level singleton is the whole
story: no cross-task contamination is possible and no plumbing is needed through
the arena. Both FOLBP and PEFA wrap their single `generate_content` call site with
`METER.timed(role)`, which is what makes their per-task LLM cost comparable at all.

Roles:
    oracle    — full-plan / next-instruction planning calls
    repair    — re-prompts issued by the LLM-repair arm after a Z3 UNSAT
    executor  — per-step grounding calls made by the robot agents
    judge     — PEFA's goal-satisfaction judge
    other     — anything unclassified

Usage:
    with METER.timed('oracle') as call:
        response = client.models.generate_content(...)
        call.observe(response)          # pulls usage_metadata if the SDK supplied it
"""
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List

ROLES = ('oracle', 'repair', 'executor', 'judge', 'other')

try:
    from bench.eventlog import EVENTS
except ImportError:          # meter used standalone, without the bench package
    EVENTS = None


@dataclass
class _RoleStats:
    calls: int = 0
    errors: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0

    def as_dict(self) -> Dict[str, Any]:
        return {
            'calls': self.calls,
            'errors': self.errors,
            'prompt_tokens': self.prompt_tokens,
            'completion_tokens': self.completion_tokens,
            'latency_s': round(self.latency_s, 3),
        }


class _Call:
    """Handle yielded by `METER.timed`; carries usage back from the SDK response."""

    def __init__(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def observe(self, response: Any):
        """Extract token usage from a google-genai response, if the SDK reported it.

        `usage_metadata` is absent on some streaming/older paths, so every field is
        optional — a missing count records as 0 rather than failing the task.
        """
        usage = getattr(response, 'usage_metadata', None)
        if usage is None:
            return
        self.prompt_tokens += int(getattr(usage, 'prompt_token_count', 0) or 0)
        completion = getattr(usage, 'candidates_token_count', 0) or 0
        if not completion:
            total = getattr(usage, 'total_token_count', 0) or 0
            completion = max(0, int(total) - self.prompt_tokens)
        self.completion_tokens += int(completion)

    def observe_counts(self, prompt_tokens: int = 0, completion_tokens: int = 0):
        self.prompt_tokens += int(prompt_tokens or 0)
        self.completion_tokens += int(completion_tokens or 0)


@dataclass
class LLMMeter:
    by_role: Dict[str, _RoleStats] = field(
        default_factory=lambda: {r: _RoleStats() for r in ROLES})
    events: List[Dict[str, Any]] = field(default_factory=list)
    keep_events: bool = True

    def reset(self):
        self.by_role = {r: _RoleStats() for r in ROLES}
        self.events = []

    @contextmanager
    def timed(self, role: str = 'other'):
        """Time one LLM call. Exceptions are counted as errors and re-raised.

        `backoff` decorates the generate functions in both codebases, so a retried
        call re-enters this context manager: each physical API attempt is counted,
        which is the number that actually costs wall-clock and quota.
        """
        stats = self.by_role.setdefault(role, _RoleStats())
        call = _Call()
        t0 = time.time()
        try:
            yield call
        except Exception:
            stats.errors += 1
            stats.latency_s += time.time() - t0
            raise
        dt = time.time() - t0
        stats.calls += 1
        stats.latency_s += dt
        stats.prompt_tokens += call.prompt_tokens
        stats.completion_tokens += call.completion_tokens
        if self.keep_events:
            self.events.append({
                'role': role,
                'latency_s': round(dt, 3),
                'prompt_tokens': call.prompt_tokens,
                'completion_tokens': call.completion_tokens,
            })
        # Every physical API call in either framework passes through here, so this
        # is the only honest place to count them for the demo video. Counting
        # stage log lines instead overstates it badly: FOLBP prints one ORACLE line
        # per plan step, so a single Oracle call looks like eleven.
        if EVENTS is not None:
            EVENTS.emit('LLM_CALL', title=role, latency_s=round(dt, 3),
                        calls_so_far=sum(x.calls for x in self.by_role.values()))

    def snapshot(self) -> Dict[str, Any]:
        roles = {r: s.as_dict() for r, s in self.by_role.items()}
        return {
            'total_calls': sum(s.calls for s in self.by_role.values()),
            'total_errors': sum(s.errors for s in self.by_role.values()),
            'prompt_tokens': sum(s.prompt_tokens for s in self.by_role.values()),
            'completion_tokens': sum(s.completion_tokens for s in self.by_role.values()),
            'total_tokens': sum(s.prompt_tokens + s.completion_tokens
                                for s in self.by_role.values()),
            'llm_latency_s': round(sum(s.latency_s for s in self.by_role.values()), 3),
            'oracle_calls': self.by_role['oracle'].calls,
            'repair_calls': self.by_role['repair'].calls,
            'executor_calls': self.by_role['executor'].calls,
            'judge_calls': self.by_role['judge'].calls,
            'by_role': roles,
        }


METER = LLMMeter()
