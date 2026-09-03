"""Process-scoped stage-event stream, for the demo video compositor.

The simulator and the planner are separate processes under different Python
interpreters (Isaac Sim's 3.7 vs. the planner's 3.10), so they cannot share
objects — but they do share a wall clock. Each side writes a timestamped stream
during the run and `tools/compose_video.py` joins them afterwards:

    sim.py        --record      -> sim.mp4 + frames.jsonl   {"i": 12, "t": ...}
    PEFA/FOLBP    --event_log   -> events.jsonl             {"t": ..., "stage": ...}

`time.time()` is the only synchronisation mechanism. Nothing here talks to the
simulator.

Shaped like METER in llm_meter.py, and for the same reason: one task per
subprocess means a module-level singleton is the whole story, with no plumbing
through the arena.

Disabled (the default) every call is one attribute check, so benchmark runs are
unaffected — `results/bench/` numbers must not move because of this module.

Usage:
    from bench.eventlog import EVENTS
    EVENTS.open(path)                        # or leave closed for a no-op
    EVENTS.emit('ORACLE', title='step 3', body=message, latency_s=2.4)

Records carry the full text. The compositor decides what to show, so re-editing
the on-screen condensation never means re-running the planner.
"""
import json
import os
import time


class EventLog:
    def __init__(self):
        self._f = None
        self._t0 = None

    def open(self, path=None):
        """Start writing to `path`. Falls back to $COHERENT_EVENT_LOG; no-op if neither."""
        path = path or os.environ.get('COHERENT_EVENT_LOG')
        if not path:
            return self
        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._f = open(path, 'a')
        self._t0 = time.time()
        self.emit('OPEN', title='event log opened', path=path)
        return self

    @property
    def enabled(self):
        return self._f is not None

    def emit(self, stage, title='', body='', **meta):
        """Append one stage record. Never raises — a broken log must not fail a run."""
        if self._f is None:
            return
        rec = {
            't': time.time(),
            'stage': str(stage),
            'title': str(title),
            'body': '' if body is None else str(body),
        }
        rec.update(meta)
        try:
            self._f.write(json.dumps(rec, ensure_ascii=False, default=str) + '\n')
            # Flushed per record: the compositor may tail this while the run is live,
            # and a crashed run should still leave a readable transcript.
            self._f.flush()
        except Exception:
            pass

    def close(self):
        if self._f is None:
            return
        self.emit('CLOSE', title='event log closed')
        try:
            self._f.close()
        finally:
            self._f = None


EVENTS = EventLog()


def add_event_log_arg(parser):
    """Register --event_log on a planner's argparse parser."""
    parser.add_argument('--event_log', default=None,
                        help='Write a stage-event JSONL stream here, for the demo '
                             'video compositor. Defaults to $COHERENT_EVENT_LOG; '
                             'omit both to disable.')
