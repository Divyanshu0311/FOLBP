"""Per-task result records — the single source of truth for every benchmark number.

One JSON file per (arm, seed, env, task). The process that ran the task writes its
own record, which is what makes `sweep.py --parallel` safe: no shared append-mode
log, no interleaving, no "keep the last occurrence" heuristic over a pile of
historical runs.

Writes are atomic (temp file + os.replace), so a killed process can never leave a
half-written record that `--resume` would mistake for a completed one.

Every record carries its own provenance (git SHA, model, temperature, resolved
flags). The run tree is deliberately *not* timestamped, so incremental invocations
accumulate into one table — and `report.py` uses the per-record provenance to warn
when rows in that table came from different code versions.
"""
import json
import os
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

SCHEMA_VERSION = 1

STATUS_COMPLETED = 'completed'
STATUS_CRASHED = 'crashed'
STATUS_TIMEOUT = 'timeout'

_GIT_CACHE: Optional[Dict[str, Any]] = None


def _git_provenance() -> Dict[str, Any]:
    """git SHA + dirty flag, resolved once per process."""
    global _GIT_CACHE
    if _GIT_CACHE is not None:
        return _GIT_CACHE
    sha, dirty = None, None
    try:
        root = Path(__file__).resolve().parent
        sha = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=root,
                             capture_output=True, text=True, timeout=10).stdout.strip() or None
        status = subprocess.run(['git', 'status', '--porcelain'], cwd=root,
                                capture_output=True, text=True, timeout=10).stdout
        dirty = bool(status.strip())
    except Exception:
        pass
    _GIT_CACHE = {'git_sha': sha, 'git_dirty': dirty}
    return _GIT_CACHE


_SECRET_FLAGS = ('--api_key', '--apikey', '--openai_api_key', '--organization')


def redact_argv(argv):
    """Drop secret values out of a recorded argv.

    Records are committed alongside the paper, so the key must never reach one.
    Handles both `--api_key KEY` and `--api_key=KEY`.
    """
    out, redact_next = [], False
    for tok in argv:
        if redact_next:
            out.append('<redacted>')
            redact_next = False
            continue
        if tok in _SECRET_FLAGS:
            out.append(tok)
            redact_next = True
        elif tok.startswith(_SECRET_FLAGS) and '=' in tok:
            out.append(tok.split('=', 1)[0] + '=<redacted>')
        else:
            out.append(tok)
    return out


def build_manifest(arm: str, seed: int, model: str, temperature: float,
                   flags: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {
        **_git_provenance(),
        'arm': arm,
        'seed': seed,
        'model': model,
        'temperature': temperature,
        'flags': flags or {},
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'host': socket.gethostname(),
        'python': platform.python_version(),
        'argv': redact_argv(sys.argv[1:]),
    }


def as_scalar_int(value: Any) -> Optional[int]:
    """Coerce a step count to an int.

    The env JSONs store `ground_truth_step_num` as a one-element list ([4], not 4).
    The old metrics scripts never saw that because they re-parsed the number out of
    log text; the records take it straight from the JSON, so it is normalized here —
    at the boundary — rather than in every consumer.
    """
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        value = value[0] if len(value) == 1 else None
        if value is None:
            return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def atomic_write_json(path: Path, payload: Dict[str, Any]):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f'.tmp.{os.getpid()}')
    with open(tmp, 'w') as f:
        json.dump(payload, f, indent=2, default=str)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def write_task_record(path: Path, *, arm: str, seed: int, env: str, task_id: int,
                      success: bool, gt_steps: Optional[int], executed_steps: Optional[int],
                      wall_clock_s: float, llm: Dict[str, Any],
                      env_id: Optional[int] = None, task_name: Optional[str] = None,
                      goal: Optional[str] = None, model: str = '', temperature: float = 0.0,
                      flags: Optional[Dict[str, Any]] = None,
                      verify: Optional[Dict[str, Any]] = None,
                      repair: Optional[Dict[str, Any]] = None,
                      cdcl: Optional[Dict[str, Any]] = None,
                      deadlock: Optional[Dict[str, Any]] = None,
                      outer_iters: Optional[int] = None,
                      failures: Optional[List[Dict[str, Any]]] = None,
                      status: str = STATUS_COMPLETED,
                      extra: Optional[Dict[str, Any]] = None) -> Path:
    """Write a completed task record. Called by the planner process itself."""
    payload = {
        'schema_version': SCHEMA_VERSION,
        'status': status,
        'arm': arm,
        'seed': seed,
        'env': env,
        'env_id': env_id,
        'task_id': task_id,
        'task_name': task_name,
        'goal': goal,
        'success': bool(success),
        'gt_steps': as_scalar_int(gt_steps),
        'executed_steps': as_scalar_int(executed_steps),
        'wall_clock_s': round(float(wall_clock_s), 3),
        'outer_iters': outer_iters,
        'llm': llm or {},
        'verify': verify or {},
        'repair': repair or {},
        'cdcl': cdcl or {},
        # Optional, like the three blocks above: records written before the deadlock
        # ablation existed simply lack the key. Adding it must NOT bump SCHEMA_VERSION —
        # read_record() rejects any other version, which would make every earlier record
        # invisible and unrender the published table.
        'deadlock': deadlock or {},
        'failures': failures or [],
        'manifest': build_manifest(arm, seed, model, temperature, flags),
    }
    if extra:
        payload.update(extra)
    atomic_write_json(path, payload)
    return Path(path)


def write_stub_record(path: Path, *, arm: str, seed: int, env: str, task_id: int,
                      status: str, wall_clock_s: float, detail: str = '',
                      model: str = '', temperature: float = 0.0,
                      flags: Optional[Dict[str, Any]] = None) -> Path:
    """Write a crashed/timeout record. Called by the *parent* sweep process.

    The child never got to write one, so without this the run would be
    indistinguishable from "never attempted" and `--resume` could not report it.
    """
    payload = {
        'schema_version': SCHEMA_VERSION,
        'status': status,
        'arm': arm,
        'seed': seed,
        'env': env,
        'task_id': task_id,
        'success': False,
        'gt_steps': None,
        'executed_steps': None,
        'wall_clock_s': round(float(wall_clock_s), 3),
        'llm': {},
        'verify': {},
        'repair': {},
        'cdcl': {},
        'deadlock': {},
        'failures': [{'kind': status, 'detail': detail[:2000]}],
        'manifest': build_manifest(arm, seed, model, temperature, flags),
    }
    atomic_write_json(path, payload)
    return Path(path)


def read_record(path: Path) -> Optional[Dict[str, Any]]:
    """Load a record, returning None if absent or unparseable (treated as not-done)."""
    try:
        with open(path) as f:
            rec = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(rec, dict) or rec.get('schema_version') != SCHEMA_VERSION:
        return None
    return rec


def record_path(root: Path, arm: str, seed: int, env: str, task_id: int) -> Path:
    """results/bench/runs/<arm>/seed<S>/<env>/task<NN>.json — stable, not timestamped."""
    return Path(root) / 'runs' / arm / f'seed{seed}' / env / f'task{task_id:02d}.json'


def log_path(root: Path, arm: str, seed: int, env: str, task_id: int) -> Path:
    return Path(root) / 'runs' / arm / f'seed{seed}' / env / f'task{task_id:02d}.log'


def iter_records(root: Path):
    """Yield every parseable record under the run tree."""
    runs = Path(root) / 'runs'
    if not runs.exists():
        return
    for path in sorted(runs.rglob('task*.json')):
        rec = read_record(path)
        if rec is not None:
            yield path, rec
