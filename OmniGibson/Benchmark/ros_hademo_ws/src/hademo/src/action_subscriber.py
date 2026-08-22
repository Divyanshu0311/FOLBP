#!/usr/bin/env python3
"""
Simulator-side ROS 2 endpoint for the COHERENT benchmark.

Ported from ROS 1 (rospy) to ROS 2 (rclpy).  Topics and semantics are unchanged:

    subscribes  /actionTopic   hademo/msg/Action    (from action_publisher.py)
    publishes   /resultTopic   hademo/msg/Result    (back to action_publisher.py)

Both use a latched-equivalent QoS profile (RELIABLE + TRANSIENT_LOCAL) so the
warmup handshake still works when one side joins after the other.

--------------------------------------------------------------------------
Why this file has two implementations
--------------------------------------------------------------------------
Under ROS 1 this module was imported straight into sim.py, because rospy and
its generated messages are pure Python and import under any interpreter.

rclpy is not: it is a C extension linked against the exact CPython that ROS 2
Humble was built for (system python3.10).  Isaac Sim 2022.2.0 embeds Python 3.7
and OmniGibson's conda env is created to match it, so `import rclpy` inside
sim.py fails with an ABI/version error no matter how PYTHONPATH is set.

So the class sim.py imports picks one of two backends at construction time:

  _InProcessRos2   used when rclpy imports cleanly (python3.10).  A real
                   rclpy.Node spun on a background thread -- the direct
                   equivalent of the old rospy code.

  _SidecarClient   used otherwise (Isaac Sim's python3.7).  Launches THIS SAME
                   FILE as a separate `--sidecar` process under a python that
                   does have rclpy, and talks to it over a UNIX socket with
                   newline-delimited JSON.  The sidecar is the actual ROS 2
                   node, so /actionTopic and /resultTopic are ordinary ROS 2
                   topics -- `ros2 topic echo`, rqt and rosbag2 all work.

Either way sim.py sees the same two members it always used:
    node._next_step_action        -> dict or None
    node.publish_feedback_result(feedback_result)
"""

import argparse
import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_WS_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))  # ros_hademo_ws

coherent_path = os.environ.get('COHERENT_PATH')
if coherent_path:
    sys.path.append(os.path.join(coherent_path, 'OmniGibson', 'Benchmark',
                                 'ros_hademo_ws', 'src'))

# `defaults_<task_name>` objects live in `tasks`.  The sidecar process does not
# need them (the simulator sends its agent list over the socket), so a failure
# here must not be fatal.
try:
    from tasks import *  # noqa: F401,F403
    _TASKS_IMPORT_ERROR = None
except Exception as _exc:  # pragma: no cover - depends on caller's sys.path
    _TASKS_IMPORT_ERROR = _exc


def _load_agent_name_list(task_name):
    if _TASKS_IMPORT_ERROR is not None:
        raise RuntimeError(
            "cannot import `tasks` to resolve defaults_%s (%s). Set COHERENT_PATH "
            "or run from the Benchmark directory." % (task_name, _TASKS_IMPORT_ERROR))
    return list(eval("defaults_%s" % task_name).agent_name_list)


# ---------------------------------------------------------------------------
# message <-> plain-dict translation (shared by both backends)
# ---------------------------------------------------------------------------

def decode_action_msg(action_msg, agent_name_list):
    """hademo/msg/Action -> {agent: [] | [func_name] | [func_name, args_dict]}

    Waypoints come out as FLAT lists of float rather than numpy arrays so the
    result is JSON-serialisable; `numpyify_action` restores the arrays that
    haskillset expects.  Field selection per func_name matches the ROS 1 code.
    """
    next_step_action = {name: [] for name in agent_name_list}

    for agent_name in agent_name_list:
        fa = getattr(action_msg, agent_name)
        if not fa.has_func:
            continue

        func_name = fa.func_name
        next_step_action[agent_name].append(func_name)

        if not fa.args.has_args:
            next_step_action[agent_name].append({"agent_name": agent_name})
            continue

        if 'pick' in func_name:
            args = {
                "agent_name": agent_name,
                "attached_prim_path": fa.args.attached_prim_path,
                "waypoint_pos": list(fa.args.waypoint_pos.data),
                "waypoint_ori": list(fa.args.waypoint_ori.data),
                "waypoint_ind": int(fa.args.waypoint_ind),
            }
        elif 'attached_prim' in func_name or 'door_open' in func_name:
            args = {
                "agent_name": agent_name,
                "attached_prim_path": fa.args.attached_prim_path,
            }
        else:
            args = {
                "agent_name": agent_name,
                "waypoint_pos": list(fa.args.waypoint_pos.data),
                "waypoint_ori": list(fa.args.waypoint_ori.data),
                "waypoint_ind": int(fa.args.waypoint_ind),
            }
        next_step_action[agent_name].append(args)

    return next_step_action


def numpyify_action(next_step_action):
    """Reshape the flat waypoint lists back into (N, 3) / (N, 4) numpy arrays."""
    for entry in next_step_action.values():
        if len(entry) < 2 or not isinstance(entry[1], dict):
            continue
        args = entry[1]
        if "waypoint_pos" in args:
            args["waypoint_pos"] = np.array(args["waypoint_pos"], dtype=np.float64).reshape(-1, 3)
        if "waypoint_ori" in args:
            args["waypoint_ori"] = np.array(args["waypoint_ori"], dtype=np.float64).reshape(-1, 4)
    return next_step_action


def encode_feedback_result(feedback_result, agent_name_list, Result, ResultInfo):
    """
        feedback_result shape (produced by sim.running_next_step_action):
        {
            "franka_0":    {"has_result": bool, "success": bool, "info": str},
            "aliengo_0":   {...},
            "quadrotor_0": {...},
            ...
        }
        Agents not present in feedback_result (or with empty dict) are reported
        as has_result=False so the downstream bridge can distinguish "this agent
        did not act this round" from "acted and failed".
    """
    result_msg = Result()
    for agent_name in agent_name_list:
        ri = ResultInfo()
        agent_fb = feedback_result.get(agent_name, {}) if feedback_result else {}
        if isinstance(agent_fb, dict) and agent_fb.get("has_result", False):
            ri.has_result = True
            ri.success = bool(agent_fb.get("success", False))
            info = agent_fb.get("info", "")
            # Force to string -- logger may return dicts / numbers
            ri.info = info if isinstance(info, str) else str(info)
        else:
            ri.has_result = False
            ri.success = False
            ri.info = ""
        setattr(result_msg, agent_name, ri)
    return result_msg


# ---------------------------------------------------------------------------
# backend A: real rclpy node, in this process
# ---------------------------------------------------------------------------

def _build_ros2_node_class():
    """Imported lazily so the module still loads under Isaac Sim's python3.7."""
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import (
        QoSProfile,
        QoSDurabilityPolicy,
        QoSHistoryPolicy,
        QoSReliabilityPolicy,
    )
    from hademo.msg import Action, Result, ResultInfo

    # ROS 1 `latch=True` -> TRANSIENT_LOCAL in ROS 2.  Must match on both ends.
    latched_qos = QoSProfile(
        depth=10,
        history=QoSHistoryPolicy.KEEP_LAST,
        reliability=QoSReliabilityPolicy.RELIABLE,
        durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
    )

    class _SimNode(Node):
        def __init__(self, task_name, agent_name_list):
            super().__init__('Sim')
            self.task_name = task_name
            self.agent_name_list = list(agent_name_list)
            self.next_step_action = None
            self._on_action_hook = None

            self._pub = self.create_publisher(Result, 'resultTopic', latched_qos)
            self._sub = self.create_subscription(
                Action, 'actionTopic', self._callback, latched_qos)

        def wait_for_llm_and_greet(self, timeout=None):
            """Replaces rospy.get_published_topics() polling for /actionTopic."""
            print("Sim side waits for communication", flush=True)
            start = time.time()
            while rclpy.ok():
                if self.count_publishers('actionTopic') > 0:
                    break
                if timeout is not None and (time.time() - start) > timeout:
                    raise RuntimeError(
                        "no publisher on /actionTopic after %ss -- is "
                        "action_publisher.py running on ROS_DOMAIN_ID=%s?"
                        % (timeout, os.environ.get('ROS_DOMAIN_ID', '0')))
                time.sleep(0.1)
            # Sim side invoke communication
            print("Sim side invoke communication", flush=True)
            self._pub.publish(Result())

        def _callback(self, action_msg):
            decoded = decode_action_msg(action_msg, self.agent_name_list)
            if self._on_action_hook is not None:
                self._on_action_hook(decoded)
            else:
                self.next_step_action = numpyify_action(decoded)

        def publish_feedback_result(self, feedback_result):
            self.next_step_action = None
            self._pub.publish(encode_feedback_result(
                feedback_result, self.agent_name_list, Result, ResultInfo))

    return rclpy, _SimNode


class _BackgroundSpin(object):
    """Spins a node on its own thread and tears it down in the right order.

    Calling node.destroy_node() while a spin thread is still servicing that node
    aborts the process ("terminate called without an active exception"), so the
    executor must be stopped and joined first.
    """

    def __init__(self, rclpy_mod, node):
        from rclpy.executors import SingleThreadedExecutor
        self._rclpy = rclpy_mod
        self._node = node
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(node)
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._closed = False
        self._thread.start()

    def _spin(self):
        try:
            self._executor.spin()
        except Exception:
            # ExternalShutdownException on SIGINT/SIGTERM, or the executor being
            # shut down from close() -- both are ordinary stop conditions here.
            pass

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            self._executor.shutdown()
        except Exception:
            pass
        self._thread.join(timeout=5)
        try:
            self._executor.remove_node(self._node)
        except Exception:
            pass
        try:
            self._node.destroy_node()
        except Exception:
            pass


class _InProcessRos2(object):
    """Backend used when rclpy is importable (system python3.10)."""

    def __init__(self, task_name, agent_name_list):
        self._rclpy, node_cls = _build_ros2_node_class()
        if not self._rclpy.ok():
            self._rclpy.init()
            self._owns_context = True
        else:
            self._owns_context = False

        self._node = node_cls(task_name, agent_name_list)
        # rclpy has no implicit spin thread; sim.py's own loop owns the main
        # thread, so callbacks have to be serviced from here.
        self._spin = _BackgroundSpin(self._rclpy, self._node)
        self._node.wait_for_llm_and_greet()

    @property
    def next_step_action(self):
        return self._node.next_step_action

    @next_step_action.setter
    def next_step_action(self, value):
        # Keep the facade's setter working on both backends.
        self._node.next_step_action = value

    def publish_feedback_result(self, feedback_result):
        self._node.publish_feedback_result(feedback_result)

    def shutdown(self):
        self._spin.close()
        if self._owns_context and self._rclpy.ok():
            self._rclpy.shutdown()


# ---------------------------------------------------------------------------
# backend B: sidecar process + UNIX socket (Isaac Sim's python3.7)
# ---------------------------------------------------------------------------

def _json_default(obj):
    """Coerce numpy scalars/arrays the stdlib json encoder rejects.

    Values in `feedback_result` come straight out of skill execution, where a
    numpy bool_/int64 is indistinguishable from a Python bool/int until json
    refuses it.  (np.float64 subclasses float, so it never reaches here -- it
    is np.bool_ and the integer types that bite, since they subclass nothing.)
    Only the sidecar backend needs this; the in-process backend assigns to
    typed ROS 2 message fields, which coerce on their own.
    """
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(
        "Object of type %s is not JSON serializable" % type(obj).__name__)


def _send_line(sock, obj):
    sock.sendall(
        (json.dumps(obj, default=_json_default) + "\n").encode("utf-8"))


def _line_reader(sock):
    """Yield decoded JSON objects from a newline-delimited stream."""
    buf = b""
    while True:
        chunk = sock.recv(65536)
        if not chunk:
            return
        buf += chunk
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            line = line.strip()
            if line:
                yield json.loads(line.decode("utf-8"))


class _SidecarClient(object):
    """Backend used when rclpy is NOT importable. Owns the sidecar process."""

    READY_TIMEOUT = float(os.environ.get("HADEMO_SIDECAR_TIMEOUT", "180"))

    def __init__(self, task_name, agent_name_list):
        self.next_step_action = None
        self._agent_name_list = list(agent_name_list)
        self._ready = threading.Event()
        self._lock = threading.Lock()

        self._sock_path = os.path.join(
            tempfile.mkdtemp(prefix="hademo_"), "sidecar.sock")

        self._server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._server.bind(self._sock_path)
        self._server.listen(1)

        self._proc = self._spawn_sidecar(task_name)

        self._server.settimeout(self.READY_TIMEOUT)
        try:
            self._conn, _ = self._server.accept()
        except socket.timeout:
            self._dump_sidecar_death()
            raise RuntimeError(
                "ROS 2 sidecar never connected within %ss. Check that "
                "%s is sourceable and that the hademo package is built "
                "(colcon build)." % (self.READY_TIMEOUT, self._env_script()))
        self._conn.settimeout(None)

        _send_line(self._conn, {
            "type": "hello",
            "task_name": task_name,
            "agent_name_list": self._agent_name_list,
        })

        self._reader_thread = threading.Thread(target=self._read_loop, daemon=True)
        self._reader_thread.start()

        print("Sim side waits for communication", flush=True)
        if not self._ready.wait(timeout=self.READY_TIMEOUT):
            self._dump_sidecar_death()
            raise RuntimeError(
                "ROS 2 sidecar did not report ready within %ss (waiting for a "
                "publisher on /actionTopic -- is action_publisher.py running?)"
                % self.READY_TIMEOUT)
        print("Sim side invoke communication", flush=True)

    @staticmethod
    def _env_script():
        return os.environ.get(
            "HADEMO_ROS2_ENV", os.path.join(_WS_ROOT, "ros2_env.sh"))

    def _spawn_sidecar(self, task_name):
        env_script = self._env_script()
        python_bin = os.environ.get("HADEMO_ROS2_PYTHON", "/usr/bin/python3")
        cmd = (
            'source "%s" && exec "%s" "%s" --sidecar --task-name "%s" --socket "%s"'
            % (env_script, python_bin, os.path.join(_HERE, "action_subscriber.py"),
               task_name, self._sock_path)
        )
        print("[hademo] launching ROS 2 sidecar: %s" % cmd, flush=True)

        # Isaac Sim's conda env exports PYTHONPATH/PYTHONHOME pointing at its own
        # python3.7 runtime; inheriting those would break the python3.10 sidecar.
        env = {k: v for k, v in os.environ.items()
               if k not in ("PYTHONPATH", "PYTHONHOME", "LD_LIBRARY_PATH")}
        env["COHERENT_PATH"] = os.environ.get("COHERENT_PATH", "")
        return subprocess.Popen(["bash", "-c", cmd], env=env)

    def _dump_sidecar_death(self):
        rc = self._proc.poll()
        if rc is not None:
            print("[hademo] sidecar exited with code %s" % rc, flush=True)

    def _read_loop(self):
        try:
            for msg in _line_reader(self._conn):
                kind = msg.get("type")
                if kind == "ready":
                    self._ready.set()
                elif kind == "action":
                    with self._lock:
                        self.next_step_action = numpyify_action(msg["data"])
                elif kind == "error":
                    print("[hademo] sidecar error: %s" % msg.get("info"), flush=True)
        except (OSError, ValueError) as exc:
            print("[hademo] sidecar connection closed: %s" % exc, flush=True)

    def publish_feedback_result(self, feedback_result):
        with self._lock:
            self.next_step_action = None
        _send_line(self._conn, {"type": "result", "data": feedback_result})

    def shutdown(self):
        try:
            _send_line(self._conn, {"type": "shutdown"})
        except OSError:
            pass
        try:
            self._proc.wait(timeout=10)
        except Exception:
            self._proc.kill()
        for closer in (getattr(self, "_conn", None), getattr(self, "_server", None)):
            try:
                closer.close()
            except Exception:
                pass
        try:
            os.unlink(self._sock_path)
            os.rmdir(os.path.dirname(self._sock_path))
        except OSError:
            pass


# ---------------------------------------------------------------------------
# public facade -- this is what sim.py imports
# ---------------------------------------------------------------------------

def _rclpy_available():
    try:
        import rclpy  # noqa: F401
        return True
    except Exception:
        return False


class ResultPublishandActionSubscribe(object):
    """Drop-in replacement for the ROS 1 class of the same name.

    sim.py only ever touches `_next_step_action` and `publish_feedback_result`,
    both of which behave exactly as before.
    """

    def __init__(self, task_name, agent_name_list=None, backend=None):
        self.task_name = task_name
        self.agent_name_list = (list(agent_name_list) if agent_name_list
                                else _load_agent_name_list(task_name))

        if backend is None:
            backend = os.environ.get("HADEMO_BACKEND")
        if backend is None:
            backend = "inprocess" if _rclpy_available() else "sidecar"

        if backend == "inprocess":
            self._impl = _InProcessRos2(task_name, self.agent_name_list)
        elif backend == "sidecar":
            why = ("requested explicitly" if _rclpy_available()
                   else "rclpy not importable under python %s" % sys.version.split()[0])
            print("[hademo] using ROS 2 sidecar (%s)" % why, flush=True)
            self._impl = _SidecarClient(task_name, self.agent_name_list)
        else:
            raise ValueError("unknown backend %r" % backend)

    @property
    def _next_step_action(self):
        return self._impl.next_step_action

    @_next_step_action.setter
    def _next_step_action(self, value):
        self._impl.next_step_action = value

    def publish_feedback_result(self, feedback_result):
        self._impl.publish_feedback_result(feedback_result)

    def shutdown(self):
        self._impl.shutdown()

    def getactionTemplate(self):
        return {name: [] for name in self.agent_name_list}


# ---------------------------------------------------------------------------
# entry points
# ---------------------------------------------------------------------------

def run_sidecar(task_name, sock_path):
    """The ROS 2 half of backend B. Runs under a python that has rclpy."""
    rclpy, node_cls = _build_ros2_node_class()

    conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    conn.connect(sock_path)

    reader = _line_reader(conn)
    hello = next(reader)
    assert hello.get("type") == "hello", "expected hello, got %r" % hello
    agent_name_list = hello["agent_name_list"]

    rclpy.init()
    node = node_cls(hello.get("task_name", task_name), agent_name_list)

    # Forward every incoming Action straight down the socket instead of caching
    # it locally -- the simulator process is the one that consumes it.
    node._on_action_hook = lambda decoded: _send_line(conn, {"type": "action", "data": decoded})

    spinner = _BackgroundSpin(rclpy, node)

    node.wait_for_llm_and_greet()
    _send_line(conn, {"type": "ready"})
    print("[hademo-sidecar] ready; bridging /actionTopic <-> /resultTopic", flush=True)

    try:
        for msg in reader:
            kind = msg.get("type")
            if kind == "result":
                node.publish_feedback_result(msg["data"])
            elif kind == "shutdown":
                break
    except (OSError, ValueError) as exc:
        print("[hademo-sidecar] socket closed: %s" % exc, flush=True)
    finally:
        spinner.close()
        if rclpy.ok():
            rclpy.shutdown()
        conn.close()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task_name', '--task-name', dest='task_name',
                        type=str, default='Merom_1_int_Task1')
    parser.add_argument('--sidecar', action='store_true',
                        help='run as the ROS 2 sidecar for a python3.7 simulator')
    parser.add_argument('--socket', type=str, default=None,
                        help='UNIX socket path (required with --sidecar)')
    return parser.parse_args()


def main():
    args = parse_args()

    if args.sidecar:
        if not args.socket:
            raise SystemExit("--sidecar requires --socket")
        run_sidecar(args.task_name, args.socket)
        return

    # Standalone: a plain ROS 2 node, useful for testing the link without Isaac Sim.
    import rclpy
    node = ResultPublishandActionSubscribe(args.task_name, backend='inprocess')
    print("[hademo] standalone Sim node running; Ctrl-C to stop", flush=True)
    try:
        while rclpy.ok():
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()


if __name__ == '__main__':
    main()
