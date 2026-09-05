#!/usr/bin/env python3
"""HTTP bridge between PEFA and crazyflie/mpc.py.

Same role as turtlebot/web_client.py but for the drone. Translates PEFA
symbolic actions into ROS service calls on the Crazyflie node.

PEFA action -> mapping:
    [takeoff_from] <X>             -> /drone/takeoff
    [movetowards] <setpoint X>(id) -> publish /drone/target, call /drone/fly_to
    [land_on] <setpoint X>(id)     -> publish /drone/target, call /drone/fly_to,
                                       then /drone/land

Setpoint poses live in drone_objects.json (keys "class(id)", values {x_m, y_m}).
"""

import argparse
import json
import re
import threading
import time
from pathlib import Path

from flask import Flask, request, jsonify

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose2D
from std_srvs.srv import Empty


# ---------------- ROS bridge ----------------

class RosBridge:
    def __init__(self):
        self.node = Node('drone_web_client')
        self.target_pub = self.node.create_publisher(Pose2D, '/drone/target', 10)
        self.cli_takeoff = self.node.create_client(Empty, '/drone/takeoff')
        self.cli_fly_to = self.node.create_client(Empty, '/drone/fly_to')
        self.cli_land = self.node.create_client(Empty, '/drone/land')
        self._spin_thread = threading.Thread(target=self._spin, daemon=True)

    def start(self):
        self._spin_thread.start()

    def _spin(self):
        while rclpy.ok():
            rclpy.spin_once(self.node, timeout_sec=0.1)

    def _publish_target(self, x_m, y_m):
        msg = Pose2D(x=float(x_m), y=float(y_m), theta=0.0)
        self.target_pub.publish(msg)
        time.sleep(0.1)
        self.target_pub.publish(msg)

    def _call(self, client, timeout_s=180.0):
        if not client.wait_for_service(timeout_sec=2.0):
            return False, f'service {client.srv_name} unavailable'
        fut = client.call_async(Empty.Request())
        deadline = time.time() + timeout_s
        while not fut.done() and time.time() < deadline:
            time.sleep(0.05)
        if fut.done():
            return True, 'ok'
        return False, 'timeout'

    def takeoff(self):
        return self._call(self.cli_takeoff, timeout_s=15.0)

    def fly_to(self, x_m, y_m):
        self._publish_target(x_m, y_m)
        return self._call(self.cli_fly_to, timeout_s=180.0)

    def land(self):
        return self._call(self.cli_land, timeout_s=30.0)


# ---------------- action parsing ----------------

VERB_RE = re.compile(r'\[(\w+)\]')
OBJ_RE = re.compile(r'<([^>]+)>\((\d+)\)')


def parse_action(s):
    m = VERB_RE.search(s)
    verb = m.group(1).lower() if m else None
    objs = [(om.group(1).strip(), int(om.group(2))) for om in OBJ_RE.finditer(s)]
    return verb, objs


def obj_key(cls, oid):
    return f'{cls}({oid})'


class ActionExecutor:
    def __init__(self, bridge: RosBridge, objects_map: dict):
        self.bridge = bridge
        self.objects = objects_map

    def lookup(self, cls, oid):
        d = self.objects.get(obj_key(cls, oid))
        if d is None:
            return None
        return float(d['x_m']), float(d['y_m'])

    def execute(self, verb, objs):
        if verb == 'takeoff_from':
            return self.bridge.takeoff()

        if verb == 'movetowards':
            if not objs:
                return False, 'movetowards needs target'
            cls, oid = objs[0]
            tgt = self.lookup(cls, oid)
            if tgt is None:
                return False, f'unknown target: {obj_key(cls, oid)} (not in drone_objects.json)'
            return self.bridge.fly_to(*tgt)

        if verb == 'land_on':
            if not objs:
                return self.bridge.land()
            cls, oid = objs[0]
            tgt = self.lookup(cls, oid)
            if tgt is not None:
                ok, info = self.bridge.fly_to(*tgt)
                if not ok:
                    return False, f'fly_to before land failed: {info}'
            return self.bridge.land()

        if verb in ('grab', 'grasp', 'put_on', 'put_inside', 'puton', 'putinto', 'open', 'close'):
            return False, f'verb [{verb}] not applicable to a drone'

        return False, f'unsupported verb: [{verb}]'


# ---------------- Flask ----------------

app = Flask(__name__)
bridge: RosBridge = None
executor: ActionExecutor = None


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'ok': True, 'role': 'drone_web_client',
                    'objects_loaded': len(executor.objects) if executor else 0})


@app.route('/execute', methods=['POST'])
def execute():
    try:
        data = request.get_json(force=True, silent=True) or {}
        action = data.get('action', '').strip()
        if not action:
            return jsonify({'success': False, 'info': 'no "action" in request'}), 400
        verb, objs = parse_action(action)
        if not verb:
            return jsonify({'success': False, 'info': f'could not parse: {action!r}'}), 400
        bridge.node.get_logger().info(f'[execute] verb={verb} objs={objs}')
        ok, info = executor.execute(verb, objs)
        bridge.node.get_logger().info(f'[execute] -> success={ok} info={info}')
        return jsonify({'success': ok, 'info': info, 'verb': verb})
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        try:
            bridge.node.get_logger().error(f'[execute] EXCEPTION: {e}\n{tb}')
        except Exception:
            print(f'[execute] EXCEPTION: {e}\n{tb}')
        return jsonify({'success': False, 'info': f'server_exception: {e}'}), 500


@app.route('/takeoff', methods=['POST'])
def http_takeoff():
    ok, info = bridge.takeoff()
    return jsonify({'success': ok, 'info': info})


@app.route('/fly_to', methods=['POST'])
def http_fly_to():
    d = request.get_json(force=True, silent=True) or {}
    ok, info = bridge.fly_to(d.get('x_m', 0), d.get('y_m', 0))
    return jsonify({'success': ok, 'info': info})


@app.route('/land', methods=['POST'])
def http_land():
    ok, info = bridge.land()
    return jsonify({'success': ok, 'info': info})


# ---------------- entry ----------------

def load_objects(path):
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        print(f'[warn] {path} not found — drone will reject named-object actions')
        return {}
    raw = json.load(open(p))
    return {k: v for k, v in raw.items() if not k.startswith('_')}


def main():
    global bridge, executor
    ap = argparse.ArgumentParser()
    ap.add_argument('--host', default='0.0.0.0')
    ap.add_argument('--port', type=int, default=8080)
    ap.add_argument('--objects-file', default='drone_objects.json')
    args = ap.parse_args()

    rclpy.init()
    bridge = RosBridge()
    bridge.start()

    objects_map = load_objects(args.objects_file)
    executor = ActionExecutor(bridge, objects_map)
    print(f'drone_web_client: loaded {len(objects_map)} objects from {args.objects_file}')
    print(f'drone_web_client: listening on http://{args.host}:{args.port}')

    try:
        # threaded=False keeps requests sequential (matches drone's one-action-at-a-time
        # operation); concurrent /health probes still work because they don't go through
        # ROS service calls and are very fast.
        app.run(host=args.host, port=args.port, threaded=True, use_reloader=False)
    finally:
        bridge.node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
