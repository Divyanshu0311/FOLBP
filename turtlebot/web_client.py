#!/usr/bin/env python3
"""HTTP bridge between PEFA and mission_controller.py.

Architecture:
    PEFA (laptop) ──HTTP POST /execute──▶ web_client.py (TurtleBot)
                                                │
                                                │ rclpy
                                                ▼
                                       mission_controller.py
                                                │
                                                ▼  /cmd_vel, /goal, etc.

PEFA generates symbolic actions like "[movetowards] <apple>(300)". This server
looks each referenced object up in objects.json (a map of "class(id)" -> x_cm/y_cm/
theta_deg) and breaks the action into mission_controller service calls:

    [movetowards] <X>   goto_abs(70 % of pose) + visual_servo
    [grab] <X>          (gripper close — placeholder until gripper services exist)
    [put_on/put_inside] goto_abs to destination + (gripper open — placeholder)
    [open/close] <X>    (placeholder for door/container)
    [takeoff_from], [land_on], etc. (quadrotor)   accepted as no-op

The /execute endpoint returns {"success": bool, "info": str} only AFTER the
underlying action completes — synchronous, blocking. PEFA's run loop
sees this response and only advances its symbolic graph on success.

Run:
    python3 web_client.py --port 8080 --objects-file objects.json
"""

import argparse
import json
import math
import re
import threading
import time
from pathlib import Path

from flask import Flask, request, jsonify

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose2D
from std_srvs.srv import Empty


# ----------------- ROS bridge -----------------

class RosBridge:
    """Owns the rclpy node + service clients. Spinning runs in a background thread."""

    def __init__(self):
        self.node = Node('web_client')
        self.target_pub = self.node.create_publisher(Pose2D, '/mission/target', 10)
        self.cli_goto_abs = self.node.create_client(Empty, '/mission/goto_abs')
        self.cli_goto_rel_body = self.node.create_client(Empty, '/mission/goto_rel_body')
        self.cli_goto_rel_world = self.node.create_client(Empty, '/mission/goto_rel_world')
        self.cli_vs = self.node.create_client(Empty, '/mission/visual_servo')
        self.cli_reset = self.node.create_client(Empty, '/mission/reset_localization')
        self.cli_pick = self.node.create_client(Empty, '/mission/pick')
        self.cli_place = self.node.create_client(Empty, '/mission/place')
        self.cli_gripper_open = self.node.create_client(Empty, '/mission/gripper_open')
        self.cli_gripper_close = self.node.create_client(Empty, '/mission/gripper_close')
        self._spin_thread = threading.Thread(target=self._spin_loop, daemon=True)

    def start(self):
        self._spin_thread.start()

    def _spin_loop(self):
        while rclpy.ok():
            rclpy.spin_once(self.node, timeout_sec=0.1)

    def _publish_target(self, x_m, y_m, theta_rad):
        msg = Pose2D(x=x_m, y=y_m, theta=theta_rad)
        self.target_pub.publish(msg)
        # Re-publish + small wait so DDS delivers before the service call.
        time.sleep(0.1)
        self.target_pub.publish(msg)

    def _call(self, client, timeout_s):
        if not client.wait_for_service(timeout_sec=2.0):
            return False, f'service {client.srv_name} unavailable'
        fut = client.call_async(Empty.Request())
        deadline = time.time() + timeout_s
        while not fut.done() and time.time() < deadline:
            time.sleep(0.05)
        if fut.done():
            return True, 'ok'
        return False, 'timeout'

    def goto_abs(self, x_m, y_m, theta_rad, timeout_s=120.0):
        self._publish_target(x_m, y_m, theta_rad)
        return self._call(self.cli_goto_abs, timeout_s)

    def goto_rel_body(self, x_m, y_m, theta_rad, timeout_s=120.0):
        self._publish_target(x_m, y_m, theta_rad)
        return self._call(self.cli_goto_rel_body, timeout_s)

    def goto_rel_world(self, x_m, y_m, theta_rad, timeout_s=120.0):
        self._publish_target(x_m, y_m, theta_rad)
        return self._call(self.cli_goto_rel_world, timeout_s)

    def visual_servo(self, timeout_s=120.0):
        return self._call(self.cli_vs, timeout_s)

    def reset_localization(self, timeout_s=10.0):
        return self._call(self.cli_reset, timeout_s)

    def pick(self, timeout_s=60.0):
        return self._call(self.cli_pick, timeout_s)

    def place(self, timeout_s=60.0):
        return self._call(self.cli_place, timeout_s)

    def gripper_open(self, timeout_s=10.0):
        return self._call(self.cli_gripper_open, timeout_s)

    def gripper_close(self, timeout_s=10.0):
        return self._call(self.cli_gripper_close, timeout_s)


# ----------------- action parsing -----------------

VERB_RE = re.compile(r'\[(\w+)\]')
OBJ_RE = re.compile(r'<([^>]+)>\((\d+)\)')


def parse_action(action_str: str):
    """Parse '[verb] <class>(id) [<class>(id) ...]' -> (verb, [(class, id), ...]).

    Returns (None, []) if no verb found.
    """
    m = VERB_RE.search(action_str)
    verb = m.group(1).lower() if m else None
    objs = [(om.group(1).strip(), int(om.group(2))) for om in OBJ_RE.finditer(action_str)]
    return verb, objs


def obj_key(cls: str, oid: int) -> str:
    return f'{cls}({oid})'


# ----------------- action handlers -----------------

class ActionExecutor:
    QUADROTOR_NOOP = ('takeoff_from', 'land_on')

    def __init__(self, bridge: RosBridge, objects_map: dict, rough_fraction: float = 0.7,
                 pre_pick_nudge_m: float = 0.15):
        self.bridge = bridge
        self.objects = objects_map
        self.rough_fraction = rough_fraction
        # Forward (body-frame +x) distance the base nudges right before [grab],
        # so the arm reaches the object cleanly. 15 cm by default; set to 0 to disable.
        self.pre_pick_nudge_m = pre_pick_nudge_m

    def lookup(self, cls: str, oid: int):
        d = self.objects.get(obj_key(cls, oid))
        if d is None:
            return None
        return {
            'x_m': float(d['x_cm']) * 0.01,
            'y_m': float(d['y_cm']) * 0.01,
            'theta_rad': math.radians(float(d.get('theta_deg', 0.0))),
            'visual_servo': bool(d.get('visual_servo', True)),  # default true (red target)
        }

    def movetowards(self, cls: str, oid: int):
        t = self.lookup(cls, oid)
        if t is None:
            return False, f'unknown target: {obj_key(cls, oid)} (not in objects.json)'
        x_m, y_m, theta_rad = t['x_m'], t['y_m'], t['theta_rad']
        log = self.bridge.node.get_logger()

        if t['visual_servo']:
            # 70% rough goto, then visual servo locks onto the red mask.
            rough_x = self.rough_fraction * x_m
            rough_y = self.rough_fraction * y_m
            facing = math.atan2(y_m - rough_y, x_m - rough_x)
            log.info(
                f'movetowards {obj_key(cls, oid)}: visual_servo=ON; rough goto -> '
                f'({rough_x:.3f}, {rough_y:.3f}, {facing:.3f} rad)'
            )
            ok, info = self.bridge.goto_abs(rough_x, rough_y, facing)
            if not ok:
                return False, f'rough goto_abs failed: {info}'
            log.info(f'movetowards {obj_key(cls, oid)}: rough goto done ({info}); calling /mission/visual_servo')
            ok, info = self.bridge.visual_servo()
            if not ok:
                return False, f'visual_servo failed: {info}'
            log.info(f'movetowards {obj_key(cls, oid)}: visual_servo done ({info})')
            return True, f'movetowards {obj_key(cls, oid)} (rough+vs) ok'

        # No visual servo — drive all the way to the recorded pose.
        log.info(
            f'movetowards {obj_key(cls, oid)}: visual_servo=OFF; goto -> '
            f'({x_m:.3f}, {y_m:.3f}, {theta_rad:.3f} rad)'
        )
        ok, info = self.bridge.goto_abs(x_m, y_m, theta_rad)
        if not ok:
            return False, f'goto_abs failed: {info}'
        return True, f'movetowards {obj_key(cls, oid)} (goto only) ok'

    def gripper_close(self, cls: str, oid: int):
        # PEFA's [grab] -> nudge forward (so the arm aligns with the object) then
        # run the full pick sequence on mission_controller (which drives the
        # OpenMANIPULATOR-X joints + closes the gripper).
        log = self.bridge.node.get_logger()
        if self.pre_pick_nudge_m > 0:
            log.info(f'gripper_close: nudging {self.pre_pick_nudge_m * 100:.0f} cm '
                     f'forward (body frame) before pick')
            ok, info = self.bridge.goto_rel_body(self.pre_pick_nudge_m, 0.0, 0.0)
            if not ok:
                return False, f'pre-pick nudge failed: {info}'
        ok, info = self.bridge.pick()
        if not ok:
            return False, f'pick sequence failed: {info}'
        return True, f'picked {obj_key(cls, oid)}'

    def gripper_open_at(self, dst_cls: str, dst_oid: int, payload_label: str = ''):
        # PEFA's [puton]/[putinto] -> first move the base to the destination
        # (which honors the visual_servo flag in objects.json), then run the place
        # sequence (rotates, lowers, opens gripper, lifts, returns home).
        ok, info = self.movetowards(dst_cls, dst_oid)
        if not ok:
            return False, f'move-to-destination failed: {info}'
        ok, info = self.bridge.place()
        if not ok:
            return False, f'place sequence failed: {info}'
        return True, (f'placed {payload_label} at {obj_key(dst_cls, dst_oid)}').strip()

    def door(self, verb: str, cls: str, oid: int):
        # TODO: replace with real door interaction service when available.
        return True, f'[door/container] {verb.upper()} (placeholder) on {obj_key(cls, oid)}'

    def execute(self, verb: str, objs):
        # Dispatch
        if verb == 'movetowards':
            if not objs:
                return False, 'movetowards needs a target object'
            return self.movetowards(*objs[0])

        if verb in ('grab', 'grasp', 'pick'):
            if not objs:
                return False, f'{verb} needs an object'
            return self.gripper_close(*objs[0])

        # Note: PEFA emits [puton] and [putinto] (no underscores) — keep both spellings.
        if verb in ('put_on', 'put_inside', 'place', 'puton', 'putinto'):
            if len(objs) < 2:
                return False, f'{verb} needs a payload and a destination'
            payload_label = obj_key(*objs[0])
            return self.gripper_open_at(objs[1][0], objs[1][1], payload_label)

        if verb in ('open', 'close'):
            if not objs:
                return False, f'{verb} needs an object'
            return self.door(verb, *objs[0])

        if verb in self.QUADROTOR_NOOP:
            tgt = obj_key(*objs[0]) if objs else '?'
            return True, f'quadrotor action [{verb}] {tgt} — TB is robot dog, no-op'

        return False, f'unsupported verb: [{verb}]'


# ----------------- Flask app -----------------

app = Flask(__name__)
bridge: RosBridge = None
executor: ActionExecutor = None


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'ok': True,
        'node': bridge.node.get_name() if bridge else None,
        'objects_loaded': len(executor.objects) if executor else 0,
    })


@app.route('/execute', methods=['POST'])
def execute():
    try:
        data = request.get_json(force=True, silent=True) or {}
        action = data.get('action', '').strip()
        if not action:
            return jsonify({'success': False, 'info': 'no "action" in request'}), 400

        verb, objs = parse_action(action)
        if not verb:
            return jsonify({'success': False, 'info': f'could not parse action: {action!r}'}), 400

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


# Direct primitives — useful for testing without going through the action parser.

@app.route('/goto_abs', methods=['POST'])
def http_goto_abs():
    d = request.get_json(force=True, silent=True) or {}
    ok, info = bridge.goto_abs(
        float(d.get('x_cm', 0)) * 0.01,
        float(d.get('y_cm', 0)) * 0.01,
        math.radians(float(d.get('theta_deg', 0))),
    )
    return jsonify({'success': ok, 'info': info})


@app.route('/goto_rel', methods=['POST'])
def http_goto_rel():
    d = request.get_json(force=True, silent=True) or {}
    ok, info = bridge.goto_rel_body(
        float(d.get('x_cm', 0)) * 0.01,
        float(d.get('y_cm', 0)) * 0.01,
        math.radians(float(d.get('theta_deg', 0))),
    )
    return jsonify({'success': ok, 'info': info})


@app.route('/visual_servo', methods=['POST'])
def http_visual_servo():
    ok, info = bridge.visual_servo()
    return jsonify({'success': ok, 'info': info})


@app.route('/reset_localization', methods=['POST'])
def http_reset():
    ok, info = bridge.reset_localization()
    return jsonify({'success': ok, 'info': info})


# ----------------- entry point -----------------

def load_objects(path: str) -> dict:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        print(f'[warn] objects file {path} not found — actions referencing objects will fail')
        return {}
    with open(p) as f:
        return json.load(f)


def main():
    global bridge, executor

    ap = argparse.ArgumentParser()
    ap.add_argument('--host', default='0.0.0.0', help='HTTP bind address')
    ap.add_argument('--port', type=int, default=8080)
    ap.add_argument('--objects-file', default='objects.json',
                    help='JSON map of "class(id)" -> {x_cm, y_cm, theta_deg}')
    ap.add_argument('--rough-fraction', type=float, default=0.7,
                    help='fraction of distance covered by goto_abs before visual_servo takes over')
    ap.add_argument('--pre-pick-nudge-cm', type=float, default=15.0,
                    help='cm to drive forward (body frame) right before [grab] runs the '
                         'pick sequence; aligns the arm with the object. 0 disables.')
    args = ap.parse_args()

    rclpy.init()
    bridge = RosBridge()
    bridge.start()

    objects_map = load_objects(args.objects_file)
    executor = ActionExecutor(
        bridge, objects_map,
        rough_fraction=args.rough_fraction,
        pre_pick_nudge_m=args.pre_pick_nudge_cm * 0.01,
    )
    print(f'web_client: loaded {len(objects_map)} objects from {args.objects_file}')
    print(f'web_client: listening on http://{args.host}:{args.port}')
    print(f'web_client: rough_fraction={args.rough_fraction}')

    try:
        # threaded=True so multiple clients can connect concurrently — e.g. PEFA running
        # an /execute (long-blocking) while a monitoring tool polls /health or /pose.
        # Concurrent /execute calls still serialize at mission_controller's action_lock
        # (the second one will get a fast-fail "another action running" response), which
        # is the correct behaviour for one physical robot.
        app.run(host=args.host, port=args.port, threaded=True, use_reloader=False)
    finally:
        bridge.node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
