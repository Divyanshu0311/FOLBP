#!/usr/bin/env python3
"""Virtual /execute endpoint — same HTTP API as web_client.py, but no ROS.

Use this when an agent (e.g. the drone) doesn't have a physical device behind it yet.
Every action is logged + returned as success: true so PEFA's symbolic graph advances
the same way it would with a real device.

Run:
    python3 virtual_client.py --port 8081 --agent drone
"""

import argparse
import re
import time

from flask import Flask, request, jsonify


VERB_RE = re.compile(r'\[(\w+)\]')
OBJ_RE = re.compile(r'<([^>]+)>\((\d+)\)')


def parse_action(action_str):
    m = VERB_RE.search(action_str)
    verb = m.group(1).lower() if m else None
    objs = [(om.group(1).strip(), int(om.group(2))) for om in OBJ_RE.finditer(action_str)]
    return verb, objs


app = Flask(__name__)
agent_label = 'virtual'
sim_delay_s = 0.5


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'ok': True, 'agent': agent_label, 'mode': 'virtual'})


@app.route('/execute', methods=['POST'])
def execute():
    data = request.get_json(force=True, silent=True) or {}
    action = data.get('action', '').strip()
    agent = data.get('agent', '?')
    if not action:
        return jsonify({'success': False, 'info': 'no "action" in request'}), 400

    verb, objs = parse_action(action)
    print(f'[virtual:{agent_label}] agent={agent} action={action}  -> verb={verb} objs={objs}')

    # Pretend the action took some time (so PEFA's pacing roughly matches reality).
    time.sleep(sim_delay_s)

    info = f'virtual {agent_label}: simulated [{verb}]' if verb else 'virtual: no-op'
    if objs:
        info += ' on ' + ', '.join(f'{c}({i})' for c, i in objs)
    return jsonify({'success': True, 'info': info, 'verb': verb})


# Stubs so the same endpoint surface as web_client.py is available — useful for tests.
@app.route('/goto_abs', methods=['POST'])
@app.route('/goto_rel', methods=['POST'])
@app.route('/visual_servo', methods=['POST'])
@app.route('/reset_localization', methods=['POST'])
def passthrough():
    print(f'[virtual:{agent_label}] {request.path} {request.get_json(silent=True)}')
    time.sleep(sim_delay_s)
    return jsonify({'success': True, 'info': f'virtual: {request.path}'})


def main():
    global agent_label, sim_delay_s
    ap = argparse.ArgumentParser()
    ap.add_argument('--host', default='0.0.0.0')
    ap.add_argument('--port', type=int, default=8081)
    ap.add_argument('--agent', default='virtual',
                    help='label that shows up in /health (purely cosmetic)')
    ap.add_argument('--sim-delay-s', type=float, default=0.5,
                    help='fake action duration so PEFA does not spin too fast')
    args = ap.parse_args()
    agent_label = args.agent
    sim_delay_s = args.sim_delay_s
    print(f'virtual_client[{agent_label}] listening on http://{args.host}:{args.port} '
          f'(sim_delay={sim_delay_s}s)')
    app.run(host=args.host, port=args.port, threaded=True, use_reloader=False)


if __name__ == '__main__':
    main()
