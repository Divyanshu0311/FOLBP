#!/usr/bin/env python3
"""CLI wrapper for mission_controller.py.

Takes cm + degrees on the command line, converts to m + rad, publishes /mission/target,
then calls the matching service. Blocks until the service returns.

Usage:
    python3 mission_cli.py goto_abs       <x_cm> <y_cm> <theta_deg>
    python3 mission_cli.py goto_rel       <fwd_cm> <left_cm> <dtheta_deg>   # body-frame
    python3 mission_cli.py goto_rel_world <dx_cm> <dy_cm> <dtheta_deg>      # world-frame
    python3 mission_cli.py visual_servo
    python3 mission_cli.py reset

Examples:
    python3 mission_cli.py goto_abs 50 30 90      # absolute (0.5, 0.3) m, face 90 deg
    python3 mission_cli.py goto_rel 50 0 0        # forward 50 cm
    python3 mission_cli.py goto_rel 0 0 90        # rotate 90 deg in place
    python3 mission_cli.py visual_servo           # CV refine
"""

import math
import sys
import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose2D
from std_srvs.srv import Empty


SERVICE_FOR = {
    'goto_abs':       '/mission/goto_abs',
    'goto_rel':       '/mission/goto_rel_body',
    'goto_rel_world': '/mission/goto_rel_world',
    'visual_servo':   '/mission/visual_servo',
    'reset':          '/mission/reset_localization',
}

NEEDS_TARGET = {'goto_abs', 'goto_rel', 'goto_rel_world'}


def usage_and_exit():
    print(__doc__, file=sys.stderr)
    sys.exit(2)


def main():
    if len(sys.argv) < 2:
        usage_and_exit()

    cmd = sys.argv[1]
    if cmd not in SERVICE_FOR:
        print(f'unknown command: {cmd}', file=sys.stderr)
        usage_and_exit()

    if cmd in NEEDS_TARGET:
        if len(sys.argv) != 5:
            print(f'{cmd} needs 3 args: <x_cm> <y_cm> <theta_deg>', file=sys.stderr)
            usage_and_exit()
        try:
            x_cm, y_cm, th_deg = (float(v) for v in sys.argv[2:5])
        except ValueError:
            print('args must be numbers', file=sys.stderr)
            usage_and_exit()
    else:
        if len(sys.argv) != 2:
            print(f'{cmd} takes no args', file=sys.stderr)
            usage_and_exit()

    rclpy.init()
    node = Node('mission_cli')

    if cmd in NEEDS_TARGET:
        pub = node.create_publisher(Pose2D, '/mission/target', 10)
        # Wait briefly for mission_controller to be subscribed.
        deadline = time.time() + 2.0
        while pub.get_subscription_count() == 0 and time.time() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
        if pub.get_subscription_count() == 0:
            node.get_logger().warn(
                'no subscriber on /mission/target — is mission_controller.py running?'
            )

        msg = Pose2D(x=x_cm * 0.01, y=y_cm * 0.01, theta=math.radians(th_deg))
        pub.publish(msg)
        # Re-publish once for safety, plus give DDS a moment to deliver.
        time.sleep(0.1)
        pub.publish(msg)
        node.get_logger().info(
            f'{cmd}: target set x={x_cm} cm ({msg.x:.3f} m), '
            f'y={y_cm} cm ({msg.y:.3f} m), theta={th_deg} deg ({msg.theta:.3f} rad)'
        )

    svc_name = SERVICE_FOR[cmd]
    cli = node.create_client(Empty, svc_name)
    if not cli.wait_for_service(timeout_sec=5.0):
        node.get_logger().error(f'service {svc_name} unavailable — is mission_controller.py running?')
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(1)

    node.get_logger().info(f'{cmd}: calling {svc_name} (blocking)…')
    fut = cli.call_async(Empty.Request())
    rclpy.spin_until_future_complete(node, fut)
    node.get_logger().info(f'{cmd}: done')

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
