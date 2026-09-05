#!/usr/bin/env python3
"""One-shot goal publisher to /goal_pose (geometry_msgs/PoseStamped).

x, y are taken in CENTIMETRES; theta in DEGREES.
Internally converted to metres + radians before publishing (ROS convention).

Modes:
  absolute (default)        target in world frame, same origin localization.py uses
  --relative / -r           args are deltas in the ROBOT BODY frame (forward, left, yaw)
                            added to the current /robot_pose. Resolved absolute target
                            is what gets published.
  --relative --world        args are deltas in WORLD frame (just added to current pose),
                            useful when you want to step in fixed-axis directions.

Usage:
    python3 send_setpoint.py 100 50 90              # absolute (1.0 m, 0.5 m), face 90 deg
    python3 send_setpoint.py 50 0 0 -r              # forward 50 cm, no rotation
    python3 send_setpoint.py 0 0 90 -r              # rotate 90 deg in place
    python3 send_setpoint.py 50 0 0 -r --world      # +50 cm along world x, regardless of heading
"""

import argparse
import math
import sys
import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose2D, PoseStamped


CM_TO_M = 0.01


def yaw_to_quat(yaw):
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def wait_for_pose(node, topic, timeout_s=5.0):
    """Spin until one Pose2D message arrives on `topic`, or timeout."""
    holder = {'pose': None}

    def on_pose(msg: Pose2D):
        holder['pose'] = (msg.x, msg.y, msg.theta)

    sub = node.create_subscription(Pose2D, topic, on_pose, 10)
    deadline = time.time() + timeout_s
    while holder['pose'] is None and time.time() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
    node.destroy_subscription(sub)
    return holder['pose']


def main():
    ap = argparse.ArgumentParser(description='Publish a goal to /goal_pose. x,y in cm; theta in deg.')
    ap.add_argument('x_cm', type=float, help='x in centimetres (forward in body frame if --relative)')
    ap.add_argument('y_cm', type=float, help='y in centimetres (left in body frame if --relative)')
    ap.add_argument('theta_deg', type=float, nargs='?', default=0.0,
                    help='yaw in degrees (delta yaw if --relative)')
    ap.add_argument('-r', '--relative', action='store_true',
                    help='interpret args as deltas added to current /robot_pose')
    ap.add_argument('--world', action='store_true',
                    help='with --relative, treat dx/dy as world-frame deltas (default: body frame)')
    ap.add_argument('--pose-topic', default='/robot_pose',
                    help='topic to read current pose from when --relative is set')
    ap.add_argument('--frame', default='odom',
                    help='frame_id for the published PoseStamped')
    args = ap.parse_args()

    dx = args.x_cm * CM_TO_M
    dy = args.y_cm * CM_TO_M
    dtheta = math.radians(args.theta_deg)

    rclpy.init()
    node = Node('send_setpoint')

    if args.relative:
        pose = wait_for_pose(node, args.pose_topic)
        if pose is None:
            print(f'ERROR: no message on {args.pose_topic} within 5 s — '
                  f'is localization.py running?', file=sys.stderr)
            node.destroy_node()
            rclpy.shutdown()
            sys.exit(1)
        X, Y, Th = pose
        if args.world:
            x_m = X + dx
            y_m = Y + dy
        else:
            # Body-frame delta: rotate (dx, dy) by current yaw before adding.
            c, s = math.cos(Th), math.sin(Th)
            x_m = X + c * dx - s * dy
            y_m = Y + s * dx + c * dy
        theta_rad = wrap(Th + dtheta)
        node.get_logger().info(
            f'relative ({"world" if args.world else "body"}-frame): '
            f'current=({X:.3f}, {Y:.3f}, {Th:.3f} rad) '
            f'+ delta=({dx:.3f}, {dy:.3f}, {dtheta:.3f}) '
            f'-> target=({x_m:.3f}, {y_m:.3f}, {theta_rad:.3f})'
        )
    else:
        x_m, y_m, theta_rad = dx, dy, dtheta

    pub = node.create_publisher(PoseStamped, '/goal_pose', 10)

    deadline = time.time() + 3.0
    while pub.get_subscription_count() == 0 and time.time() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)

    msg = PoseStamped()
    msg.header.frame_id = args.frame
    msg.header.stamp = node.get_clock().now().to_msg()
    msg.pose.position.x = x_m
    msg.pose.position.y = y_m
    qx, qy, qz, qw = yaw_to_quat(theta_rad)
    msg.pose.orientation.x = qx
    msg.pose.orientation.y = qy
    msg.pose.orientation.z = qz
    msg.pose.orientation.w = qw

    pub.publish(msg)
    if args.relative:
        node.get_logger().info(f'published /goal_pose (absolute): '
                               f'x={x_m:.3f} m, y={y_m:.3f} m, theta={math.degrees(theta_rad):.1f} deg')
    else:
        node.get_logger().info(
            f'published /goal_pose: x={args.x_cm} cm ({x_m:.3f} m), '
            f'y={args.y_cm} cm ({y_m:.3f} m), theta={args.theta_deg} deg ({theta_rad:.3f} rad), '
            f'frame={args.frame}'
        )

    time.sleep(0.2)
    pub.publish(msg)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
