#!/usr/bin/env python3
"""Reads /odom from TurtleBot3 and republishes a flat (x, y, theta) pose on /robot_pose.

The published pose is in a local frame whose origin is wherever the robot was when:
  - the node started (if `auto_zero_on_start` is True, default), or
  - the /reset_localization service (std_srvs/Empty) was last called.

So the reported pose starts at (0, 0, 0) and is reset back to (0, 0, 0) any time you
call the service. This is intentionally NOT real localization (no map, no AMCL) —
it's a "treat current pose as origin" helper for relative motion.

Reset from another terminal:
    ros2 service call /reset_localization std_srvs/srv/Empty
"""

import math

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose2D
from std_srvs.srv import Empty


def quat_to_yaw(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


class LocalizationNode(Node):
    def __init__(self):
        super().__init__('localization')

        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('pose_topic', '/robot_pose')
        self.declare_parameter('reset_service', '/reset_localization')
        self.declare_parameter('publish_rate_hz', 20.0)
        self.declare_parameter('auto_zero_on_start', True)

        odom_topic = self.get_parameter('odom_topic').value
        pose_topic = self.get_parameter('pose_topic').value
        reset_service = self.get_parameter('reset_service').value
        rate = float(self.get_parameter('publish_rate_hz').value)
        self.auto_zero = bool(self.get_parameter('auto_zero_on_start').value)

        self.latest = None      # (x, y, theta) most recent raw odom
        self.origin = None      # (X0, Y0, TH0) raw odom snapshot to subtract
        self.zero_pending = self.auto_zero  # latch first odom as origin

        self.create_subscription(Odometry, odom_topic, self.on_odom, 10)
        self.create_service(Empty, reset_service, self.on_reset)
        self.pub = self.create_publisher(Pose2D, pose_topic, 10)
        self.timer = self.create_timer(1.0 / rate, self.on_tick)

        self.get_logger().info(
            f'localization: {odom_topic} -> {pose_topic} @ {rate} Hz, '
            f'reset service <{reset_service}>, auto_zero_on_start={self.auto_zero}'
        )

    def on_odom(self, msg: Odometry):
        p = msg.pose.pose.position
        yaw = quat_to_yaw(msg.pose.pose.orientation)
        self.latest = (p.x, p.y, yaw)
        if self.zero_pending:
            self.origin = self.latest
            self.zero_pending = False
            self.get_logger().info(
                f'origin set: X0={p.x:.3f} Y0={p.y:.3f} TH0={yaw:.3f} rad'
            )

    def on_reset(self, _request: Empty.Request, response: Empty.Response):
        if self.latest is None:
            # No odom yet — latch the next one as the origin.
            self.zero_pending = True
            self.get_logger().info('reset requested — waiting for next /odom')
            return response
        self.origin = self.latest
        self.get_logger().info(
            f'reset: origin = ({self.origin[0]:.3f}, {self.origin[1]:.3f}, '
            f'{self.origin[2]:.3f} rad); pose now reads (0, 0, 0)'
        )
        return response

    def on_tick(self):
        if self.latest is None or self.origin is None:
            return
        x, y, th = self.latest
        X0, Y0, TH0 = self.origin
        dx, dy = x - X0, y - Y0
        c, s = math.cos(TH0), math.sin(TH0)
        # Rotate the offset back into the local frame defined by origin yaw.
        x_loc = c * dx + s * dy
        y_loc = -s * dx + c * dy
        th_loc = wrap(th - TH0)

        out = Pose2D()
        out.x = x_loc
        out.y = y_loc
        out.theta = th_loc
        self.pub.publish(out)


def main():
    rclpy.init()
    node = LocalizationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
