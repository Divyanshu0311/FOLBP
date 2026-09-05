#!/usr/bin/env python3
"""Go-to-goal controller.

Subscribes:
    /robot_pose  (geometry_msgs/Pose2D)        — current pose from localization.py
    /goal_pose   (geometry_msgs/PoseStamped)   — target from RViz "2D Goal Pose" or any publisher

Publishes:
    /cmd_vel     (geometry_msgs/Twist)

Services:
    /controller/pause  (std_srvs/Empty)  — stop publishing /cmd_vel until resumed
    /controller/resume (std_srvs/Empty)  — resume control

Strategy: proportional control on heading error + forward speed scaled by distance.
Stops once within position_tol; optionally aligns to goal yaw afterwards.
"""

import math

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose2D, PoseStamped, Twist
from std_srvs.srv import Empty


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def quat_to_yaw(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class GoToGoalController(Node):
    def __init__(self):
        super().__init__('go_to_goal_controller')

        self.declare_parameter('pose_topic', '/robot_pose')
        self.declare_parameter('goal_topic', '/goal_pose')
        self.declare_parameter('cmd_topic', '/cmd_vel')
        self.declare_parameter('control_rate_hz', 20.0)
        self.declare_parameter('expected_frame', 'odom')

        self.declare_parameter('k_lin', 0.5)
        self.declare_parameter('k_ang', 1.5)
        self.declare_parameter('max_lin', 0.18)   # TB3 Burger ~0.22 m/s, keep margin
        self.declare_parameter('max_ang', 1.2)
        self.declare_parameter('min_lin', 0.03)   # below this the wheels stall (stiction)
        self.declare_parameter('min_ang', 0.15)   # below this the bot just sits and hums
        self.declare_parameter('position_tol', 0.02)   # 2 cm
        self.declare_parameter('heading_gate', 0.3)    # ~17°: above this, only rotate
        self.declare_parameter('align_final', True)
        self.declare_parameter('yaw_tol', 0.02)        # ~1.1°

        pose_topic = self.get_parameter('pose_topic').value
        goal_topic = self.get_parameter('goal_topic').value
        cmd_topic = self.get_parameter('cmd_topic').value
        rate = float(self.get_parameter('control_rate_hz').value)
        self.expected_frame = str(self.get_parameter('expected_frame').value)

        self.k_lin = float(self.get_parameter('k_lin').value)
        self.k_ang = float(self.get_parameter('k_ang').value)
        self.max_lin = float(self.get_parameter('max_lin').value)
        self.max_ang = float(self.get_parameter('max_ang').value)
        self.min_lin = float(self.get_parameter('min_lin').value)
        self.min_ang = float(self.get_parameter('min_ang').value)
        self.pos_tol = float(self.get_parameter('position_tol').value)
        self.heading_gate = float(self.get_parameter('heading_gate').value)
        self.align_final = bool(self.get_parameter('align_final').value)
        self.yaw_tol = float(self.get_parameter('yaw_tol').value)

        self.pose = None
        self.goal = None
        self.reached = False
        self.paused = False

        self.create_subscription(Pose2D, pose_topic, self.on_pose, 10)
        self.create_subscription(PoseStamped, goal_topic, self.on_goal, 10)
        self.cmd_pub = self.create_publisher(Twist, cmd_topic, 10)
        self.create_service(Empty, '/controller/pause', self.on_pause)
        self.create_service(Empty, '/controller/resume', self.on_resume)
        self.timer = self.create_timer(1.0 / rate, self.step)

        self.get_logger().info(
            f'controller: pose<{pose_topic}> goal<{goal_topic}> -> {cmd_topic} @ {rate} Hz '
            f'(expected goal frame: {self.expected_frame}); '
            f'pause/resume on /controller/pause, /controller/resume'
        )

    def on_pose(self, msg: Pose2D):
        self.pose = (msg.x, msg.y, msg.theta)

    def on_goal(self, msg: PoseStamped):
        frame = msg.header.frame_id or '(unset)'
        if self.expected_frame and frame != self.expected_frame:
            # Frame mismatch is the classic foot-gun with /goal_pose from RViz (often "map"
            # while our pose is in "odom"). Warn loudly but still try — useful for testing.
            self.get_logger().warn(
                f'goal frame "{frame}" != expected "{self.expected_frame}"; '
                f'treating coordinates as-is (no TF transform applied)'
            )
        gx = msg.pose.position.x
        gy = msg.pose.position.y
        gth = quat_to_yaw(msg.pose.orientation)
        self.goal = (gx, gy, gth)
        self.reached = False
        self.get_logger().info(f'new goal: x={gx:.3f} y={gy:.3f} theta={gth:.3f} frame={frame}')

    def stop(self):
        self.cmd_pub.publish(Twist())

    def on_pause(self, _req, response):
        if not self.paused:
            self.paused = True
            # Clear any stored goal: whoever paused us (e.g. visual_servo) is about to move
            # the bot, so the old setpoint is stale. On resume the controller sits idle
            # until a fresh /goal_pose arrives.
            self.goal = None
            self.reached = False
            self.stop()  # one zero command so the bot doesn't coast
            self.get_logger().info('paused — yielding /cmd_vel; cleared stale goal')
        return response

    def on_resume(self, _req, response):
        if self.paused:
            self.paused = False
            self.get_logger().info('resumed')
        return response

    def step(self):
        if self.paused:
            return
        if self.pose is None or self.goal is None:
            return

        x, y, th = self.pose
        gx, gy, gth = self.goal
        dx, dy = gx - x, gy - y
        dist = math.hypot(dx, dy)

        cmd = Twist()

        if not self.reached and dist > self.pos_tol:
            heading_err = wrap(math.atan2(dy, dx) - th)
            ang = self._clamp_with_floor(self.k_ang * heading_err, self.min_ang, self.max_ang)
            if abs(heading_err) > self.heading_gate:
                lin = 0.0  # rotate in place when far off heading
            else:
                lin_cmd = self.k_lin * dist
                lin = self._clamp_with_floor(lin_cmd, self.min_lin, self.max_lin) if lin_cmd > 0 else 0.0
            cmd.linear.x = lin
            cmd.angular.z = ang
        elif self.align_final and abs(wrap(gth - th)) > self.yaw_tol:
            yaw_err = wrap(gth - th)
            cmd.angular.z = self._clamp_with_floor(self.k_ang * yaw_err, self.min_ang, self.max_ang)
        else:
            if not self.reached:
                self.get_logger().info(f'goal reached (dist={dist:.3f} m)')
                self.reached = True

        self.cmd_pub.publish(cmd)

    @staticmethod
    def _clamp_with_floor(v, vmin, vmax):
        # Below |vmin| the motors stall; above |vmax| we'd exceed safe speed.
        # Preserve sign, floor magnitude to vmin, ceiling to vmax.
        if v == 0.0:
            return 0.0
        sign = 1.0 if v > 0 else -1.0
        mag = max(vmin, min(vmax, abs(v)))
        return sign * mag


def main():
    rclpy.init()
    node = GoToGoalController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
