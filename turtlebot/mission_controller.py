#!/usr/bin/env python3
"""Mission controller — single node combining localization, go-to-goal, visual
servo, and OpenMANIPULATOR-X arm sequences.

Runs continuously. Each capability is exposed as a std_srvs/Empty service that blocks
until the action completes, then returns. Args (where needed) are read from the most
recent /mission/target message (geometry_msgs/Pose2D, in m + rad).

Always running:
    /odom                    -> internal localization
    /robot_pose (Pose2D)     <- published, same as localization.py would publish

Services (all std_srvs/Empty; block until done):
    /mission/reset_localization     re-zero the local frame to current /odom
    /mission/goto_abs               drive to /mission/target as absolute world goal
    /mission/goto_rel_body          treat /mission/target as deltas in robot body frame
    /mission/goto_rel_world         treat /mission/target as deltas in world frame
    /mission/visual_servo           drive toward red target until centered + close
    /mission/pick                   run the arm pick sequence + close gripper
    /mission/place                  run the arm place sequence + open gripper
    /mission/gripper_open           low-level: just open gripper
    /mission/gripper_close          low-level: just close gripper

Topic for arguments:
    /mission/target (Pose2D, m + rad)

Quick test from CLI:
    ros2 topic pub --once /mission/target geometry_msgs/Pose2D '{x: 0.5, y: 0.0, theta: 0.0}'
    ros2 service call /mission/goto_rel_body std_srvs/srv/Empty
    ros2 service call /mission/visual_servo std_srvs/srv/Empty
    ros2 service call /mission/pick std_srvs/srv/Empty
    ros2 service call /mission/place std_srvs/srv/Empty
"""

import math
import threading
import time

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose2D, Twist
from sensor_msgs.msg import Image
from std_srvs.srv import Empty
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


# ----------------- helpers -----------------

def quat_to_yaw(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def clamp_with_floor(v, vmin, vmax):
    if v == 0.0:
        return 0.0
    sign = 1.0 if v > 0 else -1.0
    return sign * max(vmin, min(vmax, abs(v)))


def msg_to_bgr(msg: Image):
    enc = msg.encoding.lower()
    buf = np.frombuffer(msg.data, dtype=np.uint8)
    if enc == 'bgr8':
        return buf.reshape(msg.height, msg.width, 3)
    if enc == 'rgb8':
        return cv2.cvtColor(buf.reshape(msg.height, msg.width, 3), cv2.COLOR_RGB2BGR)
    if enc == 'nv21':
        return cv2.cvtColor(buf.reshape(msg.height * 3 // 2, msg.width), cv2.COLOR_YUV2BGR_NV21)
    if enc == 'nv12':
        return cv2.cvtColor(buf.reshape(msg.height * 3 // 2, msg.width), cv2.COLOR_YUV2BGR_NV12)
    if enc in ('yuv420', 'i420'):
        return cv2.cvtColor(buf.reshape(msg.height * 3 // 2, msg.width), cv2.COLOR_YUV2BGR_I420)
    if enc in ('yuyv', 'yuv422'):
        return cv2.cvtColor(buf.reshape(msg.height, msg.width, 2), cv2.COLOR_YUV2BGR_YUYV)
    raise ValueError(f'unsupported image encoding: {msg.encoding}')


def rotate_frame(img, deg):
    if deg == 0:
        return img
    if deg == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if deg == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    if deg == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img


# ----------------- node -----------------

class MissionController(Node):
    MODE_IDLE = 'idle'
    MODE_GOTO = 'goto'
    MODE_VISUAL_SERVO = 'visual_servo'

    def __init__(self):
        super().__init__('mission_controller')
        self.declare_all_params()

        # ----- state -----
        self.latest_odom = None       # (x, y, th) raw odom
        self.origin = None            # (X0, Y0, TH0) zero point
        self.zero_pending = True      # latch first odom as origin
        self.robot_pose = None        # (x, y, th) in local frame

        self.target = None            # (x, y, th) latest /mission/target
        self.current_goal = None      # (x, y, th) being actively pursued
        self.mode = self.MODE_IDLE

        self.last_image = None        # latest BGR frame for visual_servo
        self.vs_arrived_since = None
        self.action_done = threading.Event()
        self.action_lock = threading.Lock()  # one action at a time

        # ----- callback groups -----
        # Reentrant group for services so they can block (using Event.wait) while the
        # tick + subscriptions still fire on the executor's other threads.
        svc_group = ReentrantCallbackGroup()

        # ----- topics -----
        self.create_subscription(Odometry, self.p('odom_topic'), self.on_odom, 10)
        self.create_subscription(Pose2D, '/mission/target', self.on_target, 10)
        self.create_subscription(Image, self.p('image_topic'), self.on_image, 10)
        self.pose_pub = self.create_publisher(Pose2D, self.p('pose_topic'), 10)
        self.cmd_pub = self.create_publisher(Twist, self.p('cmd_topic'), 10)
        self.arm_pub = self.create_publisher(
            JointTrajectory, '/arm_controller/joint_trajectory', 10)
        self.gripper_pub = self.create_publisher(
            JointTrajectory, '/gripper_controller/joint_trajectory', 10)

        # ----- services -----
        self.create_service(Empty, '/mission/reset_localization',
                            self.svc_reset_loc, callback_group=svc_group)
        self.create_service(Empty, '/mission/goto_abs',
                            self.svc_goto_abs, callback_group=svc_group)
        self.create_service(Empty, '/mission/goto_rel_body',
                            self.svc_goto_rel_body, callback_group=svc_group)
        self.create_service(Empty, '/mission/goto_rel_world',
                            self.svc_goto_rel_world, callback_group=svc_group)
        self.create_service(Empty, '/mission/visual_servo',
                            self.svc_visual_servo, callback_group=svc_group)
        self.create_service(Empty, '/mission/pick',
                            self.svc_pick, callback_group=svc_group)
        self.create_service(Empty, '/mission/place',
                            self.svc_place, callback_group=svc_group)
        self.create_service(Empty, '/mission/gripper_open',
                            self.svc_gripper_open, callback_group=svc_group)
        self.create_service(Empty, '/mission/gripper_close',
                            self.svc_gripper_close, callback_group=svc_group)

        # ----- tick -----
        rate = float(self.p('control_rate_hz'))
        self.create_timer(1.0 / rate, self.on_tick)

        self.get_logger().info(
            f'mission_controller ready — services: goto_abs, goto_rel_body, '
            f'goto_rel_world, visual_servo, reset_localization, pick, place, '
            f'gripper_open, gripper_close (all std_srvs/Empty); '
            f'args via /mission/target (Pose2D, m+rad)'
        )

    # ----- parameters -----

    def declare_all_params(self):
        # I/O
        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('pose_topic', '/robot_pose')
        self.declare_parameter('cmd_topic', '/cmd_vel')
        self.declare_parameter('image_topic', '/camera/image_raw')
        self.declare_parameter('control_rate_hz', 20.0)
        self.declare_parameter('auto_zero_on_start', True)

        # Go-to-goal gains/limits (mirror controller.py)
        self.declare_parameter('k_lin', 0.5)
        self.declare_parameter('k_ang', 1.5)
        self.declare_parameter('max_lin', 0.18)
        self.declare_parameter('max_ang', 1.2)
        self.declare_parameter('min_lin', 0.03)
        self.declare_parameter('min_ang', 0.15)
        self.declare_parameter('position_tol', 0.02)
        self.declare_parameter('heading_gate', 0.3)
        self.declare_parameter('align_final', True)
        self.declare_parameter('yaw_tol', 0.02)
        self.declare_parameter('goto_timeout_s', 60.0)

        # Visual servo gains/limits (mirror visual_servo.py)
        self.declare_parameter('vs_rotate', 0)             # 0 / 90 / 180 / 270
        self.declare_parameter('vs_invert_yaw', False)
        self.declare_parameter('vs_h_lo', 0)
        self.declare_parameter('vs_h_hi', 10)
        self.declare_parameter('vs_h_lo2', 170)            # set <0 to disable secondary range
        self.declare_parameter('vs_h_hi2', 179)
        self.declare_parameter('vs_s_lo', 120)
        self.declare_parameter('vs_s_hi', 255)
        self.declare_parameter('vs_v_lo', 70)
        self.declare_parameter('vs_v_hi', 255)
        self.declare_parameter('vs_min_area_px', 400)
        self.declare_parameter('vs_target_area_frac', 0.05)
        self.declare_parameter('vs_area_tol', 0.01)
        self.declare_parameter('vs_x_tol', 0.04)
        self.declare_parameter('vs_heading_gate', 0.25)
        self.declare_parameter('vs_kp_ang', 1.2)
        self.declare_parameter('vs_kp_lin', 0.6)
        self.declare_parameter('vs_max_ang', 0.8)
        self.declare_parameter('vs_max_lin', 0.12)
        self.declare_parameter('vs_min_ang', 0.15)
        self.declare_parameter('vs_min_lin', 0.03)
        self.declare_parameter('vs_finish_hold_s', 1.0)
        self.declare_parameter('vs_timeout_s', 60.0)

    def p(self, name):
        return self.get_parameter(name).value

    # ----- localization (always running) -----

    def on_odom(self, msg: Odometry):
        p = msg.pose.pose.position
        yaw = quat_to_yaw(msg.pose.pose.orientation)
        self.latest_odom = (p.x, p.y, yaw)
        if self.zero_pending and self.p('auto_zero_on_start'):
            self.origin = self.latest_odom
            self.zero_pending = False
            self.get_logger().info(
                f'origin set: X0={p.x:.3f} Y0={p.y:.3f} TH0={yaw:.3f}'
            )

    def update_robot_pose(self):
        if self.latest_odom is None or self.origin is None:
            return None
        x, y, th = self.latest_odom
        X0, Y0, TH0 = self.origin
        dx, dy = x - X0, y - Y0
        c, s = math.cos(TH0), math.sin(TH0)
        x_loc = c * dx + s * dy
        y_loc = -s * dx + c * dy
        th_loc = wrap(th - TH0)
        self.robot_pose = (x_loc, y_loc, th_loc)
        return self.robot_pose

    def on_target(self, msg: Pose2D):
        self.target = (msg.x, msg.y, msg.theta)
        self.get_logger().info(
            f'/mission/target set: x={msg.x:.3f} m, y={msg.y:.3f} m, theta={msg.theta:.3f} rad'
        )

    def on_image(self, msg: Image):
        try:
            bgr = msg_to_bgr(msg)
        except ValueError as e:
            self.get_logger().warn(str(e), throttle_duration_sec=2.0)
            return
        rot = int(self.p('vs_rotate'))
        if rot:
            bgr = rotate_frame(bgr, rot)
        self.last_image = bgr

    # ----- main tick -----

    def on_tick(self):
        pose = self.update_robot_pose()
        if pose is not None:
            out = Pose2D()
            out.x, out.y, out.theta = pose
            self.pose_pub.publish(out)

        if self.mode == self.MODE_GOTO:
            cmd = self.compute_goto_cmd()
            self.cmd_pub.publish(cmd)
        elif self.mode == self.MODE_VISUAL_SERVO:
            cmd = self.compute_vs_cmd()
            self.cmd_pub.publish(cmd)
        # MODE_IDLE: do not publish — TB3 stops on watchdog timeout

    # ----- go-to-goal -----

    def compute_goto_cmd(self):
        cmd = Twist()
        if self.robot_pose is None or self.current_goal is None:
            return cmd

        x, y, th = self.robot_pose
        gx, gy, gth = self.current_goal
        dx, dy = gx - x, gy - y
        dist = math.hypot(dx, dy)

        pos_tol = float(self.p('position_tol'))
        yaw_tol = float(self.p('yaw_tol'))
        heading_gate = float(self.p('heading_gate'))
        k_lin = float(self.p('k_lin'))
        k_ang = float(self.p('k_ang'))
        min_lin = float(self.p('min_lin'))
        max_lin = float(self.p('max_lin'))
        min_ang = float(self.p('min_ang'))
        max_ang = float(self.p('max_ang'))
        align_final = bool(self.p('align_final'))

        if dist > pos_tol:
            heading_err = wrap(math.atan2(dy, dx) - th)
            ang = clamp_with_floor(k_ang * heading_err, min_ang, max_ang)
            if abs(heading_err) > heading_gate:
                lin = 0.0
            else:
                lin_cmd = k_lin * dist
                lin = clamp_with_floor(lin_cmd, min_lin, max_lin) if lin_cmd > 0 else 0.0
            cmd.linear.x = lin
            cmd.angular.z = ang
        elif align_final and abs(wrap(gth - th)) > yaw_tol:
            yaw_err = wrap(gth - th)
            cmd.angular.z = clamp_with_floor(k_ang * yaw_err, min_ang, max_ang)
        else:
            self.action_done.set()
        return cmd

    # ----- visual servo -----

    def detect_red(self, bgr):
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        lo1 = np.array([self.p('vs_h_lo'), self.p('vs_s_lo'), self.p('vs_v_lo')], dtype=np.uint8)
        hi1 = np.array([self.p('vs_h_hi'), self.p('vs_s_hi'), self.p('vs_v_hi')], dtype=np.uint8)
        mask = cv2.inRange(hsv, lo1, hi1)
        h_lo2 = int(self.p('vs_h_lo2'))
        if h_lo2 >= 0:
            lo2 = np.array([h_lo2, self.p('vs_s_lo'), self.p('vs_v_lo')], dtype=np.uint8)
            hi2 = np.array([self.p('vs_h_hi2'), self.p('vs_s_hi'), self.p('vs_v_hi')], dtype=np.uint8)
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lo2, hi2))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask

    def compute_vs_cmd(self):
        cmd = Twist()
        if self.last_image is None:
            return cmd

        bgr = self.last_image
        h, w = bgr.shape[:2]

        mask = self.detect_red(bgr)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            self.vs_arrived_since = None
            return cmd

        best = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(best)
        if area < int(self.p('vs_min_area_px')):
            self.vs_arrived_since = None
            return cmd

        M = cv2.moments(best)
        if M['m00'] == 0:
            self.vs_arrived_since = None
            return cmd
        cx = M['m10'] / M['m00']

        x_err = (cx - w / 2.0) / (w / 2.0)
        area_frac = area / float(w * h)

        target_area_frac = float(self.p('vs_target_area_frac'))
        area_tol = float(self.p('vs_area_tol'))
        x_tol = float(self.p('vs_x_tol'))
        heading_gate = float(self.p('vs_heading_gate'))
        kp_ang = float(self.p('vs_kp_ang'))
        kp_lin = float(self.p('vs_kp_lin'))
        max_ang = float(self.p('vs_max_ang'))
        max_lin = float(self.p('vs_max_lin'))
        min_ang = float(self.p('vs_min_ang'))
        min_lin = float(self.p('vs_min_lin'))
        finish_hold_s = float(self.p('vs_finish_hold_s'))
        invert_yaw = bool(self.p('vs_invert_yaw'))

        size_err = target_area_frac - area_frac
        close_enough = size_err <= area_tol
        centered = abs(x_err) <= x_tol

        if close_enough and centered:
            now = time.time()
            if self.vs_arrived_since is None:
                self.vs_arrived_since = now
            elif (now - self.vs_arrived_since) >= finish_hold_s:
                self.action_done.set()
            return cmd  # zero Twist while holding
        self.vs_arrived_since = None

        if centered:
            ang = 0.0
        else:
            sign = -1.0 if invert_yaw else 1.0
            ang = sign * kp_ang * x_err
            ang = max(-max_ang, min(max_ang, ang))
            if abs(ang) < min_ang:
                ang = min_ang * (1.0 if ang > 0 else -1.0)

        if close_enough or abs(x_err) > heading_gate:
            lin = 0.0
        else:
            lin = max(0.0, min(max_lin, kp_lin * size_err))
            if 0 < lin < min_lin:
                lin = min_lin

        cmd.linear.x = lin
        cmd.angular.z = ang
        return cmd

    # ----- service handlers -----

    def svc_reset_loc(self, _req, response):
        if self.latest_odom is None:
            self.zero_pending = True
            self.get_logger().info('reset requested — waiting for next /odom')
        else:
            self.origin = self.latest_odom
            self.zero_pending = False
            self.get_logger().info(
                f'reset: origin = {self.origin}; pose now reads (0, 0, 0)'
            )
        return response

    def _wait_for_pose(self, timeout=5.0):
        deadline = time.time() + timeout
        while self.robot_pose is None and time.time() < deadline:
            time.sleep(0.05)
        return self.robot_pose is not None

    def _wait_for_target(self, timeout=2.0):
        deadline = time.time() + timeout
        while self.target is None and time.time() < deadline:
            time.sleep(0.05)
        return self.target is not None

    def _do_goto(self, mode_label, kind):
        # kind: 'abs' | 'rel_body' | 'rel_world'
        if not self.action_lock.acquire(blocking=False):
            self.get_logger().warn(f'{mode_label}: another action is already running')
            return
        try:
            if not self._wait_for_target():
                self.get_logger().error(f'{mode_label}: no /mission/target seen — aborting')
                return
            if not self._wait_for_pose():
                self.get_logger().error(f'{mode_label}: no /robot_pose available — aborting')
                return

            tx, ty, tth = self.target
            if kind == 'abs':
                gx, gy, gth = tx, ty, tth
            else:
                X, Y, Th = self.robot_pose
                if kind == 'rel_world':
                    gx, gy = X + tx, Y + ty
                else:  # rel_body
                    c, s = math.cos(Th), math.sin(Th)
                    gx = X + c * tx - s * ty
                    gy = Y + s * tx + c * ty
                gth = wrap(Th + tth)

            self.current_goal = (gx, gy, gth)
            self.action_done.clear()
            self.mode = self.MODE_GOTO
            self.get_logger().info(
                f'{mode_label}: -> ({gx:.3f}, {gy:.3f}, {gth:.3f})'
            )

            ok = self.action_done.wait(timeout=float(self.p('goto_timeout_s')))
            self.mode = self.MODE_IDLE
            self.cmd_pub.publish(Twist())
            self.current_goal = None
            self.get_logger().info(f'{mode_label}: {"reached" if ok else "TIMED OUT"}')
        finally:
            self.action_lock.release()

    def svc_goto_abs(self, _req, response):
        self._do_goto('goto_abs', 'abs')
        return response

    def svc_goto_rel_body(self, _req, response):
        self._do_goto('goto_rel_body', 'rel_body')
        return response

    def svc_goto_rel_world(self, _req, response):
        self._do_goto('goto_rel_world', 'rel_world')
        return response

    def svc_visual_servo(self, _req, response):
        if not self.action_lock.acquire(blocking=False):
            self.get_logger().warn('visual_servo: another action is already running')
            return response
        try:
            self.vs_arrived_since = None
            self.action_done.clear()
            self.mode = self.MODE_VISUAL_SERVO
            self.get_logger().info('visual_servo: started')
            ok = self.action_done.wait(timeout=float(self.p('vs_timeout_s')))
            self.mode = self.MODE_IDLE
            self.cmd_pub.publish(Twist())
            self.get_logger().info(f'visual_servo: {"arrived" if ok else "TIMED OUT"}')
        finally:
            self.action_lock.release()
        return response

    # ----- arm + gripper -----
    #
    # Direct JointTrajectory control on /arm_controller/joint_trajectory and
    # /gripper_controller/joint_trajectory — same scheme as the standalone
    # SimplePickPlace script. Each move blocks for `wait_s + 1.0` so the bot has
    # time to actually reach the commanded joint pose before we publish the next.

    ARM_JOINT_NAMES = ['joint1', 'joint2', 'joint3', 'joint4']
    GRIPPER_JOINT_NAME = 'gripper'
    GRIPPER_OPEN_VAL = 0.01
    GRIPPER_CLOSE_VAL = -0.01

    def _move_arm(self, j1, j2, j3, j4, wait_s=3.0):
        msg = JointTrajectory()
        msg.joint_names = list(self.ARM_JOINT_NAMES)
        pt = JointTrajectoryPoint()
        pt.positions = [float(j1), float(j2), float(j3), float(j4)]
        pt.time_from_start.sec = int(wait_s)
        pt.time_from_start.nanosec = int((wait_s - int(wait_s)) * 1e9)
        msg.points.append(pt)
        self.arm_pub.publish(msg)
        self.get_logger().info(f'arm -> [{j1:.3f}, {j2:.3f}, {j3:.3f}, {j4:.3f}]')
        time.sleep(wait_s + 1.0)

    def _gripper(self, value, wait_s=2.0):
        msg = JointTrajectory()
        msg.joint_names = [self.GRIPPER_JOINT_NAME]
        pt = JointTrajectoryPoint()
        pt.positions = [float(value)]
        pt.time_from_start.sec = int(wait_s)
        pt.time_from_start.nanosec = int((wait_s - int(wait_s)) * 1e9)
        msg.points.append(pt)
        self.gripper_pub.publish(msg)
        self.get_logger().info(f'gripper -> {value:+.3f}')
        time.sleep(wait_s + 1.0)

    def _open_gripper(self):
        self._gripper(self.GRIPPER_OPEN_VAL)

    def _close_gripper(self):
        self._gripper(self.GRIPPER_CLOSE_VAL)

    def _pick_sequence(self):
        # Above-object pose (two-step approach), open → close on object, lift to
        # carrying pose. Joint values courtesy of the SimplePickPlace script.
        self._move_arm(0.0, 1.0, 0.0, -0.2)
        self._move_arm(0.0, 1.0, 2.0, 0.2)
        self._open_gripper()
        self._close_gripper()
        self._move_arm(0.0, -1.054, 0.738, 0.009)

    def _place_sequence(self):
        # Carry → rotate to placement direction → lower → open gripper → lift → home.
        self._move_arm(0.0, -1.054, 0.738, 0.009)
        self._move_arm(1.466, -1.062, 0.735, 0.031)
        self._move_arm(1.447, 0.305, 0.733, 0.100)
        self._open_gripper()
        self._move_arm(1.466, -1.062, 0.735, 0.031)
        self._move_arm(0.0, -1.054, 0.738, 0.009)

    def svc_pick(self, _req, response):
        if not self.action_lock.acquire(blocking=False):
            self.get_logger().warn('pick: another action is already running')
            return response
        try:
            self.get_logger().info('pick: started')
            self._pick_sequence()
            self.get_logger().info('pick: done')
        except Exception as e:
            self.get_logger().error(f'pick: failed — {e}')
        finally:
            self.action_lock.release()
        return response

    def svc_place(self, _req, response):
        if not self.action_lock.acquire(blocking=False):
            self.get_logger().warn('place: another action is already running')
            return response
        try:
            self.get_logger().info('place: started')
            self._place_sequence()
            self.get_logger().info('place: done')
        except Exception as e:
            self.get_logger().error(f'place: failed — {e}')
        finally:
            self.action_lock.release()
        return response

    def svc_gripper_open(self, _req, response):
        if not self.action_lock.acquire(blocking=False):
            self.get_logger().warn('gripper_open: another action is already running')
            return response
        try:
            self._open_gripper()
        finally:
            self.action_lock.release()
        return response

    def svc_gripper_close(self, _req, response):
        if not self.action_lock.acquire(blocking=False):
            self.get_logger().warn('gripper_close: another action is already running')
            return response
        try:
            self._close_gripper()
        finally:
            self.action_lock.release()
        return response


def main():
    rclpy.init()
    node = MissionController()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.cmd_pub.publish(Twist())  # belt-and-suspenders stop
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
