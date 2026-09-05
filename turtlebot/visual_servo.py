#!/usr/bin/env python3
"""Visual servoing — detect a RED target and drive toward it.

Drive is ON by default — the moment the target is visible, the bot will steer + approach.
Pass --no-drive to start in observe-only mode (toggle with 'd').

Coordination with controller.py:
  On start, this calls /controller/pause so the go-to-goal controller stops publishing
  /cmd_vel (otherwise both nodes fight). When the target is held for `finish_hold_s`
  seconds, this auto-exits and calls /controller/resume so the controller is live again
  for the next /goal_pose. Pass --no-pause-controller / --no-auto-finish to disable.

Workflow:
  1. Send a rough setpoint with send_setpoint.py to get the target into the camera frame.
  2. Run this. It steers to center the target horizontally, then drives forward until the
     target's bounding-box area covers `target_area_frac` of the frame, then stops + exits.
  3. Send the next /goal_pose — controller takes over again.

Keys (in the OpenCV window):
    d   toggle drive on/off (publishing /cmd_vel)
    t   toggle tuning trackbars (HSV thresholds)
    s   save current frame to /tmp/visual_servo_<n>.png
    q / ESC  quit

Notes:
  - Default detection: red, with two hue ranges OR'd (red wraps around the hue circle).
    Tune for your specific tape with `t`. For non-red colors, pass --no-second.
  - Camera mounting orientation can be corrected with --rotate.
  - Distance estimate is "fraction of image filled" (no camera calibration needed).
    To stop closer/farther, change --target-area-frac.
"""

import argparse

import cv2
import numpy as np

import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty


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
    raise ValueError(f'unsupported encoding: {msg.encoding}')


def rotate_frame(img, deg):
    if deg == 0:
        return img
    if deg == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if deg == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    if deg == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    raise ValueError(f'rotation must be 0/90/180/270, got {deg}')


class VisualServo(Node):
    WIN_MAIN = 'visual_servo'
    WIN_MASK = 'mask'
    WIN_TUNE = 'tune'

    def __init__(self, args):
        super().__init__('visual_servo')
        self.args = args
        self.drive_enabled = not args.no_drive
        self.tune_open = False
        self.save_count = 0
        self.last_frame_shape = None
        self.arrived_since = None       # wall time of first sustained "arrived"
        self.shutdown_requested = False
        self.controller_paused = False  # we asked the controller to pause

        # Primary HSV range (defaults: lower red hue band).
        self.hsv_lo = np.array([args.h_lo, args.s_lo, args.v_lo], dtype=np.uint8)
        self.hsv_hi = np.array([args.h_hi, args.s_hi, args.v_hi], dtype=np.uint8)
        # Optional secondary range to handle red's hue wrap-around (179 -> 0).
        # Set to None for single-range colors (e.g. green, blue, white).
        if args.h_lo2 is None or args.h_hi2 is None:
            self.hsv_lo2 = None
            self.hsv_hi2 = None
        else:
            self.hsv_lo2 = np.array([args.h_lo2, args.s_lo, args.v_lo], dtype=np.uint8)
            self.hsv_hi2 = np.array([args.h_hi2, args.s_hi, args.v_hi], dtype=np.uint8)

        cv2.namedWindow(self.WIN_MAIN, cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow(self.WIN_MASK, cv2.WINDOW_AUTOSIZE)

        self.cmd_pub = self.create_publisher(Twist, args.cmd_topic, 10)
        self.pause_client = self.create_client(Empty, '/controller/pause')
        self.resume_client = self.create_client(Empty, '/controller/resume')

        # Pause the go-to-goal controller before we start publishing /cmd_vel ourselves —
        # otherwise both nodes fight over the topic.
        if args.pause_controller:
            self.pause_controller()

        # Subscribe AFTER pausing the controller so we don't race on /cmd_vel.
        self.create_subscription(Image, args.topic, self.on_image, 10)

        self.get_logger().info(
            f'visual_servo: image<{args.topic}> -> {args.cmd_topic}; '
            f'drive={"ON" if self.drive_enabled else "OFF"}; '
            f'auto_finish={args.auto_finish}; '
            f'press "d" to toggle drive, "t" for tuning trackbars, "q" to quit'
        )

    # ---------- controller coordination ----------

    def pause_controller(self):
        if not self.pause_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn(
                '/controller/pause unavailable — running standalone (controller may fight on /cmd_vel)'
            )
            return
        future = self.pause_client.call_async(Empty.Request())
        rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
        if future.done():
            self.controller_paused = True
            self.get_logger().info('controller paused')
        else:
            self.get_logger().warn('pause call timed out')

    def resume_controller(self):
        if not self.controller_paused:
            return
        if not self.resume_client.service_is_ready():
            self.resume_client.wait_for_service(timeout_sec=1.0)
        if not self.resume_client.service_is_ready():
            self.get_logger().warn('cannot reach /controller/resume — controller stays paused')
            return
        future = self.resume_client.call_async(Empty.Request())
        rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
        self.controller_paused = False
        self.get_logger().info('controller resumed')

    # ---------- detection ----------

    def detect(self, bgr):
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.hsv_lo, self.hsv_hi)
        if self.hsv_lo2 is not None:
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv, self.hsv_lo2, self.hsv_hi2))
        # Clean up speckle.
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return mask, None

        best = max(contours, key=cv2.contourArea)
        if cv2.contourArea(best) < self.args.min_area_px:
            return mask, None

        x, y, w, h = cv2.boundingRect(best)
        M = cv2.moments(best)
        if M['m00'] == 0:
            return mask, None
        cx = M['m10'] / M['m00']
        cy = M['m01'] / M['m00']
        return mask, dict(bbox=(x, y, w, h), centroid=(cx, cy), area=cv2.contourArea(best))

    # ---------- control ----------

    def control(self, det, img_w, img_h):
        cmd = Twist()
        if det is None:
            return cmd, 'no target'

        cx, _ = det['centroid']
        x_err_norm = (cx - img_w / 2.0) / (img_w / 2.0)   # in [-1, 1]
        area_frac = det['area'] / float(img_w * img_h)

        size_err = self.args.target_area_frac - area_frac  # >0 means still too far
        # Arrived if we're within area_tol of the target (or past it). The tolerance
        # band prevents getting stuck in the sub-stiction zone where commanded velocity
        # is below what the wheels can physically execute.
        close_enough = size_err <= self.args.area_tol
        centered = abs(x_err_norm) <= self.args.x_tol

        # Done — neither rotate nor drive (avoids stiction-floor twitch).
        if close_enough and centered:
            return cmd, f'arrived (area={area_frac:.3f}, x_err={x_err_norm:+.2f})'

        # Angular: P-control unless inside the deadband.
        if centered:
            ang = 0.0
        else:
            sign = -1.0 if self.args.invert_yaw else 1.0
            ang = sign * self.args.kp_ang * x_err_norm
            ang = max(-self.args.max_ang, min(self.args.max_ang, ang))
            if abs(ang) < self.args.min_ang:
                ang = self.args.min_ang * (1.0 if ang > 0 else -1.0)

        # Linear: only drive forward when not arrived AND roughly aligned.
        if close_enough or abs(x_err_norm) > self.args.heading_gate:
            lin = 0.0
        else:
            lin = max(0.0, min(self.args.max_lin, self.args.kp_lin * size_err))
            if 0 < lin < self.args.min_lin:
                lin = self.args.min_lin

        cmd.linear.x = lin
        cmd.angular.z = ang

        if close_enough:
            status = f'aligning (x_err={x_err_norm:+.2f}, area={area_frac:.3f})'
        elif abs(x_err_norm) > self.args.heading_gate:
            status = f'rotating (x_err={x_err_norm:+.2f})'
        else:
            status = (f'tracking: x_err={x_err_norm:+.2f} '
                      f'area={area_frac:.3f}/{self.args.target_area_frac:.2f}')
        return cmd, status

    # ---------- callbacks / loop ----------

    def on_image(self, msg: Image):
        try:
            bgr = msg_to_bgr(msg)
        except ValueError as e:
            self.get_logger().warn(str(e), throttle_duration_sec=2.0)
            return
        bgr = rotate_frame(bgr, self.args.rotate)
        h, w = bgr.shape[:2]
        self.last_frame_shape = (h, w)

        if self.tune_open:
            self.read_trackbars()

        mask, det = self.detect(bgr)

        cmd, status = self.control(det, w, h)
        if self.drive_enabled:
            self.cmd_pub.publish(cmd)

        # Auto-finish: once we've been "arrived" continuously for finish_hold_s, we're done.
        is_arrived = status.startswith('arrived')
        if is_arrived:
            now = time.time()
            if self.arrived_since is None:
                self.arrived_since = now
            elif self.args.auto_finish and (now - self.arrived_since) >= self.args.finish_hold_s:
                self.get_logger().info(
                    f'target held for {self.args.finish_hold_s:.1f}s — finishing'
                )
                self.shutdown_requested = True
        else:
            self.arrived_since = None

        # ---- visualization ----
        vis = bgr.copy()
        cv2.line(vis, (w // 2, 0), (w // 2, h), (255, 200, 0), 1)
        if det is not None:
            x, y, bw, bh = det['bbox']
            cx, cy = det['centroid']
            cv2.rectangle(vis, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
            cv2.circle(vis, (int(cx), int(cy)), 5, (0, 0, 255), -1)

        drive_color = (0, 255, 0) if self.drive_enabled else (0, 0, 255)
        cv2.putText(vis, f'drive={"ON" if self.drive_enabled else "OFF"}', (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, drive_color, 2)
        cv2.putText(vis, status, (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(vis, f'lin={cmd.linear.x:+.2f} ang={cmd.angular.z:+.2f}', (10, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow(self.WIN_MAIN, vis)
        cv2.imshow(self.WIN_MASK, mask)
        self.handle_keys(bgr)

    def handle_keys(self, frame):
        k = cv2.waitKey(1) & 0xFF
        if k == 255:
            return
        if k in (ord('q'), 27):
            self.publish_stop()
            self.shutdown_requested = True
        elif k == ord('d'):
            self.drive_enabled = not self.drive_enabled
            if not self.drive_enabled:
                self.publish_stop()
            self.get_logger().info(f'drive {"ENABLED" if self.drive_enabled else "DISABLED"}')
        elif k == ord('t'):
            self.toggle_tune()
        elif k == ord('s'):
            path = f'/tmp/visual_servo_{self.save_count:03d}.png'
            cv2.imwrite(path, frame)
            self.get_logger().info(f'saved {path}')
            self.save_count += 1

    def publish_stop(self):
        self.cmd_pub.publish(Twist())

    # ---------- trackbars ----------

    def toggle_tune(self):
        if self.tune_open:
            cv2.destroyWindow(self.WIN_TUNE)
            self.tune_open = False
            return
        cv2.namedWindow(self.WIN_TUNE)
        cv2.createTrackbar('H lo', self.WIN_TUNE, int(self.hsv_lo[0]), 179, lambda v: None)
        cv2.createTrackbar('H hi', self.WIN_TUNE, int(self.hsv_hi[0]), 179, lambda v: None)
        cv2.createTrackbar('S lo', self.WIN_TUNE, int(self.hsv_lo[1]), 255, lambda v: None)
        cv2.createTrackbar('S hi', self.WIN_TUNE, int(self.hsv_hi[1]), 255, lambda v: None)
        cv2.createTrackbar('V lo', self.WIN_TUNE, int(self.hsv_lo[2]), 255, lambda v: None)
        cv2.createTrackbar('V hi', self.WIN_TUNE, int(self.hsv_hi[2]), 255, lambda v: None)
        self.tune_open = True

    def read_trackbars(self):
        self.hsv_lo = np.array([
            cv2.getTrackbarPos('H lo', self.WIN_TUNE),
            cv2.getTrackbarPos('S lo', self.WIN_TUNE),
            cv2.getTrackbarPos('V lo', self.WIN_TUNE),
        ], dtype=np.uint8)
        self.hsv_hi = np.array([
            cv2.getTrackbarPos('H hi', self.WIN_TUNE),
            cv2.getTrackbarPos('S hi', self.WIN_TUNE),
            cv2.getTrackbarPos('V hi', self.WIN_TUNE),
        ], dtype=np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--topic', default='/camera/image_raw')
    ap.add_argument('--cmd-topic', default='/cmd_vel')
    ap.add_argument('--rotate', type=int, default=0, choices=[0, 90, 180, 270],
                    help='rotate incoming frame (camera mounted sideways/upside-down)')

    # HSV defaults for RED target — red wraps the hue circle, so two ranges OR'd:
    # lower band [0, h_hi] and upper band [h_lo2, 179]. Saturation/Value bounds shared.
    ap.add_argument('--h-lo', type=int, default=0)
    ap.add_argument('--h-hi', type=int, default=10)
    ap.add_argument('--h-lo2', type=int, default=170,
                    help='secondary range low (for red hue wrap); pass --no-second to disable')
    ap.add_argument('--h-hi2', type=int, default=179)
    ap.add_argument('--no-second', action='store_true',
                    help='disable the secondary hue range (use for non-red colors)')
    ap.add_argument('--s-lo', type=int, default=120)
    ap.add_argument('--s-hi', type=int, default=255)
    ap.add_argument('--v-lo', type=int, default=70)
    ap.add_argument('--v-hi', type=int, default=255)

    ap.add_argument('--no-drive', action='store_true',
                    help='start with drive disabled (default: drive ON immediately)')
    ap.add_argument('--invert-yaw', action='store_true',
                    help='flip the yaw sign — use if the bot turns the wrong way')

    ap.add_argument('--no-pause-controller', dest='pause_controller', action='store_false',
                    help='do NOT pause /controller/pause on start (default: pause it)')
    ap.add_argument('--no-auto-finish', dest='auto_finish', action='store_false',
                    help='do NOT auto-exit after the target is held (default: auto-finish)')
    ap.add_argument('--finish-hold-s', type=float, default=1.0,
                    help='seconds the target must remain "arrived" before auto-finish triggers')
    ap.set_defaults(pause_controller=True, auto_finish=True)

    ap.add_argument('--min-area-px', type=int, default=400,
                    help='ignore contours smaller than this many pixels')
    ap.add_argument('--target-area-frac', type=float, default=0.05,
                    help='stop driving when target area / image area exceeds this')
    ap.add_argument('--area-tol', type=float, default=0.01,
                    help='arrival tolerance — declare arrived when area_frac >= target - area_tol')
    ap.add_argument('--heading-gate', type=float, default=0.25,
                    help='if |x_err| > this fraction, only rotate (no forward drive)')
    ap.add_argument('--x-tol', type=float, default=0.04,
                    help='|x_err| <= this is "centered" — disables angular control + min_ang floor')

    ap.add_argument('--kp-ang', type=float, default=1.2)
    ap.add_argument('--kp-lin', type=float, default=0.6)
    ap.add_argument('--max-ang', type=float, default=0.8)
    ap.add_argument('--max-lin', type=float, default=0.12)
    ap.add_argument('--min-ang', type=float, default=0.15)
    ap.add_argument('--min-lin', type=float, default=0.03)

    args = ap.parse_args()
    if args.no_second:
        args.h_lo2 = None
        args.h_hi2 = None

    rclpy.init()
    node = VisualServo(args)
    try:
        # Manual spin loop so we can exit on a flag without calling rclpy.shutdown()
        # from inside a callback (which is flaky across rclpy versions).
        while rclpy.ok() and not node.shutdown_requested:
            rclpy.spin_once(node, timeout_sec=0.05)
    except KeyboardInterrupt:
        pass
    finally:
        node.publish_stop()
        if rclpy.ok():
            node.resume_controller()
        cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
