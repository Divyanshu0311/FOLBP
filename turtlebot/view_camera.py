#!/usr/bin/env python3
"""Subscribe to camera_ros and display frames with cv2.

Defaults to /camera/image_raw. Press 'q' or ESC in the window to quit.

Usage:
    python3 view_camera.py                          # raw, /camera/image_raw
    python3 view_camera.py --compressed             # /camera/image_raw/compressed
    python3 view_camera.py --topic /my/cam/image    # custom topic
"""

import argparse

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage


def msg_to_bgr(msg: Image):
    """Decode sensor_msgs/Image to a BGR numpy array. Handles common encodings + NV21."""
    enc = msg.encoding.lower()
    buf = np.frombuffer(msg.data, dtype=np.uint8)

    if enc == 'bgr8':
        return buf.reshape(msg.height, msg.width, 3)
    if enc == 'rgb8':
        return cv2.cvtColor(buf.reshape(msg.height, msg.width, 3), cv2.COLOR_RGB2BGR)
    if enc == 'mono8':
        return buf.reshape(msg.height, msg.width)
    if enc == 'nv21':
        return cv2.cvtColor(buf.reshape(msg.height * 3 // 2, msg.width), cv2.COLOR_YUV2BGR_NV21)
    if enc == 'nv12':
        return cv2.cvtColor(buf.reshape(msg.height * 3 // 2, msg.width), cv2.COLOR_YUV2BGR_NV12)
    if enc in ('yuv420', 'i420'):
        return cv2.cvtColor(buf.reshape(msg.height * 3 // 2, msg.width), cv2.COLOR_YUV2BGR_I420)
    if enc in ('yuyv', 'yuv422'):
        return cv2.cvtColor(buf.reshape(msg.height, msg.width, 2), cv2.COLOR_YUV2BGR_YUYV)
    raise ValueError(f'unsupported encoding: {msg.encoding}')


class Viewer(Node):
    def __init__(self, topic: str, compressed: bool):
        super().__init__('cv_viewer')
        self.window = 'camera'
        cv2.namedWindow(self.window, cv2.WINDOW_AUTOSIZE)
        self.frames = 0
        if compressed:
            self.create_subscription(CompressedImage, topic, self.on_compressed, 10)
        else:
            self.create_subscription(Image, topic, self.on_raw, 10)
        self.get_logger().info(f'subscribed: {topic} ({"compressed" if compressed else "raw"})')

    def show(self, bgr):
        if bgr is None:
            return
        cv2.imshow(self.window, bgr)
        self.frames += 1
        k = cv2.waitKey(1) & 0xFF
        if k in (ord('q'), 27):  # q or ESC
            rclpy.shutdown()

    def on_raw(self, msg: Image):
        try:
            self.show(msg_to_bgr(msg))
        except ValueError as e:
            self.get_logger().warn(str(e), throttle_duration_sec=2.0)

    def on_compressed(self, msg: CompressedImage):
        arr = np.frombuffer(msg.data, dtype=np.uint8)
        self.show(cv2.imdecode(arr, cv2.IMREAD_COLOR))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--topic', default='/camera/image_raw')
    ap.add_argument('--compressed', action='store_true',
                    help='subscribe as CompressedImage (auto-appends /compressed if not in topic)')
    args = ap.parse_args()

    topic = args.topic
    if args.compressed and not topic.endswith('/compressed'):
        topic = topic + '/compressed'

    rclpy.init()
    node = Viewer(topic, args.compressed)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
