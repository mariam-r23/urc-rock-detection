#!/usr/bin/env python3
"""Fake rover camera: publishes dummy rock/crater frames + CameraInfo, and (optionally) the ground-truth labels.

    ros2 run urc_rock_detection fake_camera_node
    ros2 run urc_rock_detection fake_camera_node --ros-args -p labels_out:=/home/$USER/dummy_labels.json

Publishes  /camera/image_raw (sensor_msgs/Image, bgr8)  and  /camera/camera_info (sensor_msgs/CameraInfo)
After ``n_scenes * hold`` frames it stops (unless loop:=true) and logs how many frames it sent, so you can
check that number against the ``frames=`` line printed by ``evaluate bag``.
"""
import json
import math
import os

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from sensor_msgs.msg import CameraInfo, Image

from .dummy_source import DummySource
from .image_utils import ndarray_to_image_fields


class FakeCameraNode(Node):
    def __init__(self):
        super().__init__("fake_camera_node")
        P = self.declare_parameter
        P("image_topic", "/camera/image_raw")
        P("camera_info_topic", "/camera/camera_info")
        P("frame_id", "camera")
        P("rate_hz", 5.0)
        P("n_scenes", 40)
        P("hold", 5)
        P("levels", "easy,medium")
        P("hfov_deg", 90.0)
        P("start_delay_s", 3.0)       # time to start `ros2 bag record` before frames flow
        P("loop", False)
        P("labels_out", "")           # write {frame_index: [[x,y,w,h],...]} here (for evaluate bag --labels)

        g = lambda n: self.get_parameter(n).value  # noqa: E731
        self.src = DummySource(int(g("n_scenes")), int(g("hold")), str(g("levels")).split(","))
        self.loop = bool(g("loop"))
        self.frame_id = str(g("frame_id"))
        self.hfov = float(g("hfov_deg"))
        out = str(g("labels_out"))
        if out:
            out = os.path.expanduser(out)
            with open(out, "w", encoding="utf-8") as f:
                json.dump(self.src.labels_by_frame(), f)
            self.get_logger().info(f"ground-truth labels for {len(self.src)} frames written to {out}")

        qos = QoSProfile(depth=10)
        self.pub_img = self.create_publisher(Image, str(g("image_topic")), qos)
        self.pub_info = self.create_publisher(CameraInfo, str(g("camera_info_topic")), qos)
        self.i = 0
        self.sent = 0
        self.timer = self.create_timer(max(float(g("start_delay_s")), 0.01), self._start)
        self.get_logger().info(
            f"fake camera ready: {len(self.src)} frames at {float(g('rate_hz'))} Hz, "
            f"starting in {float(g('start_delay_s'))} s")
        self._period = 1.0 / max(float(g("rate_hz")), 1e-3)

    def _start(self):
        self.timer.cancel()
        self.timer = self.create_timer(self._period, self._tick)

    def _info(self, w, h, stamp) -> CameraInfo:
        fx = (w / 2.0) / math.tan(math.radians(self.hfov) / 2.0)
        m = CameraInfo()
        m.header.stamp, m.header.frame_id = stamp, self.frame_id
        m.width, m.height = int(w), int(h)
        m.distortion_model = "plumb_bob"
        m.d = [0.0] * 5
        m.k = [fx, 0.0, w / 2.0, 0.0, fx, h / 2.0, 0.0, 0.0, 1.0]
        m.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        m.p = [fx, 0.0, w / 2.0, 0.0, 0.0, fx, h / 2.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        return m

    def _tick(self):
        if self.i >= len(self.src):
            if self.loop:
                self.i = 0
            else:
                self.timer.cancel()
                self.get_logger().info(f"done: published {self.sent} frames")
                return
        img, _ = self.src.frame(self.i)
        h, w = img.shape[:2]
        stamp = self.get_clock().now().to_msg()
        msg = Image()
        msg.header.stamp, msg.header.frame_id = stamp, self.frame_id
        hh, ww, enc, step, data = ndarray_to_image_fields(img)
        msg.height, msg.width, msg.encoding = hh, ww, enc
        msg.is_bigendian, msg.step, msg.data = 0, step, data
        self.pub_info.publish(self._info(w, h, stamp))
        self.pub_img.publish(msg)
        self.i += 1
        self.sent += 1


def main(args=None):
    rclpy.init(args=args)
    node = FakeCameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
