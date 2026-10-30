#!/usr/bin/env python3
"""ROS 2 (Humble) rock & crater detection node.

Subscribes : image_raw   (sensor_msgs/Image)       camera frames
             camera_info (sensor_msgs/CameraInfo)  optional intrinsics (falls back to hfov_deg)
Publishes  : hazard_points (sensor_msgs/PointCloud2)  XYZ ground points of every hazard -> Nav2 obstacle layer
             hazards       (geometry_msgs/PoseArray)  hazard centres (debug / RViz)
             detections    (std_msgs/String)          JSON list: label, confidence, bbox, x, y, rx, ry
             detection_boxes (urc_msgs/DetectionBoundingBox) one message per hazard (only if urc_msgs is built)
             debug_image   (sensor_msgs/Image)        annotated frame (only if publish_debug_image)

Only standard ROS 2 message packages are used (sensor_msgs, geometry_msgs, std_msgs); no cv_bridge.
"""
from __future__ import annotations

import json
import time
from typing import List, Optional

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import Pose, PoseArray
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField
from std_msgs.msg import String

from .boxes import detection_to_box_fields
from .detector import CRATER, Detection, DetectorParams, RockCraterDetector, SimpleTracker
from .image_utils import image_msg_to_ndarray, ndarray_to_image_fields
from .projection import (CameraModel, Hazard, detection_to_hazard, hazard_points,
                         pointcloud2_fields_and_data, yaw_to_quaternion)

# urc_msgs (RoboJackets URC software) is optional: when it is not built in the workspace the node
# still runs and simply skips the DetectionBoundingBox publisher.
try:
    from urc_msgs.msg import DetectionBoundingBox
except ImportError:  # pragma: no cover - depends on the workspace
    DetectionBoundingBox = None


class RockDetectionNode(Node):
    def __init__(self):
        super().__init__("rock_detection_node")
        d = DetectorParams()

        # ---- parameters (all overridable from config/rock_detection.yaml) ----
        P = self.declare_parameter
        P("image_topic", "image_raw")
        P("camera_info_topic", "camera_info")
        P("output_frame", "base_link")
        P("publish_debug_image", True)
        P("publish_urc_boxes", True)         # urc_msgs/DetectionBoundingBox, one message per hazard
        P("max_process_hz", 15.0)            # drop frames faster than this
        P("max_image_age_s", 1.0)             # ignore stale frames (clock skew / queue build-up)
        # camera mounting (base_link frame) and fallback optics
        P("camera_height_m", 0.5)
        P("camera_pitch_deg", 20.0)
        P("camera_x_m", 0.15)
        P("camera_y_m", 0.0)
        P("camera_yaw_deg", 0.0)
        P("hfov_deg", 90.0)
        P("max_range_m", 8.0)
        P("min_range_m", 0.2)
        # hazard geometry for the costmap
        P("inflation_m", 0.15)
        P("point_resolution_m", 0.05)
        P("min_hazard_radius_m", 0.10)
        P("max_hazard_radius_m", 1.5)
        # temporal confirmation
        P("min_hits", 2)
        P("max_age", 3)
        # detector tuning
        P("process_width", d.process_width)
        P("horizon_frac", d.horizon_frac)
        P("ignore_bottom_frac", d.ignore_bottom_frac)
        P("clahe_clip", d.clahe_clip)
        P("bg_sigma_frac", d.bg_sigma_frac)
        P("dark_thresh", d.dark_thresh)
        P("bright_thresh", d.bright_thresh)
        P("adaptive_k", d.adaptive_k)
        P("close_ksize", d.close_ksize)
        P("min_area_frac", d.min_area_frac)
        P("max_area_frac", d.max_area_frac)
        P("min_solidity", d.min_solidity)
        P("min_edge_frac", d.min_edge_frac)
        P("edge_grad_thresh", d.edge_grad_thresh)
        P("crater_min_area_frac", d.crater_min_area_frac)
        P("min_confidence", d.min_confidence)
        P("max_detections", d.max_detections)

        self.detector = RockCraterDetector(self._params_from_ros())
        self.tracker = SimpleTracker(self._i("min_hits"), self._i("max_age"))
        self._cam: Optional[CameraModel] = None
        self._cam_info = None
        self._dirty = False
        self._cam_info_seen = False
        self._last_proc = 0.0
        self._frames = 0
        self._dropped = 0

        sensor_qos = qos_profile_sensor_data
        self.create_subscription(Image, self._s("image_topic"), self._on_image, sensor_qos)
        self.create_subscription(CameraInfo, self._s("camera_info_topic"), self._on_info, sensor_qos)
        latched = QoSProfile(depth=1, history=HistoryPolicy.KEEP_LAST, reliability=ReliabilityPolicy.RELIABLE)
        self.pub_cloud = self.create_publisher(PointCloud2, "~/hazard_points", latched)
        self.pub_poses = self.create_publisher(PoseArray, "~/hazards", latched)
        self.pub_json = self.create_publisher(String, "~/detections", latched)
        self.pub_dbg = self.create_publisher(Image, "~/debug_image", 1)
        self.pub_boxes = None
        if self._b("publish_urc_boxes"):
            if DetectionBoundingBox is not None:
                self.pub_boxes = self.create_publisher(DetectionBoundingBox, "~/detection_boxes", 10)
            else:
                self.get_logger().warn(
                    "urc_msgs not found: DetectionBoundingBox output disabled "
                    "(build urc_msgs in this workspace and restart to enable it)")
        self.add_on_set_parameters_callback(self._on_params)
        self.get_logger().info(
            f"rock_detection_node ready: image='{self._s('image_topic')}' frame='{self._s('output_frame')}'")

    # ----------------------------------------------------------- parameter helpers
    def _s(self, n): return self.get_parameter(n).get_parameter_value().string_value
    def _f(self, n): return float(self.get_parameter(n).value)
    def _i(self, n): return int(self.get_parameter(n).value)
    def _b(self, n): return bool(self.get_parameter(n).value)

    def _params_from_ros(self) -> DetectorParams:
        names_f = ["horizon_frac", "ignore_bottom_frac", "clahe_clip", "bg_sigma_frac", "dark_thresh",
                   "bright_thresh", "adaptive_k", "min_area_frac", "max_area_frac", "min_solidity",
                   "min_edge_frac", "edge_grad_thresh", "crater_min_area_frac", "min_confidence"]
        names_i = ["process_width", "close_ksize", "max_detections"]
        kw = {n: self._f(n) for n in names_f}
        kw.update({n: self._i(n) for n in names_i})
        return DetectorParams(**kw)

    def _on_params(self, _params):
        """Live re-tuning: flag a rebuild; it happens on the next frame, after values are applied."""
        from rcl_interfaces.msg import SetParametersResult
        self._dirty = True
        return SetParametersResult(successful=True)

    def _rebuild_if_dirty(self):
        if not self._dirty:
            return
        self._dirty = False
        try:
            self.detector = RockCraterDetector(self._params_from_ros())
            self.tracker = SimpleTracker(self._i("min_hits"), self._i("max_age"))
            self._cam = None  # camera mounting may have changed
        except Exception as exc:
            self.get_logger().error(f"parameter rebuild failed: {exc}")

    # ----------------------------------------------------------- callbacks
    def _on_info(self, msg: CameraInfo):
        try:
            k = msg.k
            if msg.width > 0 and msg.height > 0 and k[0] > 0 and k[4] > 0:
                self._cam_info = (float(k[0]), float(k[4]), float(k[2]), float(k[5]),
                                  int(msg.width), int(msg.height))
                self._cam = None  # rebuild with real intrinsics
                self._cam_info_seen = True
        except Exception:
            pass

    def _camera_for(self, w: int, h: int) -> CameraModel:
        if self._cam is not None and (self._cam.width, self._cam.height) == (w, h):
            return self._cam
        mount = dict(height_m=self._f("camera_height_m"),
                     pitch_rad=np.deg2rad(self._f("camera_pitch_deg")),
                     x_m=self._f("camera_x_m"), y_m=self._f("camera_y_m"),
                     yaw_rad=np.deg2rad(self._f("camera_yaw_deg")),
                     max_range_m=self._f("max_range_m"), min_range_m=self._f("min_range_m"))
        if self._cam_info_seen:
            fx, fy, cx, cy, iw, ih = self._cam_info
            cam = CameraModel(fx, fy, cx, cy, iw, ih, **mount)
            if (iw, ih) != (w, h):
                cam = cam.scaled(w, h)
        else:
            cam = CameraModel.from_fov(w, h, self._f("hfov_deg"), **mount)
        self._cam = cam
        return cam

    def _on_image(self, msg: Image):
        now = time.monotonic()
        min_dt = 1.0 / max(self._f("max_process_hz"), 1e-3)
        if now - self._last_proc < min_dt:
            self._dropped += 1
            return
        # stale-frame guard (only when the stamp is set and clocks agree)
        stamp_s = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if stamp_s > 0:
            age = self.get_clock().now().nanoseconds * 1e-9 - stamp_s
            if age > self._f("max_image_age_s") and age < 1e6:
                self._dropped += 1
                return
        self._last_proc = now
        self._rebuild_if_dirty()

        img = image_msg_to_ndarray(msg)
        if img is None:
            self.get_logger().warn(f"unsupported/corrupt image (encoding='{msg.encoding}'), frame skipped",
                                   throttle_duration_sec=5.0)
            return
        try:
            dets = self.tracker.update(self.detector.detect(img))
        except Exception as exc:  # a vision failure must never kill the node
            self.get_logger().error(f"detection failed: {exc}", throttle_duration_sec=5.0)
            return
        self._frames += 1
        h, w = img.shape[:2]
        cam = self._camera_for(w, h)
        hazards: List[Hazard] = []
        kept: List[Detection] = []
        for d in dets:
            hz = detection_to_hazard(d, cam, self._f("min_hazard_radius_m"), self._f("max_hazard_radius_m"))
            if hz is not None:
                hazards.append(hz)
                kept.append(d)
        self._publish(msg.header, kept, hazards, w, h)
        if self._b("publish_debug_image") and self.pub_dbg.get_subscription_count() > 0:
            self._publish_debug(msg, img, kept, hazards)

    # ----------------------------------------------------------- publishing
    def _publish(self, header, dets: List[Detection], hazards: List[Hazard], img_w: int, img_h: int):
        stamp = header.stamp
        frame = self._s("output_frame")

        # PointCloud2 (always published, possibly empty so Nav2 can clear stale marks)
        if hazards:
            pts = np.concatenate([hazard_points(h, self._f("inflation_m"), self._f("point_resolution_m"))
                                  for h in hazards], axis=0)
        else:
            pts = np.zeros((0, 3), np.float32)
        step, fields, raw = pointcloud2_fields_and_data(pts)
        cloud = PointCloud2()
        cloud.header.stamp, cloud.header.frame_id = stamp, frame
        cloud.height, cloud.width = 1, int(pts.shape[0])
        cloud.fields = [PointField(name=n, offset=o, datatype=PointField.FLOAT32, count=1) for n, o in fields]
        cloud.is_bigendian, cloud.point_step = False, step
        cloud.row_step, cloud.data, cloud.is_dense = step * cloud.width, raw, True
        self.pub_cloud.publish(cloud)

        poses = PoseArray()
        poses.header.stamp, poses.header.frame_id = stamp, frame
        for h in hazards:
            p = Pose()
            p.position.x, p.position.y, p.position.z = h.x, h.y, 0.0
            qx, qy, qz, qw = yaw_to_quaternion(0.0)
            p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = qx, qy, qz, qw
            poses.poses.append(p)
        self.pub_poses.publish(poses)

        payload = [{"label": h.label, "confidence": h.confidence, "x": round(h.x, 3), "y": round(h.y, 3),
                    "rx": round(h.rx, 3), "ry": round(h.ry, 3), "bbox": list(d.bbox())}
                   for d, h in zip(dets, hazards)]
        self.pub_json.publish(String(data=json.dumps(payload)))

        if self.pub_boxes is not None:
            for d in dets:
                f = detection_to_box_fields(d, img_w, img_h)
                box = DetectionBoundingBox()
                box.header = header          # same stamp / camera frame as the image
                box.class_name = f["class_name"]
                box.confidence = f["confidence"]
                box.image_width, box.image_height = f["image_width"], f["image_height"]
                box.x_min, box.y_min, box.x_max, box.y_max = f["x_min"], f["y_min"], f["x_max"], f["y_max"]
                self.pub_boxes.publish(box)

    def _publish_debug(self, src: Image, img: np.ndarray, dets: List[Detection], hazards: List[Hazard]):
        vis = img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        vis = vis.copy()
        if vis.dtype != np.uint8:
            vis = cv2.normalize(vis, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        if vis.shape[2] == 4:
            vis = cv2.cvtColor(vis, cv2.COLOR_BGRA2BGR)
        for d, h in zip(dets, hazards):
            color = (0, 165, 255) if d.label == CRATER else (0, 0, 255)
            cv2.rectangle(vis, (d.x, d.y), (d.x + d.w, d.y + d.h), color, 2)
            cv2.putText(vis, f"{d.label} {d.confidence:.2f} ({h.x:.1f},{h.y:.1f}m)", (d.x, max(12, d.y - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        hh, ww, enc, step, data = ndarray_to_image_fields(vis)
        out = Image()
        out.header = src.header
        out.height, out.width, out.encoding, out.is_bigendian, out.step, out.data = hh, ww, enc, 0, step, data
        self.pub_dbg.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = RockDetectionNode()
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
