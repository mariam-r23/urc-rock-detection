"""Flat-ground camera projection and hazard footprint generation (no ROS imports).

Frame convention (REP 103): base frame x forward, y left, z up. The camera optical frame is
x right, y down, z forward. The camera sits at (cam_x, cam_y, cam_height) in the base frame, yawed by
cam_yaw and pitched DOWN by cam_pitch (radians, positive = looking toward the ground).
"""
from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .detector import CRATER, Detection


@dataclass
class CameraModel:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    height_m: float = 0.5      # camera height above ground
    pitch_rad: float = 0.35    # tilt down
    x_m: float = 0.15          # camera position in base frame
    y_m: float = 0.0
    yaw_rad: float = 0.0
    max_range_m: float = 8.0
    min_range_m: float = 0.2

    @staticmethod
    def from_fov(width: int, height: int, hfov_deg: float, **kw) -> "CameraModel":
        f = (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
        return CameraModel(f, f, width / 2.0, height / 2.0, width, height, **kw)

    def scaled(self, width: int, height: int) -> "CameraModel":
        """Return the model rescaled to another image resolution (same optics)."""
        sx, sy = width / float(self.width), height / float(self.height)
        return CameraModel(self.fx * sx, self.fy * sy, self.cx * sx, self.cy * sy, width, height,
                           self.height_m, self.pitch_rad, self.x_m, self.y_m, self.yaw_rad,
                           self.max_range_m, self.min_range_m)

    def pixel_to_ground(self, u: float, v: float) -> Optional[Tuple[float, float]]:
        """Intersect the pixel ray with the plane z=0. None if above the horizon / out of range / invalid."""
        if not (math.isfinite(u) and math.isfinite(v)) or self.fx <= 0 or self.fy <= 0:
            return None
        dx, dy = (u - self.cx) / self.fx, (v - self.cy) / self.fy
        s, c = math.sin(self.pitch_rad), math.cos(self.pitch_rad)
        # optical axes expressed in the (x fwd, y left, z up) frame, camera pitched down by pitch
        fwd = np.array([c, 0.0, -s])
        right = np.array([0.0, -1.0, 0.0])
        down = np.array([-s, 0.0, -c])
        d = dx * right + dy * down + fwd
        cyaw, syaw = math.cos(self.yaw_rad), math.sin(self.yaw_rad)
        d = np.array([cyaw * d[0] - syaw * d[1], syaw * d[0] + cyaw * d[1], d[2]])
        if d[2] >= -1e-6:
            return None  # ray never reaches the ground
        t = self.height_m / -d[2]
        x, y = self.x_m + t * d[0], self.y_m + t * d[1]
        r = math.hypot(x, y)
        if r > self.max_range_m or r < self.min_range_m:
            return None
        return float(x), float(y)


@dataclass
class Hazard:
    label: str
    confidence: float
    x: float          # centre in base frame [m]
    y: float
    rx: float         # half-extents along base x / y [m]
    ry: float


def detection_to_hazard(det: Detection, cam: CameraModel, min_radius: float = 0.10,
                        max_radius: float = 1.5) -> Optional[Hazard]:
    """Convert an image detection to a ground-plane hazard (centre + half extents)."""
    x0, y0, x1, y1 = det.x, det.y, det.x + det.w, det.y + det.h
    xm, ym = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    if det.label == CRATER:
        # a crater is flat: all four box edge midpoints lie on the ground
        pts = [cam.pixel_to_ground(xm, y1), cam.pixel_to_ground(xm, y0),
               cam.pixel_to_ground(x0, ym), cam.pixel_to_ground(x1, ym)]
        near, far, left, right = pts
        if near is None:
            return None
        if far is None:
            far = (near[0] + 2 * max_radius, near[1])
        cx, cy = (near[0] + far[0]) / 2.0, (near[1] + far[1]) / 2.0
        rx = abs(far[0] - near[0]) / 2.0
        if left is not None and right is not None:
            ry = abs(left[1] - right[1]) / 2.0
        else:
            ry = rx
    else:
        # a rock stands on the ground: only the bottom edge is on the plane
        bl, bc, br = (cam.pixel_to_ground(x0, y1), cam.pixel_to_ground(xm, y1),
                      cam.pixel_to_ground(x1, y1))
        if bc is None:
            return None
        if bl is not None and br is not None:
            width = math.hypot(br[0] - bl[0], br[1] - bl[1])
        else:
            width = 2 * min_radius
        r = max(min_radius, width / 2.0)
        # the rock's centre sits about one radius beyond its ground contact point
        cx, cy = bc[0] + r * math.cos(cam.yaw_rad), bc[1] + r * math.sin(cam.yaw_rad)
        rx = ry = r
    rx = float(np.clip(rx, min_radius, max_radius))
    ry = float(np.clip(ry, min_radius, max_radius))
    return Hazard(det.label, det.confidence, float(cx), float(cy), rx, ry)


def hazard_points(h: Hazard, inflation: float = 0.15, resolution: float = 0.05) -> np.ndarray:
    """Filled-ellipse point set (N x 3, z=0) covering the hazard plus an inflation margin."""
    rx, ry = h.rx + inflation, h.ry + inflation
    res = max(1e-3, resolution)
    xs = np.arange(-rx, rx + 1e-9, res)
    ys = np.arange(-ry, ry + 1e-9, res)
    gx, gy = np.meshgrid(xs, ys)
    inside = (gx / rx) ** 2 + (gy / ry) ** 2 <= 1.0
    pts = np.stack([gx[inside] + h.x, gy[inside] + h.y, np.zeros(int(inside.sum()))], axis=1)
    if pts.size == 0:
        pts = np.array([[h.x, h.y, 0.0]])
    return pts.astype(np.float32)


def pointcloud2_fields_and_data(points: np.ndarray):
    """Return (point_step, [(name, offset)], raw bytes) for an XYZ float32 PointCloud2 payload."""
    pts = np.ascontiguousarray(points.reshape(-1, 3), dtype=np.float32)
    return 12, [("x", 0), ("y", 4), ("z", 8)], pts.tobytes()


def yaw_to_quaternion(yaw: float) -> Tuple[float, float, float, float]:
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))
