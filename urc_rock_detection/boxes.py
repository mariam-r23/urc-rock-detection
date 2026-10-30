"""Conversion of a Detection to the fields of urc_msgs/DetectionBoundingBox (pure Python, no ROS).

urc_msgs/msg/DetectionBoundingBox.msg:
    std_msgs/Header header
    string class_name
    float32 confidence
    uint32 image_width
    uint32 image_height
    uint32 x_min
    uint32 y_min
    uint32 x_max
    uint32 y_max
"""
from __future__ import annotations

from typing import Dict

from .detector import Detection


def detection_to_box_fields(det: Detection, image_width: int, image_height: int) -> Dict:
    """Return msg field values: clamped inside the image, unsigned, x_min <= x_max, y_min <= y_max."""
    w, h = max(1, int(image_width)), max(1, int(image_height))
    x_min = max(0, min(w - 1, int(det.x)))
    y_min = max(0, min(h - 1, int(det.y)))
    x_max = max(x_min, min(w - 1, int(det.x + det.w)))
    y_max = max(y_min, min(h - 1, int(det.y + det.h)))
    conf = float(det.confidence)
    if conf != conf:  # NaN
        conf = 0.0
    return {
        "class_name": str(det.label),
        "confidence": max(0.0, min(1.0, conf)),
        "image_width": w, "image_height": h,
        "x_min": x_min, "y_min": y_min, "x_max": x_max, "y_max": y_max,
    }
