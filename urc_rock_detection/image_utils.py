"""sensor_msgs/Image <-> NumPy helpers (no cv_bridge dependency).

Works with any object exposing height, width, encoding, step, is_bigendian and data, so it can be unit
tested without ROS installed.
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

# encoding -> (dtype, channels)
_ENC = {
    "mono8": (np.uint8, 1), "8UC1": (np.uint8, 1),
    "mono16": (np.uint16, 1), "16UC1": (np.uint16, 1),
    "bgr8": (np.uint8, 3), "rgb8": (np.uint8, 3), "8UC3": (np.uint8, 3),
    "bgra8": (np.uint8, 4), "rgba8": (np.uint8, 4), "8UC4": (np.uint8, 4),
    "32FC1": (np.float32, 1),
}


def image_msg_to_ndarray(msg) -> Optional[np.ndarray]:
    """Decode an Image message into a BGR / gray ndarray. Returns None if unsupported or corrupt."""
    try:
        enc = str(msg.encoding)
        if enc not in _ENC:
            return None
        dtype, ch = _ENC[enc]
        h, w = int(msg.height), int(msg.width)
        if h <= 0 or w <= 0:
            return None
        itemsize = np.dtype(dtype).itemsize
        step = int(msg.step) if int(msg.step) > 0 else w * ch * itemsize
        buf = np.frombuffer(bytes(msg.data), dtype=np.uint8)
        if buf.size < step * h or step < w * ch * itemsize:
            return None
        rows = buf[: step * h].reshape(h, step)[:, : w * ch * itemsize]
        arr = np.ascontiguousarray(rows).view(dtype)
        if itemsize > 1 and bool(getattr(msg, "is_bigendian", False)) != (np.little_endian is False):
            arr = arr.byteswap()
        arr = arr.reshape(h, w, ch) if ch > 1 else arr.reshape(h, w)
        if enc == "rgb8":
            arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        elif enc == "rgba8":
            arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGRA)
        return arr
    except Exception:  # malformed message must never crash the node
        return None


def ndarray_to_image_fields(img: np.ndarray):
    """Return (height, width, encoding, step, bytes) for a uint8 BGR or gray image."""
    img = np.ascontiguousarray(img)
    if img.ndim == 2:
        return img.shape[0], img.shape[1], "mono8", img.shape[1], img.tobytes()
    return img.shape[0], img.shape[1], "bgr8", img.shape[1] * 3, img.tobytes()
