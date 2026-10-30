"""Offline evaluation of the detector (detection rate = recall of ground-truth hazards).

Modes
-----
  synthetic : generated scenes with ground truth (works anywhere, no ROS needed)
  images    : a folder of images + labels.json  ({"img_name.png": [[x, y, w, h], ...], ...})
  bag       : a rosbag2 directory (needs ROS 2 + rosbag2_py); optional labels.json keyed by frame
              index ({"0": [[x, y, w, h], ...]}). Without labels it prints per-frame detections only.

Examples
--------
  python3 -m urc_rock_detection.evaluate synthetic --n 60
  python3 -m urc_rock_detection.evaluate images --dir ./frames --labels labels.json
  ros2 run urc_rock_detection evaluate bag --bag ./rock_hazards_bag --topic /camera/image_raw --labels labels.json

Exit code is 0 when recall >= --target (default 0.85), else 1.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np

from .detector import DetectorParams, RockCraterDetector, iou


def match(dets: Sequence, gts: Sequence[Sequence[float]], iou_thresh: float = 0.3) -> Tuple[int, int, int]:
    """Greedy IoU matching. Returns (true_pos, false_pos, false_neg)."""
    pairs = []
    for gi, g in enumerate(gts):
        for di, d in enumerate(dets):
            v = iou(g, d.bbox())
            if v >= iou_thresh:
                pairs.append((v, gi, di))
    pairs.sort(reverse=True)
    ug, ud = set(), set()
    for _, gi, di in pairs:
        if gi in ug or di in ud:
            continue
        ug.add(gi)
        ud.add(di)
    tp = len(ug)
    return tp, len(dets) - len(ud), len(gts) - tp


def summarise(name: str, tp: int, fp: int, fn: int) -> Dict[str, float]:
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    print(f"{name:>8}: GT={tp + fn:4d}  TP={tp:4d}  FP={fp:4d}  FN={fn:4d}  "
          f"detection_rate(recall)={recall * 100:5.1f}%  precision={precision * 100:5.1f}%")
    return {"recall": recall, "precision": precision}


def run_synthetic(n: int, levels: List[str], det: RockCraterDetector) -> Dict[str, Dict[str, float]]:
    from .synthetic import generate_scene
    out = {}
    for lvl in levels:
        tp = fp = fn = 0
        for s in range(n):
            img, gt = generate_scene(1000 + s, lvl, n_rocks=3 + s % 4, n_craters=s % 2)
            a, b, c = match(det.detect(img), [g[1:] for g in gt])
            tp, fp, fn = tp + a, fp + b, fn + c
        out[lvl] = summarise(lvl, tp, fp, fn)
    return out


def run_images(folder: str, labels_path: str, det: RockCraterDetector) -> Dict[str, float]:
    with open(labels_path, "r", encoding="utf-8") as f:
        labels = json.load(f)
    tp = fp = fn = 0
    for name, boxes in sorted(labels.items()):
        img = cv2.imread(os.path.join(folder, name), cv2.IMREAD_COLOR)
        if img is None:
            print(f"[warn] cannot read {name}, skipped", file=sys.stderr)
            continue
        a, b, c = match(det.detect(img), boxes)
        tp, fp, fn = tp + a, fp + b, fn + c
    return summarise("images", tp, fp, fn)


def run_bag(bag: str, topic: str, labels_path: str, det: RockCraterDetector, storage: str = "") -> Dict[str, float]:
    import rosbag2_py  # noqa: WPS433  (only needed in bag mode)
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import Image
    from .image_utils import image_msg_to_ndarray

    labels = {}
    if labels_path:
        with open(labels_path, "r", encoding="utf-8") as f:
            labels = json.load(f)
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=bag, storage_id=storage),
                rosbag2_py.ConverterOptions("", ""))
    idx = 0
    tp = fp = fn = 0
    n_frames = n_with_det = 0
    while reader.has_next():
        tname, data, _ = reader.read_next()
        if tname != topic:
            continue
        msg = deserialize_message(data, Image)
        img = image_msg_to_ndarray(msg)
        dets = det.detect(img) if img is not None else []
        n_frames += 1
        n_with_det += 1 if dets else 0
        if labels:
            gts = labels.get(str(idx), [])
            a, b, c = match(dets, gts)
            tp, fp, fn = tp + a, fp + b, fn + c
        else:
            print(f"frame {idx:5d}: {len(dets)} hazards "
                  f"({', '.join(f'{d.label}@{d.confidence:.2f}' for d in dets)})")
        idx += 1
    print(f"frames={n_frames} frames_with_detections={n_with_det}")
    return summarise("bag", tp, fp, fn) if labels else {"recall": float("nan"), "precision": float("nan")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)
    s = sub.add_parser("synthetic")
    s.add_argument("--n", type=int, default=60)
    s.add_argument("--levels", default="easy,medium")
    i = sub.add_parser("images")
    i.add_argument("--dir", required=True)
    i.add_argument("--labels", required=True)
    b = sub.add_parser("bag")
    b.add_argument("--bag", required=True)
    b.add_argument("--topic", default="/camera/image_raw")
    b.add_argument("--labels", default="")
    b.add_argument("--storage", default="", help="rosbag2 storage id (sqlite3 / mcap); blank = auto")
    ap.add_argument("--target", type=float, default=0.85)
    args = ap.parse_args(argv)

    det = RockCraterDetector(DetectorParams())
    if args.mode == "synthetic":
        res = run_synthetic(args.n, args.levels.split(","), det)
        recall = min(v["recall"] for v in res.values())
    elif args.mode == "images":
        recall = run_images(args.dir, args.labels, det)["recall"]
    else:
        recall = run_bag(args.bag, args.topic, args.labels, det, args.storage)["recall"]
    if recall != recall:  # NaN -> no labels supplied
        return 0
    ok = recall >= args.target
    print(f"RESULT: {'PASS' if ok else 'FAIL'} (min recall {recall * 100:.1f}% vs target {args.target * 100:.0f}%)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
