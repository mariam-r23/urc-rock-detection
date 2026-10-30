"""Write a labelled folder of dummy rock/crater images (no ROS needed).

    python3 -m urc_rock_detection.make_dummy_dataset --out ~/rock_dummy --n 40
    python3 -m urc_rock_detection.evaluate images --dir ~/rock_dummy --labels ~/rock_dummy/labels.json
"""
import argparse
import json
import os
import sys

import cv2

from .dummy_source import DummySource


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="output folder (created)")
    ap.add_argument("--n", type=int, default=40, help="number of scenes")
    ap.add_argument("--levels", default="easy,medium", help="comma list of easy,medium,hard")
    ap.add_argument("--seed0", type=int, default=1000)
    a = ap.parse_args(argv)

    src = DummySource(a.n, 1, a.levels.split(","), a.seed0)
    out = os.path.expanduser(a.out)
    os.makedirs(out, exist_ok=True)
    labels = {}
    for k in range(a.n):
        img, boxes = src.scene(k)
        name = f"scene_{k:03d}.png"
        cv2.imwrite(os.path.join(out, name), img)
        labels[name] = boxes
    with open(os.path.join(out, "labels.json"), "w", encoding="utf-8") as f:
        json.dump(labels, f)
    print(f"wrote {a.n} images + labels.json to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
