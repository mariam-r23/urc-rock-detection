"""Synthetic rover-camera scenes with ground-truth boxes (used for tests and offline evaluation).

Light comes from the upper-left, so rocks cast shadows to the lower-right, and craters show a dark
inner wall on the upper-left and a bright inner wall on the lower-right.
These images only approximate real terrain; final validation must be run on the team's recorded bag.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import cv2
import numpy as np

GT = Tuple[str, int, int, int, int]  # (label, x, y, w, h)

DIFFICULTY: Dict[str, Dict[str, float]] = {
    "easy":   dict(noise=3.0, blur=0, brightness=1.0, contrast=1.0, rock_delta=45),
    "medium": dict(noise=6.0, blur=3, brightness=0.8, contrast=0.8, rock_delta=38),
    "hard":   dict(noise=10.0, blur=5, brightness=0.6, contrast=0.6, rock_delta=30),
}


def _lowfreq(rng, h, w, gx=8, gy=6, amp=10.0):
    g = rng.normal(0, amp, (gy, gx)).astype(np.float32)
    return cv2.resize(g, (w, h), interpolation=cv2.INTER_CUBIC)


def _poly(rng, cx, cy, rx, ry, n=9):
    ang = np.sort(rng.uniform(0, 2 * np.pi, n))
    rad = rng.uniform(0.7, 1.0, n)
    return np.stack([cx + rx * rad * np.cos(ang), cy + ry * rad * np.sin(ang)], 1).astype(np.int32)


def generate_scene(seed: int, difficulty: str = "easy", w: int = 640, h: int = 480,
                   n_rocks: int = 5, n_craters: int = 1) -> Tuple[np.ndarray, List[GT]]:
    cfg = DIFFICULTY[difficulty]
    rng = np.random.default_rng(seed)
    horizon = int(0.28 * h)
    g0 = rng.uniform(105, 145)

    img = np.zeros((h, w), np.float32)
    sky = np.linspace(205, 175, horizon, dtype=np.float32)[:, None] * np.ones((1, w), np.float32)
    img[:horizon] = sky
    ground = g0 + _lowfreq(rng, h - horizon, w, amp=9.0)
    ground += cv2.GaussianBlur(rng.normal(0, 7, (h - horizon, w)).astype(np.float32), (0, 0), 1.5)
    img[horizon:] = ground

    gt: List[GT] = []
    placed: List[Tuple[int, int, int, int]] = []

    def free(box):
        x, y, bw, bh = box
        for (px, py, pw, ph) in placed:
            if x < px + pw + 12 and px < x + bw + 12 and y < py + ph + 12 and py < y + bh + 12:
                return False
        return True

    def persp(y):  # object size factor grows toward the bottom of the image
        return (y - horizon) / float(h - horizon)

    # craters first (large, flat)
    for _ in range(n_craters):
        for _try in range(30):
            cy = int(rng.uniform(horizon + 0.45 * (h - horizon), h - 60))
            rx = int(rng.uniform(45, 90) * (0.6 + persp(cy)))
            ry = max(12, int(rx * 0.42))
            cx = int(rng.uniform(rx + 5, w - rx - 5))
            box = (cx - rx, cy - ry, 2 * rx, 2 * ry)
            if cy + ry < h - 10 and free(box):
                m = np.zeros((h, w), np.uint8)
                cv2.ellipse(m, (cx, cy), (rx, ry), 0, 0, 360, 255, -1)
                inner = m > 0
                img[inner] -= 6
                for sgn, delta in ((-1, -cfg["rock_delta"] * 0.9), (1, cfg["rock_delta"] * 0.7)):
                    m2 = np.zeros((h, w), np.uint8)
                    cv2.ellipse(m2, (cx + int(sgn * 0.25 * rx), cy + int(sgn * 0.22 * ry)),
                                (rx, ry), 0, 0, 360, 255, -1)
                    crescent = inner & (m2 == 0)
                    # sgn=-1 -> shifted ellipse covers lower-right => crescent on upper-left (dark)
                    # sgn=+1 -> shifted ellipse covers upper-left => crescent on lower-right (bright)
                    img[crescent] += delta
                gt.append(("crater", box[0], box[1], box[2], box[3]))
                placed.append(box)
                break

    for _ in range(n_rocks):
        for _try in range(30):
            cy = int(rng.uniform(horizon + 0.25 * (h - horizon), h - 25))
            r = max(9, int(rng.uniform(10, 30) * (0.5 + 1.4 * persp(cy))))
            cx = int(rng.uniform(r + 25, w - r - 25))
            sh = int(0.55 * r)  # shadow offset toward lower-right
            box = (cx - r, cy - r, 2 * r + sh, 2 * r + sh)
            if cy + r + sh < h - 5 and free(box):
                body = _poly(rng, cx, cy, r, int(0.8 * r))
                shadow = body + np.array([sh, int(0.4 * sh)], np.int32)
                shadow_m = np.zeros((h, w), np.uint8)
                cv2.fillPoly(shadow_m, [shadow], 255)
                body_m = np.zeros((h, w), np.uint8)
                cv2.fillPoly(body_m, [body], 255)
                dark_rock = rng.random() < 0.65
                body_level = g0 - cfg["rock_delta"] if dark_rock else g0 + cfg["rock_delta"] * 0.9
                img[shadow_m > 0] = g0 - 30 * cfg["contrast"] - 6
                img[body_m > 0] = body_level
                hl = body - np.array([int(0.25 * r), int(0.25 * r)], np.int32)
                hl_m = np.zeros((h, w), np.uint8)
                cv2.fillPoly(hl_m, [((hl - [cx, cy]) * 0.6 + [cx, cy]).astype(np.int32)], 255)
                img[(hl_m > 0) & (body_m > 0)] += 22
                ys, xs = np.nonzero((body_m > 0) | (shadow_m > 0))
                gt.append(("rock", int(xs.min()), int(ys.min()),
                           int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)))
                placed.append(gt[-1][1:])
                break

    # global degradations
    img = (img - 128.0) * cfg["contrast"] + 128.0
    img *= cfg["brightness"]
    if cfg["blur"] > 0:
        k = int(cfg["blur"]) | 1
        img = cv2.GaussianBlur(img, (k, k), 0)
    img += rng.normal(0, cfg["noise"], img.shape).astype(np.float32)
    gray = np.clip(img, 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    tint = np.array([0.92, 0.97, 1.05], np.float32)  # slightly warm regolith colour
    bgr = np.clip(bgr.astype(np.float32) * tint, 0, 255).astype(np.uint8)
    return bgr, gt
