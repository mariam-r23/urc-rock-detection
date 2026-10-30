import math
import unittest

import numpy as np

from urc_rock_detection.detector import (CRATER, ROCK, DetectorParams, RockCraterDetector,
                                         SimpleTracker, Detection, iou, to_gray_u8)
from urc_rock_detection.evaluate import match
from urc_rock_detection.synthetic import generate_scene


class TestRobustness(unittest.TestCase):
    def setUp(self):
        self.det = RockCraterDetector()

    def test_none_and_empty(self):
        self.assertEqual(self.det.detect(None), [])
        self.assertEqual(self.det.detect(np.zeros((0, 0), np.uint8)), [])

    def test_tiny_image(self):
        self.assertEqual(self.det.detect(np.zeros((10, 10, 3), np.uint8)), [])

    def test_constant_images(self):
        for v in (0, 128, 255):
            self.assertEqual(self.det.detect(np.full((480, 640, 3), v, np.uint8)), [])

    def test_bad_shapes_and_dtypes(self):
        self.assertEqual(self.det.detect(np.zeros((480, 640, 2), np.uint8)), [])
        self.assertEqual(self.det.detect(np.zeros(100, np.uint8)), [])
        img16 = (np.random.default_rng(0).random((480, 640)) * 65535).astype(np.uint16)
        self.assertIsInstance(self.det.detect(img16), list)
        imgf = np.random.default_rng(0).random((480, 640)).astype(np.float32)
        imgf[0, 0] = np.nan
        self.assertIsInstance(self.det.detect(imgf), list)

    def test_gray_input_and_bgra(self):
        img, gt = generate_scene(1, "easy", n_rocks=4, n_craters=0)
        g = to_gray_u8(img)
        self.assertEqual(g.ndim, 2)
        a = self.det.detect(g)
        bgra = np.dstack([img, np.full(img.shape[:2], 255, np.uint8)])
        b = self.det.detect(bgra)
        self.assertGreater(len(a), 0)
        self.assertGreater(len(b), 0)

    def test_pure_noise_has_no_hazards(self):
        rng = np.random.default_rng(3)
        for _ in range(5):
            img = np.clip(rng.normal(120, 6, (480, 640)), 0, 255).astype(np.uint8)
            self.assertEqual(self.det.detect(img), [], "noise must not create hazards")

    def test_smooth_gradient_has_no_hazards(self):
        ramp = np.tile(np.linspace(60, 180, 640, dtype=np.float32), (480, 1)).astype(np.uint8)
        self.assertEqual(self.det.detect(ramp), [])

    def test_sky_region_ignored(self):
        img = np.full((480, 640), 120, np.uint8)
        img[10:60, 100:160] = 20  # dark blob inside the sky band (top 30%)
        self.assertEqual([d for d in self.det.detect(img) if d.y + d.h < 0.25 * 480], [])

    def test_resolution_independence(self):
        import cv2
        img, gt = generate_scene(7, "easy", n_rocks=4, n_craters=0)
        big = cv2.resize(img, (1280, 960))
        small = cv2.resize(img, (320, 240))
        for im in (big, small):
            dets = self.det.detect(im)
            self.assertGreaterEqual(len(dets), 1)
        # boxes of the large image are reported in large-image pixels
        for d in self.det.detect(big):
            self.assertLessEqual(d.x + d.w, 1280 + 2)
            self.assertLessEqual(d.y + d.h, 960 + 2)


class TestDetectionRate(unittest.TestCase):
    def _rate(self, level, seeds):
        det = RockCraterDetector()
        tp = fp = fn = 0
        for s in seeds:
            img, gt = generate_scene(1000 + s, level, n_rocks=3 + s % 4, n_craters=s % 2)
            a, b, c = match(det.detect(img), [g[1:] for g in gt])
            tp, fp, fn = tp + a, fp + b, fn + c
        return tp / (tp + fn), tp / max(1, tp + fp)

    def test_easy_rate_above_85(self):
        r, p = self._rate("easy", range(60))
        self.assertGreaterEqual(r, 0.85, f"recall {r:.3f}")
        self.assertGreaterEqual(p, 0.85, f"precision {p:.3f}")

    def test_medium_rate_above_85(self):
        r, p = self._rate("medium", range(60))
        self.assertGreaterEqual(r, 0.85, f"recall {r:.3f}")
        self.assertGreaterEqual(p, 0.85, f"precision {p:.3f}")

    def test_craters_labelled_crater(self):
        det = RockCraterDetector()
        found = 0
        for s in range(20):
            img, gt = generate_scene(2000 + s, "easy", n_rocks=0, n_craters=1)
            if not gt:
                continue
            for d in det.detect(img):
                if iou(gt[0][1:], d.bbox()) >= 0.3 and d.label == CRATER:
                    found += 1
                    break
        self.assertGreaterEqual(found, 12)


class TestTracker(unittest.TestCase):
    def _d(self, x=100, y=200, w=40, h=30):
        return Detection(ROCK, 0.8, x, y, w, h, 1200.0, x + w / 2, y + h, x + w / 2, y + h / 2)

    def test_confirmation_and_expiry(self):
        t = SimpleTracker(min_hits=2, max_age=1)
        self.assertEqual(t.update([self._d()]), [])          # first sight: not yet confirmed
        self.assertEqual(len(t.update([self._d(102)])), 1)   # second sight: confirmed
        self.assertEqual(t.update([]), [])                   # missed
        self.assertEqual(t.update([]), [])                   # missed again -> track dropped
        self.assertEqual(t.update([self._d()]), [])          # restarts from 1 hit

    def test_min_hits_one_passes_through(self):
        t = SimpleTracker(min_hits=1)
        self.assertEqual(len(t.update([self._d()])), 1)

    def test_two_objects_tracked_separately(self):
        t = SimpleTracker(min_hits=2)
        a, b = self._d(50), self._d(400)
        t.update([a, b])
        self.assertEqual(len(t.update([a, b])), 2)


if __name__ == "__main__":
    unittest.main()
