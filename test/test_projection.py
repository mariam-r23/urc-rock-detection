import math
import unittest

import numpy as np

from urc_rock_detection.detector import CRATER, ROCK, Detection
from urc_rock_detection.projection import (CameraModel, detection_to_hazard, hazard_points,
                                           pointcloud2_fields_and_data)


def cam(**kw):
    base = dict(height_m=0.5, pitch_rad=math.radians(20), x_m=0.0, y_m=0.0, max_range_m=20.0, min_range_m=0.0)
    base.update(kw)
    return CameraModel.from_fov(640, 480, 90.0, **base)


class TestProjection(unittest.TestCase):
    def test_principal_point_distance(self):
        c = cam()
        x, y = c.pixel_to_ground(320, 240)
        self.assertAlmostEqual(x, 0.5 / math.tan(math.radians(20)), places=4)
        self.assertAlmostEqual(y, 0.0, places=6)

    def test_left_right_symmetry(self):
        c = cam()
        l = c.pixel_to_ground(100, 400)
        r = c.pixel_to_ground(540, 400)
        self.assertAlmostEqual(l[0], r[0], places=6)
        self.assertAlmostEqual(l[1], -r[1], places=6)
        self.assertGreater(l[1], 0)  # image left = +y (left of rover)

    def test_lower_pixels_are_closer(self):
        c = cam()
        self.assertLess(c.pixel_to_ground(320, 460)[0], c.pixel_to_ground(320, 300)[0])

    def test_above_horizon_is_none(self):
        c = cam(pitch_rad=0.0)
        self.assertIsNone(c.pixel_to_ground(320, 100))
        self.assertIsNone(c.pixel_to_ground(320, 240))  # exactly on the horizon

    def test_range_limits_and_invalid(self):
        c = cam(max_range_m=1.0)
        self.assertIsNone(c.pixel_to_ground(320, 241))
        self.assertIsNone(c.pixel_to_ground(float("nan"), 10))

    def test_camera_offset_and_yaw(self):
        c = cam(x_m=0.3, yaw_rad=math.radians(90))
        x, y = c.pixel_to_ground(320, 240)
        self.assertAlmostEqual(x, 0.3, places=4)
        self.assertGreater(y, 0.5)  # camera now looks to the rover's left

    def test_scaled_model_matches(self):
        c = cam()
        s = c.scaled(1280, 960)
        a = c.pixel_to_ground(200, 350)
        b = s.pixel_to_ground(400, 700)
        self.assertAlmostEqual(a[0], b[0], places=4)
        self.assertAlmostEqual(a[1], b[1], places=4)


class TestHazards(unittest.TestCase):
    def _det(self, label, x, y, w, h):
        return Detection(label, 0.9, x, y, w, h, w * h, x + w / 2, y + h, x + w / 2, y + h / 2)

    def test_rock_ahead_has_sane_size(self):
        c = cam(pitch_rad=math.radians(25))
        hz = detection_to_hazard(self._det(ROCK, 300, 330, 60, 50), c)
        self.assertIsNotNone(hz)
        self.assertGreater(hz.x, 0.3)
        self.assertLess(hz.x, 3.0)
        self.assertGreaterEqual(hz.rx, 0.1)
        self.assertLessEqual(hz.rx, 1.5)

    def test_farther_rock_is_farther(self):
        c = cam(pitch_rad=math.radians(25))
        near = detection_to_hazard(self._det(ROCK, 300, 400, 60, 50), c)
        far = detection_to_hazard(self._det(ROCK, 300, 300, 30, 25), c)
        self.assertGreater(far.x, near.x)

    def test_crater_extent(self):
        c = cam(pitch_rad=math.radians(25))
        hz = detection_to_hazard(self._det(CRATER, 150, 330, 300, 120), c)
        self.assertIsNotNone(hz)
        self.assertGreater(hz.ry, 0.1)

    def test_unprojectable_returns_none(self):
        c = cam(pitch_rad=0.0)
        self.assertIsNone(detection_to_hazard(self._det(ROCK, 300, 100, 40, 40), c))

    def test_points_cover_hazard_and_are_float32(self):
        from urc_rock_detection.projection import Hazard
        h = Hazard(ROCK, 0.9, 2.0, 0.5, 0.3, 0.3)
        pts = hazard_points(h, inflation=0.1, resolution=0.05)
        self.assertEqual(pts.dtype, np.float32)
        self.assertEqual(pts.shape[1], 3)
        self.assertLessEqual(np.hypot(pts[:, 0] - 2.0, pts[:, 1] - 0.5).max(), 0.41)
        self.assertGreater(len(pts), 20)
        step, fields, raw = pointcloud2_fields_and_data(pts)
        self.assertEqual(step, 12)
        self.assertEqual(len(raw), 12 * len(pts))

    def test_empty_cloud_payload(self):
        step, fields, raw = pointcloud2_fields_and_data(np.zeros((0, 3), np.float32))
        self.assertEqual(raw, b"")


if __name__ == "__main__":
    unittest.main()
