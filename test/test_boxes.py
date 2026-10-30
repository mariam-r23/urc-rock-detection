import unittest

from urc_rock_detection.boxes import detection_to_box_fields
from urc_rock_detection.detector import Detection


def det(x, y, w, h, conf=0.8, label="rock"):
    return Detection(label, conf, x, y, w, h, float(w * h), x + w / 2, y + h, x + w / 2, y + h / 2)


class BoxFieldTests(unittest.TestCase):
    def test_normal_box(self):
        f = detection_to_box_fields(det(10, 20, 30, 40), 640, 480)
        self.assertEqual((f["x_min"], f["y_min"], f["x_max"], f["y_max"]), (10, 20, 40, 60))
        self.assertEqual((f["image_width"], f["image_height"]), (640, 480))
        self.assertEqual(f["class_name"], "rock")

    def test_clamped_inside_image_and_unsigned(self):
        f = detection_to_box_fields(det(-15, -5, 900, 900), 640, 480)
        self.assertEqual((f["x_min"], f["y_min"]), (0, 0))
        self.assertEqual((f["x_max"], f["y_max"]), (639, 479))
        for k in ("x_min", "y_min", "x_max", "y_max"):
            self.assertGreaterEqual(f[k], 0)

    def test_ordering_holds_for_degenerate_box(self):
        f = detection_to_box_fields(det(700, 500, 10, 10), 640, 480)
        self.assertLessEqual(f["x_min"], f["x_max"])
        self.assertLessEqual(f["y_min"], f["y_max"])

    def test_confidence_clamped_and_nan_safe(self):
        self.assertEqual(detection_to_box_fields(det(1, 1, 5, 5, conf=3.0), 64, 48)["confidence"], 1.0)
        self.assertEqual(detection_to_box_fields(det(1, 1, 5, 5, conf=float("nan")), 64, 48)["confidence"], 0.0)

    def test_zero_size_image_does_not_crash(self):
        f = detection_to_box_fields(det(0, 0, 5, 5), 0, 0)
        self.assertEqual((f["image_width"], f["image_height"]), (1, 1))


if __name__ == "__main__":
    unittest.main()
