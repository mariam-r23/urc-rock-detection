import unittest

import numpy as np

from urc_rock_detection.detector import DetectorParams, RockCraterDetector
from urc_rock_detection.dummy_source import DummySource
from urc_rock_detection.evaluate import match


class DummySourceTests(unittest.TestCase):
    def test_length_and_hold(self):
        s = DummySource(n_scenes=3, hold=4)
        self.assertEqual(len(s), 12)
        a, _ = s.frame(0)
        b, _ = s.frame(3)
        c, _ = s.frame(4)
        self.assertTrue(np.array_equal(a, b))        # same scene held
        self.assertFalse(np.array_equal(a, c))       # next scene differs

    def test_labels_keyed_by_frame(self):
        s = DummySource(n_scenes=2, hold=3)
        lab = s.labels_by_frame()
        self.assertEqual(sorted(lab, key=int), [str(i) for i in range(6)])
        self.assertEqual(lab["0"], lab["2"])
        for boxes in lab.values():
            for b in boxes:
                self.assertEqual(len(b), 4)

    def test_bad_args(self):
        with self.assertRaises(ValueError):
            DummySource(n_scenes=0)
        with self.assertRaises(ValueError):
            DummySource(levels=[])

    def test_detector_finds_most_dummy_hazards(self):
        det = RockCraterDetector(DetectorParams())
        s = DummySource(n_scenes=12, hold=1)
        tp = fn = 0
        for k in range(12):
            img, boxes = s.scene(k)
            a, _b, c = match(det.detect(img), boxes)
            tp, fn = tp + a, fn + c
        self.assertGreaterEqual(tp / (tp + fn), 0.85)


if __name__ == "__main__":
    unittest.main()
