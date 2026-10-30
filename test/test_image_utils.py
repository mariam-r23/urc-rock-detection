import unittest
from types import SimpleNamespace

import numpy as np

from urc_rock_detection.image_utils import image_msg_to_ndarray, ndarray_to_image_fields


def msg(arr, enc, step=None, big=False):
    h, w = arr.shape[:2]
    data = arr.tobytes()
    return SimpleNamespace(height=h, width=w, encoding=enc, is_bigendian=big,
                           step=step if step is not None else len(data) // h, data=data)


class TestImageUtils(unittest.TestCase):
    def test_bgr8_roundtrip(self):
        a = np.random.default_rng(0).integers(0, 255, (48, 64, 3), dtype=np.uint8)
        np.testing.assert_array_equal(image_msg_to_ndarray(msg(a, "bgr8")), a)

    def test_rgb8_is_converted_to_bgr(self):
        a = np.zeros((4, 4, 3), np.uint8)
        a[..., 0] = 255  # red in RGB
        out = image_msg_to_ndarray(msg(a, "rgb8"))
        self.assertEqual(tuple(out[0, 0]), (0, 0, 255))

    def test_mono8_and_mono16(self):
        a = np.arange(48 * 64, dtype=np.uint16).reshape(48, 64)
        self.assertEqual(image_msg_to_ndarray(msg(a.astype(np.uint8), "mono8")).shape, (48, 64))
        out = image_msg_to_ndarray(msg(a, "mono16"))
        self.assertEqual(out.dtype, np.uint16)
        np.testing.assert_array_equal(out, a)

    def test_row_padding_step(self):
        a = np.random.default_rng(1).integers(0, 255, (10, 20, 3), dtype=np.uint8)
        padded = np.zeros((10, 20 * 3 + 8), np.uint8)
        padded[:, : 20 * 3] = a.reshape(10, -1)
        m = SimpleNamespace(height=10, width=20, encoding="bgr8", is_bigendian=False,
                            step=padded.shape[1], data=padded.tobytes())
        np.testing.assert_array_equal(image_msg_to_ndarray(m), a)

    def test_bad_inputs_return_none(self):
        a = np.zeros((4, 4, 3), np.uint8)
        self.assertIsNone(image_msg_to_ndarray(msg(a, "bayer_rggb8")))
        m = msg(a, "bgr8")
        m.data = m.data[:-5]
        self.assertIsNone(image_msg_to_ndarray(m))
        m2 = msg(a, "bgr8")
        m2.height = 0
        self.assertIsNone(image_msg_to_ndarray(m2))
        self.assertIsNone(image_msg_to_ndarray(None))

    def test_fields_helper(self):
        h, w, enc, step, data = ndarray_to_image_fields(np.zeros((5, 6, 3), np.uint8))
        self.assertEqual((h, w, enc, step, len(data)), (5, 6, "bgr8", 18, 90))
        self.assertEqual(ndarray_to_image_fields(np.zeros((5, 6), np.uint8))[2], "mono8")


if __name__ == "__main__":
    unittest.main()
