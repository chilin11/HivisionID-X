"""Framing regressions; run with python -m unittest discover -s test -p test_smart_crop.py."""
import unittest

import cv2
import numpy as np

from server.smart_crop import CropParams, SubjectBox, _solve_crop_rect, smart_crop


class ShoulderFramingTests(unittest.TestCase):
    def test_portrait_and_square_keep_shoulders_and_headroom(self):
        # Head: y=200..500. Shoulders begin at y=590.
        rgba = np.zeros((1400, 1200, 4), dtype=np.uint8)
        cv2.ellipse(rgba, (600, 350), (110, 150), 0, 0, 360, (90, 90, 90, 255), -1)
        cv2.rectangle(rgba, (550, 480), (650, 650), (90, 90, 90, 255), -1)
        cv2.rectangle(rgba, (330, 590), (870, 1399), (90, 90, 90, 255), -1)
        face = (510, 260, 180, 240)
        subject = SubjectBox(200, 0, 330, 329)
        for size in [(413, 295), (626, 413), (600, 600)]:
            with self.subTest(size=size):
                rect = _solve_crop_rect(1400, 1200, 200, subject, face, CropParams(size), [])
                _, y, _, h = rect
                old = _solve_crop_rect(1400, 1200, 200, subject, face,
                                       CropParams(size, head_height_fraction=0.66), [])
                self.assertGreaterEqual(y + h, old[1] + old[3] + 35)
                self.assertGreaterEqual(y + h - 590, 50)
                self.assertAlmostEqual(300 / h, 0.59, delta=0.002)
                result = smart_crop(rgba, face, size)
                self.assertEqual(result.standard.shape, (*size, 4))
                self.assertTrue(0.098 <= result.quality.top_gap <= 0.122)

    def test_floating_body_does_not_enlarge_head_to_touch_bottom(self):
        # Previously the bottom anchor shrank the crop, overriding head size.
        subject = SubjectBox(crown=200, bottom=710, left=300, right=300)
        _, y, _, h = _solve_crop_rect(
            1400, 1200, 200, subject, (510, 260, 180, 240),
            CropParams((413, 295)), [],
        )
        self.assertLessEqual(300 / h, 0.592)
        self.assertTrue(0.098 <= (200 - y) / h <= 0.122)

    def test_smaller_head_setting_reveals_more_body(self):
        subject = SubjectBox(crown=200, bottom=0, left=300, right=300)
        bottoms = []
        for fraction in (0.66, 0.59, 0.50):
            x, y, w, h = _solve_crop_rect(
                1400, 1200, 200, subject, (510, 260, 180, 240),
                CropParams((413, 295), head_height_fraction=fraction), [],
            )
            self.assertTrue(0 <= x < x + w <= 1200)
            self.assertTrue(0 <= y < y + h <= 1400)
            self.assertTrue(0.098 <= (200 - y) / h <= 0.122)
            bottoms.append(y + h)
        self.assertLess(bottoms[0], bottoms[1])
        self.assertLess(bottoms[1], bottoms[2])


if __name__ == '__main__':
    unittest.main()
