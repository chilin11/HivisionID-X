"""Print layout regressions; no model weights or inference required."""
import base64
import io
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from hivision.creator.layout_calculator import generate_layout_array, generate_layout_image
from server.routes.image import router


class PrintLayoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app = FastAPI()
        app.include_router(router)
        cls.client = TestClient(app)
        # A one-pixel pattern detects downsampling to standard size followed
        # by enlargement, which would destroy detail despite a 600 DPI tag.
        y, x = np.indices((826, 590))
        cls.source = np.stack(((x % 2) * 255, (y % 2) * 255, (x + y) % 256), axis=2).astype(np.uint8)

    def post(self, source=None, **params):
        buffer = io.BytesIO()
        Image.fromarray(self.source if source is None else source).save(buffer, format="PNG")
        response = self.client.post(
            "/generate_layout_photos",
            files={"input_image": ("hd.png", buffer.getvalue(), "image/png")},
            data=params,
        )
        self.assertEqual(response.status_code, 200)
        return response.json()

    def decode(self, payload):
        self.assertTrue(payload["status"], payload)
        image = Image.open(io.BytesIO(base64.b64decode(payload["image_base64"].split(",")[-1])))
        image.load()
        self.assertEqual(image.size, (payload["size"]["width"], payload["size"]["height"]))
        for dpi in image.info["dpi"]:
            self.assertAlmostEqual(dpi, payload["dpi"], delta=0.02)
        return image

    def test_legacy_default_still_exports_300_dpi(self):
        payload = self.post()
        image = self.decode(payload)
        self.assertEqual(image.format, "PNG")
        self.assertEqual(image.size, (1795, 1205))
        self.assertEqual(payload["dpi"], 300)

    def test_600_dpi_preserves_source_pixels_and_physical_dimensions(self):
        payload = self.post(output_dpi=600)
        image = self.decode(payload)
        self.assertEqual(image.size, (3590, 2410))
        self.assertEqual(payload["dpi"], 600)
        positions, rotated = generate_layout_array(413, 295)
        self.assertFalse(rotated)
        for x, y in positions:
            tile = np.asarray(image)[y * 2:y * 2 + 826, x * 2:x * 2 + 590]
            np.testing.assert_array_equal(tile, self.source)
        self.assertAlmostEqual(590 / payload["dpi"] * 25.4, 295 / 300 * 25.4)
        self.assertAlmostEqual(826 / payload["dpi"] * 25.4, 413 / 300 * 25.4)
        self.assertAlmostEqual(image.width / payload["dpi"], 1795 / 300)
        self.assertAlmostEqual(image.height / payload["dpi"], 1205 / 300)

    def test_larger_hd_source_is_downsampled_only_once(self):
        source = np.repeat(np.repeat(self.source, 2, axis=0), 2, axis=1)
        image = self.decode(self.post(source=source, output_dpi=600))
        positions, _ = generate_layout_array(413, 295)
        x, y = positions[0]
        np.testing.assert_array_equal(np.asarray(image)[y * 2:y * 2 + 826, x * 2:x * 2 + 590], self.source)

    def test_rotated_two_inch_photo_keeps_both_dimensions(self):
        source = cv2.resize(self.source, (826, 1252), interpolation=cv2.INTER_NEAREST)
        payload = self.post(source=source, height=626, width=413, output_dpi=600)
        image = np.asarray(self.decode(payload))
        positions, rotated = generate_layout_array(626, 413)
        self.assertTrue(rotated)
        expected = cv2.rotate(source, cv2.ROTATE_90_COUNTERCLOCKWISE)
        for x, y in positions:
            np.testing.assert_array_equal(image[y * 2:y * 2 + 826, x * 2:x * 2 + 1252], expected)

    def test_crop_lines_follow_scaled_photo_edges(self):
        image = np.asarray(self.decode(self.post(output_dpi=600, crop_line=True)))
        positions, _ = generate_layout_array(413, 295)
        x, y = positions[0]
        for edge in (x * 2, x * 2 + 590):
            np.testing.assert_array_equal(image[0, edge], [200, 200, 200])
        for edge in (y * 2, y * 2 + 826):
            np.testing.assert_array_equal(image[edge, 0], [200, 200, 200])
        np.testing.assert_array_equal(image[y * 2 + 10:y * 2 + 30, x * 2 + 10:x * 2 + 30], self.source[10:30, 10:30])

    def test_other_paper_sizes_export_real_600_dpi_pixels(self):
        for width, height in ((1500, 1051), (3508, 2479)):
            with self.subTest(paper=(width, height)):
                image = self.decode(self.post(layout_width=width, layout_height=height, output_dpi=600))
                self.assertEqual(image.size, (width * 2, height * 2))

    def test_native_size_mode_updates_dpi_too(self):
        image = self.decode(self.post(keep_original_size=True))
        self.assertEqual(image.size, (3590, 2410))
        self.assertAlmostEqual(image.info["dpi"][0], 600, delta=0.02)

    def test_explicit_dpi_takes_precedence_over_native_mode(self):
        image = self.decode(self.post(keep_original_size=True, output_dpi=300))
        self.assertEqual(image.size, (1795, 1205))

    def test_native_rounding_uses_uniform_scale_for_rotated_photos(self):
        # HD crops need not be exact integer multiples of standard dimensions.
        source = np.full((1251, 826, 3), (20, 70, 180), dtype=np.uint8)
        payload = self.post(source=source, height=626, width=413, keep_original_size=True)
        image = self.decode(payload)
        scale = 1251 / 626
        self.assertEqual(image.size, (round(1795 * scale), round(1205 * scale)))
        self.assertAlmostEqual(payload["dpi"], 300 * scale)

    def test_same_height_different_width_is_resized(self):
        source = np.full((413, 400, 3), 30, dtype=np.uint8)
        positions, rotated = generate_layout_array(413, 295)
        result = generate_layout_image(source, positions, rotated)
        self.assertEqual(result.shape, (1205, 1795, 3))

    def test_kb_opt_in_remains_jpeg_at_output_dpi(self):
        source = np.full((826, 590, 3), (20, 70, 180), dtype=np.uint8)
        image = self.decode(self.post(source=source, output_dpi=600, kb=100))
        self.assertEqual(image.format, "JPEG")
        self.assertEqual(image.size, (3590, 2410))

    def test_invalid_geometry_and_empty_layout_return_errors(self):
        for params in ({"width": 0}, {"height": -1}, {"dpi": 0}, {"output_dpi": 0}, {"output_dpi": 1201}, {"height": 2000, "width": 2000}, {"kb": -1}):
            with self.subTest(params=params):
                self.assertFalse(self.post(**params)["status"])

    def test_giant_canvas_rejected_before_rendering(self):
        with patch("server.routes.image.generate_layout_image") as render:
            payload = self.post(layout_width=100_000, layout_height=100_000, output_dpi=600)
        self.assertFalse(payload["status"])
        self.assertIn("40 megapixels", payload["error"])
        render.assert_not_called()

    def test_invalid_image_returns_error(self):
        response = self.client.post("/generate_layout_photos", files={"input_image": ("bad.png", b"not an image", "image/png")})
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json()["status"])


if __name__ == "__main__":
    unittest.main()
