"""Upload decoding regressions, without model inference."""
import base64
import io
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from server import encoding


def encoded(image, fmt="PNG", **kwargs):
    buffer = io.BytesIO()
    image.save(buffer, format=fmt, **kwargs)
    return buffer.getvalue()


class UploadDecodingTests(unittest.TestCase):
    def test_48_megapixel_photo_is_accepted_and_scaled(self):
        source = Image.new("RGB", (8064, 6048), (70, 120, 180))
        data = encoded(source, "JPEG")
        source.close()
        result = encoding.decode_bytes_to_rgb(data)
        h, w = result.shape[:2]
        self.assertLessEqual(w * h, 24_000_000)
        self.assertGreater(w * h, 23_980_000)
        self.assertAlmostEqual(w / h, 4 / 3, places=3)

    def test_small_image_is_unchanged(self):
        source = np.arange(60 * 80 * 3, dtype=np.uint8).reshape(60, 80, 3)
        result = encoding.decode_bytes_to_rgb(encoded(Image.fromarray(source)))
        np.testing.assert_array_equal(result, source)

    def test_large_base64_photo_is_rotated_after_scaling(self):
        source = Image.new("RGB", (800, 600), "red")
        exif = Image.Exif()
        exif[274] = 6
        data = encoded(source, "JPEG", exif=exif)
        with patch.object(encoding, "MAX_INPUT_PIXELS", 120_000):
            result = encoding.decode_base64_to_rgb(
                "data:image/jpeg;base64," + base64.b64encode(data).decode()
            )
        self.assertEqual(result.shape, (400, 300, 3))

    def test_rgba_keeps_alpha_and_full_frame(self):
        source = Image.new("RGBA", (800, 600), (50, 100, 150, 128))
        with patch.object(encoding, "MAX_INPUT_PIXELS", 120_000):
            result = encoding.decode_bytes_to_rgba(encoded(source))
        self.assertEqual(result.shape, (300, 400, 4))
        self.assertTrue(np.all(result[:, :, 3] == 128))


if __name__ == "__main__":
    unittest.main()
