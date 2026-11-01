"""Bound face-detection memory without shrinking the HD source image."""
import os
import unittest
from unittest.mock import patch

import numpy as np

from hivision.creator import face_detector
from hivision.creator.context import Context, Params
from hivision.creator.retinaface import inference
from hivision.error import FaceError


class RecordingSession:
    def __init__(self):
        self.shapes = []

    def run(self, _, inputs):
        tensor = inputs["input"]
        self.shapes.append(tensor.shape)
        assert tensor.dtype == np.float32
        return (
            np.zeros((1, 1, 4), np.float32),
            np.array([[[0.01, 0.99]]], np.float32),
            np.zeros((1, 1, 10), np.float32),
        )


class FaceDetectionMemoryTests(unittest.TestCase):
    def test_large_detector_input_is_bounded_and_coordinates_use_original_pixels(self):
        for height, width in ((5656, 4242), (4242, 5656), (4031, 3023), (5000, 5000)):
            with self.subTest(size=(height, width)):
                image = np.zeros((height, width, 3), dtype=np.uint8)
                image[0, 0] = (20, 60, 140)
                session = RecordingSession()
                with patch.object(inference, "PriorBox") as prior:
                    prior.return_value.forward.return_value = np.array([[0.5, 0.5, 0.25, 0.25]])
                    dets, returned_session = inference.retinaface_detect_faces(image, "unused.onnx", sess=session)
                shape = session.shapes[0]
                self.assertEqual(max(shape[2:]), 1024)
                self.assertEqual(shape[:2], (1, 3))
                self.assertLessEqual(shape[2] * shape[3], 1024 ** 2)
                self.assertAlmostEqual(shape[3] / shape[2], width / height, delta=0.002)
                prior.assert_called_once()
                self.assertEqual(prior.call_args.kwargs["image_size"], shape[2:])
                np.testing.assert_allclose(dets[0, :4], [width * .375, height * .375, width * .625, height * .625])
                np.testing.assert_allclose(dets[0, 5:], [width * .5, height * .5] * 5)
                self.assertAlmostEqual(float(dets[0, 4]), .99, places=5)
                self.assertIs(returned_session, session)
                self.assertEqual(image.shape, (height, width, 3))
                np.testing.assert_array_equal(image[0, 0], [20, 60, 140])

    def test_small_image_is_not_upscaled(self):
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        session = RecordingSession()
        with patch.object(inference, "PriorBox") as prior, patch.object(inference.cv2, "resize", wraps=inference.cv2.resize) as resize:
            prior.return_value.forward.return_value = np.array([[0.5, 0.5, 0.25, 0.25]])
            inference.retinaface_detect_faces(image, "unused.onnx", sess=session)
        self.assertEqual(session.shapes, [(1, 3, 480, 640)])
        resize.assert_not_called()

    def test_native_default_does_not_retain_detector_session(self):
        self.check_session_mode(None, False)

    def test_beast_mode_explicitly_retains_detector_session(self):
        self.check_session_mode("beast", True)

    def check_session_mode(self, mode, retained):
        ctx = Context(Params())
        source = np.zeros((2000, 1500, 3), dtype=np.uint8)
        ctx.origin_image = source
        dets = np.array([[200, 300, 600, 900, .99, 300, 400, 500, 420, 400, 500, 300, 700, 500, 700]])
        session = object()
        env = {} if mode is None else {"RUN_MODE": mode}
        with patch.dict(os.environ, env, clear=True), patch.object(face_detector, "RETINAFCE_SESS", None), patch.object(face_detector, "retinaface_detect_faces", return_value=(dets, session)) as detect:
            face_detector.detect_face_retinaface(ctx)
            self.assertIs(face_detector.RETINAFCE_SESS, session if retained else None)
            self.assertEqual(ctx.face["rectangle"], (200, 300, 401, 601))
            self.assertAlmostEqual(ctx.face["roll_angle"], np.degrees(np.arctan2(20, 200)))
            self.assertIs(ctx.origin_image, source)
            face_detector.detect_face_retinaface(ctx)
            self.assertIs(detect.call_args.kwargs["sess"], session if retained else None)

    def test_default_releases_session_when_no_face_is_found(self):
        ctx = Context(Params())
        ctx.origin_image = np.zeros((100, 100, 3), dtype=np.uint8)
        with patch.dict(os.environ, {}, clear=True), patch.object(face_detector, "RETINAFCE_SESS", object()), patch.object(face_detector, "retinaface_detect_faces", return_value=(np.empty((0, 15)), object())):
            with self.assertRaises(FaceError):
                face_detector.detect_face_retinaface(ctx)
            self.assertIsNone(face_detector.RETINAFCE_SESS)

    def test_default_releases_cached_session_on_inference_error(self):
        ctx = Context(Params())
        ctx.origin_image = np.zeros((100, 100, 3), dtype=np.uint8)
        with patch.dict(os.environ, {}, clear=True), patch.object(face_detector, "RETINAFCE_SESS", object()), patch.object(face_detector, "retinaface_detect_faces", side_effect=RuntimeError("inference failed")):
            with self.assertRaisesRegex(RuntimeError, "inference failed"):
                face_detector.detect_face_retinaface(ctx)
            self.assertIsNone(face_detector.RETINAFCE_SESS)


if __name__ == "__main__":
    unittest.main()
