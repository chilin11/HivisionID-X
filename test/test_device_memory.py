"""Verify conservative ONNX allocator settings without loading a model."""
import os
import unittest
from unittest.mock import patch

from hivision import device


class OnnxAllocatorTests(unittest.TestCase):
    def test_default_disables_large_persistent_arenas(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(device.os.path, 'exists', return_value=True), patch.object(device.ort, 'InferenceSession') as create:
            device.load_onnx_model('model.onnx', set_cpu=True)
        kwargs = create.call_args.kwargs
        self.assertFalse(kwargs['sess_options'].enable_cpu_mem_arena)
        self.assertFalse(kwargs['sess_options'].enable_mem_pattern)
        # Low-memory allocation must not silently cap CPU parallelism.
        self.assertEqual(kwargs['sess_options'].intra_op_num_threads, 0)
        self.assertEqual(kwargs['sess_options'].inter_op_num_threads, 0)
        self.assertEqual(kwargs['providers'], ['CPUExecutionProvider'])

    def test_explicit_beast_mode_keeps_runtime_defaults(self):
        with patch.dict(os.environ, {'RUN_MODE': 'beast'}, clear=True), patch.object(device.os.path, 'exists', return_value=True), patch.object(device.ort, 'InferenceSession') as create:
            device.load_onnx_model('model.onnx', set_cpu=True)
        self.assertTrue(create.call_args.kwargs['sess_options'].enable_cpu_mem_arena)
        self.assertTrue(create.call_args.kwargs['sess_options'].enable_mem_pattern)

    def test_cpu_fallback_uses_same_conservative_options(self):
        fallback = object()
        with patch.dict(os.environ, {}, clear=True), patch.object(device.os.path, 'exists', return_value=True), patch.object(device, 'get_best_providers', return_value=['CUDAExecutionProvider', 'CPUExecutionProvider']), patch.object(device.ort, 'InferenceSession', side_effect=[RuntimeError('provider failed'), fallback]) as create:
            self.assertIs(device.load_onnx_model('model.onnx'), fallback)
        first, second = create.call_args_list
        self.assertIs(first.kwargs['sess_options'], second.kwargs['sess_options'])
        self.assertFalse(second.kwargs['sess_options'].enable_cpu_mem_arena)
        self.assertEqual(second.kwargs['providers'], ['CPUExecutionProvider'])


if __name__ == '__main__':
    unittest.main()
