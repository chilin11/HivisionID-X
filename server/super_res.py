#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
Real-ESRGAN 超分增强
====================

目标：**同等尺寸下的清晰度提升**——×4 超分后用 INTER_AREA 缩回原尺寸，
重建边缘与纹理细节（比 USM 更自然的"发丝级"锐化）。

模型（位于 hivision/creator/weights/）：
- realesr-general-x4v3.onnx   紧凑版（默认，真实照片向，约 0.4s/256² 块）
- realesrgan-x4plus.onnx      完整版（可选，更高质量，CPU 上显著更慢）

实现要点：
- 分块推理（512² 块 + 16px 重叠上下文，只保留块中心区域，避免接缝）
- 输入超出 max_input_side 时先降采样再超分（快速档，速度/质量折中）
- 会话全局缓存，线程安全
"""
import hashlib
import os
import threading
from collections import OrderedDict
from typing import Optional

import cv2
import numpy as np
import onnxruntime as ort

_WEIGHTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "hivision", "creator", "weights",
)
MODELS = {
    "general": os.path.join(_WEIGHTS_DIR, "realesr-general-x4v3.onnx"),
    "x4plus": os.path.join(_WEIGHTS_DIR, "realesrgan-x4plus.onnx"),
}
DEFAULT_MODEL = "general"

_SCALE = 4
_TILE = 640
_OVERLAP = 16  # 块边缘上下文，仅保留中心避免接缝

_SESS = {}
_LOCK = threading.Lock()

# 结果缓存：同一裁剪 + 同一参数直接复用（滑杆微调场景）
_CACHE: "OrderedDict[str, np.ndarray]" = OrderedDict()
_CACHE_MAX = 4


def _session(model_key: str) -> ort.InferenceSession:
    with _LOCK:
        sess = _SESS.get(model_key)
        if sess is None:
            path = MODELS.get(model_key, MODELS[DEFAULT_MODEL])
            if not os.path.exists(path):
                raise FileNotFoundError(f"Super-resolution model missing: {path}")
            so = ort.SessionOptions()
            so.log_severity_level = 3
            sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
            _SESS[model_key] = sess
        return sess


def _run_tile(sess: ort.InferenceSession, tile: np.ndarray) -> np.ndarray:
    """BGR uint8 块 → BGR uint8 ×4 块"""
    x = tile.astype(np.float32) / 255.0
    x = x.transpose(2, 0, 1)[None]  # 1,3,h,w
    out = sess.run(None, {sess.get_inputs()[0].name: x})[0][0]
    out = np.clip(out.transpose(1, 2, 0) * 255.0, 0, 255).astype(np.uint8)
    return out


def _tiled(sr_fn, img: np.ndarray) -> np.ndarray:
    """分块超分并中心拼接"""
    h, w = img.shape[:2]
    out = np.zeros((h * _SCALE, w * _SCALE, 3), np.uint8)
    step = _TILE - 2 * _OVERLAP
    ys = list(range(0, h, step)) or [0]
    xs = list(range(0, w, step)) or [0]
    for iy, y in enumerate(ys):
        for ix, x in enumerate(xs):
            y1, x1 = max(0, y - _OVERLAP), max(0, x - _OVERLAP)
            y2, x2 = min(h, y + step + _OVERLAP), min(w, x + step + _OVERLAP)
            tile_out = sr_fn(img[y1:y2, x1:x2])
            # 块内中心区域坐标 → 输出坐标
            cy1 = (y - y1) * _SCALE
            cx1 = (x - x1) * _SCALE
            ch = min(step, h - y) * _SCALE
            cw = min(step, w - x) * _SCALE
            out[y * _SCALE : y * _SCALE + ch, x * _SCALE : x * _SCALE + cw] = (
                tile_out[cy1 : cy1 + ch, cx1 : cx1 + cw]
            )
    return out


def enhance(
    rgb: np.ndarray,
    max_input_side: int = 1400,
    model: str = DEFAULT_MODEL,
    use_cache: bool = True,
) -> np.ndarray:
    """
    同尺寸清晰度增强：×4 超分 → 缩回输入尺寸。

    :param rgb: RGB uint8
    :param max_input_side: 超分前输入最大边长上限（更大的图先降采样，
        折中速度；传 0 关闭折中以获得最高质量）
    :param model: "general"（默认，快）或 "x4plus"（高质量，慢）
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("enhance expects RGB (3 channels)")
    h, w = rgb.shape[:2]

    key = None
    if use_cache:
        key = hashlib.sha1(np.ascontiguousarray(rgb).tobytes()).hexdigest()[:16] + f"|{max_input_side}|{model}"
        with _LOCK:
            hit = _CACHE.get(key)
            if hit is not None:
                _CACHE.move_to_end(key)
                return hit

    sess = _session(model)
    work = rgb
    if max_input_side and max(h, w) > max_input_side:
        s = max_input_side / max(h, w)
        work = cv2.resize(rgb, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)

    bgr = cv2.cvtColor(work, cv2.COLOR_RGB2BGR)
    sr_bgr = _tiled(lambda t: _run_tile(sess, t), bgr)
    out = cv2.resize(sr_bgr, (w, h), interpolation=cv2.INTER_AREA)
    out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)

    if key is not None:
        with _LOCK:
            _CACHE[key] = out
            _CACHE.move_to_end(key)
            while len(_CACHE) > _CACHE_MAX:
                _CACHE.popitem(last=False)
    return out


def available() -> bool:
    """默认模型是否已就位"""
    return os.path.exists(MODELS[DEFAULT_MODEL])
