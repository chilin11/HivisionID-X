#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
Creator 服务层
==============

负责三件事：
1. 全局唯一的 IDCreator 实例（模型只加载一次）。
2. 抠图结果缓存：同一张照片 + 同一组影响抠图/美颜的参数，
   只跑一次模型；后续的构图微调（头部位置/比例/留白）零推理开销。
3. 失败降级：人脸检测失败时自动切换"无检测抠图"模式，
   交给智能裁剪引擎做兜底构图，最大限度保证出图成功率。

注意：不修改 hivision 包内的任何模型代码，
人脸检测的旁路通过运行时临时替换 handler 实现（try/finally 恢复）。
"""
import hashlib
import threading
import time
from functools import wraps
from collections import OrderedDict
from typing import Callable, Dict, Optional, Tuple

import numpy as np

from hivision import IDCreator
from hivision.error import FaceError
from hivision.creator.choose_handler import choose_handler

_LOCK = threading.RLock()
INFERENCE_LOCK = threading.RLock()


def serialized_inference(fn):
    """IDCreator owns mutable handlers/context: never share it concurrently."""
    @wraps(fn)
    def wrapped(*args, **kwargs):
        with INFERENCE_LOCK:
            return fn(*args, **kwargs)
    return wrapped
_CREATOR: Optional[IDCreator] = None

# 抠图缓存：key -> {"matting", "face", "angle", "skipped_detection"}
_CACHE: "OrderedDict[str, dict]" = OrderedDict()
_CACHE_MAX = 8
_CACHE_BYTES = 256 * 1024 * 1024


def get_creator() -> IDCreator:
    global _CREATOR
    with _LOCK:
        if _CREATOR is None:
            _CREATOR = IDCreator()
    return _CREATOR


def _noop_detect(ctx) -> None:
    """人脸检测旁路：无人脸模式下降级使用"""
    ctx.face["rectangle"] = None
    ctx.face["roll_angle"] = 0.0


def _cache_key(img: np.ndarray, matting_model: str, face_model: str, beauty: Dict, align: bool, flip: bool) -> str:
    h = hashlib.sha1(np.ascontiguousarray(img).tobytes()).hexdigest()
    return "|".join(
        [
            h,
            str(img.shape),
            str(img.dtype),
            matting_model,
            face_model,
            f"w{beauty.get('whitening', 0)}",
            f"b{beauty.get('brightness', 0)}",
            f"c{beauty.get('contrast', 0)}",
            f"s{beauty.get('saturation', 0)}",
            f"sh{beauty.get('sharpen', 0)}",
            f"a{int(align)}",
            f"f{int(flip)}",
        ]
    )


def _cache_put(key: str, entry: dict) -> None:
    with _LOCK:
        _CACHE[key] = entry
        _CACHE.move_to_end(key)
        while len(_CACHE) > _CACHE_MAX or sum(e['matting'].nbytes for e in _CACHE.values()) > _CACHE_BYTES:
            _CACHE.popitem(last=False)


def _cache_get(key: str) -> Optional[dict]:
    with _LOCK:
        entry = _CACHE.get(key)
        if entry is not None:
            _CACHE.move_to_end(key)
        return entry


class MattingResult:
    """一次抠图产物（可能来自缓存）"""

    __slots__ = ("matting", "face", "angle", "cached", "reused_matting", "skipped_detection", "result", "elapsed")

    def __init__(
        self,
        matting: np.ndarray,
        face: Optional[Tuple[int, int, int, int]],
        angle: float,
        cached: bool,
        reused_matting: bool = False,
        skipped_detection: bool = False,
        result=None,
        elapsed: float = 0.0,
    ):
        self.matting = matting  # RGBA，RGB 通道序
        self.face = face  # (x, y, w, h) 或 None
        self.angle = angle  # 人脸 roll 角度
        self.cached = cached  # 整份请求签名一致（用户视角"什么都没改"）
        self.reused_matting = reused_matting  # 抠图张量复用（参数有变，已重算构图）
        self.skipped_detection = skipped_detection  # 是否为无人脸降级结果
        self.result = result  # 旧管线 Result（仅当完整跑过 creator 时存在）
        self.elapsed = elapsed


@serialized_inference
def run_matting(
    img: np.ndarray,
    matting_model: str = "ben2",
    face_detect_model: str = "retinaface-resnet50",
    whitening: int = 0,
    brightness: int = 0,
    contrast: int = 0,
    saturation: int = 0,
    sharpen: int = 0,
    face_align: bool = False,
    horizontal_flip: bool = False,
    use_cache: bool = True,
    sig: str = "",
    size: Tuple[int, int] = (413, 295),
    head_top_range: Tuple[float, float] = (0.12, 0.10),
) -> MattingResult:
    """
    执行 抠图（含美颜、可选人脸对齐），带缓存与无人脸降级。

    缓存语义：
    - 缓存键包含所有影响抠图的参数（图像 + 抠图/人脸模型 + 美颜 + 矫正/翻转），
      任一变化即失效重跑。
    - cached 标志仅在「整份请求签名 sig 与上次完全一致」时为 True
      （用户视角的"什么都没改"）；sig 变化但抠图可复用时，
      matting_reused=True、cached=False，如实报告"已按新参数重算"。

    :param img: RGB ndarray
    :param sig: 整份请求的参数签名（由路由层拼装）
    :return: MattingResult；face 为 None 表示人脸检测失败（已自动降级）
    """
    beauty = {
        "whitening": whitening,
        "brightness": brightness,
        "contrast": contrast,
        "saturation": saturation,
        "sharpen": sharpen,
    }
    # Mirroring is performed after matting; changing it must not rerun the model.
    key = _cache_key(img, matting_model, face_detect_model, beauty, face_align, False)

    if use_cache:
        hit = _cache_get(key)
        if hit is not None:
            unchanged = sig and hit.get("sig") == sig
            hit["sig"] = sig
            return MattingResult(
                matting=hit["matting"],
                face=hit["face"],
                angle=hit["angle"],
                cached=bool(unchanged),
                reused_matting=not unchanged,
                skipped_detection=hit["skipped_detection"],
            )

    creator = get_creator()
    choose_handler(creator, matting_model, face_detect_model)

    t0 = time.time()
    face = None
    angle = 0.0
    skipped_detection = False
    legacy_result = None

    try:
        # 完整跑一遍旧管线（检测 → 预裁剪 → 抠图 → 美颜 → 裁剪），
        # 其 ctx.matting_image / ctx.face 供智能引擎使用，result 作为旧路径兜底
        legacy_result = creator(
            img,
            size=size,
            head_top_range=head_top_range,
            whitening_strength=whitening,
            brightness_strength=brightness,
            contrast_strength=contrast,
            saturation_strength=saturation,
            sharpen_strength=sharpen,
            face_alignment=face_align,
        )
        face = creator.ctx.face.get("rectangle")
        angle = creator.ctx.face.get("roll_angle") or 0.0
    except FaceError:
        # 人脸检测失败 → 旁路检测重跑抠图（不抛错，交给智能构图兜底）
        skipped_detection = True
        saved: Callable = creator.detection_handler
        creator.detection_handler = _noop_detect
        try:
            creator(
                img,
                size=(413, 295),
                change_bg_only=True,
                whitening_strength=whitening,
                brightness_strength=brightness,
                contrast_strength=contrast,
                saturation_strength=saturation,
                sharpen_strength=sharpen,
            )
        finally:
            creator.detection_handler = saved
        face = None
        angle = 0.0

    elapsed = time.time() - t0
    matting = creator.ctx.matting_image
    entry = {
        "matting": matting,
        "face": face,
        "angle": angle,
        "skipped_detection": skipped_detection,
        "sig": sig,
    }
    # 无论是否绕过读缓存，新跑出的抠图都回写缓存：
    # 强制重跑的价值就在于"重跑后后续调用拿到新结果"
    _cache_put(key, entry)

    return MattingResult(
        matting=matting,
        face=face,
        angle=angle,
        cached=False,
        reused_matting=False,
        skipped_detection=skipped_detection,
        result=legacy_result,
        elapsed=elapsed,
    )
