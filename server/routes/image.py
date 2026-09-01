#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
图像处理路由
============

- POST /add_background         换底（旧逻辑原样保留：cv2 BGR 序解码链路）
- POST /generate_layout_photos 排版照
- POST /watermark              水印
- POST /set_kb                 调整文件大小

注：这批接口历史上一直工作正常，仅做模块化归位，不改动其通道处理逻辑。
"""
import base64
from typing import Optional

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, UploadFile

from hivision.creator.layout_calculator import generate_layout_array, generate_layout_image
from hivision.utils import (
    add_background,
    add_watermark,
    bytes_2_base64,
    hex_to_rgb,
    resize_image_to_kb,
    save_image_dpi_to_bytes,
)

from server import encoding

router = APIRouter()


def _decode_unchanged(input_image: Optional[UploadFile], input_image_base64: Optional[str]) -> np.ndarray:
    """按旧逻辑解码：保留原始通道（cv2 BGR 序）"""
    if input_image_base64:
        if input_image_base64.startswith("data:image"):
            input_image_base64 = input_image_base64.split(",", 1)[1]
        arr = np.frombuffer(base64.b64decode(input_image_base64), np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if input_image is not None:
        arr = np.frombuffer(input_image.file.read(), np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    return None


def _decode_color(input_image: Optional[UploadFile], input_image_base64: Optional[str]) -> np.ndarray:
    """按旧逻辑解码为 3 通道 BGR"""
    img = _decode_unchanged(input_image, input_image_base64)
    if img is None:
        raise ValueError("No input image provided")
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


@router.post("/add_background")
async def photo_add_background(
    input_image: UploadFile = File(None),
    input_image_base64: str = Form(None),
    color: str = Form("000000"),
    kb: int = Form(None),
    dpi: int = Form(300),
    render: int = Form(0),
):
    render_choice = ["pure_color", "updown_gradient", "center_gradient"]
    img = _decode_unchanged(input_image, input_image_base64)
    if img is None or img.ndim != 3:
        return {"status": False, "error": "Invalid image (RGBA transparent image required)"}

    rgb = hex_to_rgb(color)
    bgr = (rgb[2], rgb[1], rgb[0])

    result_image = add_background(img, bgr=bgr, mode=render_choice[render]).astype(np.uint8)
    result_image = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)

    if kb:
        result_image_bytes = resize_image_to_kb(result_image, None, int(kb), dpi=dpi)
    else:
        result_image_bytes = save_image_dpi_to_bytes(result_image, None, dpi=dpi)

    return {
        "status": True,
        "image_base64": bytes_2_base64(result_image_bytes),
    }


@router.post("/generate_layout_photos")
async def generate_layout_photos(
    input_image: UploadFile = File(None),
    input_image_base64: str = Form(None),
    height: int = Form(413),
    width: int = Form(295),
    kb: int = Form(None),
    dpi: int = Form(300),
    keep_original_size: bool = Form(False),
    layout_height: int = Form(1205),
    layout_width: int = Form(1795),
    crop_line: bool = Form(False),
):
    img = _decode_color(input_image, input_image_base64)
    size = (int(height), int(width))

    typography_arr, typography_rotate = generate_layout_array(
        input_height=size[0],
        input_width=size[1],
        LAYOUT_HEIGHT=layout_height,
        LAYOUT_WIDTH=layout_width,
    )

    result_layout_image = generate_layout_image(
        img,
        typography_arr,
        typography_rotate,
        height=size[0],
        width=size[1],
        crop_line=crop_line,
        LAYOUT_HEIGHT=layout_height,
        LAYOUT_WIDTH=layout_width,
        keep_original_size=keep_original_size,
    ).astype(np.uint8)

    result_layout_image = cv2.cvtColor(result_layout_image, cv2.COLOR_RGB2BGR)
    if kb:
        result_image_bytes = resize_image_to_kb(result_layout_image, None, int(kb), dpi=dpi)
    else:
        result_image_bytes = save_image_dpi_to_bytes(result_layout_image, None, dpi=dpi)

    return {
        "status": True,
        "image_base64": bytes_2_base64(result_image_bytes),
    }


@router.post("/watermark")
async def watermark(
    input_image: UploadFile = File(None),
    input_image_base64: str = Form(None),
    text: str = Form("Hello"),
    size: int = Form(20),
    opacity: float = Form(0.5),
    angle: int = Form(30),
    color: str = Form("#000000"),
    space: int = Form(25),
    kb: int = Form(None),
    dpi: int = Form(300),
):
    try:
        img = _decode_color(input_image, input_image_base64)
        result_image = add_watermark(img, text, size, opacity, angle, color, space)
        result_image = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
        if kb:
            result_image_bytes = resize_image_to_kb(result_image, None, int(kb), dpi=dpi)
        else:
            result_image_bytes = save_image_dpi_to_bytes(result_image, None, dpi=dpi)
        return {
            "status": True,
            "image_base64": bytes_2_base64(result_image_bytes),
        }
    except Exception as e:
        return {"status": False, "error": str(e)}


@router.post("/set_kb")
async def set_kb(
    input_image: UploadFile = File(None),
    input_image_base64: str = Form(None),
    dpi: int = Form(300),
    kb: int = Form(50),
):
    try:
        img = _decode_color(input_image, input_image_base64)
        result_image_bytes = resize_image_to_kb(img, None, int(kb), dpi=dpi)
        return {
            "status": True,
            "image_base64": bytes_2_base64(result_image_bytes),
        }
    except Exception as e:
        return {"status": False, "error": str(e)}
