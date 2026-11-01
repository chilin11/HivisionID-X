#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
图像处理路由
============

- POST /add_background         换底（旧逻辑原样保留：cv2 BGR 序解码链路）
- POST /generate_layout_photos 排版照
- POST /watermark              水印
- POST /set_kb                 调整文件大小

注：旧接口保留原有通道处理；排版支持按输出 DPI 同步缩放像素与打印尺寸。
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
    output_dpi: Optional[int] = Form(None),
):
    # Input geometry is measured at `dpi`. An explicit output DPI changes
    # pixels, not paper/photo sizes; omitted output_dpi keeps legacy defaults.
    if min(height, width, layout_height, layout_width, dpi) <= 0:
        return {"status": False, "error": "Dimensions and DPI must be positive"}
    if output_dpi is not None and not 72 <= output_dpi <= 1200:
        return {"status": False, "error": "Output DPI must be within 72-1200"}
    if kb is not None and kb <= 0:
        return {"status": False, "error": "Target KB must be positive"}
    try:
        img = _decode_color(input_image, input_image_base64)
    except Exception as e:
        return {"status": False, "error": f"Invalid image: {e}"}

    if output_dpi is not None:
        scale = output_dpi / dpi
    elif keep_original_size:
        scale = min(img.shape[1] / width, img.shape[0] / height)
    else:
        scale = 1.0
    export_dpi = output_dpi if output_dpi is not None else dpi * scale
    output_width = max(1, round(layout_width * scale))
    output_height = max(1, round(layout_height * scale))
    # Includes A4 at 600 DPI (~35 MP), but rejects accidental giant canvases
    # before allocating or encoding them.
    if output_width * output_height > 40_000_000:
        return {"status": False, "error": "Layout exceeds 40 megapixels; reduce paper size or output DPI"}

    typography_arr, typography_rotate = generate_layout_array(
        input_height=height,
        input_width=width,
        LAYOUT_HEIGHT=layout_height,
        LAYOUT_WIDTH=layout_width,
    )
    if not typography_arr:
        return {"status": False, "error": "Photo does not fit on the selected paper"}

    result_layout_image = generate_layout_image(
        img,
        typography_arr,
        typography_rotate,
        height=height,
        width=width,
        crop_line=crop_line,
        LAYOUT_HEIGHT=layout_height,
        LAYOUT_WIDTH=layout_width,
        output_scale=scale,
    )

    result_layout_image = cv2.cvtColor(result_layout_image, cv2.COLOR_BGR2RGB)
    if kb:
        result_image_bytes = resize_image_to_kb(result_layout_image, None, int(kb), dpi=export_dpi)
    else:
        result_image_bytes = encoding.encode_png(result_layout_image, dpi=export_dpi)

    return {
        "status": True,
        "image_base64": bytes_2_base64(result_image_bytes),
        "size": {"height": output_height, "width": output_width},
        "dpi": export_dpi,
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
