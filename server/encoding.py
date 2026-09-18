#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
图像编解码与合成工具层
=======================

统一约定：**模块内部一律使用 RGB 通道序**。
- 输入解码：任意来源（UploadFile 字节 / base64 / numpy）→ RGB ndarray
- 输出编码：RGB(A) ndarray → 带 DPI 元数据的 PNG / JPEG 字节 → base64

与 hivision 包的边界：
- `add_background` 在 RGB 序输入 + RGB 序颜色元组下输出仍为 RGB 序，
  因此本层合成背景后无需再做任何 cvtColor。
- 旧接口 `/add_background`（cv2 解码、BGR 序）保持原有逻辑不动。
"""
import base64
import io
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageOps

from hivision.utils import add_background as _add_background, hex_to_rgb, resize_image_to_kb

# 背景渲染模式（与旧 API 的 render 序号对应）
RENDER_MODES = ["pure_color", "updown_gradient", "center_gradient"]

# 模型处理像素预算；更大的原图自动等比例缩小，不裁切画面。
MAX_INPUT_PIXELS = 24_000_000
MAX_INPUT_BYTES = 40 * 1024 * 1024


# ---------------------------------------------------------------------------
# 解码
# ---------------------------------------------------------------------------


def _pil_open_transposed(data: bytes) -> Image.Image:
    """
    PIL 解码并应用 EXIF 方向标签。
    关键：iPhone 等手机竖拍照片的 EXIF orientation=6/8，
    cv2.imdecode 会忽略该标签导致整条流水线拿到横置图像。
    """
    if len(data) > MAX_INPUT_BYTES:
        raise ValueError("图片文件不能超过 40 MB")
    with Image.open(io.BytesIO(data)) as source:
        pixels = source.width * source.height
        if pixels > MAX_INPUT_PIXELS:
            scale = (MAX_INPUT_PIXELS / pixels) ** 0.5
            size = (max(1, int(source.width * scale)), max(1, int(source.height * scale)))
            # 调色板图先转换，避免 Pillow 对 P 模式强制使用最近邻缩放。
            pil = source
            if source.mode in ("P", "PA", "1"):
                pil = source.convert("RGBA" if "transparency" in source.info or source.mode == "PA" else "RGB")
            pil = pil.resize(size, Image.Resampling.LANCZOS, reducing_gap=3.0)
            pil = ImageOps.exif_transpose(pil)
        else:
            pil = ImageOps.exif_transpose(source)
        pil.load()
    return pil


def decode_bytes_to_rgb(data: bytes) -> np.ndarray:
    """图片字节 → RGB ndarray（超大图等比例缩小 + EXIF 转正）"""
    pil = _pil_open_transposed(data)
    img = np.asarray(pil.convert("RGB"))
    return img


def decode_base64_to_rgb(b64_str: str) -> np.ndarray:
    if b64_str.startswith("data:image"):
        b64_str = b64_str.split(",", 1)[1]
    return decode_bytes_to_rgb(base64.b64decode(b64_str))


def decode_bytes_to_rgba(data: bytes) -> np.ndarray:
    """图片字节 → RGBA ndarray（保留透明通道，EXIF 转正）"""
    pil = _pil_open_transposed(data)
    if pil.mode in ("RGBA", "LA", "PA") or (pil.mode == "P" and "transparency" in pil.info):
        img = np.asarray(pil.convert("RGBA"))
    else:
        img = np.asarray(pil.convert("RGB"))
        img = np.dstack([img, np.full(img.shape[:2], 255, np.uint8)])
    return img


def _limit_side(img: np.ndarray, max_side: int = 3600) -> np.ndarray:
    h, w = img.shape[:2]
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img


# ---------------------------------------------------------------------------
# 编码
# ---------------------------------------------------------------------------


def encode_png(image: np.ndarray, dpi: int = 300) -> bytes:
    """RGB / RGBA ndarray → PNG 字节（带 DPI）"""
    pil = Image.fromarray(image)
    buf = io.BytesIO()
    pil.save(buf, format="PNG", dpi=(dpi, dpi))
    return buf.getvalue()


def encode_jpeg(image: np.ndarray, dpi: int = 300, quality: int = 95) -> bytes:
    """RGB ndarray → JPEG 字节（RGBA 会先铺白底）"""
    if image.ndim == 3 and image.shape[2] == 4:
        alpha = image[:, :, 3:4].astype(np.float32) / 255.0
        rgb = image[:, :, :3].astype(np.float32)
        image = (rgb * alpha + 255 * (1 - alpha)).astype(np.uint8)
    pil = Image.fromarray(image)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=quality, subsampling=0, dpi=(dpi, dpi))
    return buf.getvalue()


def encode_with_kb(image: np.ndarray, target_kb: int, dpi: int = 300) -> bytes:
    """RGB ndarray → 压缩到目标 KB 的 JPEG 字节（复用 hivision.utils）"""
    return resize_image_to_kb(image, None, int(target_kb), dpi=dpi)


def to_base64(data: bytes, mime: str = "image/png") -> str:
    return f"data:{mime};base64," + base64.b64encode(data).decode("utf-8")


# ---------------------------------------------------------------------------
# 合成
# ---------------------------------------------------------------------------


def compose_background(rgba: np.ndarray, hex_color: str, render: int = 0) -> np.ndarray:
    """
    为透明证件照合成背景（RGB 通道序，输入输出一致）。

    :param rgba: RGBA ndarray（RGB 序）
    :param hex_color: 十六进制颜色，如 "438edb"
    :param render: 0 纯色 / 1 上下渐变 / 2 中心渐变
    """
    rgb = hex_to_rgb(hex_color.strip().lstrip("#"))
    mode = RENDER_MODES[render] if 0 <= render < len(RENDER_MODES) else RENDER_MODES[0]
    # add_background 按输入通道序逐通道填充，RGB 序输入配 RGB 序颜色即可
    out = _add_background(rgba, bgr=rgb, mode=mode).astype(np.uint8)
    return out


def parse_color(color: Optional[str], default: str = "ffffff") -> str:
    """规整用户输入的颜色为 6 位 hex"""
    if not color:
        return default
    c = color.strip().lstrip("#")
    if len(c) == 3:
        c = "".join(x * 2 for x in c)
    if len(c) != 6:
        try:
            r, g, b = hex_to_rgb(c)
            c = f"{r:02x}{g:02x}{b:02x}"
        except Exception:
            return default
    try:
        int(c, 16)
    except ValueError:
        return default
    return c.lower()
