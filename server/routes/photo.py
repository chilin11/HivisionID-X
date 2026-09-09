#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
证件照核心路由
==============

- POST /idphoto        智能证件照（抠图 → 智能构图 → 换底 → 输出），一次请求完成
- POST /idphoto_crop   仅智能裁剪（透明底输出）
- POST /human_matting  仅抠图（保持旧逻辑不变）
"""
import time
from typing import Optional

import numpy as np
from fastapi import APIRouter, File, Form, UploadFile

from hivision.error import FaceError
from hivision.utils import bytes_2_base64, save_image_dpi_to_bytes
from hivision.creator.choose_handler import choose_handler

from server import creator_service, encoding
from server import super_res as sr_engine
from server.smart_crop import evaluate_rgba, flip_matting, matting_health, smart_crop

router = APIRouter()


def _load_input(input_image: Optional[UploadFile], input_image_base64: Optional[str]) -> np.ndarray:
    """请求图像 → RGB ndarray"""
    if input_image_base64:
        return encoding.decode_base64_to_rgb(input_image_base64)
    if input_image is not None:
        return encoding.decode_bytes_to_rgb(input_image.file.read())
    raise ValueError("No input image provided")


def _validate_size(height: int, width: int) -> Optional[str]:
    if not (100 <= height <= 1800 and 100 <= width <= 1800):
        return "Size must be within 100-1800 px (both height and width)"
    return None


def _request_sig(**kw) -> str:
    """整份请求参数签名：任何影响输出的参数变化都会改变签名"""
    return "|".join(f"{k}={v}" for k, v in sorted(kw.items()))


@router.post("/idphoto")
@creator_service.serialized_inference
def idphoto_inference(
    input_image: UploadFile = File(None),
    input_image_base64: str = Form(None),
    height: int = Form(413),
    width: int = Form(295),
    # 模型
    human_matting_model: str = Form("ben2"),
    face_detect_model: str = Form("retinaface-resnet50"),
    # 智能构图
    smart: bool = Form(True),
    head_center_ratio: float = Form(0.45),  # 人脸中心目标高度位置
    head_height_fraction: float = Form(0.66),  # 头部高度占比目标（发顶→下巴 / 画面高）
    head_measure_ratio: float = Form(0.20),  # 面积占比兜底锚点（头高不可测时）
    top_distance_max: float = Form(0.12),  # 头顶留白上限
    top_distance_min: float = Form(0.10),  # 头顶留白下限
    # 美颜
    whitening_strength: int = Form(0),
    brightness_strength: int = Form(0),
    contrast_strength: int = Form(0),
    saturation_strength: int = Form(0),
    sharpen_strength: int = Form(0),
    face_align: bool = Form(False),
    horizontal_flip: bool = Form(False),
    # 输出
    color: str = Form(None),  # 提供则直接合成底色（hex）
    render: int = Form(0),  # 0 纯色 / 1 上下渐变 / 2 中心渐变
    kb: int = Form(None),  # 目标文件大小（KB）
    jpeg: bool = Form(False),
    super_res: bool = Form(False),  # AI 超分增强（同尺寸提清晰度）
    sr_model: str = Form("general"),  # general（快）/ x4plus（高质量，慢）
    force: bool = Form(False),  # 强制重跑模型（绕过抠图缓存）
    dpi: int = Form(300),
):
    t0 = time.time()
    size_err = _validate_size(height, width)
    if size_err:
        return {"status": False, "error": size_err}

    try:
        img = _load_input(input_image, input_image_base64)
    except Exception as e:
        return {"status": False, "error": f"Invalid image: {e}"}

    # 1. 抠图（含缓存与无人脸降级）
    sig = _request_sig(
        h=height, w=width, mm=human_matting_model, fm=face_detect_model,
        smart=smart, hcr=head_center_ratio, hhf=head_height_fraction, hmr=head_measure_ratio,
        tdm=top_distance_max, tdn=top_distance_min,
        wh=whitening_strength, br=brightness_strength, ct=contrast_strength,
        st=saturation_strength, sh=sharpen_strength,
        al=face_align, fl=horizontal_flip,
        c=color, rd=render, kb=kb, jp=jpeg, sr=super_res, srm=sr_model, dpi=dpi,
    )
    try:
        mr = creator_service.run_matting(
            img,
            matting_model=human_matting_model,
            face_detect_model=face_detect_model,
            whitening=whitening_strength,
            brightness=brightness_strength,
            contrast=contrast_strength,
            saturation=saturation_strength,
            sharpen=sharpen_strength,
            face_align=face_align,
            horizontal_flip=horizontal_flip,
            sig=sig,
            use_cache=not force and smart,
            size=(height, width),
            head_top_range=(top_distance_max, top_distance_min),
        )
    except Exception as e:
        return {"status": False, "error": str(e)}

    warnings = matting_health(mr.matting)

    # 2. 智能构图
    standard_rgba = hd_rgba = None
    quality = None
    smart_ok = False
    sr_used = False
    enhancer = None
    if smart and super_res:
        try:
            if sr_engine.available():
                def enhancer(rgb, _m=sr_model):
                    # 裁剪 ≤2000px：原生全分辨率超分（零细节损失）
                    # 更大画布：降采样到 1000px 快速档（速度折中）
                    return sr_engine.enhance(rgb, max_input_side=0, model=_m)
                sr_used = True
        except FileNotFoundError:
            warnings.append("sr_missing")
    if smart:
        try:
            matting, face = flip_matting(mr.matting, mr.face) if horizontal_flip else (mr.matting, mr.face)
            cr = smart_crop(
                matting,
                face,
                (height, width),
                face_center_ratio=head_center_ratio,
                head_height_fraction=head_height_fraction,
                head_measure_ratio=head_measure_ratio,
                top_gap_max=top_distance_max,
                top_gap_min=top_distance_min,
                enhancer=enhancer,
            )
            standard_rgba, hd_rgba = cr.standard, cr.hd
            quality = cr.quality.to_dict()
            smart_ok = True
        except Exception:
            sr_used = False
            warnings.append("engine_fallback")

    # 3. 兜底：智能引擎异常时退回旧管线结果
    if not smart_ok:
        if mr.result is not None:
            standard_rgba, hd_rgba = mr.result.standard, mr.result.hd
        else:
            # 缓存命中后引擎异常的极端情况：重跑一次旧管线
            try:
                choose_handler(creator_service.get_creator(), human_matting_model, face_detect_model)
                result = creator_service.get_creator()(
                    img,
                    size=(height, width),
                    head_top_range=(top_distance_max, top_distance_min),
                    whitening_strength=whitening_strength,
                    brightness_strength=brightness_strength,
                    contrast_strength=contrast_strength,
                    saturation_strength=saturation_strength,
                    sharpen_strength=sharpen_strength,
                    face_alignment=face_align,
                )
            except FaceError:
                return {
                    "status": False,
                    "error": "Face not detected and no usable subject found — please use a clear front-facing portrait photo",
                }
            standard_rgba, hd_rgba = result.standard, result.hd
        if horizontal_flip:
            standard_rgba = np.ascontiguousarray(standard_rgba[:, ::-1])
            hd_rgba = np.ascontiguousarray(hd_rgba[:, ::-1])
        quality = evaluate_rgba(hd_rgba, (height, width)).to_dict()

    face_detected = mr.face is not None
    if mr.skipped_detection:
        warnings.append("face_fallback")
    warnings.extend((quality or {}).get("warnings") or [])

    # 4. 合成背景（kb / jpeg 必须有底色，自动铺白）
    flat_rgb = None
    if color or jpeg or kb:
        bg_color = encoding.parse_color(color, "ffffff")
        flat_rgb = encoding.compose_background(standard_rgba, bg_color, render)
        hd_rgb = encoding.compose_background(hd_rgba, bg_color, render)
    else:
        hd_rgb = None

    # 5. 编码输出
    #    standard / hd 输出所选背景的成图；matting 始终为透明底
    if kb:
        std_bytes = encoding.encode_with_kb(flat_rgb, kb, dpi)
        std_mime = "image/jpeg"
        std_ext = "jpg"
    elif jpeg:
        std_bytes = encoding.encode_jpeg(flat_rgb, dpi)
        std_mime = "image/jpeg"
        std_ext = "jpg"
    elif flat_rgb is not None:
        std_bytes = encoding.encode_png(flat_rgb, dpi)
        std_mime = "image/png"
        std_ext = "png"
    else:
        # 调用方未指定颜色：保持透明底（兼容旧 API 用法）
        std_bytes = encoding.encode_png(standard_rgba, dpi)
        std_mime = "image/png"
        std_ext = "png"

    if jpeg:
        hd_bytes = encoding.encode_jpeg(hd_rgb, dpi)
    elif hd_rgb is not None:
        hd_bytes = encoding.encode_png(hd_rgb, dpi)
    else:
        hd_bytes = encoding.encode_png(hd_rgba, dpi)
    matting_bytes = encoding.encode_png(hd_rgba, dpi)  # 透明底也保留高清像素

    return {
        "status": True,
        "image_base64_standard": encoding.to_base64(std_bytes, std_mime),
        "image_base64_hd": encoding.to_base64(hd_bytes, "image/jpeg" if jpeg else "image/png"),
        "image_base64_matting": encoding.to_base64(matting_bytes, "image/png"),
        "standard_format": std_ext,
        "hd_format": "jpg" if jpeg else "png",
        "hd_size": {"height": hd_rgba.shape[0], "width": hd_rgba.shape[1]},
        "dpi": dpi,
        "face_detected": face_detected,
        "super_res": sr_used,
        "quality": quality,
        "warnings": sorted(set(warnings)),
        "cached": mr.cached,
        "matting_reused": mr.reused_matting,
        "elapsed_ms": int((time.time() - t0) * 1000),
        "size": {"height": height, "width": width},
    }


@router.post("/idphoto_crop")
def idphoto_crop_inference(
    input_image: UploadFile = File(None),
    input_image_base64: str = Form(None),
    height: int = Form(413),
    width: int = Form(295),
    face_detect_model: str = Form("retinaface-resnet50"),
    hd: bool = Form(True),
    dpi: int = Form(300),
    head_center_ratio: float = Form(0.45),
    head_height_fraction: float = Form(0.66),
    head_measure_ratio: float = Form(0.2),
    top_distance_max: float = Form(0.12),
    top_distance_min: float = Form(0.10),
):
    """仅智能裁剪：输出透明底证件照（复用抠图缓存与兜底构图）"""
    size_err = _validate_size(height, width)
    if size_err:
        return {"status": False, "error": size_err}

    try:
        img = _load_input(input_image, input_image_base64)
    except Exception as e:
        return {"status": False, "error": f"Invalid image: {e}"}

    try:
        mr = creator_service.run_matting(
            img,
            face_detect_model=face_detect_model,
            sig=_request_sig(
                h=height, w=width, fm=face_detect_model,
                hcr=head_center_ratio, hhf=head_height_fraction, hmr=head_measure_ratio,
                tdm=top_distance_max, tdn=top_distance_min,
            ),
        )
    except Exception as e:
        return {"status": False, "error": str(e)}

    try:
        cr = smart_crop(
            mr.matting,
            mr.face,
            (height, width),
            face_center_ratio=head_center_ratio,
            head_height_fraction=head_height_fraction,
            head_measure_ratio=head_measure_ratio,
            top_gap_max=top_distance_max,
            top_gap_min=min(top_distance_min, top_distance_max),
        )
    except Exception as e:
        return {"status": False, "error": f"Smart crop failed: {e}"}

    payload = {
        "status": True,
        "image_base64_standard": bytes_2_base64(save_image_dpi_to_bytes(cr.standard, None, dpi)),
        "face_detected": mr.face is not None,
        "quality": cr.quality.to_dict(),
    }
    if hd:
        payload["image_base64_hd"] = bytes_2_base64(save_image_dpi_to_bytes(cr.hd, None, dpi))
    return payload


@router.post("/human_matting")
@creator_service.serialized_inference
def human_matting_inference(
    input_image: UploadFile = File(None),
    input_image_base64: str = Form(None),
    human_matting_model: str = Form("ben2"),
    dpi: int = Form(300),
):
    """仅抠图（旧逻辑保持不变）：输入 BGR 序解码 → creator → RGBA 输出"""
    if input_image_base64:
        img = encoding.decode_base64_to_rgb(input_image_base64)
    elif input_image is not None:
        img = encoding.decode_bytes_to_rgb(input_image.file.read())
    else:
        return {"status": False, "error": "No input image provided"}

    choose_handler(creator_service.get_creator(), human_matting_model, None)
    try:
        result = creator_service.get_creator()(img, change_bg_only=True)
    except FaceError:
        return {"status": False, "error": "Face not detected"}

    # 输入已是 RGB 序，直接编码（无需通道交换）
    result_image_bytes = save_image_dpi_to_bytes(result.standard, None, dpi)
    return {
        "status": True,
        "image_base64": bytes_2_base64(result_image_bytes),
    }
