#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
智能裁剪引擎（Smart Crop Engine）
=================================

位于 API 层的纯几何模块：接收抠图结果（RGBA，RGB 通道序）与人脸框，
通过几何约束求解直接计算最优证件照裁剪框，彻底避免旧管线中
"裁剪框越界 → 透明补边 → 底部下移 → 头顶留白过大" 的连锁问题。

本模块不包含任何模型推理代码，仅做像素级几何运算。

核心思路
--------
1. 从 alpha 通道提取人像主体 bbox（发顶 crown、肩宽、贴底情况）。
2. 以人脸框为锚点联立三个约束求解裁剪框：
   - 面积约束：人脸面积 / 裁剪框面积 ≈ head_measure_ratio
   - 位置约束：人脸中心位于裁剪框高度 face_center_ratio 处
   - 留白约束：发顶距裁剪框顶部 ∈ [top_gap_min, top_gap_max]
3. 约束冲突时按优先级自动"缩小构图"迭代（宁可人脸略小，绝不产生补边）。
4. 裁剪框始终完全落在画布内（不补透明边），身体自然贴底。
5. 无人脸时按人体比例学先验估计头部区域，兜底构图，保证出图成功率。
"""
from dataclasses import dataclass, field
from typing import Optional, Tuple, List

import numpy as np
import cv2

from hivision.creator.utils import get_box

# 头顶留白默认目标（占裁剪框高度比例），对应标准证件照 4~6mm / 35mm 规范
DEFAULT_TOP_GAP_MAX = 0.12
DEFAULT_TOP_GAP_MIN = 0.10
# 人脸中心默认高度位置（人脸中心在裁剪框高度的 45% 处）
DEFAULT_FACE_CENTER_RATIO = 0.45
# 头部高度占比默认目标（发顶→下巴 / 画面高度）：
# 中国标准 ≈ 2/3；美国签证官方要求 50%-69%；取 0.66 与画幅无关，根治方形画幅"人头过大"
DEFAULT_HEAD_HEIGHT_FRACTION = 0.66
# 人脸面积占比默认目标（仅当头高不可测时的兜底锚点）
DEFAULT_HEAD_MEASURE_RATIO = 0.20

# alpha 二值化阈值：低于该值的像素视作背景（保留发丝羽化）
_ALPHA_THRESH = 16


@dataclass
class CropParams:
    """智能裁剪参数"""

    size: Tuple[int, int]  # 目标标准照尺寸 (H, W)
    face_center_ratio: float = DEFAULT_FACE_CENTER_RATIO
    head_height_fraction: float = DEFAULT_HEAD_HEIGHT_FRACTION  # 头高/画面高（视觉规范锚点）
    head_measure_ratio: float = DEFAULT_HEAD_MEASURE_RATIO  # 旧面积占比（仅 head_h 缺失时兜底）
    top_gap_max: float = DEFAULT_TOP_GAP_MAX
    top_gap_min: float = DEFAULT_TOP_GAP_MIN


@dataclass
class SubjectBox:
    """人像主体在画布中的位置信息（距画布四边的像素距离）"""

    crown: int  # 发顶距画布顶部
    bottom: int  # 身体末端距画布底部
    left: int  # 主体距画布左侧
    right: int  # 主体距画布右侧

    def width_on(self, W: int) -> int:
        """主体宽度（画布宽 W）"""
        return W - self.left - self.right

    def height_on(self, H: int) -> int:
        """主体高度（画布高 H）"""
        return H - self.crown - self.bottom


@dataclass
class CropQuality:
    """成图质量指标"""

    top_gap: float = 0.0  # 头顶留白 / 裁剪框高
    bottom_gap: float = 0.0  # 身体底部空隙 / 裁剪框高
    left_gap: float = 0.0
    right_gap: float = 0.0
    subject_width: float = 0.0  # 主体宽度 / 裁剪框宽
    face_ratio: Optional[float] = None  # 人脸面积 / 裁剪框面积
    h_center_offset: float = 0.0  # 水平居中偏差（左差-右差）/ 裁剪框宽
    crown_intact: bool = True  # 发顶是否完整（未被上游裁掉）
    score: int = 0
    engine: str = "smart"  # smart / fallback / legacy
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "top_gap": round(self.top_gap, 4),
            "bottom_gap": round(self.bottom_gap, 4),
            "left_gap": round(self.left_gap, 4),
            "right_gap": round(self.right_gap, 4),
            "subject_width": round(self.subject_width, 4),
            "face_ratio": None if self.face_ratio is None else round(self.face_ratio, 4),
            "h_center_offset": round(self.h_center_offset, 4),
            "crown_intact": self.crown_intact,
            "score": self.score,
            "engine": self.engine,
            "warnings": self.warnings,
        }


@dataclass
class CropResult:
    """智能裁剪结果"""

    standard: np.ndarray  # 精确标准尺寸 RGBA
    hd: np.ndarray  # 高清 RGBA（保留原生分辨率，限制最大边）
    quality: CropQuality


# hd 增强器签名：RGB ndarray → RGB ndarray（同尺寸清晰度提升）
HdEnhancer = Optional[callable]


# ---------------------------------------------------------------------------
# 基础几何工具
# ---------------------------------------------------------------------------


def subject_box(rgba: np.ndarray, thresh: int = _ALPHA_THRESH) -> SubjectBox:
    """提取人像主体距画布四边的距离"""
    y_top, y_bottom, x_left, x_right = get_box(rgba.astype(np.uint8), model=2, thresh=thresh)
    return SubjectBox(crown=int(y_top), bottom=int(y_bottom), left=int(x_left), right=int(x_right))


def has_subject(rgba: np.ndarray) -> bool:
    """判断抠图结果是否有效（主体面积占比 > 0.5%）"""
    H, W = rgba.shape[:2]
    box = subject_box(rgba)
    return box.width_on(W) > 2 and box.height_on(H) > 2 and (box.width_on(W) * box.height_on(H)) / (H * W) > 0.005


def matting_health(rgba: np.ndarray) -> List[str]:
    """
    抠图健康度检查：把"结果很不对劲"变成可诊断的警告码。
    - matting_small   主体占比 <3%：模型可能没抓准人像
    - matting_full    覆盖 >90%：背景可能被误保留
    - matting_multiblob 存在 ≥2% 画布面积的独立杂块：干扰物/多人被抠进来了
    """
    H, W = rgba.shape[:2]
    alpha = rgba[:, :, 3]
    mask = (alpha > 40).astype(np.uint8)
    cov = float(mask.mean())
    warnings: List[str] = []
    if cov < 0.03:
        warnings.append("matting_small")
    if cov > 0.92:
        warnings.append("matting_full")
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    blobs = 0
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area >= 0.02 * H * W:
            blobs += 1
    if blobs >= 2:
        warnings.append("matting_multiblob")
    return warnings


def flip_matting(matting: np.ndarray, face: Optional[Tuple[int, int, int, int]]):
    """水平翻转抠图结果并同步人脸坐标"""
    flipped = matting[:, ::-1].copy()
    if face is None:
        return flipped, None
    x, y, w, h = face
    return flipped, (matting.shape[1] - x - w, y, w, h)


# ---------------------------------------------------------------------------
# 裁剪框求解
# ---------------------------------------------------------------------------


def _solve_crop_rect(
    H0: int,
    W0: int,
    crown: int,
    subject: SubjectBox,
    face: Tuple[float, float, float, float],
    p: CropParams,
    warnings: List[str],
) -> Tuple[int, int, int, int]:
    """
    联立约束求解裁剪框 (x1, y1, cw, ch)，保证完全落在画布内（零补边）。
    face 为画布坐标系下的 (x, y, w, h)（浮点）。

    纵向锚定按优先级尝试三个候选：
      A. 身体贴底锚定（y2 = 主体底部）：画面下方留有背景时最自然
      B. 人脸中心锚定（人脸中心在 face_center_ratio 高度处）：标准胸像构图
      C. 留白窗口钳制：仅保证头顶留白 ∈ [min, max]
    全部冲突时迭代缩小构图（人脸占比略降，但绝不补边）。
    """
    Ht, Wt = p.size
    aspect = Wt / Ht
    fx, fy, fw, fh = face
    fcx, fcy = fx + fw / 2.0, fy + fh / 2.0
    face_area = max(fw * fh, 1.0)
    sub_bottom = H0 - subject.bottom  # 主体底部的画布 y 坐标
    subject_h = sub_bottom - crown

    # 1) 尺寸锚定：优先「头高占比」（发顶→下巴 / 画面高，视觉规范、与画幅无关）；
    #    头高不可测（face 框异常）时退回旧的面积占比法
    chin_y = fy + fh
    head_h = chin_y - crown
    if head_h > 24:
        ch = head_h / max(p.head_height_fraction, 0.3)
    else:
        ch = float(np.sqrt(face_area / (p.head_measure_ratio * aspect)))
    cw = ch * aspect
    # 画布尺寸上限（超出即补边，绝对禁止）
    ch = min(ch, float(H0), W0 / aspect)
    cw = ch * aspect

    lo_ratio, hi_ratio = p.top_gap_min * 0.7, p.top_gap_max * 1.1

    # 2a) 双锚定精确解：主体底端悬空（画面下方有背景）时，
    #     联立「头顶留白 = 目标值」与「身体贴底」直接解出裁剪框：
    #         y1 = crown - gap·ch   且   y1 + ch = 主体底部
    #     ⇒ ch = subject_h / (1 - gap)
    #     约束：底部悬空须显著（>5% 画布高，缝隙过小不值得拉近）；
    #     人脸占比畸变 ≤1.3×（防止"拉近导致人头过大"）；
    #     全身照不收腿（>1.348×ch 时走候选法裁到胸口）
    if subject.bottom > max(8, 0.05 * H0) and subject_h > 40:
        gap_target = float(np.clip((p.top_gap_min + p.top_gap_max) / 2, 0.06, 0.16))
        ch_d = subject_h / (1 - gap_target)
        if ch * 0.877 <= ch_d <= ch * 1.348:  # sqrt(1.3), sqrt(1.8)
            ch_fit = min(ch_d, float(H0), W0 / aspect)
            if ch_fit >= ch * 0.877:
                ch = ch_fit
                cw = ch * aspect
                y1 = float(np.clip(crown - gap_target * ch, 0.0, max(0.0, H0 - ch)))
                warnings.append("zoom_adjusted")
                x1 = float(np.clip(0.55 * (fcx - cw / 2.0) + 0.45 * ((subject.left + (W0 - subject.right)) / 2.0 - cw / 2.0), 0.0, max(0.0, W0 - cw)))
                return int(round(x1)), int(round(y1)), int(round(cw)), int(round(ch))

    # 2b) 迭代求解：纵向锚定候选 → 冲突分流处理
    #     - 画布装不下（底部越界）→ 缩小构图
    #     - 头顶可用留白不足（crown 贴画布顶）→ 不缩圈，走保占比回退
    y1 = None
    for _ in range(10):
        candidates = [
            sub_bottom - ch,  # A. 身体贴底
            fcy - p.face_center_ratio * ch,  # B. 人脸中心
        ]
        accepted = False
        for cand in candidates:
            y1c = float(np.clip(cand, 0.0, max(0.0, H0 - ch)))
            gap = (crown - y1c) / ch
            if lo_ratio <= gap <= hi_ratio:
                y1 = y1c
                accepted = True
                break
        if not accepted:
            # C. 留白窗口钳制
            c_lo = max(0.0, crown - p.top_gap_max * ch)
            c_hi = min(crown - p.top_gap_min * ch, H0 - ch)
            if c_hi >= c_lo:
                y1_face = fcy - p.face_center_ratio * ch
                y1 = float(np.clip(y1_face, c_lo, c_hi))
                accepted = True
        if accepted:
            break
        if crown - p.top_gap_min * ch < 0:
            break  # 头顶空间不足：缩小只会放大人脸占比，交给回退策略
        if ch <= 24:
            break
        ch *= 0.92
        cw = ch * aspect
        warnings.append("zoom_adjusted")

    if y1 is None:
        # 发顶距画布顶的可用空间不足以满足 min 留白（人充满画面/特写源图）：
        # 优先保人脸占比（保 ch），头顶贴 y1=0 顶格利用仅有的留白
        y1 = 0.0
        gap = crown / ch if ch > 0 else 0.0
        # 上报：请求的头顶位置在本图上无调节空间（裁剪框会切头或超出画布顶）
        y1_req = fcy - p.face_center_ratio * ch
        if y1_req > crown or y1_req < 0:
            warnings.append("head_pos_pinned")
        if crown <= 2:
            warnings.append("crown_clipped")
        elif gap < 0.04:
            # 留白实在太少：仅在占比畸变 ≤1.8× 时轻微缩小 ch 换最低呼吸感
            ch_fit = min(ch, crown / 0.07)
            if ch_fit >= ch / 1.342:
                ch = ch_fit
                cw = ch * aspect
            warnings.append("top_gap_small")
        else:
            warnings.append("top_gap_small")

    # 3) 水平：人脸中心为主、主体居中为辅
    sub_cx = (subject.left + (W0 - subject.right)) / 2.0
    x1_face = fcx - cw / 2.0
    x1_sub = sub_cx - cw / 2.0
    x1 = float(np.clip(0.55 * x1_face + 0.45 * x1_sub, 0.0, max(0.0, W0 - cw)))

    return int(round(x1)), int(round(y1)), int(round(cw)), int(round(ch))


def _estimate_head_without_face(
    H0: int, W0: int, subject: SubjectBox
) -> Optional[Tuple[float, float, float, float]]:
    """
    无人脸时按人体比例学先验估计头部框 (x, y, w, h)。
    胸像构图中头部约占可见身体高度的 42%±，面部高 ≈ 0.78 头高，面宽 ≈ 0.74 面高。
    """
    body_h = H0 - subject.crown - subject.bottom
    body_w = W0 - subject.left - subject.right
    if body_h < 40 or body_w < 30:
        return None
    head_h = float(np.clip(0.42 * body_h, 0.12 * H0, 0.9 * body_h))
    face_h = head_h * 0.78
    face_w = min(face_h * 0.74, body_w * 0.9)
    cx = (subject.left + (W0 - subject.right)) / 2.0
    cy = subject.crown + head_h * 0.60  # 发顶到人脸中心 ≈ 0.6 头高
    return (cx - face_w / 2.0, cy - face_h / 2.0, face_w, face_h)


# ---------------------------------------------------------------------------
# 质量评估与评分
# ---------------------------------------------------------------------------


def _evaluate(
    crop: np.ndarray,
    face_area: Optional[float],
    crown_intact: bool,
    engine: str,
    warnings: List[str],
    p: CropParams,
) -> CropQuality:
    ch_, cw_ = crop.shape[:2]
    box = subject_box(crop)
    q = CropQuality(engine=engine, warnings=warnings, crown_intact=crown_intact)
    q.top_gap = box.crown / ch_
    q.bottom_gap = box.bottom / ch_
    q.left_gap = box.left / cw_
    q.right_gap = box.right / cw_
    q.subject_width = (cw_ - box.left - box.right) / cw_
    q.h_center_offset = (box.left - box.right) / cw_
    if face_area is not None:
        q.face_ratio = face_area / (cw_ * ch_)

    # ---- 打分 ----
    score = 100
    if q.top_gap > p.top_gap_max * 1.6 or q.top_gap < p.top_gap_min * 0.55:
        score -= 30
        if q.top_gap > p.top_gap_max * 1.6:
            warnings.append("top_gap_large")
    elif q.top_gap > p.top_gap_max or q.top_gap < p.top_gap_min:
        score -= 10
    if q.bottom_gap > 0.05:
        score -= 25
        warnings.append("body_gap")
    if q.subject_width < 0.35:
        score -= 15
    if abs(q.h_center_offset) > 0.08:
        score -= 12
        warnings.append("center_off")
    if not crown_intact:
        score -= 8
    q.score = int(np.clip(score, 0, 100))
    return q


# ---------------------------------------------------------------------------
# 对外主入口
# ---------------------------------------------------------------------------


def _sharpen_rgb(rgb: np.ndarray, amount: float = 0.45, radius: float = 2.5) -> np.ndarray:
    """USM 锐化（RGB，类 DLSS 观感的轻量方案）"""
    x = rgb.astype(np.float32)
    blur = cv2.GaussianBlur(x, (0, 0), radius)
    return np.clip(cv2.addWeighted(x, 1.0 + amount, blur, -amount, 0), 0, 255).astype(np.uint8)


def _sharpen_rgba(img: np.ndarray, amount: float = 0.40, radius: float = 2.4) -> np.ndarray:
    """
    USM 锐化：只锐化 RGB 通道，alpha 羽化边缘保持平滑，避免边缘光晕。
    """
    return np.dstack([_sharpen_rgb(img[:,:,:3], amount, radius), img[:, :, 3]])


def _finalize(
    matting: np.ndarray,
    x1: int,
    y1: int,
    cw: int,
    ch: int,
    face_area: Optional[float],
    crown_intact: bool,
    engine: str,
    warnings: List[str],
    p: CropParams,
    enhancer: HdEnhancer = None,
) -> CropResult:
    H0, W0 = matting.shape[:2]
    x2, y2 = min(x1 + cw, W0), min(y1 + ch, H0)
    crop = matting[y1:y2, x1:x2]

    Ht, Wt = p.size
    standard = cv2.resize(crop, (Wt, Ht), interpolation=cv2.INTER_AREA)

    if enhancer is not None:
        # 外部增强器（Real-ESRGAN 超分，原生全分辨率）：重建纹理细节，
        # 再叠加轻度 USM 收尾提局部对比（0.22 幅度不会产生光晕）
        sr_rgb = enhancer(np.ascontiguousarray(crop[:,:,:3]))
        sr_rgb = _sharpen_rgb(sr_rgb, amount=0.22, radius=1.5)
        hd = np.dstack([sr_rgb, crop[:, :, 3]])
    else:
        hd = crop
        max_side = max(hd.shape[:2])
        if max_side < 1200:  # 原生分辨率过低时适度放大 + 锐化补偿
            scale = 1200 / max_side
            hd = cv2.resize(hd, (int(hd.shape[1] * scale), int(hd.shape[0] * scale)), interpolation=cv2.INTER_CUBIC)
        hd = _sharpen_rgba(hd)

    quality = _evaluate(crop, face_area, crown_intact, engine, warnings, p)
    return CropResult(standard=standard, hd=hd, quality=quality)


def smart_crop(
    matting: np.ndarray,
    face: Optional[Tuple[int, int, int, int]],
    size: Tuple[int, int],
    face_center_ratio: float = DEFAULT_FACE_CENTER_RATIO,
    head_height_fraction: float = DEFAULT_HEAD_HEIGHT_FRACTION,
    head_measure_ratio: float = DEFAULT_HEAD_MEASURE_RATIO,
    top_gap_max: float = DEFAULT_TOP_GAP_MAX,
    top_gap_min: float = DEFAULT_TOP_GAP_MIN,
    enhancer: HdEnhancer = None,
) -> CropResult:
    """
    智能裁剪主入口。

    :param matting: 抠图结果 RGBA（RGB 通道序），来自 creator.ctx.matting_image
    :param face: 人脸框 (x, y, w, h)，画布坐标系；无人脸时传 None（启用兜底构图）
    :param size: 目标标准尺寸 (H, W)
    :param head_height_fraction: 头部高度占比目标（发顶→下巴 / 画面高，默认 0.66）
    :param head_measure_ratio: 人脸面积占比目标（头高不可测时的兜底锚点）
    :param enhancer: 可选高清增强器（RGB→RGB 同尺寸），如 Real-ESRGAN 超分；
        未提供时使用内置 USM 锐化
    """
    if matting.ndim != 3 or matting.shape[2] != 4:
        raise ValueError("matting must be RGBA (4 channels)")
    H0, W0 = matting.shape[:2]
    if not has_subject(matting):
        raise ValueError("matting is empty: no subject found")

    p = CropParams(
        size=size,
        face_center_ratio=face_center_ratio,
        head_height_fraction=head_height_fraction,
        head_measure_ratio=head_measure_ratio,
        top_gap_max=top_gap_max,
        top_gap_min=min(top_gap_min, top_gap_max),
    )
    subject = subject_box(matting)
    warnings: List[str] = []

    if face is not None:
        fx, fy, fw, fh = [float(v) for v in face]
        engine = "smart"
        crown_intact = subject.crown > 2
        rect = _solve_crop_rect(H0, W0, subject.crown, subject, (fx, fy, fw, fh), p, warnings)
        face_area = fw * fh
    else:
        engine = "fallback"
        warnings.append("face_fallback")
        est = _estimate_head_without_face(H0, W0, subject)
        if est is None:
            # 极端情况：主体过小，直接居中裁剪
            ch = H0
            cw = min(W0, ch * p.size[1] / p.size[0])
            rect = (int((W0 - cw) / 2), 0, int(cw), ch)
            face_area = None
        else:
            crown_intact = subject.crown > 2
            rect = _solve_crop_rect(H0, W0, subject.crown, subject, est, p, warnings)
            face_area = est[2] * est[3]

    x1, y1, cw, ch = rect
    warnings = list(dict.fromkeys(warnings))  # 去重，保持顺序
    return _finalize(matting, x1, y1, cw, ch, face_area, crown_intact, engine, warnings, p, enhancer)


def evaluate_rgba(rgba: np.ndarray, size: Tuple[int, int]) -> CropQuality:
    """对任意 RGBA 证件照做质量评估（用于旧管线结果对比）"""
    p = CropParams(size=size)
    crop = rgba if rgba.shape[2] == 4 else np.dstack([rgba, np.full(rgba.shape[:2], 255, np.uint8)])
    if not has_subject(crop):
        q = CropQuality(engine="legacy")
        q.score = 0
        q.warnings.append("matting_empty")
        return q
    box = subject_box(crop)
    ch_, cw_ = crop.shape[:2]
    q = CropQuality(engine="legacy")
    q.top_gap = box.crown / ch_
    q.bottom_gap = box.bottom / ch_
    q.left_gap = box.left / cw_
    q.right_gap = box.right / cw_
    q.subject_width = (cw_ - box.left - box.right) / cw_
    q.h_center_offset = (box.left - box.right) / cw_
    q.crown_intact = box.crown > 2
    score = 100
    if q.top_gap > p.top_gap_max * 1.6 or q.top_gap < p.top_gap_min * 0.55:
        score -= 30
    elif q.top_gap > p.top_gap_max or q.top_gap < p.top_gap_min:
        score -= 10
    if q.bottom_gap > 0.05:
        score -= 25
    if q.subject_width < 0.35:
        score -= 15
    if abs(q.h_center_offset) > 0.08:
        score -= 12
    q.score = int(np.clip(score, 0, 100))
    return q
