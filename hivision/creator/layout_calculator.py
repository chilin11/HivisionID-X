#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
@DATE: 2024/9/5 21:35
@File: layout_calculator.py
@IDE: pycharm
@Description:
    布局计算器
"""

import cv2.detail
import numpy as np


def judge_layout(
    input_width,
    input_height,
    PHOTO_INTERVAL_W,
    PHOTO_INTERVAL_H,
    LIMIT_BLOCK_W,
    LIMIT_BLOCK_H,
):
    centerBlockHeight_1, centerBlockWidth_1 = (
        input_height,
        input_width,
    )  # 由证件照们组成的一个中心区块（1 代表不转置排列）
    centerBlockHeight_2, centerBlockWidth_2 = (
        input_width,
        input_height,
    )  # 由证件照们组成的一个中心区块（2 代表转置排列）

    # 1.不转置排列的情况下：
    layout_col_no_transpose = 0  # 行
    layout_row_no_transpose = 0  # 列
    for i in range(1, 4):
        centerBlockHeight_temp = input_height * i + PHOTO_INTERVAL_H * (i - 1)
        if centerBlockHeight_temp < LIMIT_BLOCK_H:
            centerBlockHeight_1 = centerBlockHeight_temp
            layout_row_no_transpose = i
        else:
            break
    for j in range(1, 9):
        centerBlockWidth_temp = input_width * j + PHOTO_INTERVAL_W * (j - 1)
        if centerBlockWidth_temp < LIMIT_BLOCK_W:
            centerBlockWidth_1 = centerBlockWidth_temp
            layout_col_no_transpose = j
        else:
            break
    layout_number_no_transpose = layout_row_no_transpose * layout_col_no_transpose

    # 2.转置排列的情况下：
    layout_col_transpose = 0  # 行
    layout_row_transpose = 0  # 列
    for i in range(1, 4):
        centerBlockHeight_temp = input_width * i + PHOTO_INTERVAL_H * (i - 1)
        if centerBlockHeight_temp < LIMIT_BLOCK_H:
            centerBlockHeight_2 = centerBlockHeight_temp
            layout_row_transpose = i
        else:
            break
    for j in range(1, 9):
        centerBlockWidth_temp = input_height * j + PHOTO_INTERVAL_W * (j - 1)
        if centerBlockWidth_temp < LIMIT_BLOCK_W:
            centerBlockWidth_2 = centerBlockWidth_temp
            layout_col_transpose = j
        else:
            break
    layout_number_transpose = layout_row_transpose * layout_col_transpose

    if layout_number_transpose > layout_number_no_transpose:
        layout_mode = (layout_col_transpose, layout_row_transpose, 2)
        return layout_mode, centerBlockWidth_2, centerBlockHeight_2
    else:
        layout_mode = (layout_col_no_transpose, layout_row_no_transpose, 1)
        return layout_mode, centerBlockWidth_1, centerBlockHeight_1


def generate_layout_array(input_height, input_width, LAYOUT_WIDTH=1795, LAYOUT_HEIGHT=1205):
    # 1.基础参数表
    PHOTO_INTERVAL_H = 30  # 证件照与证件照之间的垂直距离
    PHOTO_INTERVAL_W = 30  # 证件照与证件照之间的水平距离
    SIDES_INTERVAL_H = 50  # 证件照与画布边缘的垂直距离
    SIDES_INTERVAL_W = 70  # 证件照与画布边缘的水平距离
    LIMIT_BLOCK_W = LAYOUT_WIDTH - 2 * SIDES_INTERVAL_W
    LIMIT_BLOCK_H = LAYOUT_HEIGHT - 2 * SIDES_INTERVAL_H

    # 3.计算照片的 layout（列、行、横竖朝向）,证件照组成的中心区块的分辨率
    layout_mode, centerBlockWidth, centerBlockHeight = judge_layout(
        input_width,
        input_height,
        PHOTO_INTERVAL_W,
        PHOTO_INTERVAL_H,
        LIMIT_BLOCK_W,
        LIMIT_BLOCK_H,
    )
    # 4.开始排列组合
    x11 = (LAYOUT_WIDTH - centerBlockWidth) // 2
    y11 = (LAYOUT_HEIGHT - centerBlockHeight) // 2
    typography_arr = []
    typography_rotate = False
    if layout_mode[2] == 2:
        input_height, input_width = input_width, input_height
        typography_rotate = True

    for j in range(layout_mode[1]):
        for i in range(layout_mode[0]):
            xi = x11 + i * input_width + i * PHOTO_INTERVAL_W
            yi = y11 + j * input_height + j * PHOTO_INTERVAL_H
            typography_arr.append([xi, yi])

    return typography_arr, typography_rotate


def generate_layout_image(
    input_image, typography_arr, typography_rotate, width=295, height=413,
    crop_line: bool = False,
    LAYOUT_WIDTH=1795,
    LAYOUT_HEIGHT=1205,
    keep_original_size: bool = False,
    output_scale=None,
):
    """Render directly from the source at the requested print resolution.

    All geometry uses the same scale, including rotated photos, gutters and
    crop lines. Never enlarge an already-rendered low-resolution sheet.
    """
    if min(width, height, LAYOUT_WIDTH, LAYOUT_HEIGHT) <= 0:
        raise ValueError("Photo and layout dimensions must be positive")
    input_h, input_w = input_image.shape[:2]
    if output_scale is None:
        output_scale = min(input_w / width, input_h / height) if keep_original_size else 1.0
    if not np.isfinite(output_scale) or output_scale <= 0:
        raise ValueError("Layout scale must be positive and finite")

    layout_w = max(1, round(LAYOUT_WIDTH * output_scale))
    layout_h = max(1, round(LAYOUT_HEIGHT * output_scale))
    photo_w = max(1, round(width * output_scale))
    photo_h = max(1, round(height * output_scale))
    if (input_w, input_h) != (photo_w, photo_h):
        interpolation = cv2.INTER_AREA if photo_w <= input_w and photo_h <= input_h else cv2.INTER_LANCZOS4
        input_image = cv2.resize(input_image, (photo_w, photo_h), interpolation=interpolation)
    if typography_rotate:
        input_image = cv2.rotate(input_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        photo_w, photo_h = photo_h, photo_w

    scaled_arr = [(round(x * output_scale), round(y * output_scale)) for x, y in typography_arr]
    white_background = np.full((layout_h, layout_w, 3), 255, dtype=np.uint8)
    for x, y in scaled_arr:
        if x < 0 or y < 0 or x + photo_w > layout_w or y + photo_h > layout_h:
            raise ValueError("Photo does not fit on the selected paper")
        white_background[y:y + photo_h, x:x + photo_w] = input_image

    if crop_line:
        line_color = (200, 200, 200)
        line_thickness = max(1, round(output_scale))
        vertical_lines = {edge for x, _ in scaled_arr for edge in (x, x + photo_w)}
        horizontal_lines = {edge for _, y in scaled_arr for edge in (y, y + photo_h)}
        for x in vertical_lines:
            cv2.line(white_background, (x, 0), (x, layout_h - 1), line_color, line_thickness)
        for y in horizontal_lines:
            cv2.line(white_background, (0, y), (layout_w - 1, y), line_color, line_thickness)

    return white_background
