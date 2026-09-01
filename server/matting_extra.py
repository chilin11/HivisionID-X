#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
扩展抠图模型（运行时注册，不改动 hivision 包文件）
==================================================

当前新增：
- BEN2 (Background Erase Network v2, PramaLLC, 2025)
  专攻人像/发丝抠图的新一代模型，质量对标 BiRefNet-portrait 而体积仅 1/4。
  输入：固定 1024×1024，ImageNet 归一化；输出：sigmoid 后的 mask。

注册方式：import 时向 hivision.creator.human_matting.MATTING_MODELS
追加条目（纯增量，复用其 BaseMattingModel 契约与 choose_handler 流程）。
"""
import os

import cv2
import numpy as np
from PIL import Image

from hivision.creator.human_matting import BaseMattingModel, MATTING_MODELS

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 项目根（server/ 的上一级）
_WEIGHTS_DIR = os.path.join(_ROOT, "hivision", "creator", "weights")

BEN2_URL = "https://huggingface.co/PramaLLC/BEN2/resolve/main/BEN2_Base.onnx"


class Ben2Model(BaseMattingModel):
    """BEN2_Base：固定 1024² 输入，ImageNet 归一化，sigmoid mask"""

    DOWNLOAD_URLS = BaseMattingModel.DOWNLOAD_URLS.copy()
    DOWNLOAD_URLS.update({"ben2": BEN2_URL})

    def __init__(self, name="ben2", checkpoint_path=os.path.join(_WEIGHTS_DIR, "BEN2_Base.onnx")):
        super().__init__(name, checkpoint_path, ref_size=1024)

    def process(self, input_image: np.ndarray) -> np.ndarray:
        self.load_model()

        orig_image = Image.fromarray(input_image)
        image = orig_image.resize((1024, 1024))
        arr = np.asarray(image, dtype=np.float32) / 255.0
        arr = (arr - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        x = np.transpose(arr, (2, 0, 1))[None].astype(np.float32)

        pred = self.session.run(None, {self.session.get_inputs()[0].name: x})[0]
        mask = np.squeeze(pred).astype(np.float32)
        # ONNX 导出已含 sigmoid（输出 ∈ [0,1]）；保险起见范围越界时再压一次
        if mask.min() < 0 or mask.max() > 1:
            mask = 1.0 / (1.0 + np.exp(-mask))
        mask = np.clip(mask, 0.0, 1.0)

        im_array = (mask * 255).astype(np.uint8)
        pil_im = Image.fromarray(im_array, mode="L")
        pil_im = pil_im.resize(orig_image.size, Image.BILINEAR)

        new_im = Image.new("RGBA", orig_image.size, (0, 0, 0, 0))
        new_im.paste(orig_image, mask=pil_im)

        self.release_model()
        return np.array(new_im)


def register_extra_matting_models() -> None:
    """向 hivision 模型注册表追加扩展模型（幂等）"""
    if "ben2" not in MATTING_MODELS:
        MATTING_MODELS["ben2"] = Ben2Model()


register_extra_matting_models()
