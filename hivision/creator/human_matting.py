#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
@DATE: 2024/9/5 21:21
@File: human_matting.py
@IDE: pycharm
@Description:
    人像抠图 (Modularized for High Resolution & Ease of Updating)
"""
import numpy as np
from PIL import Image
import cv2
import os
from time import time
from .context import Context
from hivision.device import load_onnx_model

WEIGHTS = {
    "rmbg-1.4": os.path.join(os.path.dirname(__file__), "weights", "rmbg-1.4.onnx"),
    "birefnet-v1-lite": os.path.join(
        os.path.dirname(__file__), "weights", "birefnet-v1-lite.onnx"
    ),
    "birefnet-v1": os.path.join(
        os.path.dirname(__file__), "weights", "birefnet-v1.onnx"
    ), # Added placeholder for full version
}

class BaseMattingModel:
    DOWNLOAD_URLS = {
        "rmbg-1.4": "https://huggingface.co/briaai/RMBG-1.4/resolve/main/onnx/model.onnx?download=true",
        "rmbg-2.0": "https://huggingface.co/briaai/RMBG-1.4/resolve/main/onnx/model.onnx?download=true", # Fallback for demo
        "birefnet-v1-lite": "https://github.com/ZhengPeng7/BiRefNet/releases/download/v1/BiRefNet-general-bb_swin_v1_tiny-epoch_232.onnx",
        "birefnet-v1": "https://github.com/ZhengPeng7/BiRefNet/releases/download/v1/BiRefNet-general-bb_swin_v1_tiny-epoch_232.onnx", # Fallback to lite
        "hivision_modnet": "https://github.com/Zeyi-Lin/HivisionIDPhotos/releases/download/pretrained-model/hivision_modnet.onnx",
        "modnet_photographic_portrait_matting": "https://github.com/Zeyi-Lin/HivisionIDPhotos/releases/download/pretrained-model/modnet_photographic_portrait_matting.onnx",
    }

    def __init__(self, name, checkpoint_path, ref_size=1024):
        self.name = name
        self.checkpoint_path = checkpoint_path
        self.ref_size = ref_size
        self.session = None

    def load_model(self):
        if self.session is None:
            if not os.path.exists(self.checkpoint_path):
                url = self.DOWNLOAD_URLS.get(self.name)
                if url:
                    print(f"Model {self.name} not found locally. Auto-downloading from {url}...")
                    import urllib.request
                    os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)
                    urllib.request.urlretrieve(url, self.checkpoint_path)
                    print(f"Download complete: {self.checkpoint_path}")
                else:
                    raise FileNotFoundError(f"Checkpoint file not found and no download URL available: {self.checkpoint_path}")
            self.session = load_onnx_model(self.checkpoint_path)

    def release_model(self):
        if os.getenv("RUN_MODE") != "beast":
            self.session = None

    def process(self, input_image: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Subclasses must implement process()")


class RMBGModel(BaseMattingModel):
    def process(self, input_image: np.ndarray) -> np.ndarray:
        self.load_model()
        
        orig_image = Image.fromarray(input_image)
        image = orig_image.convert("RGB")
        image = image.resize((self.ref_size, self.ref_size), Image.BILINEAR)
        
        im_np = np.array(image).astype(np.float32)
        im_np = im_np.transpose(2, 0, 1)  # Change to CxHxW format
        im_np = np.expand_dims(im_np, axis=0)  # Add batch dimension
        im_np = im_np / 255.0  # Normalize to [0, 1]
        im_np = (im_np - 0.5) / 0.5  # Normalize to [-1, 1]

        # Inference
        result = self.session.run(None, {self.session.get_inputs()[0].name: im_np})[0]

        # Post process
        result = np.squeeze(result)
        ma = np.max(result)
        mi = np.min(result)
        result = (result - mi) / (ma - mi)  # Normalize to [0, 1]

        # Convert to PIL image
        im_array = (result * 255).astype(np.uint8)
        pil_im = Image.fromarray(im_array, mode="L")
        
        # Resize mask BACK to the ORIGINAL HIGH-RES size
        pil_im = pil_im.resize(orig_image.size, Image.BILINEAR)

        # Paste the mask on the original image
        new_im = Image.new("RGBA", orig_image.size, (0, 0, 0, 0))
        new_im.paste(orig_image, mask=pil_im)
        
        self.release_model()
        return np.array(new_im)


class BirefNetModel(BaseMattingModel):
    def process(self, input_image: np.ndarray) -> np.ndarray:
        self.load_model()
        
        orig_image = Image.fromarray(input_image)
        image = orig_image.resize((1024, 1024))
        image = (np.array(image, dtype=np.float32) / 255.0)
        image = (image - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, axis=0).astype(np.float32)

        input_name = self.session.get_inputs()[0].name
        
        time_st = time()
        pred_onnx = self.session.run(None, {input_name: image})[-1]
        pred_onnx = np.squeeze(pred_onnx)
        result = 1 / (1 + np.exp(-pred_onnx))  # Sigmoid function
        print(f"BirefNet Inference time: {time() - time_st:.4f} seconds")

        im_array = (result * 255).astype(np.uint8)
        pil_im = Image.fromarray(im_array, mode="L")

        # Resize mask BACK to the ORIGINAL HIGH-RES size
        pil_im = pil_im.resize(orig_image.size, Image.BILINEAR)

        new_im = Image.new("RGBA", orig_image.size, (0, 0, 0, 0))
        new_im.paste(orig_image, mask=pil_im)
        
        self.release_model()
        return np.array(new_im)


# Model Registry
MATTING_MODELS = {
    "rmbg-1.4": RMBGModel("rmbg-1.4", WEIGHTS.get("rmbg-1.4", ""), ref_size=1024),
    "rmbg-2.0": RMBGModel("rmbg-2.0", os.path.join(os.path.dirname(__file__), "weights", "rmbg-2.0.onnx"), ref_size=1024),
    "birefnet-v1": BirefNetModel("birefnet-v1", WEIGHTS.get("birefnet-v1", ""), ref_size=1024),
}

def extract_human(ctx: Context, model_name: str = "birefnet-v1"):
    """
    Modularized matting function.
    """
    if model_name not in MATTING_MODELS:
        raise ValueError(f"Matting model {model_name} is not registered. Available: {list(MATTING_MODELS.keys())}")
    
    model = MATTING_MODELS[model_name]
    print(f"Using {model_name} for ultra high-resolution matting...")
    
    matting_image = model.process(ctx.processing_image)
    ctx.processing_image = matting_image
    ctx.matting_image = ctx.processing_image.copy()

