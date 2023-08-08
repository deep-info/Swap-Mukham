import os
import cv2
import torch
import gfpgan
from PIL import Image
from upscaler.GPEN import GPEN
from upscaler.GFPGAN import GFPGAN
from upscaler.codeformer import CodeFormer

def gfpgan_runner(img, model):
    img = model.enhance(img)
    return img


def codeformer_runner(img, model):
    img = model.enhance(img, w=0.9)
    return img


def gpen_runner(img, model):
    img = model.enhance(img)
    return img


supported_upscalers = {
    "CodeFormer": ("./assets/pretrained_models/codeformer.onnx", codeformer_runner),
    "GFPGANv1.4": ("./assets/pretrained_models/GFPGANv1.4.onnx", gfpgan_runner),
    "GFPGANv1.3": ("./assets/pretrained_models/GFPGANv1.3.onnx", gfpgan_runner),
    "GFPGANv1.2": ("./assets/pretrained_models/GFPGANv1.2.onnx", gfpgan_runner),
    "GPEN-BFR-512": ("./assets/pretrained_models/GPEN-BFR-512.onnx", gpen_runner),
    "GPEN-BFR-256": ("./assets/pretrained_models/GPEN-BFR-256.onnx", gpen_runner),
}

cv2_upscalers = ["LANCZOS4", "CUBIC", "NEAREST"]

def get_available_upscalers_names():
    available = []
    for name, data in supported_upscalers.items():
        path = os.path.join(os.path.abspath(os.path.dirname(__file__)), data[0])
        if os.path.exists(path):
            available.append(name)
    return available


def load_face_upscaler(name='GFPGAN', provider=["CPUExecutionProvider"]):
    assert name in get_available_upscalers_names() + cv2_upscalers, f"Face upscaler {name} unavailable."
    if name in supported_upscalers.keys():
        model_path, model_runner = supported_upscalers.get(name)
        model_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), model_path)
    if name == 'CodeFormer':
        model = CodeFormer(model_path=model_path, provider=provider)
    elif name.startswith('GFPGAN'):
        model = GFPGAN(model_path=model_path, provider=provider)
    elif name.startswith('GPEN'):
        model = GPEN(model_path=model_path, provider=provider)
    elif name == 'LANCZOS4':
        model = None
        model_runner = lambda img, _: cv2.resize(img, (512,512), interpolation=cv2.INTER_LANCZOS4)
    elif name == 'CUBIC':
        model = None
        model_runner = lambda img, _: cv2.resize(img, (512,512), interpolation=cv2.INTER_CUBIC)
    elif name == 'NEAREST':
        model = None
        model_runner = lambda img, _: cv2.resize(img, (512,512), interpolation=cv2.INTER_NEAREST)
    else:
        model = None
    return (model, model_runner)
