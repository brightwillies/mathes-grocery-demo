"""
Streamlit YOLOv7 Object Detector
Detects: cheerios, soup, candle
Model: yolov7-tiny custom trained
"""
import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
import os
from pathlib import Path
import subprocess
import sys

# === MONKEY PATCH: Disable YOLOv7's automatic dependency checking ===
import types

def patched_check_requirements(requirements, exclude=()):
    """Patched version that doesn't try to auto-install packages"""
    print(f"Requirements check disabled: {requirements}")
    return

# Create a mock module to prevent the subprocess calls
class MockSubprocess:
    def check_output(self, *args, **kwargs):
        return b""
    
    def run(self, *args, **kwargs):
        class Result:
            stdout = b""
            stderr = b""
            returncode = 0
        return Result()

# Apply patches before any YOLOv7 imports happen
import sys
sys.modules['subprocess'] = MockSubprocess()

# === CONFIG ===
MODEL_PATH = "yolov7_cheerios_soup_candle_best.pt"
IMG_SIZE = 640
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45

# Class names (must match training)
CLASSES = ['cheerios', 'soup', 'candle']
COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # BGR for OpenCV

# === Load Model ===
@st.cache_resource
def load_model():
    """Load YOLOv7 model with CPU fallback."""
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model not found: {MODEL_PATH}")
        st.stop()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Apply the patch to YOLOv7's utils
    try:
        # This will patch the function when YOLOv7 tries to use it
        from utils import general
        general.check_requirements = patched_check_requirements
    except ImportError:
        pass
    
    model = torch.hub.load('WongKinYiu/yolov7', 'custom', MODEL_PATH, trust_repo=True)
    model = model.to(device)
    model.eval()
    return model, device

# [Rest of your code remains the same - preprocess_image, postprocess, draw_boxes, etc.]

# === Main App ===
st.set_page_config(page_title="YOLOv7 Grocery Items Object Detector", layout="centered")
st.title("YOLOv7 Object Detector")
st.markdown("**Detects:** `cheerios`, `soup`, `candle`")
st.sidebar.header("Upload Image")

# Load model
with st.spinner("Loading model..."):
    model, device = load_model()

# [Rest of your app code...]