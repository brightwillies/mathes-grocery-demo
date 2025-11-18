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
    model = torch.hub.load('WongKinYiu/yolov7', 'custom', MODEL_PATH, trust_repo=True)
    model = model.to(device)
    model.eval()
    return model, device

# === Preprocess Image ===
def preprocess_image(image):
    """Resize and normalize image for YOLOv7."""
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]
    
    # Letterbox resize
    shape = (IMG_SIZE, IMG_SIZE)
    r = min(shape[0] / h, shape[1] / w)
    new_unpad = int(round(w * r)), int(round(h * r))
    dw, dh = shape[1] - new_unpad[0], shape[0] - new_unpad[1]
    dw, dh = dw // 2, dh // 2
    
    img_resized = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    img_input = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
    img_input[dh:dh+new_unpad[1], dw:dw+new_unpad[0]] = img_resized
    
    img_input = img_input.transpose(2, 0, 1)  # HWC to CHW
    img_input = np.ascontiguousarray(img_input)
    img_input = torch.from_numpy(img_input).float() / 255.0
    img_input = img_input.unsqueeze(0)  # Add batch dim
    return img_input, (dw, dh, r)

# === Postprocess Detections ===
def postprocess(pred, dw, dh, r, orig_shape):
    """Scale boxes back to original image size."""
    pred = pred[0].cpu().numpy()
    boxes, scores, class_ids = [], [], []
    
    h, w = orig_shape
    for *box, conf, cls in pred:
        if conf < CONF_THRESHOLD:
            continue
        # Scale box
        x1 = int((box[0] - dw) / r)
        y1 = int((box[1] - dh) / r)
        x2 = int((box[2] - dw) / r)
        y2 = int((box[3] - dh) / r)
        boxes.append([x1, y1, x2, y2])
        scores.append(conf)
        class_ids.append(int(cls))
    
    return boxes, scores, class_ids

# === Draw Boxes ===
def draw_boxes(img, boxes, scores, class_ids):
    """Draw bounding boxes with labels."""
    img = img.copy()
    for box, score, cls_id in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = box
        color = COLORS[cls_id % len(COLORS)]
        label = f"{CLASSES[cls_id]}: {score:.2f}"
        
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return img

# === Main App ===
st.set_page_config(page_title="YOLOv7 Grocery Items Object Detector", layout="centered")
st.title("YOLOv7 Object Detector")
st.markdown("**Detects:** `cheerios`, `soup`, `candle`")
st.sidebar.header("Upload Image")

# Load model
with st.spinner("Loading model..."):
    model, device = load_model()

# Upload
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    with st.spinner("Running inference..."):
        # Preprocess
        img_input, pad_info = preprocess_image(image)
        img_input = img_input.to(device)
        
        # Inference
        with torch.no_grad():
            pred = model(img_input)[0]
        
        # Postprocess
        boxes, scores, class_ids = postprocess(pred, *pad_info, image.size[::-1])
        
        # Draw
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        result_img = draw_boxes(img_cv, boxes, scores, class_ids)
        result_pil = Image.fromarray(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
        
        st.image(result_pil, caption="Detection Result", use_column_width=True)
        
        if len(boxes) == 0:
            st.success("No objects detected.")
        else:
            st.success(f"Found {len(boxes)} object(s)!")
else:
    st.info("Please upload an image to get started.")