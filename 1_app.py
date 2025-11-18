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

# === Model File Diagnostics ===
def check_model_file():
    """Diagnose model file issues."""
    if not os.path.exists(MODEL_PATH):
        return "❌ File not found"
    
    file_size = os.path.getsize(MODEL_PATH)
    
    # Read first few bytes to check if it's a valid PyTorch file
    try:
        with open(MODEL_PATH, 'rb') as f:
            header = f.read(10)
        
        if file_size < 1000000:  # Less than 1MB
            return f"❌ File too small ({file_size} bytes) - might be LFS pointer or corrupted"
        
        # Check if it has PyTorch magic number (common pickle headers)
        if header[:2] == b'\x80\x02' or header[:3] == b'\x80\x02\x90' or header[:2] == b'\x80\x03':
            return f"✅ File looks OK ({file_size / (1024*1024):.2f} MB)"
        else:
            return f"⚠️ File doesn't appear to be a valid PyTorch model ({file_size} bytes)"
            
    except Exception as e:
        return f"❌ Error reading file: {str(e)}"

# === Load Model ===
@st.cache_resource
def load_model():
    """Load YOLOv7 model directly without torch.hub."""
    st.info("Checking model file...")
    file_status = check_model_file()
    st.write(f"Model file status: {file_status}")
    
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model not found: {MODEL_PATH}")
        st.stop()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    st.info(f"Using device: {device}")
    
    try:
        # Try loading directly as a PyTorch model
        st.info("Loading model directly...")
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'model' in checkpoint:
                # Standard YOLO format with state dict
                model = checkpoint['model']
                st.info("Loaded model from checkpoint dictionary")
            else:
                # Might be a state dict directly
                model = checkpoint
                st.info("Loaded model as state dictionary")
        else:
            # Assume it's already the model object
            model = checkpoint
            st.info("Loaded model as model object")
        
        # If model is a state dict, we need to create a model architecture first
        if isinstance(model, dict):
            st.warning("Model is a state dictionary. Need model architecture to load properly.")
            # Try to infer model architecture from the state dict
            try:
                # This is a simplified approach - you might need to adjust based on your model
                from models.experimental import attempt_load
                # Create a dummy model and load state dict
                model = attempt_load(MODEL_PATH, map_location=device)
            except:
                st.error("Cannot load state dict without model architecture. Try using torch.hub approach.")
                st.stop()
        
        model = model.float()  # Ensure float32
        model.eval()
        
        st.success("✅ Model loaded successfully!")
        return model, device
        
    except Exception as e:
        st.error(f"❌ Error loading model directly: {str(e)}")
        st.info("Trying alternative loading method...")
        
        # Fallback to torch.hub with error handling
        try:
            # Apply patch to disable requirement checking
            import utils.general
            utils.general.check_requirements = lambda *args, **kwargs: None
            
            model = torch.hub.load('WongKinYiu/yolov7', 'custom', MODEL_PATH, trust_repo=True)
            model = model.to(device)
            model.eval()
            
            st.success("✅ Model loaded successfully with torch.hub!")
            return model, device
            
        except Exception as e2:
            st.error(f"❌ All loading methods failed: {str(e2)}")
            st.stop()

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
st.set_page_config(
    page_title="YOLOv7 Grocery Items Object Detector", 
    layout="centered",
    page_icon="🛒"
)

st.title("🛒 YOLOv7 Grocery Items Object Detector")
st.markdown("**Detects:** `cheerios` 🥣, `soup` 🍲, `candle` 🕯️")
st.sidebar.header("Upload Image")

# Display model info in sidebar
st.sidebar.subheader("Model Information")
st.sidebar.write(f"Model path: `{MODEL_PATH}`")
st.sidebar.write(f"Classes: {', '.join(CLASSES)}")
st.sidebar.write(f"Confidence threshold: {CONF_THRESHOLD}")
st.sidebar.write(f"Image size: {IMG_SIZE}x{IMG_SIZE}")

# Load model
with st.spinner("Loading model... This may take a moment."):
    model, device = load_model()

# File upload
uploaded_file = st.sidebar.file_uploader(
    "Choose an image...", 
    type=["jpg", "jpeg", "png"],
    help="Upload an image containing cheerios, soup, or candles"
)

if uploaded_file is not None:
    # Display original image
    image = Image.open(uploaded_file)
    st.subheader("📤 Uploaded Image")
    st.image(image, caption="Original Image", use_column_width=True)
    
    # Run detection
    with st.spinner("Running object detection..."):
        try:
            # Preprocess
            img_input, pad_info = preprocess_image(image)
            img_input = img_input.to(device)
            
            # Inference
            with torch.no_grad():
                pred = model(img_input)[0]
            
            # Postprocess
            boxes, scores, class_ids = postprocess(pred, *pad_info, image.size[::-1])
            
            # Draw results
            img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            result_img = draw_boxes(img_cv, boxes, scores, class_ids)
            result_pil = Image.fromarray(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
            
            # Display results
            st.subheader("🔍 Detection Results")
            st.image(result_pil, caption="Detection Result", use_column_width=True)
            
            # Display detection summary
            if len(boxes) == 0:
                st.success("No objects detected. Try adjusting the confidence threshold or upload a different image.")
            else:
                st.success(f"Found {len(boxes)} object(s)!")
                
                # Show detection details
                st.subheader("📊 Detection Details")
                for i, (box, score, cls_id) in enumerate(zip(boxes, scores, class_ids)):
                    st.write(f"**{i+1}. {CLASSES[cls_id]}** - Confidence: {score:.2f}")
                    
        except Exception as e:
            st.error(f"Error during detection: {str(e)}")
            st.info("This might be due to model compatibility issues. Please check the model file.")

else:
    st.info("👆 Please upload an image to get started!")
    
    # Display example usage
    st.subheader("ℹ️ How to use:")
    st.markdown("""
    1. Click **'Browse files'** in the sidebar
    2. Upload an image containing cheerios, soup, or candles
    3. Wait for the model to process the image
    4. View the detection results with bounding boxes
    """)
    
    st.subheader("🎯 Supported Objects:")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**🥣 Cheerios**")
    with col2:
        st.markdown("**🍲 Soup**")
    with col3:
        st.markdown("**🕯️ Candle**")

# Footer
st.markdown("---")
st.markdown(
    "Built with YOLOv7 | "
    "Custom trained model | "
    "Streamlit"
)