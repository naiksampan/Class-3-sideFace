import streamlit as st
import importlib, traceback, io
from PIL import Image
import numpy as np

# ------------------ OpenCV Safe Import ------------------
def try_import_cv2():
    try:
        return importlib.import_module("cv2")
    except Exception as e:
        return e

cv2_result = try_import_cv2()
if isinstance(cv2_result, Exception):
    st.set_page_config(page_title="Dependency Error")
    st.error("OpenCV (cv2) is not available.")
    st.text(traceback.format_exc())
    st.stop()

import cv2
from ultralytics import YOLO

# ------------------ Page Config ------------------
st.set_page_config(
    page_title="Side Face Shape Prediction",
    layout="centered"
)

st.title("🧠 Side Face Shape Prediction")
st.write("Upload a **side face image** to classify it as **Convex, Straight, or Concave** using YOLOv8.")

# ------------------ Load Model ------------------
@st.cache_resource
def load_model(weights_path):
    return YOLO(weights_path)

MODEL_PATH = "best.pt"

try:
    model = load_model(MODEL_PATH)
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# ------------------ Image Upload ------------------
uploaded_file = st.file_uploader(
    "Upload Side Face Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    # Read image
    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)

    st.image(image, caption="Uploaded Image", use_column_width=True)

    # ------------------ Prediction ------------------
    with st.spinner("Running prediction..."):
        results = model.predict(
            source=img_np,
            conf=0.25,
            save=False,
            verbose=False
        )

    # ------------------ Extract Prediction ------------------
    r = results[0]

    if r.boxes is not None and len(r.boxes.cls) > 0:
        class_id = int(r.boxes.cls[0])
        confidence = float(r.boxes.conf[0])

        class_name = model.names[class_id]

        st.success(f"### 🧾 Prediction: **{class_name.upper()}**")
        st.write(f"**Confidence:** `{confidence:.2f}`")

        # Annotated image
        annotated = r.plot()
        annotated_rgb = cv2.cvtColor(annotated)
        st.image(annotated_rgb, caption="Prediction Output", use_column_width=True)

    else:
        st.warning("No face detected. Please upload a clear side face image.")

st.markdown("---")
st.caption("YOLOv8 Side Profile Classification • Convex | Straight | Concave")
