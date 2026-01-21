import streamlit as st
import importlib, traceback
from PIL import Image
import numpy as np
import cv2
import dlib
from ultralytics import YOLO

# ------------------ Safe OpenCV Import ------------------
def try_import_cv2():
    try:
        return importlib.import_module("cv2")
    except Exception as e:
        return e

cv2_result = try_import_cv2()
if isinstance(cv2_result, Exception):
    st.error("OpenCV not available")
    st.stop()

# ------------------ Page Config ------------------
st.set_page_config(page_title="Facial Profile Analysis", layout="centered")
st.title("Facial Profile Analysis System")

# ======================================================
#               YOLOv8 SIDE PROFILE SECTION
# ======================================================
st.header("🧠 Side Face Profile Prediction (YOLOv8)")
st.write("Upload a **side face image** to classify as **Convex / Straight / Concave**")

@st.cache_resource
def load_yolo(weights):
    return YOLO(weights)

MODEL_PATH = "best.pt"
model = load_yolo(MODEL_PATH)

side_img = st.file_uploader("Upload Side Face Image", type=["jpg", "png", "jpeg"], key="side")

if side_img:
    image = Image.open(side_img).convert("RGB")
    img_np = np.array(image)
    st.image(image, caption="Uploaded Side Image", use_column_width=True)

    with st.spinner("Running YOLOv8 prediction..."):
        results = model.predict(source=img_np, conf=0.25, verbose=False)

    r = results[0]
    if r.boxes is not None and len(r.boxes.cls) > 0:
        cls_id = int(r.boxes.cls[0])
        conf = float(r.boxes.conf[0])
        cls_name = model.names[cls_id]

        st.success(f"### Prediction: **{cls_name.upper()}**")
        st.write(f"Confidence: `{conf:.2f}`")

        annotated = r.plot()
        st.image(annotated, caption="YOLOv8 Output", use_column_width=True)
    else:
        st.warning("No side face detected")

st.markdown("---")

# ======================================================
#           FRONTAL FACE LANDMARK SECTION
# ======================================================
st.header("📐 Frontal Face Landmark & Angle Analysis")
st.write("Upload a **frontal face image** to visualize 68 landmarks and symmetry groups")

# ---- Dlib Load ----
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def shape_to_np(shape):
    coords = np.zeros((68, 2), dtype="int")
    for i in range(68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords

def _angle_between(v1, v2):
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    cos_t = np.dot(v1, v2) / (n1 * n2)
    cos_t = np.clip(cos_t, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_t)))

def draw_analysis(im, shape):
    shape = shape.astype(np.float32)

    p27, p8 = shape[27], shape[8]
    mid_vec = p8 - p27

    # right jaw
    p11, p9 = shape[11], shape[9]
    right_vec = p9 - p11

    # left jaw
    p5, p7 = shape[5], shape[7]
    left_vec = p7 - p5

    right_angle = _angle_between(right_vec, mid_vec)
    left_angle = _angle_between(left_vec, mid_vec)
    diff = abs(left_angle - right_angle)

    # ---- group logic ----
    if diff <= 3:
        group = "Group 1 (Highly Symmetric)"
    elif diff <= 6:
        group = "Group 2 (Moderately Symmetric)"
    else:
        group = "Group 3 (Asymmetric)"

    # ---- Drawing ----
    im = im.copy()
    cv2.line(im, tuple(p27.astype(int)), tuple(p8.astype(int)), (0,255,0), 2)

    cv2.line(im, tuple(p11.astype(int)), tuple(p9.astype(int)), (0,0,255), 2)
    cv2.line(im, tuple(p5.astype(int)), tuple(p7.astype(int)), (255,0,0), 2)

    cv2.putText(im, f"L:{left_angle:.1f}°", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
    cv2.putText(im, f"R:{right_angle:.1f}°", (10,55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
    cv2.putText(im, f"Diff:{diff:.1f}°", (10,80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)

    return im, left_angle, right_angle, diff, group

front_img = st.file_uploader("Upload Frontal Face Image", type=["jpg","png","jpeg"], key="front")

if front_img:
    img = Image.open(front_img).convert("RGB")
    img_np = np.array(img)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    faces = detector(gray)

    if len(faces) == 0:
        st.warning("No frontal face detected")
    else:
        for i, face in enumerate(faces):
            shape = predictor(gray, face)
            shape_np = shape_to_np(shape)

            out_img, la, ra, diff, group = draw_analysis(img_np, shape_np)

            st.image(out_img, caption=f"Face {i+1} Analysis", use_column_width=True)
            st.success(f"**{group}**")
            st.write(f"Left Angle: `{la:.2f}°` | Right Angle: `{ra:.2f}°` | Diff: `{diff:.2f}°`")

st.markdown("---")
st.caption("YOLOv8 Side Profile + Dlib 68 Landmark Facial Symmetry Analysis")
