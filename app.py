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
#                 YOLOv8 SIDE PROFILE
# ======================================================
st.header("🧠 Side Face Profile Prediction (YOLOv8)")

@st.cache_resource
def load_yolo(weights):
    return YOLO(weights)

MODEL_PATH = "best.pt"
model = load_yolo(MODEL_PATH)

side_img = st.file_uploader(
    "Upload Side Face Image (Convex / Straight / Concave)",
    type=["jpg", "png", "jpeg"],
    key="side"
)

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
#        FRONTAL FACE – 68 LANDMARK + INTERSECTION
# ======================================================
st.header("📐 Frontal Face Landmark & Symmetry Analysis")

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def shape_to_np(shape):
    coords = np.zeros((68, 2), dtype="int")
    for i in range(68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords

def line_intersection(p1, p2, q1, q2):
    """Intersection of two infinite lines"""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = q1
    x4, y4 = q2

    denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
    if abs(denom) < 1e-6:
        return None

    px = ((x1*y2-y1*x2)*(x3-x4)-(x1-x2)*(x3*y4-y3*x4))/denom
    py = ((x1*y2-y1*x2)*(y3-y4)-(y1-y2)*(x3*y4-y3*x4))/denom
    return np.array([px, py], dtype=np.float32)

def angle_between(v1, v2):
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    cos_t = np.dot(v1, v2) / (n1 * n2)
    cos_t = np.clip(cos_t, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_t)))

def draw_vertical_midline_and_jaw(im, shape):
    h, w = im.shape[:2]

    # ---- Vertical Midline using landmark 27 ----
    p27 = shape[27].astype(np.float32)
    mid_top = np.array([p27[0], 0], dtype=np.float32)
    mid_bottom = np.array([p27[0], h], dtype=np.float32)

    mid_vec = mid_bottom - mid_top

    # ---- Jaw lines ----
    # Right side (13 -> 11 -> 9)
    p11 = shape[11].astype(np.float32)
    p9 = shape[9].astype(np.float32)

    # Left side (3 -> 5 -> 7)
    p5 = shape[5].astype(np.float32)
    p7 = shape[7].astype(np.float32)

    # ---- Intersections ----
    I_right = line_intersection(p11, p9, mid_top, mid_bottom)
    I_left = line_intersection(p5, p7, mid_top, mid_bottom)

    # ---- Angles ----
    right_angle = angle_between(p9 - p11, mid_vec)
    left_angle = angle_between(p7 - p5, mid_vec)
    diff = abs(left_angle - right_angle)

    # ---- Group ----
    if diff <= 3:
        group = "Group 1 (Highly Symmetric)"
    elif diff <= 6:
        group = "Group 2 (Moderate Symmetry)"
    else:
        group = "Group 3 (Asymmetric)"

    # ---- Drawing ----
    im = im.copy()

    # Vertical midline
    cv2.line(im, tuple(mid_top.astype(int)), tuple(mid_bottom.astype(int)), (0,255,0), 2)

    # Right jaw
    cv2.line(im, tuple(p11.astype(int)), tuple(I_right.astype(int)), (0,0,255), 2)
    cv2.circle(im, tuple(I_right.astype(int)), 5, (0,255,255), -1)

    # Left jaw
    cv2.line(im, tuple(p5.astype(int)), tuple(I_left.astype(int)), (255,0,0), 2)
    cv2.circle(im, tuple(I_left.astype(int)), 5, (0,255,255), -1)

    # Text
    cv2.putText(im, f"L:{left_angle:.1f}°", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
    cv2.putText(im, f"R:{right_angle:.1f}°", (10,55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
    cv2.putText(im, f"Diff:{diff:.1f}°", (10,80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
    cv2.putText(im, group, (10,110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,128,0), 2)

    return im, left_angle, right_angle, diff, group

# ------------------ Upload Frontal Image ------------------
front_img = st.file_uploader(
    "Upload Frontal Face Image",
    type=["jpg", "png", "jpeg"],
    key="front"
)

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

            out_img, la, ra, diff, group = draw_vertical_midline_and_jaw(img_np, shape_np)

            st.image(out_img, caption=f"Face {i+1} – Landmark Analysis", use_column_width=True)
            st.success(group)
            st.write(f"Left Angle: `{la:.2f}°` | Right Angle: `{ra:.2f}°` | Difference: `{diff:.2f}°`")

st.markdown("---")
st.caption("YOLOv8 Side Profile Classification + Vertical Midline Facial Symmetry Analysis")
