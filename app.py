import streamlit as st
import importlib, traceback
import numpy as np
from PIL import Image
import cv2
import dlib
from ultralytics import YOLO

# ------------------ Safe cv2 Import ------------------
def try_import_cv2():
    try:
        return importlib.import_module("cv2")
    except Exception as e:
        return e

cv2_result = try_import_cv2()
if isinstance(cv2_result, Exception):
    st.error("OpenCV not available")
    st.text(traceback.format_exc())
    st.stop()

# ------------------ Page Config ------------------
st.set_page_config(page_title="Face Profile Analyzer", layout="centered")
st.title("Face Profile Analyzer")
st.write("Side Profile Classification + Frontal Landmark Symmetry")

# ------------------ Load YOLOv8 Model ------------------
@st.cache_resource
def load_yolo(path):
    return YOLO(path)

model = load_yolo("best.pt")

# ------------------ Load Dlib ------------------
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# ------------------ Helper Functions ------------------
def shape_to_np(shape):
    coords = np.zeros((68, 2), dtype="int")
    for i in range(68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords

def _line_intersection(p1, p2, q1, q2):
    x1,y1 = p1; x2,y2 = p2
    x3,y3 = q1; x4,y4 = q2
    denom = (x1-x2)*(y3-y4)-(y1-y2)*(x3-x4)
    if abs(denom) < 1e-6:
        return None
    px = ((x1*y2-y1*x2)*(x3-x4)-(x1-x2)*(x3*y4-y3*x4))/denom
    py = ((x1*y2-y1*x2)*(y3-y4)-(y1-y2)*(x3*y4-y3*x4))/denom
    return np.array([px, py], dtype=np.float32)

def angle(v1, v2):
    v1 = v1.astype(np.float32)
    v2 = v2.astype(np.float32)
    cos = np.dot(v1, v2) / (np.linalg.norm(v1)*np.linalg.norm(v2)+1e-6)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))

def draw_and_classify(im, shape):
    p27, p8 = shape[27], shape[8]
    v_mid = p8 - p27

    p11,p9 = shape[11],shape[9]
    p5,p7 = shape[5],shape[7]

    v_r = p9 - p11
    v_l = p7 - p5

    right_angle = angle(v_r, v_mid)
    left_angle = angle(v_l, v_mid)
    diff = abs(left_angle - right_angle)

    if diff <= 3:
        group = "Group 1"
    elif diff <= 6:
        group = "Group 2"
    else:
        group = "Group 3"

    # draw landmarks
    for (x,y) in shape:
        cv2.circle(im,(int(x),int(y)),1,(0,255,0),-1)

    return im, left_angle, right_angle, diff, group

# ================== SIDEBAR ==================
st.sidebar.header("Upload Images")

side_img_file = st.sidebar.file_uploader("Side Face Image", ["jpg","png","jpeg"])
front_img_file = st.sidebar.file_uploader("Frontal Face Image", ["jpg","png","jpeg"])

# ================== SIDE PROFILE ==================
if side_img_file:
    st.subheader("Side Profile Prediction (YOLOv8)")
    image = Image.open(side_img_file).convert("RGB")
    img_np = np.array(image)
    st.image(image, caption="Uploaded Side Face", use_column_width=True)

    results = model.predict(img_np, conf=0.25, verbose=False)
    r = results[0]

    if r.boxes is not None and len(r.boxes.cls) > 0:
        cid = int(r.boxes.cls[0])
        conf = float(r.boxes.conf[0])
        label = model.names[cid]

        st.success(f"Prediction: **{label.upper()}**")
        st.write(f"Confidence: `{conf:.2f}`")
        st.image(r.plot(), caption="YOLO Output", use_column_width=True)
    else:
        st.warning("No side face detected")

# ================== FRONTAL LANDMARK ==================
if front_img_file:
    st.subheader("Frontal Landmark Symmetry Analysis")
    image = Image.open(front_img_file).convert("RGB")
    img = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    faces = detector(gray)

    if len(faces)==0:
        st.warning("No frontal face detected")
    else:
        for i, face in enumerate(faces):
            shape = predictor(gray, face)
            shape_np = shape_to_np(shape)

            out_img, l_ang, r_ang, diff, group = draw_and_classify(img.copy(), shape_np)

            st.image(out_img, caption=f"Face {i+1} Landmarks", use_column_width=True)
            st.write(f"Left Angle: `{l_ang:.2f}°`")
            st.write(f"Right Angle: `{r_ang:.2f}°`")
            st.write(f"Difference: `{diff:.2f}°`")
            st.success(f"Category: **{group}**")

st.markdown("---")
st.caption("YOLOv8 Side Profile • Dlib 68 Landmark • Group Classification")
