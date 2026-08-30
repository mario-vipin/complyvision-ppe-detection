import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

# --- Page setup ---
st.set_page_config(page_title="ComplyVision", page_icon="🦺", layout="centered")
st.title("🦺 ComplyVision")
st.subheader("Real-Time PPE & Mask Compliance Detection")

# --- Load model (cached so it doesn't reload on every interaction) ---
@st.cache_resource
def load_model():
    return YOLO("models/best.pt")

model = load_model()

# --- Sidebar controls ---
st.sidebar.header("Settings")
conf_threshold = st.sidebar.slider("Confidence threshold", 0.1, 1.0, 0.4, 0.05)

# --- Input mode ---
option = st.radio("Choose input method:", ["Upload Image", "Use Webcam"])

def run_inference(image: np.ndarray):
    results = model.predict(image, conf=conf_threshold)
    annotated = results[0].plot()  # draws boxes + labels
    annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    return annotated, results[0]

if option == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)

        with st.spinner("Detecting..."):
            annotated, result = run_inference(image_np)

        st.image(annotated, caption="Detection Result", use_container_width=True)

        # Compliance summary
        class_names = result.names
        detected = [class_names[int(cls)] for cls in result.boxes.cls]
        if detected:
            st.subheader("Detected Objects")
            for cls in set(detected):
                st.write(f"- **{cls}**: {detected.count(cls)}")
        else:
            st.info("No objects detected above the confidence threshold.")

elif option == "Use Webcam":
    camera_image = st.camera_input("Take a photo")
    if camera_image is not None:
        image = Image.open(camera_image).convert("RGB")
        image_np = np.array(image)

        with st.spinner("Detecting..."):
            annotated, result = run_inference(image_np)

        st.image(annotated, caption="Detection Result", use_container_width=True)

        class_names = result.names
        detected = [class_names[int(cls)] for cls in result.boxes.cls]
        if detected:
            st.subheader("Detected Objects")
            for cls in set(detected):
                st.write(f"- **{cls}**: {detected.count(cls)}")
        else:
            st.info("No objects detected above the confidence threshold.")