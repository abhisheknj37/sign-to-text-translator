from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input

BASE = Path(__file__).resolve().parent
st.set_page_config(page_title="Sign Language to Text", page_icon="🤟", layout="centered")
st.title("🤟 Sign Language to Text")
st.caption("Capture or upload one hand-sign image. The trained CNN models predict a letter.")

def build_model(outputs):
    return Sequential([
        Input(shape=(128, 128, 1)),
        Conv2D(32, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),
        Conv2D(32, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),
        Conv2D(32, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.4),
        Dense(96, activation="relu"),
        Dropout(0.4),
        Dense(64, activation="relu"),
        Dense(outputs, activation="softmax"),
    ])

@st.cache_resource
def load_models():
    specs = [
        ("model-bw.h5", 27),
        ("model-bw_dru.h5", 3),
        ("model-bw_tkdi.h5", 4),
        ("model-bw_smn.h5", 3),
    ]
    models = []
    for filename, outputs in specs:
        model = build_model(outputs)
        model.load_weights(str(BASE / "model" / filename))
        models.append(model)
    return models

def predict(image):
    rgb = np.array(image.convert("RGB"))
    frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    height, width = frame.shape[:2]
    x1, y1 = int(0.5 * width), 10
    x2, y2 = width - 10, min(int(0.5 * width), height)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Image is too small. Use a landscape image with the hand on the right half.")
    crop = frame[y1:y2, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 2)
    adaptive = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    _, processed = cv2.threshold(
        adaptive, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    sample = cv2.resize(processed, (128, 128)).astype("float32")
    sample = sample.reshape(1, 128, 128, 1)
    main, dru, tkdi, smn = [m.predict(sample, verbose=0)[0] for m in load_models()]
    letters = ["blank"] + list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    symbol = letters[int(np.argmax(main))]
    if symbol in "DRU":
        symbol = ["D", "R", "U"][int(np.argmax(dru))]
    if symbol in "DIKT":
        symbol = ["D", "I", "K", "T"][int(np.argmax(tkdi))]
    if symbol in "MNS":
        candidate = ["M", "N", "S"][int(np.argmax(smn))]
        if candidate == "S":
            symbol = candidate
    return symbol, processed

source = st.radio("Image source", ["Use camera", "Upload image"], horizontal=True)
file = st.camera_input("Show one hand sign to the camera") if source == "Use camera" else st.file_uploader("Upload a hand-sign image", type=["png", "jpg", "jpeg"])
if file:
    image = Image.open(BytesIO(file.getvalue()))
    st.image(image, caption="Input image", use_container_width=True)
    if st.button("Recognize sign", type="primary"):
        try:
            with st.spinner("Loading model and recognizing sign..."):
                symbol, processed = predict(image)
            st.success("Predicted sign: " + ("No sign / blank" if symbol == "blank" else symbol))
            st.image(processed, caption="Preprocessed hand region", clamp=True, width=260)
        except Exception as exc:
            st.error(f"Could not recognize this image: {exc}")
else:
    st.info("Place your hand in the right half of the image, matching the original model's capture region.")
