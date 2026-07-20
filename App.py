"""
Left / Right Hand Detection -- Streamlit app

Deploy-ready version of the Colab notebook. Uses MediaPipe's Tasks API
(HandLandmarker) since the older mp.solutions API is deprecated/broken
on current MediaPipe releases.
"""

import os
import urllib.request

import cv2
import numpy as np
import streamlit as st
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_PATH = "hand_landmarker.task"

# 21-point hand landmark connections (thumb, fingers, palm)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17),                                 # palm base
]


@st.cache_resource(show_spinner="Loading hand landmark model...")
def load_detector():
    if not os.path.exists(MODEL_PATH):
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    return mp_vision.HandLandmarker.create_from_options(options)


def detect_hands(img_bgr, detector):
    img_bgr = cv2.flip(img_bgr, 1)  # mirror, so Left/Right match the user's own perspective
    h, w, _ = img_bgr.shape

    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_image)

    labels_found = []

    if result.hand_landmarks:
        if len(result.hand_landmarks) == 2:
            cv2.putText(img_bgr, "Both Hands", (250, 50),
                        cv2.FONT_HERSHEY_COMPLEX, 0.9, (0, 255, 0), 2)

        for landmarks, handedness in zip(result.hand_landmarks, result.handedness):
            label = handedness[0].category_name  # 'Left' or 'Right'
            labels_found.append(label)

            pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

            for a, b in HAND_CONNECTIONS:
                cv2.line(img_bgr, pts[a], pts[b], (255, 255, 255), 2)
            for p in pts:
                cv2.circle(img_bgr, p, 4, (0, 140, 255), -1)

            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            x_min, x_max = min(xs) - 20, max(xs) + 20
            y_min, y_max = min(ys) - 20, max(ys) + 20
            cv2.rectangle(img_bgr, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            color = (0, 255, 0) if label == "Left" else (0, 0, 255)
            cv2.putText(img_bgr, label + " Hand", (x_min, max(y_min - 10, 20)),
                        cv2.FONT_HERSHEY_COMPLEX, 0.9, color, 2)

    return img_bgr, labels_found


def main():
    st.set_page_config(page_title="Hand Detection", page_icon="🖐️")
    st.title("🖐️ Left / Right Hand Detection")
    st.caption(
        "Take a photo or upload an image. The app detects each hand, "
        "draws its skeleton + bounding box, and labels it Left or Right."
    )

    detector = load_detector()

    source = st.radio("Image source", ["Camera", "Upload"], horizontal=True)

    if source == "Camera":
        file = st.camera_input("Take a photo")
    else:
        file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if file is not None:
        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        with st.spinner("Detecting hands..."):
            annotated, labels = detect_hands(img_bgr, detector)

        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)

        if labels:
            st.success(f"Detected: {', '.join(labels)}")
        else:
            st.warning("No hands detected. Try again with your hand(s) clearly visible.")


if __name__ == "__main__":
    main()