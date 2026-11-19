import os
import requests
import streamlit as st

from PIL import Image, ImageDraw
from io import BytesIO


API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="NeuroVision AI", layout="wide")

st.title("NeuroVision AI")
st.subheader("Brain MRI Tumor Detection")
st.write("Upload a brain MRI image to detect and locate potential tumors.")

st.sidebar.header("Settings")

confidence = st.sidebar.slider(
    "Confidence Threshold", min_value=0.0, max_value=1.0, value=0.25, step=0.05
)

uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "jpeg", "png", "bmp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original MRI")
        st.image(image, use_container_width=True)

    if st.button("Detect Tumors", type="primary"):
        with st.spinner("Analyzing MRI image..."):
            try:
                response = requests.post(
                    f"{API_URL}/predict",
                    files={
                        "file": (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            uploaded_file.type,
                        )
                    },
                    data={"confidence": confidence},
                    timeout=120,
                )

                if response.status_code != 200:
                    st.error(f"API Error {response.status_code}: {response.text}")
                    st.stop()

                result = response.json()

                detections = result["detections"]

                annotated_image = image.copy()
                draw = ImageDraw.Draw(annotated_image)

                class_colors = {
                    "Glioma": "red",
                    "Meningioma": "blue",
                    "No Tumor": "green",
                    "Pituitary": "orange",
                }

                for detection in detections:
                    bbox = detection["bbox"]

                    x1 = bbox["x1"]
                    y1 = bbox["y1"]
                    x2 = bbox["x2"]
                    y2 = bbox["y2"]

                    class_name = detection["class_name"]
                    score = detection["confidence"]

                    color = class_colors.get(class_name, "yellow")

                    draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

                    label = f"{class_name}: {score:.2%}"

                    draw.text((x1, max(0, y1 - 20)), label, fill=color)

                with col2:
                    st.subheader("Detection Results")
                    st.image(annotated_image, use_container_width=True)

                st.divider()

                if not detections:
                    st.success("No tumor detected.")

                else:
                    st.warning(f"{len(detections)} detection(s) found.")

                    st.subheader("Detected Objects")

                    for index, detection in enumerate(detections, start=1):
                        st.write(f"**{index}. {detection['class_name']}**")

                        st.write(f"Confidence: {detection['confidence']:.2%}")

                with st.expander("View API Response"):
                    st.json(result)

            except requests.exceptions.ConnectionError:
                st.error(
                    "Cannot connect to FastAPI. Make sure the API server is running."
                )

            except requests.exceptions.Timeout:
                st.error("The API request timed out.")

            except Exception as error:
                st.error(f"Unexpected error: {error}")

st.divider()

st.caption(
    "NeuroVision AI is an experimental decision-support tool. "
    "It does not replace professional medical diagnosis."
)
