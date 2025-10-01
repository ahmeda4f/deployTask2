import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import tempfile
import os
import pandas as pd

st.title("Parking Space Detection App")
model = YOLO("best (1).pt")

conf_threshold = st.slider("Confidence Threshold", 0.0, 0.7, 0.2)

uploaded_file = st.file_uploader("Upload an image or video", type=["jpg", "jpeg", "png", "mp4", "avi"])

custom_labels = {0: "Empty", 1: "Occupied"}
custom_colors = {0: (0, 255, 0), 1: (255, 0, 0)}

def annotate_frame(frame, results):
    for box, cls in zip(results[0].boxes.xyxy.cpu().numpy(), results[0].boxes.cls.cpu().numpy()):
        x1, y1, x2, y2 = map(int, box)
        label = custom_labels.get(int(cls), "Unknown")
        color = custom_colors.get(int(cls), (255, 255, 255))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return frame

if uploaded_file is not None:
    if uploaded_file.type.startswith("image"):
        image = Image.open(uploaded_file)
        results = model.predict(image, conf=conf_threshold)
        annotated_image = cv2.cvtColor(results[0].orig_img, cv2.COLOR_BGR2RGB)
        annotated_image = annotate_frame(annotated_image, results)
        st.image(annotated_image, caption="Detected Parking Spaces", use_container_width=True)

        boxes = results[0].boxes.xyxy.cpu().numpy()
        scores = results[0].boxes.conf.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()
        if len(boxes) > 0:
            df = pd.DataFrame({
                "x1": boxes[:,0], "y1": boxes[:,1],
                "x2": boxes[:,2], "y2": boxes[:,3],
                "confidence": scores, "class": classes
            })
            st.dataframe(df)

    elif uploaded_file.type.startswith("video"):
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        cap = cv2.VideoCapture(tfile.name)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_path = "output.mp4"
        out = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = model.predict(frame_rgb, conf=conf_threshold)
            annotated_frame = annotate_frame(frame_rgb, results)
            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
            if out is None:
                h, w, _ = annotated_frame.shape
                out = cv2.VideoWriter(output_path, fourcc, 20.0, (w, h))
            out.write(annotated_frame)

        cap.release()
        out.release()

        with open(output_path, "rb") as f:
            video_bytes = f.read()

        st.video(video_bytes)

        st.download_button(
            label="Download Processed Video",
            data=video_bytes,
            file_name="processed_output.mp4",
            mime="video/mp4"
        )

        os.remove(tfile.name)
        os.remove(output_path)
