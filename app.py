import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import tempfile
import os
import pandas as pd

st.title("Parking Space Detection App")

model = YOLO("best (1).pt")
model.names = {0: "empty", 1: "occupied"}

conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.25)

uploaded_file = st.file_uploader("Upload an image or video", type=["jpg", "jpeg", "png", "mp4", "avi"])

colors = {0: (0, 255, 0), 1: (0, 0, 255)}

if uploaded_file is not None:
    if uploaded_file.type.startswith("image"):
        image = Image.open(uploaded_file)
        results = model.predict(image, conf=conf_threshold)

        annotated = results[0].plot(line_width=2, labels=True, conf=True)
        annotated_custom = annotated.copy()
        for box, cls in zip(results[0].boxes.xyxy.cpu().numpy(), results[0].boxes.cls.cpu().numpy()):
            x1, y1, x2, y2 = map(int, box)
            color = colors[int(cls)]
            label = model.names[int(cls)]
            cv2.rectangle(annotated_custom, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated_custom, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, color, 2)

        st.image(annotated_custom, caption="Detected Parking Spaces", use_column_width=True)

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

            annotated_frame = frame.copy()
            for box, cls in zip(results[0].boxes.xyxy.cpu().numpy(), results[0].boxes.cls.cpu().numpy()):
                x1, y1, x2, y2 = map(int, box)
                color = colors[int(cls)]
                label = model.names[int(cls)]
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.7, color, 2)

            if out is None:
                h, w, _ = annotated_frame.shape
                out = cv2.VideoWriter(output_path, fourcc, 20.0, (w, h))
            out.write(annotated_frame)

        cap.release()
        out.release()
        st.video(output_path)
        os.remove(tfile.name)
