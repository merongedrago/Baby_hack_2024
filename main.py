import streamlit as st
import tempfile
import time
import cv2
from ultralytics import YOLO 
from model.model import split_video_into_memory_chunks , run_model

CUSTOM_FRAME_RATE = 32
CONF_THRESHOLD = 0.5
EXPANSION_RATE = 0.3




st.title("Real-Time Object Detection with YOLO")

# Step 1: Video Upload
uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    # Save uploaded video to a temporary file
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    # Display video in Streamlit without frame-by-frame processing
    st.video(tfile.name)

    # Run YOLO model on the whole video file
    # Replace `run_yolo_model` with the actual function for your model
    # Example: results = run_yolo_model(tfile.name)
    # For this example, simulate model output with a loop
    notification_placeholder = st.empty()
    cap = cv2.VideoCapture(tfile)

    chunks, width_length = split_video_into_memory_chunks(cap, chunk_duration= 3)

    for chunk in chunks:

        output_chunk = run_model(
            cap = cap,
            yolo_path='model/yolov8s.pt',
            chunk = chunk,
            output_path='output_video_structured.avi',
            frame_width_height=width_length,
            conf_threshold=CONF_THRESHOLD
        )


    cap.release()



    # Simulate model output notifications for demonstration purposes
    for i in range(100):
        if i % 5 == 0:
            notification_placeholder.error("DANGER: Immediate threat detected!")
        elif i % 3 == 0:
            notification_placeholder.warning("WARNING: Potential danger approaching.")
        else:
            notification_placeholder.info("You are OK!")

        # Simulate delay based on model processing time
        time.sleep(1)  # Adjust as needed for real model latency

    st.write("Video processing completed.")
