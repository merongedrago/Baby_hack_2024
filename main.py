import streamlit as st
import tempfile
import time
import cv2
from ultralytics import YOLO
from model.model import split_video_into_memory_chunks, run_model
from lib.obj_functions import data_transforming, output
from lib.danger_detector import check_dangerous_items, get_completion_gemini
from dotenv import load_dotenv

load_dotenv()

CUSTOM_FRAME_RATE = 32
CONF_THRESHOLD = 0.5
EXPANSION_RATE = 0.3

st.title("Real-Time Object Detection with YOLO")

# Step 1: Video Upload
uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    # Save uploaded video to a temporary file for OpenCV processing
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_video_path = temp_file.name

    # Display the uploaded video in Streamlit
    st.video(temp_video_path)

    # Run YOLO model on the whole video file
    notification_placeholder = st.empty()
    cap = cv2.VideoCapture(temp_video_path)

    # Process video in chunks
    chunks, width_length = split_video_into_memory_chunks(cap, chunk_duration=3)
    dic_hazard = {}
    t = 0
    for chunk in chunks:
        # Run the YOLO model on each chunk
        output_chunk = run_model(
            cap=cap,
            yolo_path="model/yolov8s.pt",
            chunk=chunk,
            output_path="output_video_structured.avi",
            frame_width_height=width_length,
            conf_threshold=CONF_THRESHOLD,
        )

        # Initialize dictionaries for data processing
        dic_observed = {}
        raw_data = {}
        labels = []

        for i in range(len(output_chunk)):
            raw_data[i] = output_chunk[i]
            if raw_data[i]["class_name"] not in labels:
                labels.append(raw_data[i]["class_name"])

        # Check for dangerous items
        dic_hazard = check_dangerous_items(labels, dic_hazard)
        data = data_transforming(raw_data, dic_observed, dic_hazard)

        # Generate output for each chunk and display notification
        final_output = output(data, t)
        t += 1

        notification_placeholder.info(f"{final_output}")

        # Optional: add delay for model processing simulation
        # time.sleep(1)  # Adjust for real model latency if needed

    st.write("Video processing completed.")
    cap.release()
