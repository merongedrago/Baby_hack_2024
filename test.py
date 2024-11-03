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


# Display video in Streamlit without frame-by-frame processing


# Run YOLO model on the whole video file
# Replace `run_yolo_model` with the actual function for your model
# Example: results = run_yolo_model(tfile.name)
# For this example, simulate model output with a loop
notification_placeholder = st.empty()
cap = cv2.VideoCapture("test_mov.mp4")

chunks, width_length = split_video_into_memory_chunks(cap, chunk_duration=3)
dic_hazard = {}
t = 0
for chunk in chunks:

    output_chunk = run_model(
        cap=cap,
        yolo_path="model/yolov8s.pt",
        chunk=chunk,
        output_path="output_video_structured.avi",
        frame_width_height=width_length,
        conf_threshold=CONF_THRESHOLD,
    )
    dic_observed = {}

    raw_data = {}
    labels = []
    for i in range(len(output_chunk)):
        raw_data[i] = output_chunk[i]
        if raw_data[i]["class_name"] not in labels:
            labels.append(raw_data[i]["class_name"])

    dic_hazard = check_dangerous_items(labels, dic_hazard)
    data = data_transforming(raw_data, dic_observed, dic_hazard)

    print(data)
    output(data, t)
    t += 1
