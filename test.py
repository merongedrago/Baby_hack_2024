import streamlit as st
import tempfile
import time
import cv2
from ultralytics import YOLO
from model.model import split_video_into_memory_chunks, run_model
from lib.obj_functions import data_transforming, output, circle_overlap_percentage
from lib.danger_detector import check_dangerous_items, get_completion_gemini
from dotenv import load_dotenv
import numpy as np

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
hazard = {}
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
    data = {}
    labels = []
    for i in range(len(output_chunk)):
        data[i] = output_chunk[i]
        if data[i]["class_name"] not in labels:
            labels.append(data[i]["class_name"])

    hazard = check_dangerous_items(labels, hazard)
    dic_observed = data_transforming(data, dic_observed, hazard)

    total = []

    result = output(data, t)
    total.append(total)

    t += 1
