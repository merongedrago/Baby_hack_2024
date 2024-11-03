import cv2
from ultralytics import YOLO
import os

# from moviepy.editor import VideoFileClip, concatenate_videoclips

CUSTOM_FRAME_RATE = 32
CONF_THRESHOLD = 0.5
EXPANSION_RATE = 0.3


def split_video_into_memory_chunks(cap, chunk_duration):
    # Open video file
    # cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    chunk_frames = int(fps * chunk_duration)  # Frames per chunk

    chunks = []  # List to hold all chunks
    current_chunk = []  # List to hold frames for the current chunk
    cnt = 1
    # Loop through frames and add them to chunks in memory
    while True:
        ret, frame_val = cap.read()

        frame = {"frame_data": frame_val, "frame_id": cnt}

        if not ret:
            # If we're at the end of the video, add any remaining frames to chunks
            if current_chunk:
                chunks.append(current_chunk)
            break

        current_chunk.append(frame)

        # When reaching the chunk frame limit, store the chunk and reset
        if len(current_chunk) == chunk_frames:
            chunks.append(current_chunk)
            current_chunk = []
        cnt += 1
    width_height = (int(cap.get(3)), int(cap.get(4)))
    # cap.release()
    print(f"Video split into {len(chunks)} chunks in memory")
    return chunks, width_height  # (width, height)
    return chunks, width_height  # (width, height)


def run_model(cap, yolo_path, chunk, output_path, frame_width_height, conf_threshold):

    model = YOLO(yolo_path)
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(output_path, fourcc, CUSTOM_FRAME_RATE, frame_width_height)
    output_matrix = []

    # Get frame dimensions for normalization
    # Get frame dimensions for normalization
    for frame_data in chunk:
        frame = frame_data.get("frame_data")

        frame_height, frame_width = frame.shape[:2]

        # Run inference
        results = model(frame, conf=conf_threshold)
        # Process detections
        for result in results:
            for bbox in result.boxes:  # Get bounding boxes
                x1, y1, x2, y2 = bbox.xyxy[0]  # Get coordinates
                conf = bbox.conf.item()  # Confidence score as a float
                cls = int(bbox.cls.item())  # Class index

                if conf > conf_threshold:  # Filter by confidence

                    # Calculate width and height of the bounding box
                    width = x2 - x1
                    height = y2 - y1

                    # Calculate normalized coordinates and dimensions
                    normalized_x = x1 / frame_width
                    normalized_y = y1 / frame_height
                    normalized_width = width / frame_width
                    normalized_height = height / frame_height

                    # Check if the detected object is a person
                    if model.names[cls] == "person":
                        # Calculate expanded width and height
                        expand_width = normalized_width * (1 + EXPANSION_RATE)
                        expand_height = normalized_height * (1 + EXPANSION_RATE)

                        # Adjust normalized x and y coordinates
                        expand_x = normalized_x - (
                            (expand_width - normalized_width) / 2
                        )
                        expand_y = normalized_y - (
                            (expand_height - normalized_height) / 2
                        )
                        expand_x = normalized_x - (
                            (expand_width - normalized_width) / 2
                        )
                        expand_y = normalized_y - (
                            (expand_height - normalized_height) / 2
                        )

                        # Ensure bounding box stays within frame boundaries
                        norm_x = max(0, expand_x)
                        norm_y = max(0, expand_y)
                        norm_width = min(1, expand_width)
                        norm_height = min(1, expand_height)
                    else:
                        # Use the original normalized coordinates and dimensions for non-person objects
                        norm_x = normalized_x
                        norm_y = normalized_y
                        norm_width = normalized_width
                        norm_height = normalized_height

                    # Store information in the output matrix
                    output_matrix.append(
                        {
                            "frame": frame_data.get("frame_id"),  # Current frame number
                            "class_name": model.names[cls],  # Object name
                            "norm_x": float(norm_x),  # Normalized X coordinate
                            "norm_y": float(norm_y),  # Normalized Y coordinate
                            "norm_width": float(norm_width),  # Normalized Width
                            "norm_height": float(norm_height),  # Normalized Height
                            # 'confidence': float(conf)  # Confidence score
                        }
                    )
                    output_matrix.append(
                        {
                            "frame": frame_data.get("frame_id"),  # Current frame number
                            "class_name": model.names[cls],  # Object name
                            "norm_x": float(norm_x),  # Normalized X coordinate
                            "norm_y": float(norm_y),  # Normalized Y coordinate
                            "norm_width": float(norm_width),  # Normalized Width
                            "norm_height": float(norm_height),  # Normalized Height
                            # 'confidence': float(conf)  # Confidence score
                        }
                    )

                    # Draw bounding box on the frame
                    color = (0, 255, 0)  # Use green color for the box
                    cv2.rectangle(
                        frame,
                        (int(norm_x * frame_width), int(norm_y * frame_height)),
                        (
                            int((norm_x + norm_width) * frame_width),
                            int((norm_y + norm_height) * frame_height),
                        ),
                        color,
                        2,
                    )
                    cv2.rectangle(
                        frame,
                        (int(norm_x * frame_width), int(norm_y * frame_height)),
                        (
                            int((norm_x + norm_width) * frame_width),
                            int((norm_y + norm_height) * frame_height),
                        ),
                        color,
                        2,
                    )

                    # Prepare detailed label text
                    label = (
                        f"{model.names[cls]}, "
                        f"Norm X: {norm_x:.2f}, Norm Y: {norm_y:.2f}, "
                        f"Norm W: {norm_width:.2f}, Norm H: {norm_height:.2f}"
                    )
                    label = (
                        f"{model.names[cls]}, "
                        f"Norm X: {norm_x:.2f}, Norm Y: {norm_y:.2f}, "
                        f"Norm W: {norm_width:.2f}, Norm H: {norm_height:.2f}"
                    )

                    # Display text above the bounding box
                    cv2.putText(
                        frame,
                        label,
                        (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        2,
                    )
                    cv2.putText(
                        frame,
                        label,
                        (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        2,
                    )

    # Write the frame to output video
    out.write(frame)

    return output_matrix


# def concatavi():
#     output_dir = "outputvid"

#     all_avi_files = []
#     clips = []

#     for vid in os.listdir(output_dir):
#         file_path = os.path.join(output_dir, vid)
#         if os.path.isfile(file_path):
#             all_avi_files.append(file_path)

#     # Load each AVI file into a VideoFileClip object
#     for file in all_avi_files:
#         clip = VideoFileClip(file)
#         clips.append(clip)

#     final_clip = concatenate_videoclips(clips)
#     final_clip.write_videofile("final_output.avi", codec="libx264")

def concatavi():
    dir = "outputvid"  # Directory containing the AVI files
    output_file = "final_output.avi"
    # Collect all .avi files from the specified directory
    video_files = [
        os.path.join(dir, file) for file in os.listdir(dir) if file.endswith(".avi")
    ]
    # Check if there are any video files to process
    if not video_files:
        print("No video files found in the specified directory.")
        return
    video_caps = [cv2.VideoCapture(file) for file in video_files]
    # Get properties from the first video to define codec and frame size
    first_video = video_caps[0]
    fps = first_video.get(cv2.CAP_PROP_FPS)
    width = int(first_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(first_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # Create a VideoWriter object to write the merged video
    fourcc = cv2.VideoWriter_fourcc(*"XVID")  # Codec for the output video
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
    # Read frames from each video and write to output
    for cap in video_caps:
        while True:
            ret, frame = cap.read()
            if not ret:
                break  # Break if there are no frames left
            # Resize frame to ensure it matches the output video dimensions
            frame = cv2.resize(
                frame, (width, height)
            )  # Resize frame to (width, height)
            out.write(frame)  # Write the resized frame to the output video
        cap.release()  # Release the video capture object
    out.release()  # Release the VideoWriter
    print(f"Merged video saved as: {output_file}")





if __name__ == "__main__":
    video_path = "istockphoto-163887417-640_adpp_is.mp4"
    cap = cv2.VideoCapture(video_path)

    chunks, width_length = split_video_into_memory_chunks(cap, chunk_duration=0.0625)

    for index, chunk in enumerate(chunks):

        output_chunk = run_model(
            cap=cap,
            yolo_path="yolov8s.pt",
            chunk=chunk,
            output_path=f"outputvid/output_video_structured_{index}.avi",
            frame_width_height=width_length,
            conf_threshold=CONF_THRESHOLD,
        )

    # print(output_chunk)
    cap.release()

    concatavi()
