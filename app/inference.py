import numpy as np
import cv2
import time
import logging
from typing import Callable, Generator
import inspect
from PIL import Image
import requests
import base64
import io
import json
from . import app
import supervision as sv
# Set up logging
logger = logging.getLogger(__name__)

# Global variables for inference
latest_detections = []  # Store latest detections
user_prompts = ["person", "spade", "hard hat", "digger", "machinery", "vest", "building", "robot", "truck", "car", "dirt"]  # Default prompts
class_colors = {}  # Dictionary to store consistent colors for each class
frame_counter = 0  # Frame counter for running detections
last_results = None  # Store the most recent results for reuse
is_paused = False  # Flag to pause inference
show_segmentation = False  # Flag to toggle between bounding boxes and segmentation
last_inference_time = 0  # Track the last time inference was run
visual_prompts = None  # Store visual prompts (bboxes and classes)
is_on_visual_prompt = False  # Flag to pause detections when on the visual prompt page
video_file_path = None  # Store the path to the uploaded video file
video_cap = None  # VideoCapture object for the video file

# Define class-specific colors (BGR format for OpenCV)
POSITIVE_CLASSES = {"worker with helmet", "safety vest"}
NEGATIVE_CLASSES = {"worker without helmet"}
GREEN = (0, 255, 0)  # Positive color
RED = (0, 0, 255)    # Negative color
CONFIDENCE_THRESHOLD = 0.25  # Adjusted for YOLOE

# Model service URL
MODEL_SERVICE_URL = "http://localhost:8000"

# Desired frame rate (24 FPS)
TARGET_FPS = 600
FRAME_INTERVAL = 1.0 / TARGET_FPS  # Time per frame in seconds (≈ 0.04167 seconds for 24 FPS)

def encode_image_to_base64(image):
    """Encode a PIL Image to base64 string."""
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def generate_frames(get_frame: Callable[[], tuple[bool, np.ndarray | None]]) -> Generator[bytes, None, None]:
    """Generator function to yield video frames with their detections at 24 FPS."""
    global is_paused, video_file_path, video_cap, latest_detections, user_prompts, is_on_visual_prompt, show_segmentation
    # Check if get_frame is callable
    if not inspect.isfunction(get_frame) and not inspect.ismethod(get_frame):
        logger.error(f"get_frame must be a callable function, got {type(get_frame)}: {get_frame}")
        raise TypeError(f"get_frame must be a callable function, got {type(get_frame)}: {get_frame}")

    # Determine the frame source
    if video_file_path and not is_on_visual_prompt:
        # Use video file if available and not on visual prompt page
        if video_cap is None or not video_cap.isOpened():
            logger.debug("Attempting to open video file: %s", video_file_path)
            video_cap = cv2.VideoCapture(video_file_path)
            if not video_cap.isOpened():
                logger.error("Could not open video file: %s", video_file_path)
                video_file_path = None  # Reset to fall back to webcam
                video_cap = None
            else:
                logger.debug("Using video file as frame source: %s", video_file_path)
                # Get the video's native frame rate
                native_fps = video_cap.get(cv2.CAP_PROP_FPS)
                logger.debug("Video native FPS: %s", native_fps)

    frame_count = 0
    start_time = time.time()
    while True:
        try:
            frame_start_time = time.time()

            # Read the frame
            if video_file_path and video_cap and video_cap.isOpened() and not is_on_visual_prompt:
                # Read frames from the video file
                ret, frame = video_cap.read()
                if not ret or frame is None:
                    # Loop back to the beginning of the video
                    logger.debug("Reached end of video, looping back to start")
                    video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = video_cap.read()
                    if not ret or frame is None:
                        logger.error("Could not read frame from video file after looping")
                        break
                logger.debug("Using video file frame for rendering and inference")
            else:
                # Fall back to webcam feed
                frame_source = get_frame()  # Initialize the frame source (this is a generator)
                ret, frame = next(frame_source)
                if not ret or frame is None:
                    logger.error("Could not retrieve frame in generate_frames")
                    break
                logger.debug("Using webcam frame for rendering and inference")

            # Resize frame to 640x480 for consistency
            frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)

            # Run inference on the frame
            detections = []
            start_time = time.time()
            if not is_paused and user_prompts and not is_on_visual_prompt:
                logger.debug("Running YOLOE inference on frame")
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                target_image = Image.fromarray(frame_rgb)
                image_base64 = encode_image_to_base64(target_image)
                response = requests.post(
                    f"{MODEL_SERVICE_URL}/predict",
                    json={"image_base64": image_base64, "user_prompts": user_prompts}
                )
                response_time = time.time()-start_time
                if response.status_code == 200:
                    detections = response.json()
                    latest_detections = detections  # Update global detections
                    logger.info(f"Inference completed, results: {len(detections)} objects detected")
                else:
                    logger.error(f"Failed to get inference results: {response.text}")
            else:
                # detections = latest_detections if latest_detections is not None else []
                logger.debug("Inference skipped: is_paused=%s, user_prompts=%s, is_on_visual_prompt=%s", is_paused, user_prompts, is_on_visual_prompt)
            try:
                sv_detections = sv.Detections(
                    xyxy=np.array(detections['xyxy'], dtype=np.float32),
                    mask=np.array(detections['mask'], dtype=bool) if detections['mask'] is not None else None,
                    confidence=np.array(detections['confidence'], dtype=np.float32),
                    class_id=np.array(detections['class_id'], dtype=np.int32),
                    data={'class_names': detections['class_names']} if user_prompts else {}
                )

                # labels = [
                #         f"{class_name}" for class_name in detections['class_names']
                # ]

                annotated_image = frame.copy()
                annotated_image = sv.ColorAnnotator().annotate(scene=annotated_image, detections=sv_detections)
                annotated_image = sv.BoxAnnotator().annotate(scene=annotated_image, detections=sv_detections)
                # annotated_image = sv.LabelAnnotator().annotate(scene=annotated_image, detections=sv_detections, labels=labels)
                annotated_image = sv.LabelAnnotator().annotate(scene=annotated_image, detections=sv_detections)
                ret, buffer = cv2.imencode('.jpg', annotated_image, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
                image_time = time.time()-response_time-start_time
            except Exception as o:
                logger.info(f"Error {o}")
                logger.info(detections)
                ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
                image_time = time.time()-response_time-start_time

            # Encode frame as JPEG for streaming

            if not ret:
                logger.error("Failed to encode frame as JPEG in generate_frames")
                continue
            frame_bytes = buffer.tobytes()

            # Yield frame and detections in a custom multipart format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n'
                   b'\r\n' + frame_bytes + b'\r\n')
            logger.debug("Sent video frame with detections via /video_feed")

            # Calculate elapsed time and adjust delay to maintain 24 FPS
            frame_count += 1
            elapsed_time = time.time() - frame_start_time
            # delay = max(0, FRAME_INTERVAL - elapsed_time)
            # time.sleep(delay)

            logger.info("response time: %s", response_time*1000)
            logger.info("image time: %s", image_time*1000)
            logger.info("total time: %s", elapsed_time*1000)
            # Log the actual frame rate every 24 frames (approximately every second at 24 FPS)
            if frame_count % 24 == 0:
                total_time = time.time() - start_time
                actual_fps = frame_count / total_time

                logger.info("Actual playback FPS: %s", actual_fps)
                frame_count = 0
                start_time = time.time()

        except Exception as e:
            logger.error(f"Error in frame generation: {e}")
            break

# Remove process_detections since inference is now handled in generate_frames
def process_detections(get_frame_func: Callable[[], tuple[bool, np.ndarray | None]]):
    pass