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
from . import app
import queue
import threading

# Set up logging
logger = logging.getLogger(__name__)

# Global variables for inference
latest_detections = []  # Store latest detections
user_prompts = ["person", "hard hat", "spade", "digger", "excavator", "high vis vest", "shovel"]  # Default prompts
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

# Thread-safe queue for sharing frames between generate_frames and process_detections
frame_queue = queue.Queue(maxsize=10)  # Buffer up to 10 frames
inference_thread_running = False  # Flag to control the inference thread
inference_thread = None  # Thread for process_detections

# Desired frame rate (24 FPS)
TARGET_FPS = 30
FRAME_INTERVAL = 1.0 / TARGET_FPS  # Time per frame in seconds (≈ 0.04167 seconds for 24 FPS)

# Define class-specific colors (BGR format for OpenCV)
POSITIVE_CLASSES = {"worker with helmet", "safety vest"}
NEGATIVE_CLASSES = {"worker without helmet"}
GREEN = (0, 255, 0)  # Positive color
RED = (0, 0, 255)    # Negative color
CONFIDENCE_THRESHOLD = 0.3  # Adjusted for YOLOE

# Model service URL
MODEL_SERVICE_URL = "http://localhost:8000"

def encode_image_to_base64(image):
    """Encode a PIL Image to base64 string."""
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def process_detections():
    """Background thread to run YOLOE inference on frames from the queue."""
    global latest_detections, user_prompts, frame_counter, last_results, is_paused, show_segmentation, last_inference_time, visual_prompts, is_on_visual_prompt, inference_thread_running
    logger.info("Inside process_detections: Thread has started")

    while inference_thread_running:
        try:
            logger.debug("Inference thread loop iteration")
            if is_paused or not user_prompts or is_on_visual_prompt:
                logger.debug("Inference paused: is_paused=%s, user_prompts=%s, is_on_visual_prompt=%s", is_paused, user_prompts, is_on_visual_prompt)
                time.sleep(0.1)
                continue

            # Get the next frame from the queue
            try:
                frame = frame_queue.get(timeout=1.0)  # Blocking get with timeout
                logger.debug("Retrieved frame from queue for inference")
            except queue.Empty:
                logger.warning("Frame queue is empty, waiting for frames")
                continue

            # Convert the frame to a PIL Image (no resizing)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            target_image = Image.fromarray(frame_rgb)

            # Run YOLOE inference every 100ms (10Hz)
            if time.time() - last_inference_time >= 0.1:  # 100ms interval
                logger.debug("Running YOLOE inference")
                # Encode the image to base64
                image_base64 = encode_image_to_base64(target_image)

                # Send request to the model service
                response = requests.post(
                    f"{MODEL_SERVICE_URL}/predict",
                    json={"image_base64": image_base64, "user_prompts": user_prompts}
                )
                if response.status_code != 200:
                    logger.error(f"Failed to get inference results: {response.text}")
                    continue

                results = response.json().get("detections", [])
                last_results = results  # Store the latest results
                last_inference_time = time.time()  # Update the last inference time
                logger.info(f"Inference completed, results: {len(results)} objects detected")
            else:
                results = last_results if last_results is not None else []  # Use the last results if available

            # Update latest detections
            latest_detections = results

        except Exception as e:
            logger.error(f"Error in process_detections: {e}")
            time.sleep(1)  # Wait before retrying to avoid spamming errors

def start_inference_thread():
    """Start the inference thread."""
    global inference_thread_running, inference_thread
    if not inference_thread_running:
        inference_thread_running = True
        inference_thread = threading.Thread(target=process_detections)
        inference_thread.daemon = True  # Daemon thread will exit when the main program exits
        inference_thread.start()
        logger.debug("Inference thread started")

def stop_inference_thread():
    """Stop the inference thread."""
    global inference_thread_running, inference_thread
    if inference_thread_running:
        inference_thread_running = False
        if inference_thread is not None:
            inference_thread.join()
            inference_thread = None
        # Clear the frame queue
        while not frame_queue.empty():
            try:
                frame_queue.get_nowait()
            except queue.Empty:
                break
        logger.debug("Inference thread stopped and frame queue cleared")

def generate_frames(get_frame: Callable[[], tuple[bool, np.ndarray | None]]) -> Generator[bytes, None, None]:
    """Generator function to yield video frames from webcam or video file at 24 FPS."""
    global is_paused, video_file_path, video_cap
    # Check if get_frame is callable
    if not inspect.isfunction(get_frame) and not inspect.ismethod(get_frame):
        logger.error(f"get_frame must be a callable function, got {type(get_frame)}: {get_frame}")
        raise TypeError(f"get_frame must be a callable function, got {type(get_frame)}: {get_frame}")

    # Start the inference thread if not already running
    start_inference_thread()

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
                logger.debug("Using video file frame for rendering")
            else:
                # Fall back to webcam feed
                frame_source = get_frame()  # Initialize the frame source (this is a generator)
                ret, frame = next(frame_source)
                if not ret or frame is None:
                    logger.error("Could not retrieve frame in generate_frames")
                    break
                logger.debug("Using webcam frame for rendering")

            # Resize frame to 640x480 for consistency
            frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)

            # Pass the frame to process_detections via the queue
            try:
                frame_queue.put(frame, timeout=0.1)  # Non-blocking put with timeout
                logger.debug("Frame added to queue for inference, queue size: %s", frame_queue.qsize())
            except queue.Full:
                logger.warning("Frame queue is full, dropping frame for inference")
                # Remove the oldest frame to make space
                try:
                    frame_queue.get_nowait()
                    frame_queue.put(frame, timeout=0.1)
                except queue.Empty:
                    pass

            # Encode frame as JPEG for streaming
            ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
            if not ret:
                logger.error("Failed to encode frame as JPEG in generate_frames")
                break
            frame_bytes = buffer.tobytes()

            # Yield frame in the format expected by the browser
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            logger.debug("Sent video frame via /video_feed")

            # Calculate elapsed time and adjust delay to maintain 24 FPS
            frame_count += 1
            elapsed_time = time.time() - frame_start_time
            delay = max(0, FRAME_INTERVAL - elapsed_time)
            time.sleep(delay)

            # Log the actual frame rate every 24 frames (approximately every second at 24 FPS)
            if frame_count % 24 == 0:
                total_time = time.time() - start_time
                actual_fps = frame_count / total_time
                logger.debug("Actual playback FPS: %s", actual_fps)
                frame_count = 0
                start_time = time.time()

        except Exception as e:
            logger.error(f"Error in frame generation: {e}")
            break