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

# Set up logging
logger = logging.getLogger(__name__)

# Global variables for inference
latest_detections = []  # Store latest detections
user_prompts = ["croc", "jetson"]  # Default prompts
class_colors = {}  # Dictionary to store consistent colors for each class
frame_counter = 0  # Frame counter for running detections
last_results = None  # Store the most recent results for reuse
is_paused = False  # Flag to pause inference
show_segmentation = False  # Flag to toggle between bounding boxes and segmentation
last_inference_time = 0  # Track the last time inference was run
visual_prompts = None  # Store visual prompts (bboxes and classes)
is_on_visual_prompt = False  # Flag to pause detections when on the visual prompt page

# Define class-specific colors (BGR format for OpenCV)
POSITIVE_CLASSES = {"worker with helmet", "safety vest"}
NEGATIVE_CLASSES = {"worker without helmet"}
GREEN = (0, 255, 0)  # Positive color
RED = (0, 0, 255)    # Negative color
CONFIDENCE_THRESHOLD = 0.75  # Adjusted for YOLOE

# Model service URL
MODEL_SERVICE_URL = "http://localhost:8000"

def encode_image_to_base64(image):
    """Encode a PIL Image to base64 string."""
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def generate_frames(get_frame: Callable[[], tuple[bool, np.ndarray | None]]) -> Generator[bytes, None, None]:
    """Generator function to yield raw video frames without rendering."""
    global is_paused
    # Check if get_frame is callable
    if not inspect.isfunction(get_frame) and not inspect.ismethod(get_frame):
        logger.error(f"get_frame must be a callable function, got {type(get_frame)}: {get_frame}")
        raise TypeError(f"get_frame must be a callable function, got {type(get_frame)}: {get_frame}")
    
    frame_source = get_frame()  # Initialize the frame source (this is a generator)
    while True:
        try:
            # Get the next frame from the generator
            ret, frame = next(frame_source)
            
            if not ret or frame is None:
                logger.error("Could not retrieve frame in generate_frames")
                break

            # Encode raw frame as JPEG for streaming
            ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 50])  # Reduced JPEG quality
            if not ret:
                logger.error("Failed to encode frame as JPEG in generate_frames")
                break
            frame_bytes = buffer.tobytes()

            # Yield frame in the format expected by the browser
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            logger.debug("Sent video frame via /video_feed")

        except Exception as e:
            logger.error(f"Error in frame generation: %s", str(e))
            break

def process_detections(get_frame_func: Callable[[], tuple[bool, np.ndarray | None]]):
    """Background thread to run YOLOE inference and update detections."""
    global latest_detections, user_prompts, frame_counter, last_results, is_paused, show_segmentation, last_inference_time, visual_prompts, is_on_visual_prompt
    logger.info("Inside process_detections: Thread has started")
    retry_attempts = 5
    retry_delay = 1  # Seconds
    # Initialize the frame generator
    frame_gen = get_frame_func()
    while True:
        try:
            logger.debug("Detection thread loop iteration")
            if is_paused or not user_prompts or is_on_visual_prompt:
                logger.debug("Inference paused: is_paused=%s, user_prompts=%s, is_on_visual_prompt=%s", is_paused, user_prompts, is_on_visual_prompt)
                time.sleep(0.1)
                continue

            # Retry frame retrieval if it fails
            for attempt in range(retry_attempts):
                ret, frame = next(frame_gen)
                if ret and frame is not None:
                    break
                logger.warning(f"Could not retrieve frame in detection thread (attempt {attempt + 1}/{retry_attempts})")
                time.sleep(retry_delay)
            else:
                logger.error("Failed to retrieve frame after multiple attempts, skipping this iteration")
                time.sleep(1)
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
                    json={"image_base64": image_base64,
                            "user_prompts": user_prompts
                    }
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
            logger.error(f"Error in detection thread: {e}")
            time.sleep(1)  # Wait before retrying to avoid spamming errors