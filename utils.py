import cv2
import numpy as np
from flask import Flask, Response, render_template, jsonify
from ultralytics import YOLOE
from typing import Callable, Generator
import inspect
import threading
import logging
import sys
import time

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG for more detailed output
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)  # Ensure logs go to stdout
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Load YOLOE model
try:
    model = YOLOE("yoloe-11l-seg.pt")  # Load the pre-downloaded model
    model.to('cuda')  # Ensure the model runs on GPU
    logger.info("YOLOE model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load YOLOE model: {e}")
    raise

# Global variables
latest_detections = []  # Store latest detections
user_prompts = ["worker with helmet", "worker without helmet", "safety vest", "heavy machinery"]  # Default prompts
class_colors = {}  # Dictionary to store consistent colors for each class
frame_counter = 0  # Frame counter for running detections
last_results = None  # Store the most recent results for reuse
is_paused = False  # Flag to pause inference
show_segmentation = False  # Flag to toggle between bounding boxes and segmentation
get_frame_func = None  # Store the get_frame function for the detection thread
last_inference_time = 0  # Track the last time inference was run

# Define class-specific colors (BGR format for OpenCV)
POSITIVE_CLASSES = {"worker with helmet", "safety vest"}
NEGATIVE_CLASSES = {"worker without helmet"}
GREEN = (0, 255, 0)  # Positive color
RED = (0, 0, 255)    # Negative color
CONFIDENCE_THRESHOLD = 0.3  # Adjusted for YOLOE

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
            logger.error(f"Error in frame generation: {e}")
            break

# Background thread to process detections
# Remove this line from the global variables
# get_frame_func = None  # Store the get_frame function for the detection thread

# Background thread to process detections

# Background thread to process detections
def process_detections(get_frame_func: Callable[[], tuple[bool, np.ndarray | None]]):
    """Background thread to run YOLOE inference and update detections."""
    global latest_detections, user_prompts, frame_counter, last_results, is_paused, show_segmentation, last_inference_time
    logger.info("Inside process_detections: Thread has started")
    retry_attempts = 5
    retry_delay = 1  # Seconds
    # Initialize the frame generator
    frame_gen = get_frame_func()
    while True:
        try:
            logger.debug("Detection thread loop iteration")
            if is_paused or not user_prompts:
                logger.debug("Inference paused or no prompts, skipping detection")
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

            # Resize frame for faster inference (e.g., 640x640)
            inference_frame = cv2.resize(frame, (640, 640))

            # Run YOLOE inference every 100ms (10Hz)
            if time.time() - last_inference_time >= 0.1:  # 100ms interval
                logger.debug("Running YOLOE inference")
                model.set_classes(user_prompts, model.get_text_pe(user_prompts))
                results = model.predict(inference_frame, conf=CONFIDENCE_THRESHOLD)
                last_results = results  # Store the latest results
                last_inference_time = time.time()  # Update the last inference time
                logger.debug(f"Inference completed, results: {len(results)} objects detected")
            else:
                results = last_results if last_results is not None else []  # Use the last results if available

            # Process detections
            filtered_predictions = []
            for r in results:
                boxes = r.boxes
                masks = r.masks if show_segmentation and hasattr(r, 'masks') else None  # Get masks if available
                for i, box in enumerate(boxes):
                    # Extract bounding box coordinates and labels
                    xyxy = box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
                    conf = box.conf.cpu().numpy().item()  # Convert to scalar using .item()
                    cls = int(box.cls.cpu().numpy())  # Class index
                    label = user_prompts[cls]  # Map class index to prompt

                    # Scale bounding box coordinates back to original frame size
                    orig_h, orig_w = frame.shape[:2]
                    infer_h, infer_w = inference_frame.shape[:2]
                    scale_x, scale_y = orig_w / infer_w, orig_h / infer_h
                    xyxy[0] *= scale_x  # x1
                    xyxy[1] *= scale_y  # y1
                    xyxy[2] *= scale_x  # x2
                    xyxy[3] *= scale_y  # y2

                    # Prepare mask data if in segmentation mode
                    mask_data = None
                    if show_segmentation and masks is not None:
                        mask = masks[i].data[0].cpu().numpy()  # Get the mask for this detection
                        mask = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)  # Resize mask to original frame size
                        mask = mask.astype(np.uint8)  # Convert to binary mask (0 or 1)
                        mask_data = mask.tolist()  # Convert to list for JSON serialization

                    # Create prediction dictionary
                    pred = {
                        "x": int((xyxy[0] + xyxy[2]) / 2),  # Center x
                        "y": int((xyxy[1] + xyxy[3]) / 2),  # Center y
                        "width": int(xyxy[2] - xyxy[0]),
                        "height": int(xyxy[3] - xyxy[1]),
                        "x1": int(xyxy[0]),  # Top-left x
                        "y1": int(xyxy[1]),  # Top-left y
                        "x2": int(xyxy[2]),  # Bottom-right x
                        "y2": int(xyxy[3]),  # Bottom-right y
                        "class": label,
                        "confidence": conf,
                        "color": GREEN if label in POSITIVE_CLASSES else RED if label in NEGATIVE_CLASSES else class_colors.setdefault(label, (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))),
                        "mask": mask_data  # Include mask data if in segmentation mode
                    }
                    filtered_predictions.append(pred)

            latest_detections = filtered_predictions  # Update global detections
            logger.debug(f"Updated detections: {len(latest_detections)} objects detected")

        except Exception as e:
            logger.error(f"Error in detection thread: {e}")
            time.sleep(1)  # Wait before retrying to avoid spamming errors

@app.route('/')
def index():
    """Render the main web page."""
    logger.debug("Serving / endpoint")
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Stream the video feed with raw frames."""
    logger.debug("Serving /video_feed endpoint")
    return Response(generate_frames(app.config['GET_FRAME_FUNC']), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detections')
def get_detections():
    """Return the latest detections as JSON."""
    logger.debug("Serving /detections endpoint")
    return jsonify({"detections": latest_detections})

@app.route('/update_prompts/<prompts>')
def update_prompts(prompts):
    """Update the user prompts for YOLOE detection."""
    global user_prompts, last_results
    user_prompts = prompts.split(",")  # Split comma-separated prompts
    user_prompts = [p.strip() for p in user_prompts]  # Clean up whitespace
    last_results = None  # Reset last_results to avoid index mismatches
    logger.info(f"Updated prompts: {user_prompts}")
    return jsonify({"status": "success", "prompts": user_prompts})

@app.route('/clear_prompts')
def clear_prompts():
    """Clear the user prompts and toggle the pause state."""
    global user_prompts, is_paused, last_results
    user_prompts = []  # Clear prompts
    last_results = None  # Reset last_results to avoid index mismatches
    is_paused = not is_paused  # Toggle pause state
    logger.info(f"Prompts cleared, pause state: {is_paused}")
    return jsonify({"status": "success", "paused": is_paused, "prompts": user_prompts})

@app.route('/toggle_display_mode')
def toggle_display_mode():
    """Toggle between showing bounding boxes and segmentation masks."""
    global show_segmentation
    show_segmentation = not show_segmentation