# ... (imports) ...
import threading
from collections import deque
import os

# Set up logging
logger = logging.getLogger(__name__)

# --- Configuration (Move to Flask app config later) ---

# Get the FastAPI service URL from environment variable, with a default
MODEL_SERVICE_URL = os.environ.get("MODEL_SERVICE_URL", "http://localhost:8000")
logger.info(f"Using Model Service URL: {MODEL_SERVICE_URL}")


# --- State Management (Using globals - be mindful of concurrency) ---
latest_detections = deque(maxlen=1) # Use deque for thread-safe(ish) single item storage
latest_annotated_frame = deque(maxlen=1)
user_prompts = ["person"]
is_paused = False
show_segmentation = False # Flag (implement usage or remove)
visual_prompts = None
is_on_visual_prompt = False
video_file_path = None
video_cap = None
inference_running = False # Flag to control inference loop
stop_inference_event = threading.Event() # To signal the inference thread to stop

# --- Annotators (Initialize once) ---
box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()
trace_annotator = sv.TraceAnnotator()
# mask_annotator = sv.MaskAnnotator() # Uncomment and use if show_segmentation is implemented

# --- Helper Functions ---
def encode_image_to_base64(image: Image.Image) -> str:
    """Encode a PIL Image to base64 string."""
    buffered = io.BytesIO()
    # Use PNG for potentially better quality if needed, or keep JPEG for size
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def parse_detections_from_response(response_json: dict) -> Optional[sv.Detections]:
    """Parses the list-of-dicts response from the FastAPI service."""
    detections_list = response_json.get("detections", [])
    if not detections_list:
        return None

    # Extract data into lists for sv.Detections
    xyxy = []
    masks = []
    confidence = []
    class_id = []
    tracker_id = []
    class_names = [] # Assuming class_name is in each detection dict

    has_masks = False
    has_tracker_ids = False

    for det in detections_list:
        xyxy.append(det.get("xyxy", [0,0,0,0])) # Default if missing
        confidence.append(det.get("confidence", 0.0))
        class_id.append(det.get("class_id", -1))
        class_names.append(det.get("class_name", "N/A"))
        if "mask" in det and det["mask"] is not None:
            masks.append(det["mask"])
            has_masks = True
        if "tracker_id" in det and det["tracker_id"] is not None:
            tracker_id.append(det["tracker_id"])
            has_tracker_ids = True

    # Create sv.Detections object
    sv_dets = sv.Detections(
        xyxy=np.array(xyxy, dtype=np.float32),
        mask=np.array(masks) if has_masks and len(masks) == len(xyxy) else None, # Ensure mask consistency
        confidence=np.array(confidence, dtype=np.float32),
        class_id=np.array(class_id, dtype=np.int32),
        # Store class names directly if needed later, Supervision might infer them
        # data={'class_names': class_names}
    )

    # Add tracker_id if present and consistent
    if has_tracker_ids and len(tracker_id) == len(xyxy):
        sv_dets.tracker_id = np.array(tracker_id, dtype=np.int32)

    return sv_dets

def annotate_frame(frame: np.ndarray, detections: Optional[sv.Detections]) -> np.ndarray:
    """Annotates a frame with the given detections."""
    if detections is None or len(detections) == 0:
        return frame # Return original frame if no detections

    annotated_image = frame.copy()
    try:
        # Base annotations
        annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections)

        # Generate labels (adjust format as needed)
        labels = [
            f"#{det_idx} {detections.data['class_names'][det_idx]} {detections.confidence[det_idx]:0.2f}"
            if 'class_names' in detections.data and len(detections.data['class_names']) == len(detections)
            else f"#{det_idx} ID:{detections.class_id[det_idx]} {detections.confidence[det_idx]:0.2f}"
            for det_idx in range(len(detections))
        ]
        # Add tracker ID to label if available
        if detections.tracker_id is not None:
             labels = [
                 f"T:{detections.tracker_id[i]} {labels[i]}"
                 for i in range(len(detections))
             ]

        annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)

        if detections.tracker_id is not None:
            annotated_image = trace_annotator.annotate(scene=annotated_image, detections=detections)

        # Optional: Segmentation Mask Annotation
        # if show_segmentation and detections.mask is not None:
        #     annotated_image = mask_annotator.annotate(scene=annotated_image, detections=detections)

    except Exception as e:
        logger.error(f"Error during frame annotation: {e}", exc_info=True)
        return frame # Return original frame on annotation error
    return annotated_image


# --- Background Inference Thread ---
def run_inference_loop(get_frame_func: Callable[[], tuple[bool, np.ndarray | None]]):
    """Background thread function to continuously run inference."""
    global latest_detections, inference_running, video_cap, video_file_path, user_prompts, is_paused, is_on_visual_prompt

    # Initialize frame source (camera or video)
    current_video_path = video_file_path
    if current_video_path:
        logger.info(f"[InferenceThread] Using video file: {current_video_path}")
        cap = cv2.VideoCapture(current_video_path)
        if not cap.isOpened():
            logger.error(f"[InferenceThread] Failed to open video: {current_video_path}. Stopping thread.")
            inference_running = False
            return
    else:
        logger.info("[InferenceThread] Using webcam.")
        # Assuming get_frame_func returns a generator for webcam
        frame_source = get_frame_func()
        cap = None # Indicate webcam usage

    last_inference_call_time = 0
    inference_interval = 0.5 # Run inference every 0.5 seconds (adjust as needed)

    while not stop_inference_event.is_set():
        try:
            frame_start_time = time.time()

            # --- Get Frame ---
            ret, frame = None, None
            if cap: # Reading from video file
                ret, frame = cap.read()
                if not ret or frame is None:
                    logger.info("[InferenceThread] Video ended, looping.")
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = cap.read()
                    if not ret: # Handle case where looping fails immediately
                         logger.error("[InferenceThread] Failed to read frame after looping video.")
                         time.sleep(0.1)
                         continue
            else: # Reading from webcam generator
                 try:
                     ret, frame = next(frame_source)
                 except StopIteration:
                     logger.error("[InferenceThread] Webcam stream ended unexpectedly.")
                     break # Exit loop if webcam stops
                 if not ret:
                     logger.warning("[InferenceThread] Failed to get frame from webcam.")
                     time.sleep(0.1) # Wait briefly before retrying
                     continue

            if frame is None:
                logger.warning("[InferenceThread] Got None frame, skipping.")
                time.sleep(0.1)
                continue

            frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)
            # Store the latest raw frame for annotation by the streaming thread
            latest_annotated_frame.append(frame) # Temporarily store raw, will be replaced by annotated

            # --- Run Inference Periodically ---
            should_run_inference = (
                not is_paused and
                not is_on_visual_prompt and
                user_prompts and
                (time.time() - last_inference_call_time > inference_interval)
            )

            current_detections = None # Holds results from *this* iteration's inference call

            if should_run_inference:
                last_inference_call_time = time.time()
                logger.debug("[InferenceThread] Running inference...")
                try:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    target_image = Image.fromarray(frame_rgb)
                    image_base64 = encode_image_to_base64(target_image)

                    # --- Call CORRECT FastAPI Endpoint ---
                    response = requests.post(
                        f"{MODEL_SERVICE_URL}/predict_text", # Use text prediction endpoint
                        json={"image_base64": image_base64},
                        timeout=5 # Add a timeout
                    )
                    response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

                    response_data = response.json()
                    # --- Parse CORRECT Response Format ---
                    sv_dets = parse_detections_from_response(response_data)
                    latest_detections.append(sv_dets) # Store the latest valid detections
                    current_detections = sv_dets # Use for annotation *this* cycle
                    logger.debug(f"[InferenceThread] Inference successful, {len(sv_dets) if sv_dets else 0} detections.")

                except requests.exceptions.RequestException as e:
                    logger.error(f"[InferenceThread] API request failed: {e}")
                    latest_detections.append(None) # Clear detections on error
                except Exception as e:
                    logger.error(f"[InferenceThread] Inference processing error: {e}", exc_info=True)
                    latest_detections.append(None) # Clear detections on error
            else:
                 # If not running inference, keep using the last known detections
                 if latest_detections:
                      current_detections = latest_detections[-1]


            # --- Annotate Frame ---
            # Use detections from this cycle if inference ran, otherwise use last known
            frame_to_annotate = frame # Use the raw frame captured this cycle
            annotated_frame = annotate_frame(frame_to_annotate, current_detections)
            latest_annotated_frame.append(annotated_frame) # Update the shared annotated frame


            # --- Control Loop Speed (Optional - prevents busy-waiting) ---
            # Remove the TARGET_FPS logic, just prevent overly tight loop
            time.sleep(0.01) # Small sleep to yield CPU

        except Exception as e:
            logger.error(f"[InferenceThread] Unhandled error in loop: {e}", exc_info=True)
            time.sleep(1) # Avoid rapid looping on persistent error

    # Cleanup
    if cap:
        cap.release()
    logger.info("[InferenceThread] Exiting.")
    inference_running = False

# --- Frame Generation for Streaming ---
def generate_frames_for_stream() -> Generator[bytes, None, None]:
    """Generator that yields the latest annotated frame bytes."""
    logger.info("Starting frame streaming generator.")
    while True:
        if not latest_annotated_frame:
            # Wait briefly if no frame is available yet
            time.sleep(0.02)
            continue

        frame = latest_annotated_frame[-1] # Get the most recent annotated frame

        # Encode frame as JPEG for streaming
        ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70]) # Quality 70
        if not ret:
            logger.error("Failed to encode frame as JPEG in generate_frames_for_stream")
            time.sleep(0.02)
            continue

        frame_bytes = buffer.tobytes()

        # Yield frame in multipart format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n'
               b'\r\n' + frame_bytes + b'\r\n')
        # Minimal sleep to allow other threads/requests to run
        time.sleep(0.01) # Adjust if needed, prevents hogging CPU


# --- Control Functions ---
_inference_thread = None

def start_inference_thread(get_frame_func):
    global inference_running, _inference_thread, stop_inference_event
    if not inference_running:
        stop_inference_event.clear()
        _inference_thread = threading.Thread(target=run_inference_loop, args=(get_frame_func,), daemon=True)
        _inference_thread.start()
        inference_running = True
        logger.info("Inference thread started.")
    else:
        logger.warning("Inference thread already running.")

def stop_inference_thread():
    global inference_running, _inference_thread, stop_inference_event
    if inference_running and _inference_thread is not None:
        logger.info("Stopping inference thread...")
        stop_inference_event.set()
        _inference_thread.join(timeout=5) # Wait for thread to finish
        if _inference_thread.is_alive():
             logger.warning("Inference thread did not stop gracefully.")
        inference_running = False
        _inference_thread = None
        latest_detections.clear() # Clear state when stopped
        latest_annotated_frame.clear()
        logger.info("Inference thread stopped.")
    else:
        logger.info("Inference thread not running or already stopped.")

# Remove the empty process_detections function
# def process_detections(...):
#    pass