# Inside flask_app/inference.py (or wherever frame grabbing happens)
import cv2
import os
import time
import logging

logger = logging.getLogger(__name__)

# Get the RTSP stream URL from environment variable
# This will be set by Docker Compose to point to the rtsp-server service
RTSP_STREAM_URL = os.environ.get("RTSP_STREAM_URL", "rtsp://localhost:8554/webcam") # Default for local testing

def get_video_capture():
    """Opens the video source (RTSP stream or potentially file later)."""
    logger.info(f"Attempting to connect to video stream: {RTSP_STREAM_URL}")
    # Add options for better RTSP handling with OpenCV
    # Using FFMPEG backend and TCP transport can improve reliability
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
    cap = cv2.VideoCapture(RTSP_STREAM_URL, cv2.CAP_FFMPEG)

    if not cap.isOpened():
        logger.error(f"Error: Could not open video stream: {RTSP_STREAM_URL}")
        return None
    logger.info(f"Successfully connected to video stream: {RTSP_STREAM_URL}")
    return cap

# Modify your background inference thread (run_inference_loop)
def run_inference_loop(): # Removed get_frame_func argument
    global latest_detections, inference_running, stop_inference_event
    global latest_annotated_frame # Make sure this is accessible

    cap = None
    retry_delay = 5 # seconds between connection retries

    while not stop_inference_event.is_set():
        if cap is None or not cap.isOpened():
            logger.warning(f"Stream not open. Attempting connection to {RTSP_STREAM_URL}...")
            cap = get_video_capture()
            if cap is None:
                logger.error(f"Connection failed. Retrying in {retry_delay} seconds...")
                stop_inference_event.wait(retry_delay) # Wait before retrying
                continue
            else:
                 logger.info("Stream connection successful.")

        try:
            frame_start_time = time.time()

            # --- Get Frame ---
            ret, frame = cap.read() # Read frame from RTSP stream

            if not ret or frame is None:
                logger.warning("Failed to grab frame from RTSP stream. Connection might be lost.")
                cap.release() # Release potentially broken connection
                cap = None # Trigger reconnect attempt in next loop iteration
                time.sleep(1) # Small pause before retry
                continue

            # --- Existing logic ---
            frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)
            latest_raw_frame_for_annotation = frame.copy() # Store raw frame

            # Check if inference should run (periodically)
            # ... (your existing logic using is_paused, is_on_visual_prompt, user_prompts, timer) ...
            should_run_inference = ... # Your condition here

            current_detections = None
            if should_run_inference:
                 # --- Call FastAPI ---
                 try:
                      # ... (prepare image, call API endpoint /predict_text) ...
                      response = requests.post(f"{MODEL_SERVICE_URL}/predict_text", ...) # Use correct URL
                      # ... (parse response using parse_detections_from_response) ...
                      sv_dets = parse_detections_from_response(response.json())
                      latest_detections.append(sv_dets)
                      current_detections = sv_dets
                 except Exception as e:
                     logger.error(f"[InferenceThread] API/Inference error: {e}", exc_info=True)
                     latest_detections.append(None) # Clear on error
            else:
                 if latest_detections:
                     current_detections = latest_detections[-1]

            # --- Annotate Frame ---
            annotated_frame = annotate_frame(latest_raw_frame_for_annotation, current_detections)
            latest_annotated_frame.append(annotated_frame) # Update shared frame for streamer

            time.sleep(0.01) # Yield CPU

        except Exception as e:
            logger.error(f"[InferenceThread] Unhandled error in loop: {e}", exc_info=True)
            if cap:
                cap.release() # Release on error
            cap = None # Force reconnect attempt
            time.sleep(retry_delay) # Wait before continuing

    # Cleanup
    if cap and cap.isOpened():
        cap.release()
    logger.info("[InferenceThread] Exiting.")
    inference_running = False

# Make sure generate_frames_for_stream remains the same, yielding from latest_annotated_frame

# Update start_inference_thread - no longer needs get_frame_func
def start_inference_thread():
    global inference_running, _inference_thread, stop_inference_event
    if not inference_running:
        stop_inference_event.clear()
         # Pass NO arguments to the target function now
        _inference_thread = threading.Thread(target=run_inference_loop, daemon=True)
        _inference_thread.start()
        inference_running = True
        logger.info("Inference thread started (using RTSP stream).")
    # ... rest of start/stop logic ...

# Remove or adapt any routes/logic that directly tried to access get_frame
# e.g., the /visual_prompt route needs a different way to get a static frame.
# Option 1: Have it call the inference service with a dummy request to get a recent frame?
# Option 2: Store the latest raw frame globally and use that? (Simpler)
# Assuming option 2: Ensure latest_raw_frame_for_annotation is accessible by the route.