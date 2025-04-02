import cv2
import os
from utils import app, process_detections
import threading
import time
import logging

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize the camera once
device_index = int(os.getenv('VIDEO_DEVICE_INDEX', 0))
cap = None
max_index_to_try = 5  # Try indices 0 to 4
for index in range(max_index_to_try):
    logger.debug(f"Attempting to open camera at index {index} (/dev/video{index})")
    cap = cv2.VideoCapture(index)
    if cap.isOpened():
        logger.info(f"Successfully opened camera at index {index} (/dev/video{index})")
        break
    else:
        logger.warning(f"Could not open camera at index {index} (/dev/video{index})")
        cap.release()
        cap = None

if cap is None or not cap.isOpened():
    logger.error(f"Failed to open any camera after trying indices 0 to {max_index_to_try-1}")

def get_frame():
    """Generator function to yield frames from the camera."""
    global cap
    if cap is None or not cap.isOpened():
        logger.error("Camera is not open")
        while True:
            yield False, None  # Always yield a 2-tuple, even on failure

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.error("Failed to read frame from camera")
                yield False, None
            else:
                yield True, frame
    except Exception as e:
        logger.error(f"Error in get_frame: {e}")
        yield False, None

# Set the frame function
app.config['GET_FRAME_FUNC'] = get_frame

if __name__ == "__main__":
    logger.info("Starting inference_usb.py")

    # Test the frame source
    logger.debug("Testing frame source")
    try:
        frame_gen = get_frame()
        ret, frame = next(frame_gen)
        if ret and frame is not None:
            logger.info(f"Frame source test successful, frame shape: {frame.shape}")
        else:
            logger.warning("Frame source test failed: Could not retrieve frame, proceeding anyway")
    except Exception as e:
        logger.warning(f"Frame source test failed: {e}, proceeding anyway")

    # Start the detection thread, passing the get_frame function
    logger.info("Starting detection thread from inference_usb.py")
    detection_thread = threading.Thread(target=process_detections, args=(get_frame,), daemon=True)
    detection_thread.start()
    logger.info("Detection thread started")
    
    # Verify the thread is running
    time.sleep(1)  # Give the thread a moment to start
    if detection_thread.is_alive():
        logger.info("Detection thread is running")
    else:
        logger.error("Detection thread failed to start or exited immediately")

    # Run the Flask app
    logger.info("Starting Flask app on port 5000")
    try:
        app.run(host="0.0.0.0", port=5000, debug=False)
    finally:
        # Ensure the camera is released when the app exits
        if cap is not None and cap.isOpened():
            logger.debug("Releasing camera on app exit")
            cap.release()