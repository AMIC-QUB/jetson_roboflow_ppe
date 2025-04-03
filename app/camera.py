import cv2
import os
import logging

# Set up logging
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

def release_camera():
    """Release the camera resource."""
    global cap
    if cap is not None and cap.isOpened():
        logger.debug("Releasing camera")
        cap.release()