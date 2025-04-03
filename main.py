from app import app
from app.camera import get_frame, release_camera
from app.inference import process_detections
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

if __name__ == "__main__":
    logger.info("Starting main.py")

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
    logger.info("Starting detection thread from main.py")
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
        release_camera()