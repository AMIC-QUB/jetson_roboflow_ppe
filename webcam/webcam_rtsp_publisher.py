# webcam_rtsp_publisher.py
import cv2
import subprocess
import time
import os
import logging
import sys

# --- Configuration ---
# These will usually be overridden by environment variables in Docker Compose
RTSP_URL = os.environ.get("RTSP_PUBLISH_URL", "rtsp://localhost:8554/webcam")
CAMERA_DEVICE_INDEX = int(os.environ.get("CAMERA_DEVICE_INDEX", 0))
FRAME_WIDTH = int(os.environ.get("FRAME_WIDTH", 640))
FRAME_HEIGHT = int(os.environ.get("FRAME_HEIGHT", 480))
FPS = int(os.environ.get("FPS", 20)) # Target FPS for publishing

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("WebcamRTSPPublisher")

def main():
    logger.info(f"Attempting to open camera device index: {CAMERA_DEVICE_INDEX}")
    cap = cv2.VideoCapture(CAMERA_DEVICE_INDEX)

    if not cap.isOpened():
        logger.error(f"Error: Could not open video device {CAMERA_DEVICE_INDEX}.")
        sys.exit(1)

    # Set camera properties (optional, might not be supported by all cameras)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS)

    # Get actual dimensions (camera might not support requested ones)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    if actual_fps == 0:
        logger.warning(f"Camera reported 0 FPS, using target FPS {FPS} for FFmpeg.")
        actual_fps = FPS

    logger.info(f"Camera opened successfully: {width}x{height} @ {actual_fps:.2f} FPS (target {FPS} FPS)")
    logger.info(f"Publishing stream to: {RTSP_URL}")

    # --- FFmpeg Command ---
    # Command to push raw video frames read from stdin to the RTSP server
    command = [
        'ffmpeg',
        '-re',  # Read input at native frame rate (-re is important for streaming)
        # Input options - raw video frames from stdin
        '-f', 'rawvideo',
        '-pix_fmt', 'bgr24',  # Pixel format from OpenCV
        '-s', f'{width}x{height}',  # Dimensions of input frames
        '-r', str(actual_fps), # Input frame rate
        '-i', '-',  # Input comes from stdin

        # Output options - encode to H.264 and publish via RTSP
        '-c:v', 'libx264',    # Video codec H.264
        '-pix_fmt', 'yuv420p', # Common pixel format for H.264
        '-preset', 'ultrafast', # Encoding speed (tradeoff for quality/latency)
        '-tune', 'zerolatency', # Optimize for low latency
        '-b:v', '1500k',      # Target video bitrate (adjust as needed)
        '-maxrate', '2000k',  # Max video bitrate
        '-bufsize', '3000k',  # Encoding buffer size
        '-g', str(int(actual_fps * 2)), # Keyframe interval (e.g., every 2 seconds)
        '-an',                # No audio
        '-f', 'rtsp',         # Output format RTSP
        # '-rtsp_transport', 'tcp', # Use TCP (more reliable over lossy networks) - optional
        RTSP_URL              # The destination RTSP URL
    ]

    logger.info(f"Starting FFmpeg command: {' '.join(command)}")

    # Start FFmpeg process
    process = subprocess.Popen(command, stdin=subprocess.PIPE)

    frame_count = 0
    start_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Failed to grab frame from camera. Retrying...")
                time.sleep(0.1)
                # Optional: Try reopening camera if it fails consistently
                # cap.release()
                # cap = cv2.VideoCapture(CAMERA_DEVICE_INDEX)
                # if not cap.isOpened():
                #    logger.error("Failed to reopen camera. Exiting.")
                #    break
                continue

            # Write frame to FFmpeg's stdin
            try:
                process.stdin.write(frame.tobytes())
                frame_count += 1
            except BrokenPipeError:
                logger.error("FFmpeg process pipe broken. FFmpeg might have crashed.")
                break
            except Exception as e:
                 logger.error(f"Error writing to FFmpeg stdin: {e}")
                 # Might indicate FFmpeg has exited, break loop
                 break

            # Log FPS occasionally
            if frame_count % (int(actual_fps) * 5) == 0: # Log every 5 seconds approx
                 elapsed = time.time() - start_time
                 fps_calc = frame_count / elapsed if elapsed > 0 else 0
                 logger.info(f"Published {frame_count} frames. Avg FPS: {fps_calc:.2f}")


            # Optional small sleep to prevent 100% CPU if camera reading is too fast
            # time.sleep(1 / (actual_fps * 1.5)) # Sleep slightly less than frame interval

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        logger.info("Cleaning up...")
        if cap.isOpened():
            cap.release()
            logger.info("Camera released.")
        if process and process.stdin:
            process.stdin.close()
        if process:
            process.wait() # Wait for FFmpeg to exit
            logger.info(f"FFmpeg process exited with code: {process.returncode}")

if __name__ == "__main__":
    main()