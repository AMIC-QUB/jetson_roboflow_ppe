import cv2
import os
from utils import app

def get_frame():
    # Get the video device index from an environment variable (default to 0)
    device_index = int(os.getenv('VIDEO_DEVICE_INDEX', 0))
    
    # Use the specified camera index
    cap = cv2.VideoCapture(device_index)
    if not cap.isOpened():
        print(f"Error: Could not open camera at index {device_index} (/dev/video{device_index}).")
        while True:
            yield False, None  # Always yield a 2-tuple, even on failure

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to read frame from camera.")
                yield False, None
            else:
                yield True, frame
    finally:
        cap.release()

# Set the frame function
app.config['GET_FRAME_FUNC'] = get_frame

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)