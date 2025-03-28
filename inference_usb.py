import cv2
from utils import app

# Open USB camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open USB camera.")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

def get_usb_frame():
    """Get a frame from the USB camera."""
    return cap.read()

app.config['GET_FRAME_FUNC'] = get_usb_frame

if __name__ == "__main__":
    try:
        app.run(host='0.0.0.0', port=5000, threaded=True)
    finally:
        cap.release()