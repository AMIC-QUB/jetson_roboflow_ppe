import pyrealsense2 as rs
import cv2
import numpy as np
import requests
import base64
from flask import Flask, Response, render_template

# Initialize Flask app
app = Flask(__name__)

# Initialize Roboflow (replace with your API key and model details)
ROBOFLOW_API_KEY = "YOUR_API_KEY"
ROBOFLOW_MODEL_URL = "http://localhost:9001/YOUR_MODEL/YOUR_VERSION"

# Configure RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # RGB stream at 640x480, 30 FPS

# Start streaming
pipeline.start(config)

def generate_frames():
    """Generator function to yield video frames with detections."""
    while True:
        try:
            # Wait for frames
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()

            if not color_frame:
                print("Error: Could not retrieve frame.")
                continue

            # Convert to numpy array
            frame = np.asanyarray(color_frame.get_data())

            # Encode frame as JPEG and then to base64 for Roboflow
            _, buffer = cv2.imencode('.jpg', frame)
            jpg_as_text = base64.b64encode(buffer).decode('utf-8')

            # Send frame to Roboflow inference server
            response = requests.post(
                f"{ROBOFLOW_MODEL_URL}?api_key={ROBOFLOW_API_KEY}",
                data=jpg_as_text,
                headers={"Content-Type": "application/x-www-form-urlencoded"}
            )

            # Process response and draw detections
            if response.status_code == 200:
                predictions = response.json()
                for pred in predictions.get("predictions", []):
                    x, y, w, h = pred["x"], pred["y"], pred["width"], pred["height"]
                    label = pred["class"]
                    confidence = pred["confidence"]
                    cv2.rectangle(frame, (x - w//2, y - h//2), (x + w//2, y + h//2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{label} ({confidence:.2f})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Encode frame as JPEG for streaming
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()

            # Yield frame in the format expected by the browser
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        except Exception as e:
            print(f"Error in frame generation: {e}")
            break

@app.route('/')
def index():
    """Render the main web page."""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Stream the video feed with detections."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    # Run Flask app on host 0.0.0.0 (accessible from network) and port 5000
    app.run(host='0.0.0.0', port=5000, threaded=True)