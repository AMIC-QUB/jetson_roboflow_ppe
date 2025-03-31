import cv2
import numpy as np
import requests
import base64
from flask import Flask, Response, render_template, jsonify
from typing import Callable, Generator
import inspect

# Initialize Flask app
app = Flask(__name__)

# Initialize Roboflow (replace with your API key and model details)
ROBOFLOW_API_KEY = "ENZzcL3rs3i1hPuQqXSW"
ROBOFLOW_MODEL_URL = "http://localhost:9001/construction-safety-gsnvb/1"

# Global variable to store latest detections
latest_detections = []

# Define class-specific colors (BGR format for OpenCV)
POSITIVE_CLASSES = {"helmet", "vest", "person"}
NEGATIVE_CLASSES = {"no-helmet", "no-vest"}
GREEN = (0, 255, 0)  # Positive color
RED = (0, 0, 255)    # Negative color
CONFIDENCE_THRESHOLD = 0.75  # Minimum confidence to display predictions

def generate_frames(get_frame: Callable[[], tuple[bool, np.ndarray | None]]) -> Generator[bytes, None, None]:
    """Generator function to yield video frames with detections from a provided frame source."""
    global latest_detections
    # Check if get_frame is callable
    if not inspect.isfunction(get_frame) and not inspect.ismethod(get_frame):
        raise TypeError(f"get_frame must be a callable function, got {type(get_frame)}: {get_frame}")
    
    frame_source = get_frame()  # Initialize the frame source
    while True:
        try:
            # Handle generator case
            if inspect.isgenerator(frame_source):
                result = next(frame_source)
            else:
                result = frame_source()
            
            # Ensure result is a 2-tuple
            if not isinstance(result, tuple) or len(result) != 2:
                print(f"Error: get_frame() must return a 2-tuple (ret, frame), got {result}")
                break
            ret, frame = result
            
            if not ret or frame is None:
                print("Error: Could not retrieve frame.")
                break

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
                filtered_predictions = [pred for pred in predictions.get("predictions", []) 
                                      if pred.get("confidence", 0.0) >= CONFIDENCE_THRESHOLD]
                latest_detections = filtered_predictions  # Update global detections with filtered list
                for pred in filtered_predictions:
                    try:
                        # Extract and validate bounding box coordinates
                        x = int(pred.get("x", 0))  # Center x
                        y = int(pred.get("y", 0))  # Center y
                        w = int(pred.get("width", 0))
                        h = int(pred.get("height", 0))
                        label = pred.get("class", "unknown")
                        confidence = pred.get("confidence", 0.0)

                        # Determine color based on class
                        color = GREEN if label in POSITIVE_CLASSES else RED if label in NEGATIVE_CLASSES else (255, 255, 255)  # White for unknown

                        # Calculate top-left and bottom-right corners
                        pt1 = (x - w // 2, y - h // 2)
                        pt2 = (x + w // 2, y + h // 2)

                        # Draw rectangle and label
                        cv2.rectangle(frame, pt1, pt2, color, 2)
                        cv2.putText(frame, f"{label} ({confidence:.2f})", (x, y - 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                    except (ValueError, TypeError) as e:
                        print(f"Error processing prediction: {pred}, {e}")
                        continue

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
    return Response(generate_frames(app.config['GET_FRAME_FUNC']), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detections')
def get_detections():
    """Return the latest detections as JSON."""
    return jsonify({"detections": latest_detections})