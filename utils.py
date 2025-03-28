import cv2
import numpy as np
import requests
import base64
from flask import Flask, Response, render_template, jsonify, request
from typing import Callable, Generator
from datetime import datetime
# Initialize Flask app
app = Flask(__name__)

# Initialize Roboflow (replace with your API key and model details)
ROBOFLOW_API_KEY = "ENZzcL3rs3i1hPuQqXSW"
ROBOFLOW_MODEL_URL = "http://localhost:9001/construction-safety-gsnvb/1"

# Global variables
latest_detections = []
rois = []  # List of dicts: {"id": int, "name": str, "coords": [(x1, y1), (x2, y2)], "color": (r, g, b), "detections": [{"time": str}]}
roi_id_counter = 0

# Colors (BGR format for OpenCV)
ROI_COLORS = [
    (0, 255, 0),   # Green
    (255, 0, 0),   # Blue
    (0, 255, 255), # Yellow
    (255, 0, 255), # Magenta
    (255, 165, 0)  # Orange
]
GREEN = (0, 255, 0)
RED = (0, 0, 255)
CONFIDENCE_THRESHOLD = 0.5
DETECTION_INTERVAL = 5  # Seconds between detections

def box_in_rectangle(box: tuple[int, int, int, int], rect: tuple[tuple[int, int], tuple[int, int]]) -> bool:
    """Check if a bounding box overlaps with a rectangle."""
    box_x1, box_y1, box_x2, box_y2 = box
    rect_p1, rect_p2 = rect
    rect_x1, rect_y1 = rect_p1
    rect_x2, rect_y2 = rect_p2
    return not (box_x2 < rect_x1 or box_x1 > rect_x2 or box_y2 < rect_y1 or box_y1 > rect_y2)

def generate_frames(get_frame: Callable[[], tuple[bool, np.ndarray | None]]) -> Generator[bytes, None, None]:
    """Generator function to yield video frames with ROIs and person detections."""
    global latest_detections, rois
    while True:
        try:
            ret, frame = get_frame()
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
                                      if pred.get("confidence", 0.0) >= CONFIDENCE_THRESHOLD and pred.get("class") == "person"]
                latest_detections = filtered_predictions

                # Draw ROIs as rectangles
                for roi in rois:
                    cv2.rectangle(frame, roi["coords"][0], roi["coords"][1], roi["color"], 2)
                    cv2.putText(frame, roi["name"], (roi["coords"][0][0], roi["coords"][0][1] - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.9, roi["color"], 2)

                # Draw person detections and check for ROI overlap
                current_time = datetime.now()
                current_time_str = current_time.strftime("%Y-%m-%d %H:%M:%S")
                for pred in filtered_predictions:
                    try:
                        x = int(pred.get("x", 0))
                        y = int(pred.get("y", 0))
                        w = int(pred.get("width", 0))
                        h = int(pred.get("height", 0))
                        label = pred.get("class", "person")
                        confidence = pred.get("confidence", 0.0)

                        pt1 = (x - w // 2, y - h // 2)
                        pt2 = (x + w // 2, y + h // 2)
                        box = (pt1[0], pt1[1], pt2[0], pt2[1])

                        in_roi = False
                        for roi in rois:
                            if box_in_rectangle(box, roi["coords"]):
                                in_roi = True
                                last_detection_time = (datetime.strptime(roi["detections"][-1]["time"], "%Y-%m-%d %H:%M:%S") 
                                                      if roi["detections"] else datetime.min)
                                time_diff = (current_time - last_detection_time).total_seconds()
                                if time_diff >= DETECTION_INTERVAL:
                                    roi["detections"].append({"time": current_time_str})
                                    print(f"Detection logged for {roi['name']} at {current_time_str}, time_diff: {time_diff}")
                        color = RED if in_roi else GREEN

                        cv2.rectangle(frame, pt1, pt2, color, 2)
                        cv2.putText(frame, f"{label} ({confidence:.2f})", (x, y - 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                    except (ValueError, TypeError) as e:
                        print(f"Error processing prediction: {pred}, {e}")
                        continue

            # Encode frame as JPEG for streaming
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()

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
    """Return the latest detections and ROI data as JSON."""
    detections_with_roi = []
    current_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for pred in latest_detections:
        detection = pred.copy()
        detection["in_roi"] = []
        try:
            box = (int(pred["x"] - pred["width"] / 2), int(pred["y"] - pred["height"] / 2),
                   int(pred["x"] + pred["width"] / 2), int(pred["y"] + pred["height"] / 2))
            for roi in rois:
                if box_in_rectangle(box, roi["coords"]):
                    detection["in_roi"].append({"name": roi["name"], "time": current_time_str})
        except (ValueError, TypeError) as e:
            print(f"Error in detections endpoint: {e}")
        detections_with_roi.append(detection)
    return jsonify({"detections": detections_with_roi, "rois": rois})

@app.route('/add_roi', methods=['POST'])
def add_roi():
    """Add a new ROI based on client coordinates."""
    global rois, roi_id_counter
    data = request.get_json()
    start = data['start']
    end = data['end']
    coords = [(int(start['x']), int(start['y'])), (int(end['x']), int(end['y']))]
    roi = {
        "id": roi_id_counter,
        "name": f"ROI {roi_id_counter}",
        "coords": coords,
        "color": ROI_COLORS[roi_id_counter % len(ROI_COLORS)],
        "detections": []
    }
    rois.append(roi)
    roi_id_counter += 1
    return jsonify({"status": "ROI added", "roi": roi})

@app.route('/update_roi_name', methods=['POST'])
def update_roi_name():
    """Update the name of an existing ROI."""
    data = request.get_json()
    roi_id = data['id']
    new_name = data['name']
    for roi in rois:
        if roi["id"] == roi_id:
            roi["name"] = new_name
            break
    return jsonify({"status": "ROI name updated", "id": roi_id, "name": new_name})

@app.route('/delete_roi', methods=['POST'])
def delete_roi():
    """Delete an ROI by ID."""
    global rois
    data = request.get_json()
    roi_id = data['id']
    rois = [roi for roi in rois if roi["id"] != roi_id]
    return jsonify({"status": "ROI deleted", "id": roi_id})