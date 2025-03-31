import cv2
import numpy as np
import requests
import base64
from flask import Flask, Response, render_template, jsonify, request,send_from_directory
from typing import Callable, Generator
from datetime import datetime, timedelta
import inspect
import psycopg2
import paho.mqtt.client as mqtt
import json
import os
import logging


app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)

ROBOFLOW_API_KEY = "ENZzcL3rs3i1hPuQqXSW"
ROBOFLOW_MODEL_URL = "http://roboflow-inference:9001/construction-safety-gsnvb/1"
MQTT_PORT = 1883
MQTT_TOPIC = "roi/detections"
mqtt_client = None
mqtt_connected = False
DB_HOST = os.getenv("DB_HOST", "postgres")
DB_NAME = os.getenv("DB_NAME", "detections_db")
DB_USER = os.getenv("DB_USER", "user")
DB_PASSWORD = os.getenv("DB_PASSWORD", "password")
DB_PORT = os.getenv("DB_PORT", "5432")

latest_detections = []
rois = []
roi_id_counter = 0

ROI_COLORS = [(0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255), (255, 165, 0)]
GREEN = (0, 255, 0)
RED = (0, 0, 255)
CONFIDENCE_THRESHOLD = 0.5
DETECTION_INTERVAL = 10

def get_db_connection():
    return psycopg2.connect(
        host=DB_HOST,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        port=DB_PORT
    )

def init_db():
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS detections
                 (id SERIAL PRIMARY KEY,
                  roi_id INTEGER,
                  roi_name TEXT,
                  detection_time TIMESTAMP,
                  image_path TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS rois
                 (id INTEGER PRIMARY KEY,
                  name TEXT,
                  coords_x1 INTEGER, coords_y1 INTEGER, coords_x2 INTEGER, coords_y2 INTEGER,
                  color_r INTEGER, color_g INTEGER, color_b INTEGER,
                  enabled BOOLEAN DEFAULT TRUE)''')
    conn.commit()
    conn.close()

def load_rois():
    global rois, roi_id_counter
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT id, name, coords_x1, coords_y1, coords_x2, coords_y2, color_r, color_g, color_b, enabled FROM rois")
    rows = c.fetchall()
    rois = [
        {
            "id": row[0], "name": row[1],
            "coords": [(row[2], row[3]), (row[4], row[5])],
            "color": (row[6], row[7], row[8]),
            "enabled": row[9],
            "detections": []  # In-memory for now
        }
        for row in rows
    ]
    roi_id_counter = max([row[0] for row in rows] + [0]) + 1 if rows else 1
    conn.close()
    app.logger.debug(f"Loaded ROIs: {rois}")

def save_roi(roi):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('''INSERT INTO rois (id, name, coords_x1, coords_y1, coords_x2, coords_y2, color_r, color_g, color_b, enabled)
                 VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                 ON CONFLICT (id) DO UPDATE SET
                 name = EXCLUDED.name,
                 coords_x1 = EXCLUDED.coords_x1, coords_y1 = EXCLUDED.coords_y1,
                 coords_x2 = EXCLUDED.coords_x2, coords_y2 = EXCLUDED.coords_y2,
                 color_r = EXCLUDED.color_r, color_g = EXCLUDED.color_g, color_b = EXCLUDED.color_b,
                 enabled = EXCLUDED.enabled''',
              (roi["id"], roi["name"],
               roi["coords"][0][0], roi["coords"][0][1], roi["coords"][1][0], roi["coords"][1][1],
               roi["color"][0], roi["color"][1], roi["color"][2], roi["enabled"]))
    conn.commit()
    conn.close()

def log_detection(roi_id, roi_name, detection_time, image_path):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("INSERT INTO detections (roi_id, roi_name, detection_time, image_path) VALUES (%s, %s, %s, %s)",
              (roi_id, roi_name, detection_time, image_path))
    conn.commit()
    conn.close()

def publish_mqtt(roi_id, roi_name, detection_time):
    global mqtt_client, mqtt_connected
    if mqtt_client and mqtt_connected:
        message = json.dumps({"roi_id": roi_id, "roi_name": roi_name, "time": detection_time})
        mqtt_client.publish(MQTT_TOPIC, message)

def box_in_rectangle(box: tuple[int, int, int, int], rect: tuple[tuple[int, int], tuple[int, int]]) -> bool:
    box_x1, box_y1, box_x2, box_y2 = box
    rect_p1, rect_p2 = rect
    rect_x1, rect_y1 = rect_p1
    rect_x2, rect_y2 = rect_p2
    return not (box_x2 < rect_x1 or box_x1 > rect_x2 or box_y2 < rect_y1 or box_y1 > rect_y2)

# ... (other imports and setup unchanged)

def generate_frames(get_frame: Callable[[], tuple[bool, np.ndarray | None]]) -> Generator[bytes, None, None]:
    global latest_detections
    if not inspect.isfunction(get_frame) and not inspect.ismethod(get_frame):
        raise TypeError(f"get_frame must be a callable function, got {type(get_frame)}: {get_frame}")
    
    frame_source = get_frame()
    while True:
        try:
            # Get frame
            if inspect.isgenerator(frame_source):
                result = next(frame_source)
            else:
                result = frame_source()
            
            if not isinstance(result, tuple) or len(result) != 2:
                app.logger.error(f"Error: get_frame() must return a 2-tuple (ret, frame), got {result}")
                continue
            ret, frame = result
            
            if not ret or frame is None:
                app.logger.error("Error: Could not retrieve frame from camera.")
                continue

            # Encode frame for Roboflow
            _, buffer = cv2.imencode('.jpg', frame)
            jpg_as_text = base64.b64encode(buffer).decode('utf-8')

            # Send to Roboflow
            try:
                response = requests.post(
                    f"{ROBOFLOW_MODEL_URL}?api_key={ROBOFLOW_API_KEY}",
                    data=jpg_as_text,
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                    timeout=5
                )
                response.raise_for_status()
            except requests.exceptions.RequestException as e:
                app.logger.error(f"Error contacting Roboflow API: {str(e)}")
                # Draw error message on the frame (we'll keep this as a fallback)
                cv2.putText(frame, "Roboflow API Error", (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                ret, buffer = cv2.imencode('.jpg', frame)
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                continue

            if response.status_code == 200:
                predictions = response.json()
                filtered_predictions = [pred for pred in predictions.get("predictions", []) 
                                      if pred.get("confidence", 0.0) >= CONFIDENCE_THRESHOLD and pred.get("class") == "person"]
                latest_detections = filtered_predictions

                current_time = datetime.now()
                filename_time_str = current_time.strftime("%Y-%m-%d_%H-%M-%S")
                db_time_str = current_time.strftime("%Y-%m-%d %H:%M:%S")
                for pred in filtered_predictions:
                    try:
                        x = int(pred.get("x", 0))
                        y = int(pred.get("y", 0))
                        w = int(pred.get("width", 0))
                        h = int(pred.get("height", 0))
                        box = (x - w // 2, y - h // 2, x + w // 2, y + h // 2)

                        in_roi = False
                        for roi in [r for r in rois if r["enabled"]]:
                            if box_in_rectangle(box, roi["coords"]):
                                in_roi = True
                                last_detection_time = (datetime.strptime(roi["detections"][-1]["time"], "%Y-%m-%d %H:%M:%S") 
                                                      if roi["detections"] else datetime.min)
                                time_diff = (current_time - last_detection_time).total_seconds()
                                if time_diff >= DETECTION_INTERVAL:
                                    try:
                                        image_filename = f"{roi['name']}_{filename_time_str}.jpg".replace(" ", "_")
                                        image_path = f"/app/detections/{image_filename}"
                                        os.makedirs(os.path.dirname(image_path), exist_ok=True)
                                        cv2.imwrite(image_path, frame)
                                    except Exception as e:
                                        app.logger.error(f"Error saving detection image: {str(e)}")
                                        image_path = ""
                                    roi["detections"].append({"time": db_time_str})
                                    log_detection(roi["id"], roi["name"], db_time_str, image_path)
                                    publish_mqtt(roi["id"], roi["name"], db_time_str)
                                    print(f"Detection logged for {roi['name']} at {db_time_str}")
                                break
                    except (ValueError, TypeError) as e:
                        app.logger.error(f"Error processing prediction: {pred}, {e}")
                        continue

            # Send the raw frame without drawing
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                app.logger.error("Error encoding frame to JPEG")
                continue
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        except Exception as e:
            app.logger.error(f"Error in frame generation: {str(e)}")
            if frame is None:
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "Stream Error", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            ret, buffer = cv2.imencode('.jpg', frame)
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# ... (rest of the file unchanged)
@app.route('/')
def index():
    global mqtt_connected
    load_rois()  # Refresh ROIs from DB
    return render_template('index.html', mqtt_connected=mqtt_connected, rois=rois)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(app.config['GET_FRAME_FUNC']), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detections')
def get_detections():
    load_rois()
    detections_with_roi = []
    current_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for pred in latest_detections:
        detection = pred.copy()
        detection["in_roi"] = []
        try:
            box = (int(pred["x"] - pred["width"] / 2), int(pred["y"] - pred["height"] / 2),
                   int(pred["x"] + pred["width"] / 2), int(pred["y"] + pred["height"] / 2))
            for roi in [r for r in rois if r["enabled"]]:
                if box_in_rectangle(box, roi["coords"]):
                    detection["in_roi"].append({"name": roi["name"], "time": current_time_str})
        except (ValueError, TypeError) as e:
            print(f"Error in detections endpoint: {e}")
        detections_with_roi.append(detection)
    return jsonify({"detections": detections_with_roi, "rois": rois})

@app.route('/add_roi', methods=['POST'])
def add_roi():
    global rois, roi_id_counter
    load_rois()
    data = request.get_json()
    start = data['start']
    end = data['end']
    coords = [(int(start['x']), int(start['y'])), (int(end['x']), int(end['y']))]
    roi = {
        "id": roi_id_counter,
        "name": f"ROI {roi_id_counter}",
        "coords": coords,
        "color": ROI_COLORS[roi_id_counter % len(ROI_COLORS)],
        "enabled": True,
        "detections": []
    }
    rois.append(roi)
    save_roi(roi)
    roi_id_counter += 1
    return jsonify({"status": "ROI added", "roi": roi})

@app.route('/update_roi_name', methods=['POST'])
def update_roi_name():
    data = request.get_json()
    roi_id = data['id']
    new_name = data['name']
    load_rois()
    for roi in rois:
        if roi["id"] == roi_id:
            roi["name"] = new_name
            save_roi(roi)
            break
    return jsonify({"status": "ROI name updated", "id": roi_id, "name": new_name})

@app.route('/toggle_roi', methods=['POST'])
def toggle_roi():
    data = request.get_json()
    roi_id = data['id']
    enabled = data['enabled']
    load_rois()
    for roi in rois:
        if roi["id"] == roi_id:
            roi["enabled"] = enabled
            save_roi(roi)
            break
    return jsonify({"status": "ROI toggled", "id": roi_id, "enabled": enabled})

@app.route('/delete_roi', methods=['POST'])
def delete_roi():
    global rois
    data = request.get_json()
    roi_id = data['id']
    load_rois()  # Ensure in-memory list is up-to-date
    rois = [roi for roi in rois if roi["id"] != roi_id]
    
    # Delete from database (both rois and associated detections)
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("DELETE FROM detections WHERE roi_id = %s", (roi_id,))
    c.execute("DELETE FROM rois WHERE id = %s", (roi_id,))
    conn.commit()
    conn.close()
    
    app.logger.debug(f"Deleted ROI {roi_id} and its detections")
    return jsonify({"status": "ROI deleted", "id": roi_id})

@app.route('/reset_db', methods=['POST'])
def reset_db():
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("TRUNCATE TABLE detections")
    conn.commit()
    conn.close()
    app.logger.debug("Database reset: detections table truncated")
    return jsonify({"status": "Database reset"})

@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    try:
        app.logger.debug("Entering dashboard route")
        conn = get_db_connection()
        c = conn.cursor()
        app.logger.debug("Executing SQL query")
        # Check if image_path exists, fall back if not
        c.execute("SELECT column_name FROM information_schema.columns WHERE table_name = 'detections' AND column_name = 'image_path'")
        has_image_path = c.fetchone() is not None
        if has_image_path:
            c.execute("SELECT roi_id, roi_name, detection_time, image_path FROM detections ORDER BY detection_time DESC")
        else:
            c.execute("SELECT roi_id, roi_name, detection_time, NULL AS image_path FROM detections ORDER BY detection_time DESC")
        rows = c.fetchall()
        app.logger.debug(f"Fetched {len(rows)} rows: {rows}")
        conn.close()

        # Group by ROI
        roi_groups = {}
        all_roi_names = set()
        for row in rows:
            roi_id = row[0]
            roi_name = row[1] if row[1] is not None else f"Unnamed ROI {roi_id}"
            detection_time = row[2]
            image_path = row[3]  # May be None
            all_roi_names.add(roi_name)
            if roi_id not in roi_groups:
                roi_groups[roi_id] = {"name": roi_name, "detections": []}
                app.logger.debug(f"Initialized roi_groups[{roi_id}] = {{'name': '{roi_name}', 'detections': []}}")
            roi_groups[roi_id]["detections"].append({"time": detection_time, "image": image_path})

        # Filter by selected ROIs
        selected_rois = request.form.getlist('rois') if request.method == 'POST' else request.args.getlist('rois')
        if not selected_rois or 'all' in selected_rois:
            selected_rois = list(all_roi_names)
        app.logger.debug(f"Selected ROIs: {selected_rois}")

        filtered_roi_groups = {k: v for k, v in roi_groups.items() if v["name"] in selected_rois}
        app.logger.debug(f"Filtered roi_groups: {filtered_roi_groups}")

        # Prepare data for template
        detections_by_roi = [
            {"roi_id": roi_id, "roi_name": group["name"], 
             "detections": [{"time": d["time"].strftime("%Y-%m-%d %H:%M:%S"), "image": d["image"] or ""} for d in group["detections"]]}
            for roi_id, group in filtered_roi_groups.items()
        ]
        app.logger.debug(f"Prepared detections_by_roi: {detections_by_roi}")

        # Detection counts per ROI for bar chart
        chart_labels = [entry["roi_name"] for entry in detections_by_roi]
        chart_data = [len(entry["detections"]) for entry in detections_by_roi]
        app.logger.debug(f"Chart labels: {chart_labels}, Chart data: {chart_data}")

        # Stats: Per Hour, Per Day (date), Per Week
        hour_stats = {}
        day_stats = {}
        week_stats = {}
        total_hourly = {str(i): 0 for i in range(24)}
        total_daily = {}
        total_weekly = {}

        for roi_id, group in filtered_roi_groups.items():
            roi_name = group["name"]
            hour_stats[roi_name] = {str(i): 0 for i in range(24)}
            day_stats[roi_name] = {}
            week_stats[roi_name] = {}

            for d in group["detections"]:
                dt = d["time"]
                hour_key = str(dt.hour)
                hour_stats[roi_name][hour_key] += 1
                total_hourly[hour_key] += 1

                day_key = dt.strftime("%Y-%m-%d")
                day_stats[roi_name][day_key] = day_stats[roi_name].get(day_key, 0) + 1
                total_daily[day_key] = total_daily.get(day_key, 0) + 1

                week_key = dt.strftime("%Y-W%W")
                week_stats[roi_name][week_key] = week_stats[roi_name].get(week_key, 0) + 1
                total_weekly[week_key] = total_weekly.get(week_key, 0) + 1

        total_hourly_sum = sum(total_hourly.values())
        total_daily_sum = sum(total_daily.values())
        total_weekly_sum = sum(total_weekly.values())

        hour_labels = [str(i) for i in range(24)]
        day_labels = sorted(total_daily.keys())
        week_labels = sorted(total_weekly.keys())

        hour_datasets = [
            {"label": roi_name, "data": [hour_stats[roi_name][label] for label in hour_labels], "backgroundColor": f"rgba({i*50 % 255}, {i*100 % 255}, {i*150 % 255}, 0.6)"}
            for i, roi_name in enumerate(hour_stats.keys())
        ]
        day_datasets = [
            {"label": roi_name, "data": [day_stats[roi_name].get(label, 0) for label in day_labels], "backgroundColor": f"rgba({i*50 % 255}, {i*100 % 255}, {i*150 % 255}, 0.6)"}
            for i, roi_name in enumerate(day_stats.keys())
        ]
        week_datasets = [
            {"label": roi_name, "data": [week_stats[roi_name].get(label, 0) for label in week_labels], "backgroundColor": f"rgba({i*50 % 255}, {i*100 % 255}, {i*150 % 255}, 0.6)"}
            for i, roi_name in enumerate(week_stats.keys())
        ]

        return render_template(
            'dashboard.html',
            detections_by_roi=detections_by_roi,
            chart_labels=chart_labels,
            chart_data=chart_data,
            hour_stats=hour_stats,
            day_stats=day_stats,
            week_stats=week_stats,
            hour_labels=hour_labels,
            day_labels=day_labels,
            week_labels=week_labels,
            hour_datasets=hour_datasets,
            day_datasets=day_datasets,
            week_datasets=week_datasets,
            total_hourly=total_hourly,
            total_daily=total_daily,
            total_weekly=total_weekly,
            total_hourly_sum=total_hourly_sum,
            total_daily_sum=total_daily_sum,
            total_weekly_sum=total_weekly_sum,
            roi_options=sorted(list(all_roi_names)),
            selected_rois=selected_rois
        )
    except Exception as e:
        app.logger.error(f"Error in dashboard route: {str(e)}")
        return "Internal Server Error: " + str(e), 500
    
@app.route('/images/<path:filename>')
def serve_image(filename):
    try:
        return send_from_directory('/app/detections', filename)
    except Exception as e:
        app.logger.error(f"Error serving image {filename}: {str(e)}")
        return "Image not found", 404
# Initialize database and load ROIs on startup
init_db()
load_rois()