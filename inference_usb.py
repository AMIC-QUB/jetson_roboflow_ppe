import cv2
import os
from utils import app
import psycopg2
import time
from psycopg2 import OperationalError
import requests

def wait_for_postgres():
    db_host = os.getenv("DB_HOST", "postgres")
    db_name = os.getenv("DB_NAME", "detections_db")
    db_user = os.getenv("DB_USER", "user")
    db_password = os.getenv("DB_PASSWORD", "password")
    db_port = os.getenv("DB_PORT", "5432")
    
    while True:
        try:
            conn = psycopg2.connect(
                host=db_host,
                database=db_name,
                user=db_user,
                password=db_password,
                port=db_port
            )
            conn.close()
            print("PostgreSQL is ready!")
            break
        except OperationalError as e:
            print(f"Waiting for PostgreSQL to be ready... ({e})")
            time.sleep(1)

def wait_for_roboflow():
    roboflow_url = "http://roboflow-inference:9001"  # Match updated ROBOFLOW_MODEL_URL base
    while True:
        try:
            response = requests.get(roboflow_url)
            if response.status_code in [200, 404]:
                print("Roboflow inference server is ready!")
                break
        except requests.ConnectionError as e:
            print(f"Waiting for Roboflow inference server... ({e})")
            time.sleep(1)

def get_frame():
    device_index = int(os.getenv('VIDEO_DEVICE_INDEX', 0))
    cap = cv2.VideoCapture(device_index)
    if not cap.isOpened():
        print(f"Error: Could not open camera at index {device_index} (/dev/video{device_index}).")
        while True:
            yield False, None

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

app.config['GET_FRAME_FUNC'] = get_frame

if __name__ == "__main__":
    wait_for_postgres()
    wait_for_roboflow()
    app.run(host="0.0.0.0", port=5000, debug=False)  # Back to 5000