# Real-Time Construction Site PPE Detection (YOLOE)

This project provides a web-based system for real-time monitoring of Personal Protective Equipment (PPE) compliance on construction sites using the YOLOE object detection model. It features both text-based and interactive visual prompting for defining objects of interest.

**Core Components:**

1.  **FastAPI Inference Service:** Hosts the YOLOE model(s) and performs efficient object detection inference via API endpoints. Supports separate models for text and visual prompts.
2.  **Flask Web Server:** Serves the frontend application, manages video input (webcam or file), handles user interactions (setting prompts), and communicates with the FastAPI service for inference results.
3.  **React Frontend:** A modern, interactive user interface built with React, allowing users to view the annotated video stream, manage prompts, and interact with the visual prompting feature.

## Features

*   **Real-Time Detection:** Processes video streams (webcam or uploaded file) to detect objects.
*   **YOLOE Model:** Leverages the powerful YOLOE model for object detection/segmentation.
*   **Text Prompts:** Define target object classes using text input (e.g., "person, hard hat, safety vest").
*   **Visual Prompts:** Interactively draw bounding boxes on a frame to define specific objects visually.
*   **Annotated Stream:** Displays the video feed with bounding boxes, class labels, confidence scores, and optional tracking IDs overlaid.
*   **Web Interface:** User-friendly interface built with React for controlling prompts and viewing results.
*   **Decoupled Architecture:** Separate services for inference and web serving allow for scalability and independent development.

## Architecture

The system follows a microservice-like architecture:

+-----------------+ HTTP Requests +-----------------+ HTTP API Calls +-------------------------+
| React Frontend | <--------------------> | Flask Web App | <------------------------> | FastAPI Inference Service |
| (Browser) | (Video Stream) | (Port 5000) | (e.g., /predict) | (YOLOE Model - Port 8000) |
+-----------------+ +-----------------+ +-------------------------+
| | |
| Displays annotated stream, | Serves Frontend Files, | Loads YOLOE model(s)
| Sends prompt updates/VP data | Handles /video_feed endpoint, | Runs inference on images
| | Manages webcam/video input, | Provides /predict_text,
| | Calls Inference Service API | /predict_vp, /set_text_classes
| | | endpoints


1.  The **User** interacts with the **React Frontend** in their browser.
2.  The **React Frontend** makes API calls to the **Flask Web App** (e.g., to update prompts, upload video, request visual prompt data).
3.  The **Flask Web App** serves the static React files and handles the `/video_feed` stream.
4.  For inference, the **Flask Web App** (specifically its background thread) sends image data to the **FastAPI Inference Service** API endpoints (`/predict_text` or `/predict_vp`).
5.  The **FastAPI Service** runs the YOLOE model and returns detection results (bounding boxes, classes, confidence, etc.) to Flask.
6.  The **Flask Web App** receives the detections, annotates the video frame (using Supervision library), and sends the annotated frame via the `/video_feed` stream.
7.  The **React Frontend** displays the annotated `/video_feed`.

## Technology Stack

*   **Backend:** Python
    *   **Inference Service:** FastAPI, Uvicorn, Ultralytics YOLOE, Supervision, PyTorch, Pydantic
    *   **Web Server:** Flask, Requests
*   **Frontend:** JavaScript, React, Vite (build tool), CSS
*   **Machine Learning:** YOLOE (Ultralytics implementation)
*   **Environment:** Python Virtual Environments, Node.js/npm (or yarn)

## Setup and Installation

**Prerequisites:**

*   Python 3.8+
*   Node.js 16+ and npm (or yarn)
*   Git
*   **GPU with CUDA:** Strongly recommended for acceptable inference performance. Ensure you have compatible NVIDIA drivers and CUDA Toolkit installed if using GPU.

**Steps:**

1.  **Clone the Repository:**
    ```bash
    git clone <your-repository-url>
    cd <your-repository-name>
    ```

2.  **Setup FastAPI Inference Service:**
    *   Navigate to the FastAPI service directory (e.g., `cd inference_service` - *adjust if named differently*).
    *   Create and activate a Python virtual environment:
        ```bash
        python -m venv venv
        source venv/bin/activate  # On Windows use `venv\Scripts\activate`
        ```
    *   Install Python dependencies:
        ```bash
        pip install -r requirements.txt # Ensure this file exists in the service dir
        ```
    *   **Download YOLOE Model:** Download the required YOLOE model weights file (e.g., `yoloe-l-seg.pt`). Place it in a location accessible by the service (e.g., within the service directory).
    *   **Configuration:** Create a `.env` file in the service directory based on the configuration settings (see `main_two_models.py` or similar):
        ```env
        # Example .env for FastAPI service
        TEXT_MODEL_PATH=yoloe-l-seg.pt
        VP_MODEL_PATH=yoloe-l-seg.pt # Can be the same or different
        DEVICE=cuda # or cpu
        DEFAULT_CONFIDENCE=0.3
        LOG_LEVEL=INFO
        PORT=8000
        ```
        *Adjust `*_MODEL_PATH` to the actual path of your downloaded model file.*

3.  **Setup Flask Web Server & React Frontend:**
    *   Navigate back to the main project directory (or the directory containing the Flask app module).
    *   Create and activate a *separate* Python virtual environment for Flask:
        ```bash
        python -m venv venv_flask
        source venv_flask/bin/activate # On Windows use `venv_flask\Scripts\activate`
        ```
    *   Install Flask Python dependencies:
        ```bash
        pip install -r requirements.txt # Ensure this file exists for Flask deps
        ```
    *   Navigate to the React frontend directory (e.g., `cd frontend`).
    *   Install Node.js dependencies:
        ```bash
        npm install
        # OR: yarn install
        ```
    *   **Build the React App:** Create the optimized production build.
        ```bash
        npm run build
        # OR: yarn build
        ```
        This will create a `dist` folder inside `frontend` containing `index.html` and static assets. The Flask app is configured to serve files from this `dist` directory.

## Running the Application

You need to run both the FastAPI service and the Flask web server.

1.  **Start the FastAPI Inference Service:**
    *   Navigate to the FastAPI service directory.
    *   Activate its virtual environment (`source venv/bin/activate`).
    *   Run Uvicorn (adjust `main_two_models:app` if your file/app variable is named differently):
        ```bash
        uvicorn main_two_models:app --host 0.0.0.0 --port 8000
        ```
    *   Keep this terminal running. You should see logs indicating the models are loading.

2.  **Start the Flask Web Server:**
    *   Navigate to the main project directory (where the Flask app module/`__init__.py` is).
    *   Activate the Flask virtual environment (`source venv_flask/bin/activate`).
    *   Run Flask (adjust `your_flask_app_module:app` if needed):
        ```bash
        # Development server (easier for testing)
        flask run --host 0.0.0.0 --port 5000

        # OR Production server (using Gunicorn)
        # pip install gunicorn
        # gunicorn 'your_flask_app_module:app' -b 0.0.0.0:5000 --workers 2 --threads 4 # Adjust workers/threads
        ```
    *   Keep this terminal running.

3.  **Access the Application:**
    *   Open your web browser and navigate to the Flask server's address: `http://localhost:5000` (or your server's IP address if running remotely).

## Configuration

*   **FastAPI Service:** Configure model paths, device (`cuda`/`cpu`), confidence threshold, and logging level via the `.env` file in the FastAPI service directory.
*   **Flask Web Server:**
    *   The URL for the FastAPI service (`MODEL_SERVICE_URL`) might be hardcoded or configurable within the Flask app (e.g., using environment variables or `app.config`). Check `inference.py` or `routes.py`.
    *   The paths to the React build output (`frontend/dist`) are configured in the Flask app's `__init__.py`. Adjust if your directory structure differs.

## Troubleshooting

*   **`CUDA out of memory`:** You may be trying to load models that are too large for your GPU VRAM. Try smaller model variants or run on `cpu` (slower). Ensure only one instance of the FastAPI service is running.
*   **`Model file not found`:** Verify the `*_MODEL_PATH` in the FastAPI `.env` file points correctly to your downloaded `.pt` file.
*   **React App Not Loading (404 errors in browser console):**
    *   Ensure you ran `npm run build` in the `frontend` directory.
    *   Verify the `REACT_DIST_FOLDER` and `REACT_STATIC_FOLDER` paths in Flask's `__init__.py` are correct relative to the Flask app's location.
    *   Check that the Flask catch-all route is correctly serving `index.html`.
*   **Connection Errors (Flask to FastAPI):** Ensure the FastAPI service is running and accessible from the Flask server. Check firewalls and that the `MODEL_SERVICE_URL` (default `http://localhost:8000`) in Flask points to the correct address/port of the FastAPI service.
*   **Dependency Issues:** Make sure all dependencies in `requirements.txt` (for both services) and `package.json` (for frontend) are installed correctly in their respective environments.


<!-- # jetson_roboflow_ppe



Build docker image
```bash
chmod +x build.sh
./build.sh
```

Run the Containers:
Inference container:
```bash
docker run -d --runtime nvidia --network host roboflow/roboflow-inference-server-trt-jetson-5.1.1
```

    For USB Camera (default):
```bash
docker run --runtime nvidia --device /dev/video0:/dev/video0 --network host jetson-roboflow-ppe-web
```
    For RealSense Camera:

```bash
docker run --runtime nvidia --privileged --network host -e CAMERA_TYPE=realsense jetson-roboflow-ppe-web
```

Go to 
http://{JETSON_IP}:5000


## Testing on laptop
```bash
chmod +x build.sh
./build.sh
docker run -d --gpus all --network host roboflow/roboflow-inference-server-gpu:latest
```
Inference container:
```bash
docker run -d --gpus all --network host roboflow/roboflow-inference-server-gpu:latest
```

For USB Camera (default):
```bash
docker run -it --gpus all --privileged --device /dev/video4:/dev/video4 --network host \
    -e CAMERA_TYPE=usb \
    -e VIDEO_DEVICE_INDEX=4 \
    roboflow-ppe-web
```
For RealSense Camera:

```bash
docker run --gpus all --privileged --network host -e CAMERA_TYPE=realsense roboflow-ppe-web
```


docker run --gpus all --network host \
    -e CAMERA_TYPE=tcp_server \
    -e TCP_SERVER_HOST="0.0.0.0" \
    -e TCP_SERVER_PORT="8080" \
    roboflow-ppe-web -->