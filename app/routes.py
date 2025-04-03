from flask import render_template, jsonify, request, Response
from . import app, inference
from .camera import get_frame
import cv2
import base64
import numpy as np
from PIL import Image
import requests
import io
# Set up logging
logger = app.logger

MODEL_SERVICE_URL = "http://localhost:8000"

def encode_image_to_base64(image):
    """Encode a PIL Image to base64 string."""
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

@app.route('/')
def index():
    """Render the main web page."""
    logger.debug("Serving / endpoint")
    inference.is_on_visual_prompt = False  # Resume detections when on the main page
    logger.debug("is_on_visual_prompt set to False")
    return render_template('index.html', visual_prompts_active=inference.visual_prompts is not None)

@app.route('/visual_prompt')
def visual_prompt():
    """Render the visual prompt page."""
    logger.debug("Serving /visual_prompt endpoint")
    inference.is_on_visual_prompt = True  # Pause detections when on the visual prompt page
    logger.debug("is_on_visual_prompt set to True")
    # Capture a single frame to display on the visual prompt page
    frame_gen = get_frame()  # This is a generator
    try:
        ret, frame = next(frame_gen)  # Get the next frame
        if not ret or frame is None:
            logger.error("Could not capture frame for visual prompt: ret=%s, frame=%s", ret, frame)
            return jsonify({"error": "Could not capture frame"}), 500
        logger.debug("Frame captured successfully, shape: %s", frame.shape)
    except Exception as e:
        logger.error("Exception while capturing frame for visual prompt: %s", str(e))
        return jsonify({"error": "Could not capture frame: " + str(e)}), 500

    # Encode the frame as a base64 string to display in the browser
    try:
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            logger.error("Failed to encode frame for visual prompt")
            return jsonify({"error": "Failed to encode frame"}), 500
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        logger.debug("Frame encoded to base64 successfully")
    except Exception as e:
        logger.error("Exception while encoding frame for visual prompt: %s", str(e))
        return jsonify({"error": "Failed to encode frame: " + str(e)}), 500

    return render_template('visual_prompt.html', frame_base64=frame_base64, user_prompts=inference.user_prompts)

@app.route('/process_visual_prompt', methods=['POST'])
def process_visual_prompt():
    """Process the visual prompt, crop the image, and run inference."""
    logger.debug("Serving /process_visual_prompt endpoint")
    data = request.json
    bboxes = np.array(data['bboxes'])  # List of [x1, y1, x2, y2]
    classes = np.array(data['classes'], dtype=np.int32)  # List of class indices

    # Validate input
    if len(bboxes) != len(classes):
        logger.error("Mismatch between number of bounding boxes and classes")
        return jsonify({"error": "Mismatch between number of bounding boxes and classes"}), 400

    # Log the visual prompts for debugging
    logger.debug("Visual prompts received: bboxes=%s, bboxes_shape=%s, classes=%s, classes_shape=%s", bboxes, bboxes.shape, classes, classes.shape)

    # Capture a frame for inference
    frame_gen = get_frame()
    ret, frame = next(frame_gen)
    if not ret or frame is None:
        logger.error("Could not capture frame for visual prompt inference")
        return jsonify({"error": "Could not capture frame"}), 500

    # Convert the frame to a PIL Image (no resizing)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    source_image = Image.fromarray(frame_rgb)
    target_image = source_image.copy()  # Use the same image for both source and target for now
    logger.debug("Input image shape: %s", frame_rgb.shape)

    # Crop the image using each bounding box in prompts
    cropped_images = []
    for bbox in bboxes:
        x1, y1, x2, y2 = map(int, bbox)  # Convert to integers
        # Ensure coordinates are within image bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame_rgb.shape[1], x2)
        y2 = min(frame_rgb.shape[0], y2)
        if x2 <= x1 or y2 <= y1:
            logger.warning("Invalid bounding box coordinates: x1=%d, y1=%d, x2=%d, y2=%d", x1, y1, x2, y2)
            continue
        # Crop the image
        cropped = frame_rgb[y1:y2, x1:x2]
        # Encode the cropped image as base64
        ret, buffer = cv2.imencode('.jpg', cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR))
        if not ret:
            logger.error("Failed to encode cropped image")
            continue
        cropped_base64 = base64.b64encode(buffer).decode('utf-8')
        cropped_images.append(cropped_base64)
        logger.debug("Cropped image encoded, shape: %s", cropped.shape)

    # Create the prompts dictionary
    prompts = dict(
        bboxes=bboxes.tolist(),  # Convert to list for JSON serialization
        cls=classes.tolist()     # Convert to list for JSON serialization
    )

    # Run the two-stage prediction process using the model service
    try:
        # Encode the source image to base64
        source_image_base64 = encode_image_to_base64(source_image)

        # First stage: Generate visual prompt embeddings and set classes
        response = requests.post(
            f"{MODEL_SERVICE_URL}/predict_with_visual_prompts",
            json={
                "image_base64": source_image_base64,
                "prompts": prompts,
                "user_prompts": inference.user_prompts
            }
        )
        if response.status_code != 200:
            logger.error(f"Failed to set visual prompts: {response.text}")
            return jsonify({"error": f"Failed to set visual prompts: {response.text}"}), 500

        # Second stage: Run inference on the target image
        target_image_base64 = encode_image_to_base64(target_image)
        response = requests.post(
            f"{MODEL_SERVICE_URL}/predict",
            json={"image_base64": target_image_base64}
        )
        if response.status_code != 200:
            logger.error(f"Failed to get inference results: {response.text}")
            return jsonify({"error": f"Failed to get inference results: {response.text}"}), 500

        filtered_predictions = response.json().get("detections", [])
    except Exception as e:
        logger.error(f"Failed to run inference with visual prompts: {e}")
        return jsonify({"error": f"Failed to run inference: {str(e)}"}), 500

    # Store the visual prompts globally
    inference.visual_prompts = dict(
        bboxes=bboxes,
        cls=classes
    )
    logger.info(f"Visual prompts set: {inference.visual_prompts}")

    # Encode the frame with results for display
    ret, buffer = cv2.imencode('.jpg', frame)
    if not ret:
        logger.error("Failed to encode result frame")
        return jsonify({"error": "Failed to encode result frame"}), 500
    result_frame_base64 = base64.b64encode(buffer).decode('utf-8')

    return jsonify({
        "detections": filtered_predictions,
        "result_frame_base64": result_frame_base64,
        "cropped_images": cropped_images  # Return the cropped images
    })

@app.route('/clear_visual_prompts')
def clear_visual_prompts():
    """Clear the visual prompts."""
    inference.visual_prompts = None
    inference.is_on_visual_prompt = False  # Resume detections when clearing visual prompts
    # Reset the model's state using the model service
    logger.info("Visual prompts cleared, model state reset, is_on_visual_prompt set to False")
    return jsonify({"status": "success"})

@app.route('/video_feed')
def video_feed():
    """Stream the video feed with raw frames."""
    logger.debug("Serving /video_feed endpoint")
    return Response(inference.generate_frames(get_frame), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detections')
def get_detections():
    """Return the latest detections as JSON."""
    logger.debug("Serving /detections endpoint")
    return jsonify({"detections": inference.latest_detections})

@app.route('/update_prompts/<prompts>')
def update_prompts(prompts):
    """Update the user prompts for YOLOE detection."""
    inference.user_prompts = prompts.split(",")  # Split comma-separated prompts
    inference.user_prompts = [p.strip() for p in inference.user_prompts]  # Clean up whitespace
    inference.last_results = None  # Reset last_results to avoid index mismatches
    # Reset the model's state using ModelManager
    logger.info(f"Updated prompts: {inference.user_prompts}, model state reset")
    return jsonify({"status": "success", "prompts": inference.user_prompts})

@app.route('/clear_prompts')
def clear_prompts():
    """Clear the user prompts and toggle the pause state."""
    inference.user_prompts = []  # Clear prompts
    inference.last_results = None  # Reset last_results to avoid index mismatches
    inference.is_paused = not inference.is_paused  # Toggle pause state
    # Reset the model's state using ModelManager
    logger.info(f"Prompts cleared, pause state: {inference.is_paused}, model state reset")
    return jsonify({"status": "success", "paused": inference.is_paused, "prompts": inference.user_prompts})

@app.route('/toggle_display_mode')
def toggle_display_mode():
    """Toggle between showing bounding boxes and segmentation masks."""
    inference.show_segmentation = not inference.show_segmentation
    logger.info(f"Display mode toggled, show_segmentation: {inference.show_segmentation}")
    return jsonify({"status": "success", "show_segmentation": inference.show_segmentation})