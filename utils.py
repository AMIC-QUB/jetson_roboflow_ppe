import cv2
import numpy as np
from flask import Flask, Response, render_template, jsonify
from ultralytics import YOLOE
from typing import Callable, Generator
import inspect

# Initialize Flask app
app = Flask(__name__)

# Load YOLO-World model
model = YOLOE("yoloe-11l-seg.pt")  # You can use 'yolo-world-m.pt' or 'yolo-world-l.pt' for better accuracy
model.to('cuda')  # Ensure the model runs on GPU
# model.half()  # Enable FP16 inference
# Global variables
latest_detections = []  # Store latest detections
user_prompts = ["worker with helmet", "worker without helmet", "safety vest", "heavy machinery"]  # Default prompts

# Define class-specific colors (BGR format for OpenCV)
POSITIVE_CLASSES = {"worker with helmet", "safety vest"}
NEGATIVE_CLASSES = {"worker without helmet"}
GREEN = (0, 255, 0)  # Positive color
RED = (0, 0, 255)    # Negative color
CONFIDENCE_THRESHOLD = 0.75  # Adjusted for YOLO-World (you can tweak this)
class_colors = {}  # Dictionary to store consistent colors for each class
frame_counter = 0  # Frame counter for running detections every 5th frame
last_results = None  # Store the most recent results for reuse
is_paused = False
# Flag to toggle between bounding boxes and segmentation
show_segmentation = False
def generate_frames(get_frame: Callable[[], tuple[bool, np.ndarray | None]]) -> Generator[bytes, None, None]:
    """Generator function to yield video frames with detections from a provided frame source."""
    global latest_detections, user_prompts, frame_counter, last_results, is_paused, show_segmentation
    # Check if get_frame is callable
    if not inspect.isfunction(get_frame) and not inspect.ismethod(get_frame):
        raise TypeError(f"get_frame must be a callable function, got {type(get_frame)}: {get_frame}")
    frame_counter=0
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
            if is_paused or not user_prompts:
                # Encode the raw frame without detections
                ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 50])  # Reduced JPEG quality
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                continue  # Skip the rest of the loop
            if not ret or frame is None:
                print("Error: Could not retrieve frame.")
                break

            # Run YOLO-World inference with user-defined prompts
            inference_frame = cv2.resize(frame, (640, 640))
            model.set_classes(user_prompts, model.get_text_pe(user_prompts))
            frame_counter += 1

            # Run YOLO-World inference only every 5th frame
            if frame_counter % 5 == 0:
                # inference_frame = inference_frame.astype(np.float16) / 255.0  #
                results = model.predict(inference_frame, conf=CONFIDENCE_THRESHOLD)
                last_results = results  # Store the latest results
            else:
                results = last_results if last_results is not None else []  #
            # Process detections
# Process detections
            filtered_predictions = []
            for r in results:
                boxes = r.boxes
                masks = r.masks if show_segmentation and hasattr(r, 'masks') else None  # Get masks if available and in segmentation mode
                for i, box in enumerate(boxes):
                    # Extract bounding box coordinates and labels
                    xyxy = box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
                    conf = box.conf.cpu().numpy().item()  # Convert to scalar using .item()
                    cls = int(box.cls.cpu().numpy())  # Class index
                    label = user_prompts[cls]  # Map class index to prompt

                    # Scale bounding box coordinates back to original frame size
                    orig_h, orig_w = frame.shape[:2]
                    infer_h, infer_w = inference_frame.shape[:2]
                    scale_x, scale_y = orig_w / infer_w, orig_h / infer_h
                    xyxy[0] *= scale_x  # x1
                    xyxy[1] *= scale_y  # y1
                    xyxy[2] *= scale_x  # x2
                    xyxy[3] *= scale_y  # y2

                    # Create prediction dictionary (similar to Roboflow format for compatibility)
                    pred = {
                        "x": int((xyxy[0] + xyxy[2]) / 2),  # Center x
                        "y": int((xyxy[1] + xyxy[3]) / 2),  # Center y
                        "width": int(xyxy[2] - xyxy[0]),
                        "height": int(xyxy[3] - xyxy[1]),
                        "class": label,
                        "confidence": conf  # Store as scalar
                    }
                    filtered_predictions.append(pred)

                    # Determine color based on class, assign random color for new classes
                    color = GREEN if label in POSITIVE_CLASSES else RED if label in NEGATIVE_CLASSES else class_colors.setdefault(label, (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256)))

                    if show_segmentation and masks is not None:
                        # Draw segmentation mask
                        mask = masks[i].data[0].cpu().numpy()  # Get the mask for this detection
                        mask = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)  # Resize mask to original frame size
                        mask = mask.astype(np.uint8) * 255  # Convert to binary mask
                        colored_mask = np.zeros_like(frame)  # Create a colored mask
                        colored_mask[mask > 0] = color  # Apply the class color to the mask
                        frame = cv2.addWeighted(frame, 0.7, colored_mask, 0.3, 0)  # Overlay the mask on the frame
                        # Optionally, draw the label above the mask
                        cv2.putText(frame, f"{label} ({conf:.2f})", (int(xyxy[0]), int(xyxy[1]) - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                    else:
                        # Draw bounding box and label
                        pt1 = (int(xyxy[0]), int(xyxy[1]))
                        pt2 = (int(xyxy[2]), int(xyxy[3]))
                        cv2.rectangle(frame, pt1, pt2, color, 2)
                        cv2.putText(frame, f"{label} ({conf:.2f})", (int(xyxy[0]), int(xyxy[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            latest_detections = filtered_predictions  # Update global detections

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

@app.route('/update_prompts/<prompts>')
def update_prompts(prompts):
    """Update the user prompts for YOLO-World detection."""
    global user_prompts, last_results
    user_prompts = prompts.split(",")  # Split comma-separated prompts
    user_prompts = [p.strip() for p in user_prompts]  # Clean up whitespace
    last_results = None  # Reset last_results to avoid index mismatches
    return jsonify({"status": "success", "prompts": user_prompts})

@app.route('/clear_prompts')
def clear_prompts():
    """Clear the user prompts and toggle the pause state."""
    global user_prompts, is_paused, last_results
    user_prompts = []  # Clear prompts
    last_results = None  # Reset last_results to avoid index mismatches
    is_paused = not is_paused  # Toggle pause state
    return jsonify({"status": "success", "paused": is_paused, "prompts": user_prompts})

@app.route('/toggle_display_mode')
def toggle_display_mode():
    """Toggle between showing bounding boxes and segmentation masks."""
    global show_segmentation
    show_segmentation = not show_segmentation
    return jsonify({"status": "success", "show_segmentation": show_segmentation})