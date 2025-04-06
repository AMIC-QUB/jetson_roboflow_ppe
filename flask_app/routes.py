# ... (imports) ...
from .camera import get_frame # Make sure this provides the webcam generator
from . import inference # Import the refactored inference module

# ... (logger, helper functions like encode_image_to_base64) ...

MODEL_SERVICE_URL = "http://localhost:8000" # Get from app.config ideally

# --- Route Implementations ---

@app.route('/', methods=['GET', 'POST'])
def index():
    """Render the main web page, handle video upload, manage inference thread."""
    logger.debug("Serving / endpoint")

    # Ensure inference thread uses correct source (webcam/video)
    # Stop/start logic might need refinement based on desired behavior on page load/reload

    if request.method == 'POST':
        # Handle video file upload
        if 'video' in request.files:
            file = request.files['video']
            if file and file.filename != '':
                video_path = f"/tmp/uploaded_{file.filename}" # Use a safe path
                try:
                    file.save(video_path)
                    logger.info(f"Video file saved to: {video_path}")
                    # Stop existing thread, set new path, start new thread
                    inference.stop_inference_thread()
                    inference.video_file_path = video_path
                    if inference.video_cap: # Release old capture if exists
                        inference.video_cap.release()
                        inference.video_cap = None
                    inference.start_inference_thread(get_frame)
                    logger.debug("Switched inference to video file.")
                except Exception as e:
                    logger.error(f"Failed to save or process video file: {e}", exc_info=True)
                    return jsonify({"error": f"Failed to process video file: {str(e)}"}), 500
            else:
                 logger.warning("Video upload POST request with no valid file.")
                 # Decide how to handle this - maybe just reload the page?
        else:
             logger.warning("POST request to / without 'video' file part.")

    elif request.method == 'GET':
        # If currently using video and accessing via GET, switch back to webcam?
        # This logic might be too simplistic depending on desired UX.
        # Maybe only switch back if a specific button/query param indicates it.
        # For now, assume GET means webcam unless video_file_path is already set.
        if inference.video_file_path is None and not inference.inference_running:
             # Start with webcam if no video path and thread not running
             logger.info("GET request: Starting inference with webcam.")
             inference.stop_inference_thread() # Ensure clean state
             inference.video_file_path = None
             inference.start_inference_thread(get_frame)

    # Set visual prompt flag correctly
    inference.is_on_visual_prompt = False

    return render_template(
        'index.html',
        # Pass necessary state to the template
        user_prompts=inference.user_prompts,
        is_paused=inference.is_paused,
        # Add other flags if needed by template
        using_video=(inference.video_file_path is not None)
    )


@app.route('/visual_prompt')
def visual_prompt():
    """Render the visual prompt page with a static frame."""
    logger.debug("Serving /visual_prompt endpoint")
    inference.is_on_visual_prompt = True # Pause inference updates in background

    # Get a single static frame
    # Ensure get_frame() can be called to just grab one frame if needed,
    # or use the latest frame from the inference thread if running.
    frame_to_display = None
    if inference.latest_annotated_frame:
         # Use the latest raw frame if available from the running thread
         # Note: latest_annotated_frame now stores raw frame in the thread before annotation
         # This needs adjustment - maybe store latest raw frame separately?
         # Let's assume get_frame can provide a single frame for simplicity here:
         frame_gen = get_frame()
         try:
              ret, frame = next(frame_gen)
              if ret and frame is not None:
                   frame_to_display = frame
              else:
                   logger.error("Could not capture single frame for visual prompt")
                   return "Error capturing frame", 500
         except Exception as e:
              logger.error(f"Error getting frame for VP: {e}", exc_info=True)
              return "Error capturing frame", 500
         # Release webcam if get_frame() holds it open
         if hasattr(frame_gen, 'release'): frame_gen.release()

    else: # Fallback if thread isn't running or didn't provide frame
        logger.warning("Inference thread not running or no frame available, trying direct capture.")
        # Direct capture logic (similar to original)
        # ... (ensure proper camera release) ...
        return "Error: Cannot get frame for visual prompt", 500

    if frame_to_display is None:
         return "Error: Failed to get frame for visual prompt", 500

    # Encode frame for display
    ret, buffer = cv2.imencode('.jpg', frame_to_display)
    if not ret:
        logger.error("Failed to encode frame for visual prompt")
        return "Error encoding frame", 500
    frame_base64 = base64.b64encode(buffer).decode('utf-8')

    return render_template(
        'visual_prompt.html',
        frame_base64=frame_base64,
        user_prompts=inference.user_prompts
    )


@app.route('/process_visual_prompt', methods=['POST'])
def process_visual_prompt():
    """Process visual prompts using the dedicated VP model."""
    logger.debug("Serving /process_visual_prompt endpoint")
    data = request.json
    # --- Get image that was *actually* annotated ---
    frame_base64 = data.get('frame_base64')
    bboxes_data = data.get('bboxes')
    classes_data = data.get('classes')
    vp_user_prompts = data.get('user_prompts') # Names for the classes used in VP

    if not frame_base64 or not bboxes_data or not classes_data or not vp_user_prompts:
        logger.error("Missing data in /process_visual_prompt request")
        return jsonify({"error": "Missing frame_base64, bboxes, classes, or user_prompts"}), 400

    try:
        # Decode the frame the user annotated
        img_bytes = base64.b64decode(frame_base64)
        img = Image.open(io.BytesIO(img_bytes))
        img_rgb = img.convert("RGB") # Ensure RGB
        # Re-encode for sending to service (or pass decoded numpy array if service accepts)
        image_to_send_base64 = encode_image_to_base64(img_rgb)

        prompts = dict(
            bboxes=bboxes_data,
            cls=classes_data
        )

        # --- Call the CORRECT VP Endpoint ---
        api_payload = {
            "target_image_base64": image_to_send_base64,
            "reference_image_base64": image_to_send_base64, # Use same image
            "prompts": prompts,
            "user_prompts_for_names": vp_user_prompts # Pass the names
        }

        logger.debug(f"Calling /predict_vp with payload keys: {api_payload.keys()}")
        response = requests.post(
            f"{MODEL_SERVICE_URL}/predict_vp", # Call the VP endpoint
            json=api_payload,
            timeout=15 # VP might take longer
        )
        response.raise_for_status() # Check for HTTP errors

        response_data = response.json()
        # Parse the response (list of dicts)
        sv_detections = inference.parse_detections_from_response(response_data)

        # Annotate the original frame with VP results for display
        np_frame = cv2.cvtColor(np.array(img_rgb), cv2.COLOR_RGB2BGR)
        annotated_vp_frame = inference.annotate_frame(np_frame, sv_detections)

        # Encode result frame
        ret, buffer = cv2.imencode('.jpg', annotated_vp_frame)
        result_frame_base64 = base64.b64encode(buffer).decode('utf-8') if ret else None

        # Optionally store VP prompts if needed globally (though stateless is better)
        # inference.visual_prompts = prompts # Be cautious with global state

        logger.info("Visual prompt processed successfully.")
        return jsonify({
            "detections": response_data.get("detections", []), # Return raw detections list
            "result_frame_base64": result_frame_base64
            # "cropped_images": [] # Add cropped images if needed
        })

    except requests.exceptions.RequestException as e:
        logger.error(f"API request failed during VP processing: {e}")
        return jsonify({"error": f"Model service request failed: {e}"}), 502 # Bad Gateway
    except Exception as e:
        logger.error(f"Error processing visual prompt: {e}", exc_info=True)
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


@app.route('/clear_visual_prompts')
def clear_visual_prompts():
    """Clear visual prompts (if stored globally) and reset model state."""
    logger.info("Clearing visual prompts")
    inference.visual_prompts = None # Clear global VP state if used
    inference.is_on_visual_prompt = False # Allow background inference again

    # No need to call the model service here unless you want to explicitly
    # reset the text model's classes back to the current inference.user_prompts
    try:
        # --- Call CORRECT endpoint to set text classes ---
        logger.debug(f"Resetting text model classes to: {inference.user_prompts}")
        response = requests.post(
            f"{MODEL_SERVICE_URL}/set_text_classes",
            json={"classes": inference.user_prompts},
            timeout=5
        )
        response.raise_for_status()
        logger.info("Text model classes reset successfully.")
        return jsonify({"status": "success"})
    except requests.exceptions.RequestException as e:
         logger.error(f"Failed to reset text classes via API: {e}")
         # Decide if this is a critical error for the user
         return jsonify({"error": f"Failed reset text classes on model service: {e}"}), 502
    except Exception as e:
        logger.error(f"Error clearing visual prompts state: {e}", exc_info=True)
        return jsonify({"error": f"Internal error: {str(e)}"}), 500


@app.route('/video_feed')
def video_feed():
    """Stream annotated video frames from the background thread."""
    logger.debug("Serving /video_feed endpoint")
    # Use the new generator that yields pre-annotated frames
    return Response(inference.generate_frames_for_stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/detections')
def get_detections():
    """Return the latest detections (if needed by JS)."""
    # This might be less necessary if annotations happen server-side
    logger.debug("Serving /detections endpoint")
    dets = None
    if inference.latest_detections:
        dets = inference.latest_detections[-1] # Get the last sv.Detections object

    # Convert sv.Detections back to JSON serializable format if needed
    # Or return the raw format from the API stored earlier
    # For now, return empty if not available
    return jsonify({"detections": [] if dets is None else "..."}) # Adjust format


@app.route('/update_prompts/<prompts>')
def update_prompts(prompts):
    """Update text prompts and reset the text model."""
    new_prompts = [p.strip() for p in prompts.split(",") if p.strip()]
    inference.user_prompts = new_prompts
    logger.info(f"Updating text prompts to: {inference.user_prompts}")
    inference.latest_detections.clear() # Clear old detections

    try:
        # --- Call CORRECT endpoint ---
        response = requests.post(
            f"{MODEL_SERVICE_URL}/set_text_classes",
            json={"classes": inference.user_prompts},
            timeout=5
        )
        response.raise_for_status()
        logger.info("Text model classes updated successfully.")
        return jsonify({"status": "success", "prompts": inference.user_prompts})
    except requests.exceptions.RequestException as e:
         logger.error(f"Failed to update text classes via API: {e}")
         return jsonify({"error": f"Failed update text classes on model service: {e}"}), 502
    except Exception as e:
        logger.error(f"Error updating prompts state: {e}", exc_info=True)
        return jsonify({"error": f"Internal error: {str(e)}"}), 500


@app.route('/clear_prompts')
def clear_prompts():
    """Clear text prompts and reset the text model."""
    logger.info("Clearing text prompts.")
    inference.user_prompts = []
    inference.latest_detections.clear()

    try:
        # --- Call CORRECT endpoint ---
        response = requests.post(
            f"{MODEL_SERVICE_URL}/set_text_classes",
            json={"classes": []}, # Send empty list
            timeout=5
        )
        response.raise_for_status()
        logger.info("Text model classes cleared successfully.")
        # Toggle pause state if desired (keep original logic)
        inference.is_paused = not inference.is_paused
        logger.info(f"Pause state toggled to: {inference.is_paused}")
        return jsonify({"status": "success", "paused": inference.is_paused, "prompts": inference.user_prompts})
    except requests.exceptions.RequestException as e:
         logger.error(f"Failed to clear text classes via API: {e}")
         return jsonify({"error": f"Failed clear text classes on model service: {e}"}), 502
    except Exception as e:
        logger.error(f"Error clearing prompts state: {e}", exc_info=True)
        return jsonify({"error": f"Internal error: {str(e)}"}), 500


@app.route('/toggle_pause')
def toggle_pause():
    """Toggle the pause state for inference."""
    inference.is_paused = not inference.is_paused
    logger.info(f"Pause state toggled to: {inference.is_paused}")
    return jsonify({"status": "success", "paused": inference.is_paused})


# Remove /toggle_display_mode unless segmentation annotation is implemented
# @app.route('/toggle_display_mode')
# def toggle_display_mode():
#     inference.show_segmentation = not inference.show_segmentation
#     logger.info(f"Display mode toggled, show_segmentation: {inference.show_segmentation}")
#     return jsonify({"status": "success", "show_segmentation": inference.show_segmentation})

# --- Add application context handling for thread start/stop ---
@app.before_request
def before_request():
    # Start inference thread if not running (e.g., on first request)
    # Be careful with this - might start multiple times if not checked properly
    if not inference.inference_running:
         logger.info("Starting inference thread from before_request hook.")
         # Determine initial source (webcam or video?)
         inference.video_file_path = None # Default to webcam initially? Or check config?
         inference.start_inference_thread(get_frame)

# Consider adding a shutdown hook if running directly with Flask's dev server
# import atexit
# atexit.register(inference.stop_inference_thread)