import base64
import io
import logging
import os
import time
import binascii
from typing import List, Dict, Any, Optional

import cv2
import numpy as np
import supervision as sv
import torch
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor # Assuming needed for VP

# --- Configuration ---
class Settings(BaseSettings):
    """Manages application settings."""
    # Allow potentially different model files, default to same one
    text_model_path: str = Field("yoloe-11l-seg.pt", description="Path to the model file for text prompts.")
    vp_model_path: str = Field("yoloe-11l-seg.pt", description="Path to the model file for visual prompts.")
    device: str = Field("cuda", description="Device to run models on ('cuda' or 'cpu'). Check GPU memory!")
    default_confidence: float = Field(0.3, description="Default confidence threshold.")
    log_level: str = Field("INFO", description="Logging level.")
    port: int = Field(8000, description="Server port.")
    host: str = Field("0.0.0.0", description="Server host.")

    class Config:
        env_file = '.env'
        env_file_encoding = 'utf-8'

settings = Settings()

# --- Logging Setup ---
logging.basicConfig(
    level=settings.log_level.upper(),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# --- Model Management ---
class MultiModelManager:
    """Handles loading and running inference with two separate YOLOE models."""
    def __init__(self, text_model_path: str, vp_model_path: str, device: str):
        self.text_model_path = text_model_path
        self.vp_model_path = vp_model_path
        self.device = device
        self.text_model: Optional[YOLOE] = None
        self.vp_model: Optional[YOLOE] = None
        self.current_text_classes: Optional[List[str]] = None

        logger.warning(f"Attempting to load two models ('{text_model_path}', '{vp_model_path}') onto device '{device}'. Ensure sufficient GPU memory.")
        self._load_models()

    def _load_models(self):
        """Loads both YOLOE models."""
        try:
            logger.info(f"Loading TEXT model from: {self.text_model_path}")
            self.text_model = YOLOE(self.text_model_path)
            self.text_model.to(self.device)
            logger.info(f"TEXT model loaded successfully to device: {self.device}")
        except Exception as e:
            logger.error(f"Failed to load TEXT model: {e}", exc_info=True)
            # Decide if failure is critical, here we allow partial success
            self.text_model = None

        try:
            logger.info(f"Loading VISUAL PROMPT model from: {self.vp_model_path}")
            self.vp_model = YOLOE(self.vp_model_path)
            # VP model might require specific predictor setup during/after loading
            # self.vp_model.predictor = YOLOEVPSegPredictor(...) # If needed
            self.vp_model.to(self.device)
            logger.info(f"VISUAL PROMPT model loaded successfully to device: {self.device}")
        except Exception as e:
            logger.error(f"Failed to load VISUAL PROMPT model: {e}", exc_info=True)
            self.vp_model = None

        if not self.text_model and not self.vp_model:
             logger.critical("Failed to load BOTH models. Service unusable.")
             # raise RuntimeError("Failed to load any models.") # Optional: Hard fail

    def set_text_classes(self, classes: List[str]):
        """Sets the classes ONLY for the text-prompt model."""
        if not self.text_model:
            raise RuntimeError("Text model is not loaded.")
        try:
            logger.debug(f"Setting text classes for TEXT model: {classes}")
            text_pe = self.text_model.get_text_pe(classes)
            self.text_model.set_classes(classes, text_pe)
            self.current_text_classes = classes
            logger.info(f"Text classes set successfully for TEXT model: {classes}")
        except Exception as e:
            logger.error(f"Failed to set text classes for TEXT model: {e}", exc_info=True)
            raise RuntimeError(f"Failed to set text classes: {e}")

    def predict_text_prompt(self, image: Image.Image, confidence_threshold: float) -> List[Dict[str, Any]]:
        """Runs prediction using the dedicated TEXT model."""
        if not self.text_model:
            raise RuntimeError("Text model is not loaded.")
        try:
            results = self.text_model.predict(image, conf=confidence_threshold, verbose=False)
            return self._process_results(results, self.text_model.names)
        except Exception as e:
            logger.error(f"Error during text prediction: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Text prediction failed: {e}")

    def predict_visual_prompt(
        self,
        target_image: Image.Image,
        reference_image: Image.Image,
        prompts: Dict[str, Any],
        user_prompts_for_names: List[str],
        confidence_threshold: float
    ) -> List[Dict[str, Any]]:
        """Runs prediction using the dedicated VISUAL PROMPT model."""
        if not self.vp_model:
            raise RuntimeError("Visual Prompt model is not loaded.")
        try:
            # Prepare prompts (ensure numpy arrays, correct types)
            if 'bboxes' in prompts:
                 prompts['bboxes'] = np.array(prompts['bboxes'])
            if 'cls' in prompts:
                 prompts['cls'] = np.array(prompts['cls'], dtype=np.int32)

            # Create temporary names map for this VP call based on user input
            temp_names = {i: name for i, name in enumerate(user_prompts_for_names)}
            # We modify the names dict of the VP model instance for this call
            original_names = self.vp_model.names
            self.vp_model.names = temp_names

            logger.debug("Running inference with VISUAL PROMPT model.")
            results = self.vp_model.predict(
                target_image,
                refer_image=reference_image,
                visual_prompts=prompts,
                predictor=YOLOEVPSegPredictor, # Apply the VP predictor
                conf=confidence_threshold,
                verbose=False,
            )

            self.vp_model.names = original_names # Restore VP model's original names

            # Process results using the temporary names map
            return self._process_results(results, temp_names)

        except Exception as e:
            logger.error(f"Error during visual prompt prediction: {e}", exc_info=True)
            if 'original_names' in locals() and self.vp_model: # Ensure names are restored on error
                 self.vp_model.names = original_names
            raise HTTPException(status_code=500, detail=f"Visual prompt prediction failed: {e}")

    def _process_results(self, results, names_map: Dict) -> List[Dict[str, Any]]:
        """Helper function to process Ultralytics results into desired format."""
        if not results or len(results) == 0:
            return []

        sv_detections = sv.Detections.from_ultralytics(results[0])
        detections_list = []
        for i in range(len(sv_detections)):
            class_id = sv_detections.class_id[i]
            detections_list.append({
                "xyxy": sv_detections.xyxy[i].tolist(),
                "confidence": float(sv_detections.confidence[i]),
                "class_id": int(class_id),
                "class_name": names_map.get(class_id, "N/A"), # Use provided names map
                "tracker_id": int(sv_detections.tracker_id[i]) if sv_detections.tracker_id is not None else None,
                "mask": sv_detections.mask[i].tolist() if sv_detections.mask is not None else None,
            })
        return detections_list

# --- Initialize Model Manager ---
# Done globally so models are loaded only once on startup
try:
    model_manager = MultiModelManager(
        settings.text_model_path,
        settings.vp_model_path,
        settings.device
    )
except Exception:
    logger.critical("Application failed to start: Could not initialize MultiModelManager.", exc_info=True)
    model_manager = None # Indicate failure

# --- FastAPI App ---
app = FastAPI(title="YOLOE Multi-Model Service", version="1.1.0")

# --- Pydantic Request/Response Models ---
# (Using the same models as before, but endpoints will dictate which model is used)
class PredictTextRequest(BaseModel):
    image_base64: str = Field(..., description="Base64 encoded target image string.")
    confidence_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Optional confidence override.")

class SetClassesRequest(BaseModel):
    classes: List[str] = Field(..., description="List of class names to set for the text prompting model.")

class VisualPrompt(BaseModel):
    bboxes: List[List[float]]
    cls: List[int]

class PredictVPRequest(BaseModel):
    target_image_base64: str = Field(..., description="Base64 encoded target image.")
    reference_image_base64: str = Field(..., description="Base64 encoded reference image.")
    prompts: VisualPrompt = Field(..., description="Visual prompts.")
    user_prompts_for_names: List[str] = Field(..., description="Class names for prompt 'cls' indices.")
    confidence_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Optional confidence override.")

class DetectionResult(BaseModel):
    xyxy: List[float]
    confidence: float
    class_id: int
    class_name: str
    tracker_id: Optional[int] = None
    mask: Optional[List[Any]] = None

class PredictResponse(BaseModel):
    detections: List[DetectionResult]

class StatusResponse(BaseModel):
    status: str
    message: Optional[str] = None

# --- Helper Functions ---
def decode_base64_image(image_base64: str) -> Image.Image:
    """Decodes a base64 encoded image string to a PIL Image."""
    try:
        image_data = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_data))
        if image.mode != 'RGB':
             image = image.convert('RGB')
        return image
    except (binascii.Error, io.UnsupportedOperation, Exception) as e:
        logger.error(f"Failed to decode base64 image: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid base64 image data or format: {e}")

# --- API Endpoints ---
@app.post("/set_text_classes", response_model=StatusResponse)
def set_text_classes(request: SetClassesRequest):
    """
    Sets the target classes ONLY for the TEXT-prompted detection model.
    """
    if not model_manager or not model_manager.text_model:
         raise HTTPException(status_code=503, detail="Text Model not available.")
    try:
        model_manager.set_text_classes(request.classes)
        return {"status": "success", "message": f"Text classes set for TEXT model: {request.classes}"}
    except Exception as e:
        logger.error(f"Error in set_text_classes: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_text", response_model=PredictResponse)
def predict_text(request: PredictTextRequest):
    """
    Performs detection using the TEXT model with its currently set classes.
    """
    if not model_manager or not model_manager.text_model:
         raise HTTPException(status_code=503, detail="Text Model not available.")

    start_time = time.monotonic()
    try:
        target_image = decode_base64_image(request.image_base64)
        confidence = request.confidence_threshold if request.confidence_threshold is not None else settings.default_confidence
        results = model_manager.predict_text_prompt(target_image, confidence)
        end_time = time.monotonic()
        logger.info(f"Text prediction processed {len(results)} detections in {end_time - start_time:.4f} seconds.")
        return {"detections": results}
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.error(f"Error in /predict_text endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error during text prediction.")

@app.post("/predict_vp", response_model=PredictResponse)
def predict_vp(request: PredictVPRequest):
    """
    Performs detection using the dedicated VISUAL PROMPT model.
    """
    if not model_manager or not model_manager.vp_model:
         raise HTTPException(status_code=503, detail="Visual Prompt Model not available.")

    start_time = time.monotonic()
    try:
        target_image = decode_base64_image(request.target_image_base64)
        reference_image = decode_base64_image(request.reference_image_base64)
        prompts_dict = request.prompts.model_dump()
        confidence = request.confidence_threshold if request.confidence_threshold is not None else settings.default_confidence

        results = model_manager.predict_visual_prompt(
            target_image=target_image,
            reference_image=reference_image,
            prompts=prompts_dict,
            user_prompts_for_names=request.user_prompts_for_names,
            confidence_threshold=confidence
        )
        end_time = time.monotonic()
        logger.info(f"Visual prompt prediction processed {len(results)} detections in {end_time - start_time:.4f} seconds.")
        return {"detections": results}
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.error(f"Error in /predict_vp endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error during visual prompt prediction.")

@app.get("/health", response_model=StatusResponse)
def health_check():
    """Checks if both models required by the service are loaded."""
    if not model_manager:
         status = "error"
         message = "Model manager failed to initialize."
         status_code = 503
    elif model_manager.text_model and model_manager.vp_model:
        status = "ok"
        message = "Both Text and Visual Prompt models are loaded."
        status_code = 200
    elif model_manager.text_model:
        status = "partial"
        message = "Only Text model loaded. Visual Prompt model failed."
        status_code = 503 # Service partially unavailable
    elif model_manager.vp_model:
        status = "partial"
        message = "Only Visual Prompt model loaded. Text model failed."
        status_code = 503 # Service partially unavailable
    else:
        status = "error"
        message = "Neither model could be loaded."
        status_code = 503

    if status_code != 200:
        # For non-OK statuses, it's conventional to raise HTTPException for monitoring tools
        raise HTTPException(status_code=status_code, detail=message)
    else:
        return {"status": status, "message": message}

# --- Run Server ---
if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting multi-model server on {settings.host}:{settings.port}")
    # Ensure model_manager initialized somewhat before running
    if not model_manager:
         logger.critical("Cannot start server, Model Manager failed initialization.")
    else:
         uvicorn.run(app, host=settings.host, port=settings.port)