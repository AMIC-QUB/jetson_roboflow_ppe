from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor
import torch
import logging
from PIL import Image
import io
import base64
import numpy as np
import cv2
# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()
class_colors = {}  # Dictionary to store consistent colors for each class

# Model Manager class to encapsulate the YOLOE model
class ModelManager:
    def __init__(self):
        try:
            self.model = YOLOE("yoloe-11l-seg.pt")
            self.model.to('cuda')
            logger.info("YOLOE model loaded successfully in ModelManager")
            self.vp=False
        except Exception as e:
            logger.error(f"Failed to load YOLOE model: {e}")
            raise
        self.class_colours = {}
        self.refer_frame = None
    def predict_with_visual_prompts(self, source_image, prompts, user_prompts):
        """Generate visual prompt embeddings (vpe) and set classes."""
        if self.refer_frame is None:  # Set reference frame only once
            self.refer_frame = np.array(source_image)
        self.refer_image = Image.fromarray(cv2.cvtColor(self.refer_frame, cv2.COLOR_BGR2RGB))
        prompts['bboxes'] = np.array(prompts['bboxes'])
        prompts['cls'] = np.array(prompts['cls'], dtype=np.int32)
        self.prompts = prompts.copy()
        results = self.model.predict(
                    source_image,
                    refer_image=self.refer_image,
                    visual_prompts=self.prompts,
                    predictor=YOLOEVPSegPredictor,
                )
        self.vp=True
        return results
        

    def predict(self, target_image, user_prompts):
        """Run inference on the target image."""
        # Ensure the predictor is set before inference
        if self.vp:
            logger.debug("running inference with vpe embeddings")
            results = self.model.predict(
                    target_image,
                    refer_image=self.refer_image,
                    visual_prompts=self.prompts,
                    predictor=YOLOEVPSegPredictor,
                )
        else:
            self.model.set_classes(user_prompts, self.model.get_text_pe(user_prompts))
            results = self.model.predict(target_image, conf=0.3, verbose=True)
        logger.debug("Inference completed, results: %s", len(results))
        return results

    def set_classes_without_vpe(self, classes):
        """Set classes without visual prompt embeddings."""
        # Ensure the predictor is set before setting classes
        # self.model.predictor=YOLOEVPSegPredictor
        self.model = YOLOE("yoloe-11l-seg.pt").cuda()
        self.model.set_classes(classes, self.model.get_text_pe(classes))
        logger.debug("Classes set without vpe: %s", classes)

# Instantiate the ModelManager
model_manager = ModelManager()

# Pydantic models for request validation
class VisualPromptRequest(BaseModel):
    image_base64: str
    prompts: dict
    user_prompts: list

class PredictRequest(BaseModel):
    image_base64: str
    user_prompts: list

class SetClassesRequest(BaseModel):
    classes: list

# Helper function to decode base64 image
def decode_base64_image(image_base64: str) -> Image.Image:
    try:
        image_data = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_data))
        return image
    except Exception as e:
        logger.error(f"Failed to decode base64 image: {e}")
        raise HTTPException(status_code=400, detail="Invalid base64 image")

# API endpoints
@app.post("/predict_with_visual_prompts")
async def predict_with_visual_prompts(request: VisualPromptRequest):
    # try:
        # Decode the base64 image
    source_image = decode_base64_image(request.image_base64)
    prompts = request.prompts
    user_prompts = request.user_prompts

    # Run visual prompting
    model_manager.predict_with_visual_prompts(source_image, prompts, user_prompts)
    return {"status": "success"}
    # except Exception as e:
    #     logger.error(f"Error in predict_with_visual_prompts: {e}")
    #     raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict")
async def predict(request: PredictRequest):
    global class_colors
    try:
        # Decode the base64 image
        target_image = decode_base64_image(request.image_base64)

        # Run inference
        results = model_manager.predict(target_image, request.user_prompts)

        # Process results into a JSON-serializable format
        filtered_predictions = []
        for r in results:
            boxes = r.boxes
            masks = r.masks if hasattr(r, 'masks') else None
            for i, box in enumerate(boxes):
                xyxy = box.xyxy[0].cpu().numpy()
                conf = box.conf.cpu().numpy().item()
                cls = int(box.cls.cpu().numpy())
                label = request.user_prompts[cls] if cls < len(request.user_prompts) else "Unknown"

                mask_data = None
                if masks is not None:
                    mask = masks[i].data[0].cpu().numpy()
                    mask = cv2.resize(mask, (target_image.size[0], target_image.size[1]), interpolation=cv2.INTER_NEAREST)
                    mask = mask.astype(np.uint8)
                    mask_data = mask.tolist()

                pred = {
                    "x": int((xyxy[0] + xyxy[2]) / 2),
                    "y": int((xyxy[1] + xyxy[3]) / 2),
                    "width": int(xyxy[2] - xyxy[0]),
                    "height": int(xyxy[3] - xyxy[1]),
                    "x1": int(xyxy[0]),
                    "y1": int(xyxy[1]),
                    "x2": int(xyxy[2]),
                    "y2": int(xyxy[3]),
                    "class": label,
                    "confidence": conf,
                    "color": class_colors.setdefault(label, (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))),
                    "mask": mask_data
                }
                filtered_predictions.append(pred)

        return {"detections": filtered_predictions}
    except Exception as e:
        logger.error(f"Error in predict: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/set_classes_without_vpe")
async def set_classes_without_vpe(request: SetClassesRequest):
    try:
        classes = request.classes
        model_manager.set_classes_without_vpe(classes)
        return {"status": "success"}
    except Exception as e:
        logger.error(f"Error in set_classes_without_vpe: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)