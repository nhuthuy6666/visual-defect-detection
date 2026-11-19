import io
import os
import time
from typing import Dict, List

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

ONNX_MODEL_PATH = os.getenv("ONNX_MODEL_PATH", "./checkpoints/model.onnx")
IMG_SIZE = int(os.getenv("IMG_SIZE", "224"))

DEFAULT_CLASS_NAMES: List[str] = [
    "crazing", 
    "iclusion", 
    "patches", 
    "pitted_surface", 
    "rolled-in_scale", 
    "scratches"
]

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    latency_ms: float
    all_probabilities: Dict[str, float]


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp_vals = np.exp(shifted)
    return exp_vals / np.sum(exp_vals, axis=-1, keepdims=True)


def preprocess_image(image_bytes: bytes, img_size: int = 224) -> np.ndarray:
    """
    Chuyển bytes ảnh đầu vào thành tensor (1, 3, H, W) float32 đã chuẩn hóa.
    """
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="File tải lên không phải là ảnh hợp lệ.") from exc

    resize_target = int(img_size * 256 / 224)
    image = image.resize((resize_target, resize_target), Image.Resampling.BILINEAR)

    left = (resize_target - img_size) // 2
    top = (resize_target - img_size) // 2
    image = image.crop((left, top, left + img_size, top + img_size))

    arr = np.asarray(image, dtype=np.float32) / 255.0  
    chw = np.transpose(arr, (2, 0, 1))                 
    normalized = (chw - MEAN) / STD
    return np.expand_dims(normalized, axis=0).astype(np.float32)  


app = FastAPI(
    title="Visual Defect Detection API",
    description="API for classifying surface defects on components, utilizing an ONNX Runtime model.",
    version="1.0.0",
)

ort_session: ort.InferenceSession | None = None
input_name: str = "input"
class_names: List[str] = DEFAULT_CLASS_NAMES.copy()


@app.on_event("startup")
def load_onnx_model():
    global ort_session, input_name, class_names

    if not os.path.exists(ONNX_MODEL_PATH):
        print("file not found")
        return

    available_providers = ort.get_available_providers()
    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if "CUDAExecutionProvider" in available_providers
        else ["CPUExecutionProvider"]
    )

    ort_session = ort.InferenceSession(ONNX_MODEL_PATH, providers=providers)
    input_name = ort_session.get_inputs()[0].name
    output_shape = ort_session.get_outputs()[0].shape


    num_outputs = output_shape[-1] if isinstance(output_shape[-1], int) else len(DEFAULT_CLASS_NAMES)
    if num_outputs != len(class_names):
        class_names = [f"class_{i}" for i in range(num_outputs)]


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model_loaded": ort_session is not None,
        "onnx_path": ONNX_MODEL_PATH,
        "classes": class_names,
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_defect(file: UploadFile = File(...)):
    if ort_session is None:
        raise HTTPException(
            status_code=503,
            detail=f"The ONNX model has not been loaded. Please check the file at: {ONNX_MODEL_PATH}",
        )

    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image file formats (image/*) are accepted.")

    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Empty image file.")

    input_tensor = preprocess_image(raw_bytes, img_size=IMG_SIZE)

    t0 = time.perf_counter()
    logits = ort_session.run(None, {input_name: input_tensor})[0][0]
    latency_ms = (time.perf_counter() - t0) * 1000.0

    probs = softmax(logits)
    pred_idx = int(np.argmax(probs))
    predicted_class = class_names[pred_idx] if pred_idx < len(class_names) else str(pred_idx)
    confidence = float(probs[pred_idx])

    all_probs = {
        (class_names[i] if i < len(class_names) else f"class_{i}"): round(float(probs[i]), 6)
        for i in range(len(probs))
    }

    return JSONResponse(
        content={
            "predicted_class": predicted_class,
            "confidence": round(confidence, 6),
            "latency_ms": round(latency_ms, 2),
            "all_probabilities": all_probs,
        }
    )