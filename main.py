"""
Gloriosa Seed Classifier — FastAPI + ONNX Runtime
==================================================
Runs on Render free tier (512MB RAM).
No PyTorch needed on the server — only onnxruntime (~150MB).

Endpoints:
  GET  /           → status
  GET  /health     → keep-alive ping (used by UptimeRobot)
  POST /predict    → image file upload  → JSON result
  POST /predict_base64 → base64 JSON   → JSON result  (for ESP32-CAM)
"""

import os
import io
import json
import base64
import numpy as np
import onnxruntime as ort
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Load metadata ──────────────────────────────────────────
META_PATH  = os.path.join(os.path.dirname(__file__), "model_meta.json")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.onnx")

with open(META_PATH) as f:
    meta = json.load(f)

idx_to_class = {int(k): v for k, v in meta["idx_to_class"].items()}
num_classes  = meta["num_classes"]
image_size   = meta["image_size"]
mean         = np.array(meta["imagenet_mean"], dtype=np.float32)
std          = np.array(meta["imagenet_std"],  dtype=np.float32)
best_acc     = meta.get("best_val_accuracy", 0.0)

# ── Load ONNX model ONCE at startup ────────────────────────
print("[Startup] Loading ONNX model...")
session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
input_name  = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
print(f"[Startup] Ready — Accuracy: {best_acc:.2%} | Classes: {list(idx_to_class.values())}")

# ── FastAPI ─────────────────────────────────────────────────
app = FastAPI(
    title="Gloriosa Seed Classifier API",
    description="EfficientNet-B0 (ONNX) — Diseased vs Healthy Gloriosa seed classification",
    version="1.0.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ── Preprocessing ───────────────────────────────────────────
def preprocess(pil_image: Image.Image) -> np.ndarray:
    img = pil_image.convert("RGB").resize((image_size, image_size))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - mean) / std                          # ImageNet normalisation
    arr = np.transpose(arr, (2, 0, 1))                # HWC → CHW
    return np.expand_dims(arr, axis=0).astype(np.float32)   # → (1,3,H,W)


# ── Inference ───────────────────────────────────────────────
def run_inference(pil_image: Image.Image) -> dict:
    inp     = preprocess(pil_image)
    outputs = session.run([output_name], {input_name: inp})[0][0]  # shape: (num_classes,)

    # Softmax
    e = np.exp(outputs - outputs.max())
    probs = e / e.sum()

    pred_idx = int(np.argmax(probs))
    return {
        "prediction":    idx_to_class[pred_idx],
        "confidence":    round(float(probs[pred_idx]) * 100, 2),
        "probabilities": {idx_to_class[i]: round(float(probs[i]), 4) for i in range(num_classes)},
        "status":        "success",
    }


# ── Routes ──────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "service":  "Gloriosa Seed Classifier",
        "model":    "EfficientNet-B0 (ONNX)",
        "accuracy": f"{best_acc:.2%}",
        "classes":  list(idx_to_class.values()),
        "status":   "running",
    }

@app.get("/health")
def health():
    """Lightweight ping — UptimeRobot calls this every 5 min to prevent sleep."""
    return {"status": "ok"}

@app.post("/predict")
async def predict_file(file: UploadFile = File(...)):
    """Accept image file — used by browser or ESP32-CAM (multipart/form-data)."""
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload must be an image.")
    try:
        data = await file.read()
        img  = Image.open(io.BytesIO(data))
        return run_inference(img)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class B64Request(BaseModel):
    image_base64: str   # plain base64 or data URI

@app.post("/predict_base64")
def predict_base64(req: B64Request):
    """Accept base64 JSON — alternative for ESP32 if multipart is too heavy."""
    try:
        b64 = req.image_base64
        if "," in b64:
            b64 = b64.split(",", 1)[1]
        img = Image.open(io.BytesIO(base64.b64decode(b64)))
        return run_inference(img)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
