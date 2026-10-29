from __future__ import annotations

import io
from pathlib import Path

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from torchvision import transforms

from model import build_model

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "artifacts" / "best_model.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = FastAPI(
    title="Deepfake Detection AI",
    version="0.1.0",
    description="Educational real/fake image classifier. Prediction is not forensic proof.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
image_size = 224
preprocess = None


def load_checkpoint():
    global model, image_size, preprocess
    if not MODEL_PATH.exists():
        model = None
        preprocess = None
        return
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    image_size = int(checkpoint.get("image_size", 224))
    model = build_model(pretrained=False).to(DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    preprocess = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


load_checkpoint()


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None, "device": str(DEVICE)}


@app.get("/model-info")
def model_info():
    return {
        "architecture": "EfficientNet-B0",
        "classes": ["fake", "real"],
        "trained_checkpoint_available": model is not None,
        "image_size": image_size,
        "warning": "Educational prototype; output is not proof of authenticity.",
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None or preprocess is None:
        raise HTTPException(
            status_code=503,
            detail="No trained model found. Prepare the dataset and run backend/train.py first.",
        )
    if file.content_type not in {"image/jpeg", "image/png", "image/webp", "image/bmp"}:
        raise HTTPException(status_code=415, detail="Upload a JPG, PNG, WEBP, or BMP image.")
    raw = await file.read()
    if len(raw) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image must be 10 MB or smaller.")
    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="Could not read this image.")
    tensor = preprocess(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probabilities = torch.softmax(model(tensor), dim=1)[0].cpu()
    fake_probability = float(probabilities[0].item())
    real_probability = float(probabilities[1].item())
    return {
        "prediction": "fake" if fake_probability >= 0.5 else "real",
        "fake_probability": round(fake_probability, 4),
        "real_probability": round(real_probability, 4),
        "threshold": 0.5,
        "filename": file.filename,
        "caveat": "Probabilistic model output may be wrong; it is not proof of authenticity.",
    }
