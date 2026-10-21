"""FastAPI service: upload a skin image, get back a predicted lesion mask."""
import base64
import io
import os
from contextlib import asynccontextmanager

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image

from model import IMG_SIZE, load_model

MODEL_PATH = os.getenv("MODEL_PATH", "models/best_model.pth")
state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["model"] = load_model(MODEL_PATH)  # loaded once at startup
    yield
    state.clear()


app = FastAPI(title="ISIC Lesion Segmentation API", version="1.0", lifespan=lifespan)


def run_inference(data: bytes):
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="File is not a valid image.")

    original_size = img.size
    x = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    x = torch.from_numpy(np.asarray(x, dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)

    with torch.no_grad():
        prob = torch.sigmoid(state["model"](x))[0, 0].numpy()

    mask = (prob > 0.5).astype(np.uint8) * 255
    area_percent = float((mask > 0).mean() * 100)
    mask_img = Image.fromarray(mask).resize(original_size, Image.NEAREST)
    return mask_img, area_percent


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in state}


@app.post("/predict")
def predict(file: UploadFile = File(...)):
    mask_img, area = run_inference(file.file.read())
    buf = io.BytesIO()
    mask_img.save(buf, format="PNG")
    return {
        "filename": file.filename,
        "lesion_area_percent": round(area, 2),
        "mask_png_base64": base64.b64encode(buf.getvalue()).decode("utf-8"),
        "disclaimer": "Research prototype. Not for clinical use.",
    }


@app.post("/predict/mask")
def predict_mask(file: UploadFile = File(...)):
    mask_img, _ = run_inference(file.file.read())
    buf = io.BytesIO()
    mask_img.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")