
from __future__ import annotations

import os
import uuid
from pathlib import Path
from io import BytesIO

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from .inference import CrowdCounterService


# --------------------------------------------------
# Project configuration
# --------------------------------------------------

# main.py location:
# D:\aerial-crowd-counting\aerial-crowd-backend\backend\main.py
#
# parents[0] = backend
# parents[1] = aerial-crowd-backend
# parents[2] = aerial-crowd-counting

PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Stage 1 trained checkpoint
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "error_aware"
    / "checkpoints_stage1_holdout80"
    / "coarse_best.pth"
)

# Allow checkpoint override through environment variable
CHECKPOINT = Path(
    os.getenv(
        "COARSE_CHECKPOINT",
        str(DEFAULT_CHECKPOINT),
    )
).expanduser()

# Resolve relative environment paths against project root
if not CHECKPOINT.is_absolute():
    CHECKPOINT = PROJECT_ROOT / CHECKPOINT

DEVICE = os.getenv("MODEL_DEVICE", "auto")
TILE_SIZE = int(os.getenv("TILE_SIZE", "512"))


# --------------------------------------------------
# FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="AirCount API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount(
    "/outputs",
    StaticFiles(directory=str(OUTPUT_DIR)),
    name="outputs",
)


# --------------------------------------------------
# Model service
# --------------------------------------------------

service: CrowdCounterService | None = None


@app.on_event("startup")
def load_model() -> None:
    global service

    print(f"[AirCount] Project root: {PROJECT_ROOT}")
    print(f"[AirCount] Expected checkpoint: {CHECKPOINT}")

    if not CHECKPOINT.exists():
        print(f"[AirCount] Checkpoint not found: {CHECKPOINT}")
        print(
            "[AirCount] API will start, but /predict "
            "will return a configuration error."
        )
        return

    try:
        service = CrowdCounterService(
            checkpoint_path=CHECKPOINT,
            device_name=DEVICE,
            tile_size=TILE_SIZE,
        )

        print(
            f"[AirCount] Model loaded on {service.device}: "
            f"{CHECKPOINT}"
        )

    except Exception as exc:
        service = None
        print(f"[AirCount] Model loading failed: {exc}")
        raise


# --------------------------------------------------
# Root endpoint
# --------------------------------------------------

@app.get("/")
def root():
    return {
        "name": "AirCount API",
        "status": "running",
    }


# --------------------------------------------------
# Health endpoint
# --------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "ok" if service is not None else "model_not_loaded",
        "device": str(service.device) if service is not None else None,
        "checkpoint": str(CHECKPOINT),
        "checkpoint_exists": CHECKPOINT.exists(),
    }


# --------------------------------------------------
# Crowd prediction endpoint
# --------------------------------------------------

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # Check model availability
    if service is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Model is not loaded. "
                f"Check checkpoint path: {CHECKPOINT}"
            ),
        )

    # Validate uploaded file type
    allowed = {
        "image/jpeg",
        "image/png",
        "image/webp",
    }

    if file.content_type not in allowed:
        raise HTTPException(
            status_code=415,
            detail="Upload a JPG, PNG, or WebP image.",
        )

    # Read uploaded image
    content = await file.read()

    if not content:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty.",
        )

    # Maximum upload size: 20 MB
    if len(content) > 20 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="Maximum image size is 20 MB.",
        )

    # Decode image
    try:
        image = Image.open(
            BytesIO(content)
        ).convert("RGB")

    except (UnidentifiedImageError, OSError):
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid image.",
        )

    # Run crowd counting inference
    try:
        result = service.predict(image)

    except Exception as exc:
        print(f"[AirCount] Inference error: {exc}")

        raise HTTPException(
            status_code=500,
            detail=f"Inference failed: {exc}",
        )

    # Save density heatmap
    output_name = f"{uuid.uuid4().hex}.png"
    output_path = OUTPUT_DIR / output_name

    result["density_map"].save(output_path)

    # Return prediction response
    return {
        "predicted_count": result["predicted_count"],
        "inference_time_ms": round(
            result["inference_time_ms"],
            2,
        ),
        "image_width": image.width,
        "image_height": image.height,
        "density_map_url": f"/outputs/{output_name}",
        "tiles_processed": result["tiles_processed"],
        "model": "Stage-1 CoarseCounter",
    }
