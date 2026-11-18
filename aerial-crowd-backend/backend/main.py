
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
from .detection import HybridCounterService, configure_detector


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

# Retain the original as a fallback when validated fine-tuned weights are absent.
BASELINE_CHECKPOINT = (
    PROJECT_ROOT
    / "error_aware"
    / "checkpoints_stage1_holdout80"
    / "coarse_best.pth"
)
VALIDATED_CHECKPOINT = (
    PROJECT_ROOT / "error_aware" / "checkpoints_balanced_low_vram" / "coarse_validated.pth"
)
DEFAULT_CHECKPOINT = VALIDATED_CHECKPOINT if VALIDATED_CHECKPOINT.exists() else BASELINE_CHECKPOINT

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
# The sparse counting policy is pinned to the evaluated checkpoint's SHA-256.
COUNT_POLICY_PATH = Path(__file__).with_name("count_policy.json")
COUNTING_MODE = os.getenv("COUNTING_MODE", "auto")
# Auto mode is enabled only by a benchmarked policy pinned to both model files.
DETECTOR_POLICY_PATH = Path(__file__).with_name("person_detector_policy.json")
DETECTOR_CHECKPOINT = Path(os.getenv("PERSON_DETECTOR_CHECKPOINT", str(
    PROJECT_ROOT / "checkpoints/person_detection/fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth"
))).expanduser()
if not DETECTOR_CHECKPOINT.is_absolute():
    DETECTOR_CHECKPOINT = PROJECT_ROOT / DETECTOR_CHECKPOINT


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

service: CrowdCounterService | HybridCounterService | None = None
detector_status = "not_loaded"


@app.on_event("startup")
def load_model() -> None:
    global service, detector_status

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
            count_policy_path=COUNT_POLICY_PATH if COUNT_POLICY_PATH.exists() else None,
        )
        service, detector_status = configure_detector(
            service, COUNTING_MODE, DETECTOR_CHECKPOINT, DETECTOR_POLICY_PATH, CHECKPOINT
        )

        print(
            f"[AirCount] Model loaded on {service.device}: "
            f"{CHECKPOINT}"
        )
        print(f"[AirCount] Person detector: {detector_status}")

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
        "count_policy": service.count_policy if service is not None else None,
        "counting_mode": COUNTING_MODE,
        "detector_status": detector_status,
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
    detection_overlay_url = None
    if result.get("detection_overlay") is not None:
        overlay_name = f"{Path(output_name).stem}_detections.png"
        result["detection_overlay"].save(OUTPUT_DIR / overlay_name)
        detection_overlay_url = f"/outputs/{overlay_name}"

    # Return prediction response
    return {
        "predicted_count": result["predicted_count"],
        "raw_count": result["raw_count"],
        "estimated_count": result["estimated_count"],
        "count_method": result["count_method"],
        "inference_time_ms": round(
            result["inference_time_ms"],
            2,
        ),
        "image_width": image.width,
        "image_height": image.height,
        "density_map_url": f"/outputs/{output_name}",
        "detection_overlay_url": detection_overlay_url,
        "detections": result.get("detections", []),
        "density_predicted_count": result.get("density_predicted_count", result["predicted_count"]),
        "detector_selection_reason": result.get("detector_selection_reason"),
        "tiles_processed": result["tiles_processed"],
        "model": result.get("model", "Stage-1 CoarseCounter"),
    }
