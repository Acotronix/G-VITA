
import os
import tempfile

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from ml.yogamat_inference import (
    run_yogamat_inference,
    get_model_info,
)

app = FastAPI(title="YOGAMAT ML API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

MAX_IMAGE_SIZE = 8 * 1024 * 1024
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "YOGAMAT ML",
        "model": get_model_info(),
    }


@app.post("/api/ml/predict")
async def predict_pose(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=415,
            detail="Upload a JPEG, PNG, or WebP image.",
        )

    image_bytes = await file.read()

    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail="The uploaded image is empty.",
        )

    if len(image_bytes) > MAX_IMAGE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="Image must be 8 MB or smaller.",
        )

    temp_path = None

    try:
        suffix = {
            "image/jpeg": ".jpg",
            "image/png": ".png",
            "image/webp": ".webp",
        }[file.content_type]

        with tempfile.NamedTemporaryFile(
            suffix=suffix,
            delete=False,
        ) as temp:
            temp.write(image_bytes)
            temp_path = temp.name

        result = run_yogamat_inference(temp_path)

        # Convert NumPy arrays into JSON-compatible values.
        if result.get("keypoints") is not None:
            result["keypoints"] = np.asarray(
                result["keypoints"]
            ).tolist()

        return result

    except HTTPException:
        raise
    except Exception:
        import logging
        logging.exception("YOGAMAT inference failed")

        raise HTTPException(
            status_code=500,
            detail="Pose inference failed. Check the backend logs.",
        )
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        await file.close()
@app.get("/")
def root():
    return {
        "message": "YOGAMAT ML API is running",
        "docs": "/docs",
        "health": "/health"
    }