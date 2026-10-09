from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import os
import tempfile

from ml.yogamat_inference import run_yogamat_inference


app = FastAPI(
    title="YOGAMAT ML API",
    version="1.0.0"
)


# --------------------------------------------------
# CORS
# --------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# Health check
# --------------------------------------------------

@app.get("/")
def root():
    return {
        "message": "YOGAMAT ML backend is running"
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "YOGAMAT ML"
    }


# --------------------------------------------------
# Pose prediction
# --------------------------------------------------

@app.post("/predict")
async def predict_pose(file: UploadFile = File(...)):

    if not file.content_type:
        raise HTTPException(
            status_code=400,
            detail="Invalid file type."
        )

    allowed_types = [
        "image/jpeg",
        "image/png",
        "image/jpg"
    ]

    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail="Please upload a JPG or PNG image."
        )

    file_bytes = await file.read()

    if not file_bytes:
        raise HTTPException(
            status_code=400,
            detail="Uploaded image is empty."
        )

    temp_path = None

    try:

        suffix = ".jpg"

        if file.filename:
            extension = os.path.splitext(
                file.filename
            )[1].lower()

            if extension in [".jpg", ".jpeg", ".png"]:
                suffix = extension

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix
        ) as temp_file:

            temp_file.write(file_bytes)
            temp_path = temp_file.name

        result = run_yogamat_inference(
            temp_path
        )

        if not result.get("success"):
            raise HTTPException(
                status_code=422,
                detail=result.get(
                    "message",
                    "Pose could not be detected."
                )
            )

        return {
            "success": True,
            "predicted_pose": result["predicted_pose"],
            "pose_confidence": result["pose_confidence"],
            "form_analysis": result["form_analysis"]
        }

    except HTTPException:
        raise

    except Exception as error:

        print(
            "Prediction error:",
            str(error)
        )

        raise HTTPException(
            status_code=500,
            detail="Failed to process the image."
        )

    finally:

        if temp_path and os.path.exists(temp_path):

            os.remove(temp_path)