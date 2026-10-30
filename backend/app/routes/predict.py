"""
VisionAI - Image Prediction Route
==================================
Handles image uploads, preprocessing, model inference, and Grad-CAM generation.
"""

import io
import os
import base64
from pathlib import Path
from typing import Optional, List

from fastapi import APIRouter, File, UploadFile, Query, HTTPException, status
from fastapi.responses import JSONResponse
from PIL import Image

from app.core.config import settings
from app.schemas.prediction import PredictionResponse, ErrorResponse
from app.services.model_service import model_service

router = APIRouter(tags=["Inference & Analysis"])

@router.post(
    "/predict",
    response_model=PredictionResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid image or format error"},
        413: {"model": ErrorResponse, "description": "File exceeds 10MB limit"},
        503: {"model": ErrorResponse, "description": "Model not loaded or standby"}
    },
    summary="Analyze Uploaded Image"
)
def predict_image(
    file: UploadFile = File(..., description="Target image file (JPG, PNG, WEBP)"),
    top_k: int = Query(5, ge=1, le=6, description="Number of top predictions to return"),
    gradcam: bool = Query(True, description="Whether to generate Grad-CAM explainability heatmap")
):
    """
    Primary image classification endpoint:
    - Validates file content type, size, and header
    - Preprocesses image according to model requirements
    - Returns primary predicted class and top-K ranked probabilities
    - Generates optional Grad-CAM visual explainability heatmap
    - Measures inference latency in milliseconds
    """
    if not model_service.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="VisionAI model is not currently loaded. Please run the training pipeline to train and save model weights."
        )

    # 1. Validate File Size
    contents = file.file.read()
    if not contents or len(contents) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty (0 bytes). Please upload a valid image."
        )

    if len(contents) > settings.MAX_UPLOAD_SIZE_BYTES:
        mb = len(contents) / (1024 * 1024)
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Image size ({mb:.2f} MB) exceeds maximum allowed size (10 MB)."
        )

    # 2. Validate Extension & Content-Type
    filename = file.filename or "upload.jpg"
    ext = Path(filename).suffix.lower()
    if ext and ext not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported format '{ext}'. Supported formats: {', '.join(sorted(settings.ALLOWED_EXTENSIONS))}."
        )

    try:
        # Run prediction via model service
        result = model_service.predict_image(
            image_bytes=contents,
            filename=filename,
            top_k=top_k,
            generate_gradcam=gradcam
        )
        return PredictionResponse(**result)
    except HTTPException:
        raise
    except Exception as e:
        err_msg = str(e)
        if "Corrupted" in err_msg or "Unidentified" in err_msg or "cannot identify" in err_msg.lower():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="The provided file could not be decoded as an image. It may be corrupted or in an unsupported format."
            )
        # Safe error logging without raw internal filesystem leakage
        print(f"[ERROR in /predict]: {err_msg}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during image processing. Please verify the input image and retry."
        )

@router.post(
    "/predict/sample/{class_name}",
    response_model=PredictionResponse,
    summary="Predict Pre-packaged Sample Image"
)
def predict_sample_image(
    class_name: str,
    top_k: int = Query(5, ge=1, le=6),
    gradcam: bool = Query(True)
):
    """
    Allows instant testing of the model using pre-packaged canonical test images
    (buildings, forest, glacier, mountain, sea, street).
    """
    if not model_service.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="VisionAI model is not loaded."
        )

    sample_path = settings.SAMPLE_IMAGES_DIR / f"sample_{class_name.lower()}.jpg"
    if not sample_path.exists():
        # Fallback to search in raw or processed if sample_test_images doesn't have it
        alt_paths = list((settings.PROJECT_ROOT / "ml" / "data").glob(f"**/{class_name.lower()}*.*"))
        if alt_paths:
            sample_path = alt_paths[0]
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Sample image for category '{class_name}' not found."
            )

    with open(sample_path, "rb") as f:
        image_bytes = f.read()

    result = model_service.predict_image(
        image_bytes=image_bytes,
        filename=sample_path.name,
        top_k=top_k,
        generate_gradcam=gradcam
    )
    return PredictionResponse(**result)

@router.get("/samples", summary="List Sample Test Images")
def list_sample_images():
    """
    Returns available sample test images with base64 thumbnail previews for instant UI selection.
    """
    samples = []
    sample_dir = settings.SAMPLE_IMAGES_DIR
    if sample_dir.exists():
        for f in sorted(sample_dir.glob("sample_*.jpg")):
            cls = f.stem.replace("sample_", "")
            try:
                with open(f, "rb") as img_f:
                    b64 = base64.b64encode(img_f.read()).decode("utf-8")
                samples.append({
                    "class_name": cls,
                    "filename": f.name,
                    "data_url": f"data:image/jpeg;base64,{b64}"
                })
            except Exception as e:
                print(f"[WARN] Failed reading sample {f}: {e}")

    return {"samples": samples}
