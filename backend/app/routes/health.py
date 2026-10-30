"""
VisionAI - Health Route
"""

from datetime import datetime, timezone
from fastapi import APIRouter
from app.core.config import settings
from app.services.model_service import model_service

router = APIRouter(tags=["Health"])

@router.get("/health", summary="System Health Status")
def health_check():
    """
    Returns operational health status, active model information, and system metadata.
    """
    is_ready = model_service.is_ready()
    active_model = model_service.predictor.model_name if (is_ready and model_service.predictor) else "None"

    return {
        "status": "healthy",
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_loaded": is_ready,
        "active_model": active_model
    }
