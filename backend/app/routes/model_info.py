"""
VisionAI - Model Information Route
"""

from fastapi import APIRouter
from app.schemas.model_info import ModelInfoResponse
from app.services.model_service import model_service

router = APIRouter(tags=["Model Info"])

@router.get("/model-info", response_model=ModelInfoResponse, summary="Inspect Active Model Architecture")
def get_model_information():
    """
    Returns architecture specs, parameter counts, input requirements,
    and serving status for the current active neural network.
    """
    return model_service.get_model_info()
