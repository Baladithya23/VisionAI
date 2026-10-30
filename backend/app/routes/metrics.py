"""
VisionAI - Metrics & Reports Route
===================================
Serves evaluation metrics, confusion matrices, and comparison figures.
"""

from pathlib import Path
from fastapi import APIRouter, HTTPException, status
from fastapi.responses import FileResponse

from app.core.config import settings
from app.schemas.metrics import MetricsResponse
from app.services.model_service import model_service

router = APIRouter(tags=["Evaluation Metrics"])

@router.get("/metrics", response_model=MetricsResponse, summary="Get Model Evaluation Metrics")
def get_metrics():
    """
    Returns empirical evaluation metrics (test accuracy, weighted/macro F1,
    per-class precision/recall, and confusion matrix) computed on the unseen test split.
    Displays 'not_available' if models have not yet been evaluated.
    """
    return model_service.get_evaluation_metrics()

@router.get("/metrics/plots/{plot_name}", summary="Download Evaluation Visualizations")
def get_metric_plot(plot_name: str):
    """
    Serves generated report plots safely (confusion_matrix, model_comparison, training_curves).
    Guards strictly against directory traversal.
    """
    allowed_plots = {
        "confusion_matrix_baseline.png",
        "confusion_matrix_mobilenetv2.png",
        "model_comparison.png",
        "baseline_training_curves.png",
        "transfer_learning_training_curves.png",
        "class_distribution.png"
    }

    clean_name = Path(plot_name).name
    if clean_name not in allowed_plots:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Report visualization '{clean_name}' not recognized."
        )

    plot_path = settings.REPORTS_DIR / clean_name
    if not plot_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Visualization '{clean_name}' has not been generated yet."
        )

    return FileResponse(path=plot_path, media_type="image/png", filename=clean_name)
