"""
VisionAI - Metrics Schemas
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

class PerClassMetric(BaseModel):
    precision: float
    recall: float
    f1_score: float = Field(..., alias="f1-score")
    support: int

    class Config:
        populate_by_name = True

class ModelEvaluationMetrics(BaseModel):
    model_name: str
    test_loss: float
    test_accuracy: float
    weighted_precision: float
    weighted_recall: float
    weighted_f1: float
    macro_f1: float
    per_class: Dict[str, PerClassMetric]
    confusion_matrix: List[List[int]]
    classes: List[str]

class MetricsResponse(BaseModel):
    status: str = Field(..., description="'available' or 'not_available'")
    evaluation_dataset: Optional[str] = None
    num_test_samples: Optional[int] = None
    classes: Optional[List[str]] = None
    models: Optional[Dict[str, ModelEvaluationMetrics]] = None
    message: Optional[str] = None
