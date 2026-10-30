"""
VisionAI - Prediction Pydantic Schemas
"""

from typing import List, Optional
from pydantic import BaseModel, Field

class PredictionItem(BaseModel):
    rank: int = Field(..., description="Rank in top predictions (1 is primary)")
    class_id: int = Field(..., description="Numeric class index")
    class_name: str = Field(..., description="Human-readable category label")
    confidence: float = Field(..., description="Calculated softmax probability between 0.0 and 1.0")
    confidence_percentage: str = Field(..., description="Formatted percentage string")

class ModelMetadata(BaseModel):
    name: str = Field(..., description="Name of the active model")
    type: str = Field(..., description="Model architecture type (transfer_learning or baseline_cnn)")
    input_resolution: str = Field(..., description="Input tensor dimension (e.g. 224x224)")
    original_resolution: str = Field(..., description="Original dimensions of the uploaded image")

class GradCamResult(BaseModel):
    status: str = Field(..., description="Status of explainability computation ('success' or 'unavailable')")
    method: str = Field(..., description="Algorithm used (Grad-CAM)")
    target_class: str = Field(..., description="Target predicted class for activation mapping")
    overlay_base64: Optional[str] = Field(None, description="Base64 data URL of the overlaid heatmap")
    explanation: str = Field(..., description="Plain-language description of what the heatmap reveals")
    disclaimer: str = Field(..., description="Scientific disclaimer on feature attention vs causality")

class PredictionResponse(BaseModel):
    success: bool = True
    predicted_class: str = Field(..., description="Primary classification outcome")
    confidence: float = Field(..., description="Top confidence score")
    confidence_percentage: str = Field(..., description="Top confidence score percentage")
    top_predictions: List[PredictionItem] = Field(..., description="Top-K predicted categories")
    inference_time_ms: float = Field(..., description="Pure model inference duration in milliseconds")
    model_info: ModelMetadata = Field(..., description="Metadata of the serving model")
    explainability: Optional[GradCamResult] = Field(None, description="Grad-CAM visual attribution details")
    disclaimer: str = Field(..., description="Statistical confidence disclaimer")

class ErrorResponse(BaseModel):
    success: bool = False
    detail: str = Field(..., description="Safe human-readable error description")
    error_code: str = Field(..., description="Application error classification code")
