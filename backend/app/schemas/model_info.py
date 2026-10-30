"""
VisionAI - Model Info Schemas
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class LayerSummary(BaseModel):
    name: str
    type: str
    output_shape: str
    param_count: int

class ModelInfoResponse(BaseModel):
    is_model_loaded: bool = Field(..., description="Whether a trained model is currently loaded in memory")
    active_model_name: str = Field(..., description="Name of the model serving predictions")
    model_architecture: str = Field(..., description="Architecture description")
    input_shape: List[int] = Field(..., description="Expected input shape, e.g. [224, 224, 3]")
    num_classes: int = Field(..., description="Total output classes")
    classes: List[str] = Field(..., description="List of recognized class labels")
    total_parameters: Optional[int] = Field(None, description="Total parameters in model")
    trainable_parameters: Optional[int] = Field(None, description="Trainable parameter count")
    supported_formats: List[str] = Field(..., description="Supported image upload formats")
    max_file_size_mb: int = Field(..., description="Max upload limit in MB")
    status_message: str = Field(..., description="Current operational status message")
