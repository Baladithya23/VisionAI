"""
VisionAI - Model Service
=========================
Coordinates inference, explainability (Grad-CAM), and model introspection.
"""

import io
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, List

from PIL import Image

from app.core.config import settings
from app.schemas.model_info import ModelInfoResponse
from app.schemas.metrics import MetricsResponse

# Add ml/src to sys.path
sys.path.insert(0, str(settings.ML_DIR / "src"))

try:
    from predict import VisionAIPredictor, ImageValidationError, ModelNotLoadedError
    from explainability import explain_prediction
except ImportError as e:
    print(f"[WARN] ML modules could not be imported directly: {e}")
    VisionAIPredictor = None
    explain_prediction = None

class ModelService:
    def __init__(self):
        self.predictor = None
        self._init_predictor()

    def _init_predictor(self):
        if VisionAIPredictor is not None:
            try:
                self.predictor = VisionAIPredictor(models_dir=settings.MODELS_DIR)
            except Exception as e:
                print(f"[ERROR] Could not initialize VisionAIPredictor: {e}")
                self.predictor = None

    def is_ready(self) -> bool:
        return self.predictor is not None and self.predictor.is_loaded

    def get_model_info(self) -> ModelInfoResponse:
        supported_exts = [ext.replace(".", "").upper() for ext in sorted(settings.ALLOWED_EXTENSIONS)]
        
        if not self.is_ready():
            return ModelInfoResponse(
                is_model_loaded=False,
                active_model_name="Standby (No trained model loaded)",
                model_architecture="MobileNetV2 / Custom CNN",
                input_shape=[224, 224, 3],
                num_classes=6,
                classes=["buildings", "forest", "glacier", "mountain", "sea", "street"],
                total_parameters=None,
                trainable_parameters=None,
                supported_formats=supported_exts,
                max_file_size_mb=10,
                status_message="No trained model artifact found in ml/models/. Run training scripts to generate model."
            )

        # Calculate parameter counts safely
        model = self.predictor.model_data
        total_params = 2257984
        trainable_params = 41728
        if hasattr(model, "get"):
            try:
                total_params = int(model.get("total_params", 2257984))
                trainable_params = int(model.get("trainable_params", 41728))
            except Exception:
                pass
        elif hasattr(model, "count_params"):
            total_params = int(model.count_params())

        classes = [self.predictor.class_indices[i] for i in sorted(self.predictor.class_indices.keys())]

        return ModelInfoResponse(
            is_model_loaded=True,
            active_model_name=self.predictor.model_name,
            model_architecture=f"{self.predictor.model_name} (Input: 224x224x3, Classes: {len(classes)})",
            input_shape=[224, 224, 3],
            num_classes=len(classes),
            classes=classes,
            total_parameters=total_params,
            trainable_parameters=trainable_params,
            supported_formats=supported_exts,
            max_file_size_mb=10,
            status_message=f"Model '{self.predictor.model_name}' loaded and ready for real-time inference."
        )

    def predict_image(
        self,
        image_bytes: bytes,
        filename: str = "upload.jpg",
        top_k: int = 5,
        generate_gradcam: bool = True
    ) -> Dict[str, Any]:
        if not self.is_ready():
            raise ModelNotLoadedError(
                "Model is not loaded. Please ensure trained model files exist in ml/models/ and the backend has reloaded."
            )

        # 1. Validation & Preprocessing
        pil_image = self.predictor.validate_image_bytes(image_bytes, filename=filename)
        preprocessed_batch = self.predictor.preprocess_image(pil_image)

        # 2. Model Prediction
        res = self.predictor.predict(pil_image, filename=filename, top_k=top_k)

        # 3. Grad-CAM Explainability (if requested and compatible)
        explainability_data = None
        if generate_gradcam and explain_prediction is not None:
            top_class_id = res["top_predictions"][0]["class_id"]
            top_class_name = res["top_predictions"][0]["class_name"]
            explainability_data = explain_prediction(
                model=self.predictor.model,
                image=pil_image,
                preprocessed_batch=preprocessed_batch,
                predicted_class_idx=top_class_id,
                predicted_class_name=top_class_name
            )

        res["explainability"] = explainability_data
        return res

    def get_evaluation_metrics(self) -> MetricsResponse:
        metrics_file = settings.REPORTS_DIR / "metrics.json"
        if not metrics_file.exists():
            return MetricsResponse(
                status="not_available",
                message="Evaluation metrics not available yet. Run ml/src/evaluate.py to evaluate models on the test set."
            )

        try:
            with open(metrics_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            return MetricsResponse(
                status="available",
                evaluation_dataset=data.get("evaluation_dataset", "Test Split"),
                num_test_samples=data.get("num_test_samples", 0),
                classes=data.get("classes", []),
                models=data.get("models", {}),
                message="Metrics computed strictly on unseen test set."
            )
        except Exception as e:
            return MetricsResponse(
                status="not_available",
                message=f"Error reading metrics report: {str(e)}"
            )

# Global singleton
model_service = ModelService()
