"""
VisionAI - Core API & Endpoint Logic Tests
===========================================
Direct endpoint unit & integration tests:
- health_check()
- get_model_information()
- get_metrics()
- list_sample_images()
- predict_sample_image()
- predict validation and error handling
"""

import io
import pytest
from PIL import Image

import sys
from pathlib import Path

# Add backend and ml/src to sys.path
backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))
sys.path.insert(0, str(backend_dir.parent / "ml" / "src"))

from app.routes.health import health_check
from app.routes.model_info import get_model_information
from app.routes.metrics import get_metrics
from app.routes.predict import list_sample_images, predict_sample_image
from app.services.model_service import model_service
from predict import ImageValidationError

def create_test_image_bytes(format="JPEG", size=(100, 100), color=(120, 180, 220)) -> bytes:
    img = Image.new("RGB", size, color)
    buf = io.BytesIO()
    img.save(buf, format=format)
    return buf.getvalue()

def test_health_endpoint():
    data = health_check()
    assert data["status"] == "healthy"
    assert "version" in data
    assert "model_loaded" in data
    assert data["model_loaded"] is True

def test_model_info_endpoint():
    data = get_model_information()
    assert data.is_model_loaded is True
    assert "MobileNetV2" in data.active_model_name or "CNN" in data.active_model_name
    assert data.num_classes == 6
    assert "buildings" in data.classes

def test_metrics_endpoint():
    data = get_metrics()
    assert data.status in ["available", "not_available"]
    if data.status == "available":
        assert data.models is not None
        assert "mobilenetv2" in data.models
        assert data.models["mobilenetv2"].test_accuracy > 0.85

def test_list_samples_endpoint():
    data = list_sample_images()
    assert "samples" in data
    assert len(data["samples"]) > 0

def test_predict_sample_preset():
    data = predict_sample_image("forest", top_k=5, gradcam=True)
    assert data.success is True
    assert data.predicted_class in ["buildings", "forest", "glacier", "mountain", "sea", "street"]
    assert data.confidence > 0.0
    assert len(data.top_predictions) == 5
    assert data.inference_time_ms >= 0.0
    assert data.explainability is not None
    assert data.explainability.status == "success"
    assert data.explainability.overlay_base64.startswith("data:image/jpeg;base64,")

def test_predict_valid_image_bytes():
    img_bytes = create_test_image_bytes("JPEG", (224, 224))
    res = model_service.predict_image(img_bytes, filename="test.jpg", top_k=5, generate_gradcam=True)
    assert res["predicted_class"] is not None
    assert len(res["top_predictions"]) == 5
    assert res["explainability"]["status"] == "success"

def test_predict_empty_bytes():
    with pytest.raises(ImageValidationError):
        model_service.predict_image(b"", filename="empty.jpg")

def test_predict_unsupported_format():
    with pytest.raises(ImageValidationError):
        model_service.predict_image(b"fake-content", filename="doc.pdf")

def test_predict_corrupted_image():
    with pytest.raises(ImageValidationError):
        model_service.predict_image(b"corrupted-image-data-string", filename="corrupted.jpg")

def test_predict_oversized_image():
    huge_bytes = b"0" * (11 * 1024 * 1024)
    with pytest.raises(ImageValidationError):
        model_service.predict_image(huge_bytes, filename="large.jpg")
