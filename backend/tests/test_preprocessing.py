"""
VisionAI - Preprocessing & Unit Tests
======================================
Tests:
- Input image validation
- Preprocessing dimension and normalization bounds
- Softmax probability distribution properties (sum == 1.0)
- Grad-CAM heatmap array dimensions and normalization in [0, 1]
"""

import io
import pytest
import numpy as np
from PIL import Image

import sys
from pathlib import Path

backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir.parent / "ml" / "src"))

from predict import VisionAIPredictor, ImageValidationError
from explainability import compute_spatial_activation_map, generate_gradcam_overlay

def test_preprocessing_shape_and_bounds():
    predictor = VisionAIPredictor()
    test_img = Image.new("RGB", (640, 480), (200, 150, 100))
    preprocessed = predictor.preprocess_image(test_img, target_size=(224, 224))

    assert preprocessed.shape == (1, 224, 224, 3)
    assert preprocessed.dtype == np.float32
    assert preprocessed.min() >= 0.0
    assert preprocessed.max() <= 1.0

def test_softmax_probabilities_sum_to_one():
    predictor = VisionAIPredictor()
    test_img = Image.new("RGB", (224, 224), (50, 180, 50)) # Forest green
    res = predictor.predict(test_img, top_k=6)

    probs = [item["confidence"] for item in res["top_predictions"]]
    assert len(probs) == 6
    assert abs(sum(probs) - 1.0) < 0.01

def test_image_dimension_limits():
    predictor = VisionAIPredictor()
    tiny_img = Image.new("RGB", (8, 8), (255, 255, 255))
    buf = io.BytesIO()
    tiny_img.save(buf, format="JPEG")

    with pytest.raises(ImageValidationError):
        predictor.validate_image_bytes(buf.getvalue(), "tiny.jpg")

def test_gradcam_heatmap_computation():
    dummy_input = np.random.rand(224, 224, 3).astype(np.float32)
    heatmap = compute_spatial_activation_map(dummy_input, "forest")

    assert heatmap.shape == (14, 14)
    assert heatmap.min() >= 0.0
    assert heatmap.max() <= 1.0

    pil_img = Image.fromarray((dummy_input * 255).astype(np.uint8))
    overlay = generate_gradcam_overlay(pil_img, heatmap)
    assert overlay.size == (224, 224)
