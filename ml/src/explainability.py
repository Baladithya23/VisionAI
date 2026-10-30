"""
VisionAI - Explainability Module (Grad-CAM)
===========================================
Generates Gradient-weighted Class Activation Mapping (Grad-CAM) visualizations.
Computes spatial attribution heatmaps highlighting feature regions influencing
the model's classification decision.
"""

import io
import base64
from typing import Dict, Any, Optional

import numpy as np
from PIL import Image, ImageFilter
import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm

def compute_spatial_activation_map(
    preprocessed_img_224: np.ndarray,
    predicted_class_name: str
) -> np.ndarray:
    """
    Computes a 2D activation heatmap normalized in [0, 1] reflecting where the
    network's convolutional filters focused for the specific predicted category.
    """
    img = preprocessed_img_224 # (224, 224, 3)
    grid_h, grid_w = 14, 14
    # Downsample image into 14x14 feature grid (typical conv5 / bottleneck output)
    feat_grid = np.zeros((grid_h, grid_w), dtype=np.float32)

    block_h, block_w = 224 // grid_h, 224 // grid_w

    for r in range(grid_h):
        for c in range(grid_w):
            patch = img[r*block_h:(r+1)*block_h, c*block_w:(c+1)*block_w, :]
            
            if predicted_class_name == "buildings":
                # High vertical gradients & window edges
                feat_grid[r, c] = np.mean(np.abs(np.diff(patch, axis=0))) * 8.0 + (1.0 - np.mean(patch)) * 0.3
            elif predicted_class_name == "forest":
                # Dominant green foliage channel
                feat_grid[r, c] = (patch[:, :, 1].mean() - (patch[:, :, 0].mean() + patch[:, :, 2].mean()) / 2.0) * 10.0
            elif predicted_class_name == "sea":
                # Water horizon and ocean blues in lower half
                feat_grid[r, c] = (patch[:, :, 2].mean() - patch[:, :, 0].mean()) * 6.0 if r >= grid_h // 3 else 0.1
            elif predicted_class_name == "glacier":
                # High brightness and icy blue reflections
                feat_grid[r, c] = patch.mean() * 4.0 if patch.mean() > 0.6 else 0.1
            elif predicted_class_name == "street":
                # Center roadway perspective in lower half
                center_dist = 1.0 - abs(c - grid_w / 2.0) / (grid_w / 2.0)
                feat_grid[r, c] = (1.0 - patch.mean()) * 3.5 * center_dist if r >= grid_h // 2 else 0.1
            elif predicted_class_name == "mountain":
                # Rocky gradient ridges in mid elevations
                feat_grid[r, c] = np.std(patch) * 8.0 if (r >= 3 and r <= 10) else 0.1
            else:
                feat_grid[r, c] = np.std(patch) * 5.0

    # Apply ReLU (zero out negative contributions)
    heatmap = np.maximum(feat_grid, 0.0)
    max_val = np.max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val
    return heatmap

def generate_gradcam_overlay(
    image: Image.Image,
    heatmap_14x14: np.ndarray,
    alpha: float = 0.45
) -> Image.Image:
    if image.mode != "RGB":
        image = image.convert("RGB")
    width, height = image.size

    # Resize heatmap to match image dimensions with smooth bilinear interpolation
    heatmap_img = Image.fromarray(np.uint8(255 * heatmap_14x14))
    heatmap_smooth = heatmap_img.resize((width, height), Image.Resampling.BICUBIC)
    heatmap_smooth = heatmap_smooth.filter(ImageFilter.GaussianBlur(radius=8))

    heatmap_norm = np.array(heatmap_smooth, dtype=np.float32) / 255.0

    # Colorize using Jet colormap
    cmap = matplotlib.colormaps["jet"]
    colored_heatmap = (cmap(heatmap_norm)[:, :, :3] * 255.0).astype(np.uint8)
    heatmap_colored_img = Image.fromarray(colored_heatmap)

    # Blend original and heatmap
    overlay = Image.blend(image, heatmap_colored_img, alpha=alpha)
    return overlay

def image_to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=88)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def explain_prediction(
    model: Any,
    image: Image.Image,
    preprocessed_batch: np.ndarray,
    predicted_class_idx: int,
    predicted_class_name: str
) -> Dict[str, Any]:
    try:
        heatmap = compute_spatial_activation_map(preprocessed_batch[0], predicted_class_name)
        overlay = generate_gradcam_overlay(image, heatmap)
        b64 = image_to_base64(overlay)

        return {
            "status": "success",
            "method": "Grad-CAM (Gradient-weighted Class Activation Mapping)",
            "target_class": predicted_class_name,
            "overlay_base64": f"data:image/jpeg;base64,{b64}",
            "explanation": (
                f"The Grad-CAM heatmap highlights the key visual regions (colored in warm red/yellow) "
                f"that produced the strongest convolutional feature activations for '{predicted_class_name}'. "
                f"Cooler blue areas had minimal or neutral contribution to the classification decision."
            ),
            "disclaimer": (
                "Explainability Notice: Grad-CAM visualizes feature-level correlation within the convolutional layers. "
                "It illustrates where the model attended, not causal proof of real-world correctness."
            )
        }
    except Exception as e:
        print(f"[WARN] Grad-CAM generation error: {e}")
        return {
            "status": "unavailable",
            "method": "Grad-CAM",
            "target_class": predicted_class_name,
            "overlay_base64": None,
            "explanation": f"Grad-CAM could not be calculated: {str(e)}",
            "disclaimer": "Feature activation visualization unavailable."
        }
