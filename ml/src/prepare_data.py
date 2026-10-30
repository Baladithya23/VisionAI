"""
VisionAI - Dataset Preparation Module
=====================================
Dataset: Intel Image Classification (Natural Scenes)
Source: Kaggle (https://www.kaggle.com/datasets/puneet6060/intel-image-classification)
Original Host: Intel Corporation / Analytics Vidhya
License: CC0 - Public Domain
Image Resolution: ~150x150 original, normalized to 224x224 for MobileNetV2
Total Classes (6):
  0: buildings
  1: forest
  2: glacier
  3: mountain
  4: sea
  5: street

This script handles:
- Verification and loading of raw dataset
- Optional automated starter dataset generation for rapid development and automated testing
- Train / Validation / Test split (70% / 15% / 15%) without data leakage
- Class distribution visualization saved to ml/reports/
- Class index mapping serialization to ml/models/class_indices.json
"""

import os
import sys
import json
import random
import shutil
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Canonical Classes
CLASSES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]
RANDOM_SEED = 42

def set_seed(seed: int = RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)

def get_class_indices() -> Dict[int, str]:
    return {i: label for i, label in enumerate(CLASSES)}

def save_class_indices(output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mapping = {str(i): label for i, label in enumerate(CLASSES)}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2)
    print(f"[OK] Saved class index mapping to {output_path}")

def generate_synthetic_scene_image(category: str, width: int = 224, height: int = 224) -> Image.Image:
    """
    Generates a procedurally structured image exhibiting distinct visual patterns,
    palettes, and textures characteristic of each natural/urban scene class.
    This provides realistic RGB samples for baseline training & testing.
    """
    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    if category == "buildings":
        # Sky background (gradient blue/grey)
        for y in range(height):
            sky_r = int(120 + 40 * (y / height))
            sky_g = int(150 + 50 * (y / height))
            sky_b = int(200 + 40 * (y / height))
            draw.line([(0, y), (width, y)], fill=(sky_r, sky_g, sky_b))
        # Draw multiple building silhouettes with window grids
        num_buildings = random.randint(3, 6)
        x = 5
        while x < width - 10:
            b_w = random.randint(30, 60)
            b_h = random.randint(80, 170)
            top_y = height - b_h
            b_color = (random.randint(50, 90), random.randint(50, 90), random.randint(60, 100))
            draw.rectangle([x, top_y, min(x + b_w, width), height], fill=b_color)
            # Windows
            for wx in range(x + 5, min(x + b_w - 5, width), 8):
                for wy in range(top_y + 10, height - 10, 12):
                    win_color = (240, 230, 140) if random.random() > 0.4 else (40, 40, 50)
                    draw.rectangle([wx, wy, wx + 4, wy + 6], fill=win_color)
            x += b_w + random.randint(2, 8)

    elif category == "forest":
        # Green / earthy palette with pine and deciduous silhouettes
        for y in range(height):
            c_val = int(30 + 40 * (y / height))
            draw.line([(0, y), (width, y)], fill=(20, c_val + 20, 20))
        for _ in range(random.randint(15, 30)):
            tx = random.randint(10, width - 10)
            th = random.randint(70, 160)
            tw = random.randint(20, 50)
            base_y = height - random.randint(0, 40)
            peak_y = base_y - th
            # Triangle pine shape
            green_shade = (random.randint(10, 40), random.randint(80, 150), random.randint(20, 50))
            draw.polygon([(tx, peak_y), (tx - tw // 2, base_y), (tx + tw // 2, base_y)], fill=green_shade)

    elif category == "glacier":
        # Icy blues, white crystalline masses, crevasses
        for y in range(height):
            sky_b = int(180 + 60 * (1.0 - y / height))
            draw.line([(0, y), (width, y)], fill=(160, 210, sky_b))
        # Ice peaks
        pts = [(0, height)]
        cx = 0
        while cx < width:
            step = random.randint(20, 45)
            cy = random.randint(height // 3, int(height * 0.75))
            pts.append((cx, cy))
            cx += step
        pts.append((width, height))
        draw.polygon(pts, fill=(230, 245, 255))
        # Ice fissures/shading
        for _ in range(8):
            fx = random.randint(20, width - 20)
            fy = random.randint(height // 2, height - 20)
            draw.line([(fx, fy), (fx + random.randint(-15, 15), fy + random.randint(20, 50))], fill=(100, 160, 210), width=2)

    elif category == "mountain":
        # Rocky peaks, earthy brown/stone and snowcaps
        for y in range(height):
            sky_val = int(140 + 80 * (1.0 - y / height))
            draw.line([(0, y), (width, y)], fill=(sky_val, sky_val + 20, 240))
        # Mountain range polygon
        m_pts = [(0, height), (0, height // 2)]
        x = 0
        while x < width:
            nx = x + random.randint(25, 60)
            ny = random.randint(50, 120)
            m_pts.append((min(nx, width), ny))
            x = nx
        m_pts.append((width, height))
        draw.polygon(m_pts, fill=(80, 75, 70))
        # Snow caps
        for pt in m_pts[1:-1]:
            px, py = pt
            draw.polygon([(px, py), (px - 15, py + 25), (px + 15, py + 25)], fill=(245, 245, 250))

    elif category == "sea":
        # Horizon, water layers, ripples and sunset/sky
        horizon = height // 2
        for y in range(horizon):
            sky_ratio = y / horizon
            draw.line([(0, y), (width, y)], fill=(int(220 - 40 * sky_ratio), int(180 - 30 * sky_ratio), int(140 + 60 * sky_ratio)))
        # Ocean
        for y in range(horizon, height):
            sea_ratio = (y - horizon) / (height - horizon)
            draw.line([(0, y), (width, y)], fill=(int(10 + 20 * sea_ratio), int(60 + 50 * sea_ratio), int(120 + 80 * sea_ratio)))
        # Waves/ripples
        for _ in range(25):
            wx = random.randint(0, width - 40)
            wy = random.randint(horizon + 5, height - 5)
            draw.arc([wx, wy, wx + random.randint(20, 50), wy + 4], start=0, end=180, fill=(200, 230, 255), width=1)

    elif category == "street":
        # Road perspective with lane markings and curbs
        for y in range(height):
            draw.line([(0, y), (width, y)], fill=(130, 140, 160))
        # Road polygon
        road_pts = [(int(width * 0.35), height // 2), (int(width * 0.65), height // 2), (width, height), (0, height)]
        draw.polygon(road_pts, fill=(45, 45, 50))
        # Center dashed yellow line
        mid_top = (width // 2, height // 2)
        mid_bot = (width // 2, height)
        for t in range(5):
            y1 = int(height // 2 + t * (height // 10))
            y2 = int(y1 + 10)
            draw.line([(width // 2, y1), (width // 2, y2)], fill=(255, 215, 0), width=3)
        # Sidewalks / curbs
        draw.line([(0, height), (int(width * 0.35), height // 2)], fill=(180, 180, 180), width=4)
        draw.line([(width, height), (int(width * 0.65), height // 2)], fill=(180, 180, 180), width=4)

    # Add light gaussian blur and texture noise
    img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
    return img

def create_starter_dataset(
    output_raw_dir: Path,
    num_samples_per_class: int = 50,
    sample_test_dir: Path = None
) -> None:
    """
    Generates a starter dataset split into class directories.
    Provides immediate self-contained reproducibility.
    """
    set_seed(RANDOM_SEED)
    output_raw_dir.mkdir(parents=True, exist_ok=True)
    print(f"[*] Generating starter dataset ({num_samples_per_class} images/class across {len(CLASSES)} classes)...")

    for cls in CLASSES:
        cls_dir = output_raw_dir / cls
        cls_dir.mkdir(parents=True, exist_ok=True)
        for idx in range(num_samples_per_class):
            img = generate_synthetic_scene_image(cls, 224, 224)
            img.save(cls_dir / f"{cls}_{idx:04d}.jpg", quality=90)

    # Save a few designated standalone test samples for user live evaluation
    if sample_test_dir:
        sample_test_dir.mkdir(parents=True, exist_ok=True)
        for cls in CLASSES:
            img = generate_synthetic_scene_image(cls, 224, 224)
            img.save(sample_test_dir / f"sample_{cls}.jpg", quality=95)
        print(f"[OK] Saved standalone test samples to {sample_test_dir}")

    print(f"[OK] Starter dataset created at {output_raw_dir}")

def split_dataset(
    raw_dir: Path,
    processed_dir: Path,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15
) -> Dict[str, Dict[str, int]]:
    """
    Splits images from raw_dir into processed_dir/train, processed_dir/val, and processed_dir/test.
    Ensures zero data leakage between splits.
    """
    set_seed(RANDOM_SEED)
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-4, "Ratios must sum to 1.0"

    splits = ["train", "val", "test"]
    for split in splits:
        for cls in CLASSES:
            (processed_dir / split / cls).mkdir(parents=True, exist_ok=True)

    distribution_counts: Dict[str, Dict[str, int]] = {s: {c: 0 for c in CLASSES} for s in splits}

    for cls in CLASSES:
        cls_raw = raw_dir / cls
        if not cls_raw.exists():
            print(f"[WARN] Class directory {cls_raw} does not exist. Skipping.")
            continue

        images = [f for f in cls_raw.iterdir() if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]]
        random.shuffle(images)

        n_total = len(images)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        # Remainder goes to test to account for integer rounding
        train_files = images[:n_train]
        val_files = images[n_train:n_train + n_val]
        test_files = images[n_train + n_val:]

        split_map = {"train": train_files, "val": val_files, "test": test_files}

        for split_name, files in split_map.items():
            for f in files:
                dest = processed_dir / split_name / cls / f.name
                shutil.copy2(f, dest)
                distribution_counts[split_name][cls] += 1

    print("[OK] Dataset split completed successfully without data leakage.")
    for s in splits:
        total_s = sum(distribution_counts[s].values())
        print(f"  - {s.upper()}: {total_s} images across {len(CLASSES)} classes")

    return distribution_counts

def plot_class_distribution(distribution_counts: Dict[str, Dict[str, int]], output_path: Path):
    """
    Plots and saves class distribution across train, validation, and test splits.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(CLASSES))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    train_counts = [distribution_counts["train"].get(c, 0) for c in CLASSES]
    val_counts = [distribution_counts["val"].get(c, 0) for c in CLASSES]
    test_counts = [distribution_counts["test"].get(c, 0) for c in CLASSES]

    ax.bar(x - width, train_counts, width, label="Train", color="#3b82f6")
    ax.bar(x, val_counts, width, label="Validation", color="#10b981")
    ax.bar(x + width, test_counts, width, label="Test", color="#f59e0b")

    ax.set_ylabel("Number of Images", fontsize=12)
    ax.set_title("VisionAI - Dataset Class Distribution Across Splits", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(CLASSES, fontsize=11)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    fig.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"[OK] Class distribution chart saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="VisionAI Dataset Preparation")
    parser.add_argument("--raw-dir", type=str, default="ml/data/raw", help="Path to raw dataset")
    parser.add_argument("--processed-dir", type=str, default="ml/data/processed", help="Path for train/val/test splits")
    parser.add_argument("--models-dir", type=str, default="ml/models", help="Path for models and class mapping")
    parser.add_argument("--reports-dir", type=str, default="ml/reports", help="Path for charts and reports")
    parser.add_argument("--generate-starter", action="store_true", help="Generate synthetic starter dataset")
    parser.add_argument("--samples-per-class", type=int, default=60, help="Samples per class if generating starter")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent.parent
    raw_dir = project_root / args.raw_dir
    processed_dir = project_root / args.processed_dir
    models_dir = project_root / args.models_dir
    reports_dir = project_root / args.reports_dir
    sample_test_dir = project_root / "ml/data/sample_test_images"

    print("==================================================")
    print(" VisionAI - Dataset Preparation Pipeline")
    print("==================================================")
    print(f"Project root : {project_root}")
    print(f"Target classes: {CLASSES}")

    # Check if raw data exists
    has_raw_data = raw_dir.exists() and any((raw_dir / c).exists() and len(list((raw_dir / c).glob("*.*"))) > 0 for c in CLASSES)

    if not has_raw_data or args.generate_starter:
        print("[INFO] No raw images found or --generate-starter requested. Generating starter dataset...")
        create_starter_dataset(raw_dir, num_samples_per_class=args.samples_per_class, sample_test_dir=sample_test_dir)
    else:
        print(f"[INFO] Using existing raw images found in {raw_dir}")

    # Clean existing processed directory if re-preparing
    if processed_dir.exists():
        shutil.rmtree(processed_dir)

    # Perform train/val/test split
    dist = split_dataset(raw_dir, processed_dir, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)

    # Save class indices
    save_class_indices(models_dir / "class_indices.json")

    # Plot class distribution
    plot_class_distribution(dist, reports_dir / "class_distribution.png")

    print("==================================================")
    print("[SUCCESS] Data preparation completed cleanly.")
    print("==================================================")

if __name__ == "__main__":
    main()
