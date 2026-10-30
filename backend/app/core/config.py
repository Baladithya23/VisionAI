"""
VisionAI - Application Configuration
"""

from pathlib import Path
from pydantic import BaseModel

class Settings(BaseModel):
    PROJECT_NAME: str = "VisionAI — Intelligent Image Recognition & Analysis Platform"
    PROJECT_DESCRIPTION: str = (
        "Production-grade deep learning platform for real-time image intelligence, "
        "model introspection, Grad-CAM visual explainability, and comparative metrics."
    )
    VERSION: str = "1.0.0"
    API_PREFIX: str = ""
    
    # Path configurations
    PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent.parent
    ML_DIR: Path = PROJECT_ROOT / "ml"
    MODELS_DIR: Path = ML_DIR / "models"
    REPORTS_DIR: Path = ML_DIR / "reports"
    SAMPLE_IMAGES_DIR: Path = ML_DIR / "data" / "sample_test_images"

    # Upload and security bounds
    MAX_UPLOAD_SIZE_BYTES: int = 10 * 1024 * 1024  # 10 MB
    ALLOWED_EXTENSIONS: set = {".jpg", ".jpeg", ".png", ".webp"}
    ALLOWED_CONTENT_TYPES: set = {
        "image/jpeg",
        "image/png",
        "image/webp",
        "image/jpg",
        "application/octet-stream"  # When sent via raw binary upload
    }

    # CORS configuration
    CORS_ORIGINS: list = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "*"
    ]

settings = Settings()
