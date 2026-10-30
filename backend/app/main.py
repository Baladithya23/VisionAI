"""
VisionAI - FastAPI Application Entry Point
==========================================
Intelligent Image Recognition & Analysis Platform API.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from app.core.config import settings
from app.routes import health, model_info, predict, metrics
from app.services.model_service import model_service

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan context:
    - Pre-loads trained model weights on startup
    - Frees resources cleanly on shutdown
    """
    print("==================================================")
    print(f"[*] Starting {settings.PROJECT_NAME} (v{settings.VERSION})")
    print(f"[*] Models directory: {settings.MODELS_DIR}")
    if model_service.is_ready():
        print(f"[OK] Active Model: {model_service.predictor.model_name}")
    else:
        print("[INFO] Model not yet loaded. API in standby mode.")
    print("==================================================")
    yield
    print("[*] VisionAI API shutting down gracefully.")

app = FastAPI(
    title=settings.PROJECT_NAME,
    description=settings.PROJECT_DESCRIPTION,
    version=settings.VERSION,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Exception Handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "success": False,
            "error_code": "VALIDATION_ERROR",
            "detail": "Request payload validation failed.",
            "errors": exc.errors()
        }
    )

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    # Log securely without exposing internal filesystem paths or stack trace to client
    print(f"[UNHANDLED EXCEPTION on {request.url.path}]: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "error_code": "INTERNAL_SERVER_ERROR",
            "detail": "An internal server error occurred while processing the request."
        }
    )

# Register API Routers
app.include_router(health.router, prefix=settings.API_PREFIX)
app.include_router(model_info.router, prefix=settings.API_PREFIX)
app.include_router(predict.router, prefix=settings.API_PREFIX)
app.include_router(metrics.router, prefix=settings.API_PREFIX)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
