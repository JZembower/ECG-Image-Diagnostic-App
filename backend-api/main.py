"""
FastAPI Backend for ECG Diagnosis System
Converts ECG images to diagnostic results using digitizer and classifier services.
"""

import os
import logging
from typing import Optional
from pathlib import Path
import json

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Service imports
# We use try/except to handle cases where dependencies might be missing during initial setup
try:
    from services.digitizer import extract_signal_from_image, assess_signal_quality
    from services.classifier import classify_ecg, load_model
except ImportError as e:
    print(f"Warning: Service modules not found: {e}")
    # Define dummy functions so the app can at least start for health checks
    def extract_signal_from_image(*args, **kwargs): raise NotImplementedError("Digitizer service missing")
    def assess_signal_quality(*args, **kwargs): return "unknown"
    def classify_ecg(*args, **kwargs): raise NotImplementedError("Classifier service missing")
    def load_model(*args, **kwargs): pass

# Configure logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="ECG Diagnosis API",
    description="Production-ready API for converting ECG images to diagnostic results",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# DATA MODELS
# =============================================================================

class PredictionResponse(BaseModel):
    """Response model for ECG diagnosis prediction"""
    diagnosis: str = Field(..., description="Arrhythmia classification")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Model confidence score")
    bpm: int = Field(..., description="Heart rate in beats per minute")
    signal_quality: str = Field(..., description="Quality of extracted signal")
    all_probabilities: dict = Field(..., description="Probabilities for all classes")

class ErrorResponse(BaseModel):
    """Error response model"""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")

# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/", tags=["Documentation"])
async def root():
    """API root endpoint with basic information and documentation links."""
    return {
        "message": "ECG Diagnosis API",
        "version": "1.0.0",
        "status": "online",
        "docs": "/docs"
    }

@app.get("/health", tags=["Health"])
async def health_check():
    """
    Health check endpoint to verify API is running and model is loaded.
    """
    try:
        # --- DYNAMIC PATH LOGIC ---
        BASE_DIR = Path(__file__).resolve().parent
        # Look for the new 'model.weights.h5' filename
        default_model_path = BASE_DIR / "models" / "model.weights.h5"
        default_classes_path = BASE_DIR / "models" / "classes.npy"

        model_path = os.getenv("MODEL_PATH", str(default_model_path))
        classes_path = os.getenv("CLASSES_PATH", str(default_classes_path))
        
        model_exists = os.path.exists(model_path)
        classes_exist = os.path.exists(classes_path)
        
        # Check if loaded in memory
        model_loaded = False
        try:
            from services.classifier import _model
            if _model is not None:
                model_loaded = True
            else:
                # Try to trigger load
                load_model()
                model_loaded = True
        except Exception as e:
            logger.warning(f"Model load check warning: {e}")
        
        status_str = "healthy" if (model_exists and classes_exist and model_loaded) else "degraded"
        
        return {
            "status": status_str,
            "model_loaded": model_loaded,
            "model_file_exists": model_exists,
            "classes_file_exists": classes_exist,
            "model_path": str(model_path)
        }
    
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "unhealthy", "error": str(e)}
        )

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(
    image: UploadFile = File(..., description="ECG image file (JPG, PNG, etc.)"),
    calibration_points: Optional[str] = Form(None, description="Optional JSON string of calibration points")
):
    """
    Upload an ECG image and get diagnostic prediction.
    """
    try:
        # --- FIX: Safe Content-Type Check ---
        # Some clients (like simple python scripts) might not send a content-type.
        # We handle None safely and also check the filename extension as a fallback.
        
        is_valid_type = False
        
        # 1. Check MIME type if available
        if image.content_type and image.content_type.startswith('image/'):
            is_valid_type = True
            
        # 2. Fallback: Check file extension
        if not is_valid_type and image.filename:
            ext = image.filename.lower().split('.')[-1]
            if ext in ['jpg', 'jpeg', 'png', 'bmp', 'webp', 'tiff']:
                is_valid_type = True
                
        if not is_valid_type:
            # We log a warning but try to proceed anyway, because OpenCV is robust.
            # If it fails later, the digitizer will catch it.
            logger.warning(f"Unknown content type: {image.content_type} for file {image.filename}. Proceeding cautiously.")

        # Read image bytes
        image_bytes = await image.read()
        if len(image_bytes) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Empty image file received"
            )
        
        logger.info(f"Received image: {image.filename}, size: {len(image_bytes)} bytes")
        
        # Parse calibration points if provided
        calibration_data = None
        if calibration_points:
            try:
                calibration_data = json.loads(calibration_points)
            except json.JSONDecodeError:
                pass # Ignore invalid JSON, treat as None

        # Step 1: Extract signal from image
        logger.info("Extracting signal from image...")
        try:
            signal = extract_signal_from_image(image_bytes, calibration_data)
        except Exception as e:
            logger.error(f"Digitization failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Failed to process image. Ensure it is a clear ECG image. Error: {str(e)}"
            )
        
        # Step 2: Assess quality
        signal_quality = assess_signal_quality(signal)
        
        # Step 3: Classify
        logger.info("Running classifier...")
        try:
            result = classify_ecg(signal)
        except Exception as e:
            logger.error(f"Classification failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"AI Model Error: {str(e)}"
            )
        
        # Step 4: Return response
        return PredictionResponse(
            diagnosis=result["diagnosis"],
            confidence=float(result["confidence"]),
            bpm=int(result["bpm"]),
            signal_quality=signal_quality,
            all_probabilities=result["all_probabilities"]
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Internal server error processing request"
        )

# Global Exception Handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled Error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal Server Error", "detail": str(exc)}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)