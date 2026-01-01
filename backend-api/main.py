"""
FastAPI Backend for ECG Diagnosis System - Phase 2 Dual-Stream
Converts ECG images to diagnostic results using dual-stream classifier.

Phase 2 Updates:
- Accepts both ECG signal data and ECG images
- Handles cases where only signal or only image is provided
- Uses dual-stream model (ResNet1D + EfficientNetB0)
- Enhanced preprocessing for 500Hz signals and 224x224 images
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
try:
    from services.digitizer import (
        extract_signal_from_image, 
        assess_signal_quality,
        preprocess_image_for_model,
        preprocess_dual_inputs
    )
    from services.classifier import classify_ecg, load_model
except ImportError as e:
    print(f"Warning: Service modules not found: {e}")
    # Define dummy functions for health checks
    def extract_signal_from_image(*args, **kwargs): raise NotImplementedError("Digitizer service missing")
    def assess_signal_quality(*args, **kwargs): return "unknown"
    def preprocess_image_for_model(*args, **kwargs): raise NotImplementedError("Digitizer service missing")
    def preprocess_dual_inputs(*args, **kwargs): raise NotImplementedError("Digitizer service missing")
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
    title="ECG Diagnosis API - Phase 2 Dual-Stream",
    description="Production-ready API for ECG arrhythmia classification using dual-stream deep learning",
    version="2.0.0",
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

# ====
# DATA MODELS
# ====

class PredictionResponse(BaseModel):
    """Response model for ECG diagnosis prediction"""
    diagnosis: str = Field(..., description="Arrhythmia classification")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Model confidence score")
    bpm: int = Field(..., description="Heart rate in beats per minute")
    signal_quality: str = Field(..., description="Quality of extracted/provided signal")
    all_probabilities: dict = Field(..., description="Probabilities for all classes")
    input_mode: str = Field(..., description="Input mode: 'image_only', 'signal_only', or 'dual'")

class ErrorResponse(BaseModel):
    """Error response model"""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")

# ====
# API ENDPOINTS
# ====

@app.get("/", tags=["Documentation"])
async def root():
    """API root endpoint with basic information and documentation links."""
    return {
        "message": "ECG Diagnosis API - Phase 2 Dual-Stream",
        "version": "2.0.0",
        "architecture": "ResNet1D + EfficientNetB0 Fusion",
        "status": "online",
        "docs": "/docs",
        "phase": "Phase 2 - Dual-Stream Fusion",
        "features": [
            "500Hz signal processing (5000 samples)",
            "224x224 ECG image processing",
            "Dual-stream fusion architecture",
            "14-class arrhythmia detection",
            "Supports image-only, signal-only, or dual inputs"
        ]
    }

@app.get("/health", tags=["Health"])
async def health_check():
    """
    Health check endpoint to verify API is running and model is loaded.
    """
    try:
        BASE_DIR = Path(__file__).resolve().parent
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
            "model_path": str(model_path),
            "architecture": "Dual-Stream (ResNet1D + EfficientNetB0)",
            "phase": "Phase 2"
        }
    
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "unhealthy", "error": str(e)}
        )

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(
    image: Optional[UploadFile] = File(None, description="ECG image file (JPG, PNG, etc.)"),
    calibration_points: Optional[str] = Form(None, description="Optional JSON string of calibration points")
):
    """
    Upload an ECG image and get diagnostic prediction using Phase 2 dual-stream model.
    
    **Phase 2 Dual-Stream Architecture:**
    - Extracts signal from image (500Hz, 5000 samples)
    - Preprocesses image for EfficientNetB0 (224x224 RGB)
    - Fuses both modalities for classification
    
    **Input Modes:**
    - Image only: Signal extracted from image, both used for classification
    
    **Returns:**
    - Diagnosis, confidence, BPM, signal quality, all class probabilities
    """
    try:
        # Validate input
        if image is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="ECG image is required for Phase 2 dual-stream model"
            )
        
        # Validate image format
        is_valid_type = False
        
        if image.content_type and image.content_type.startswith('image/'):
            is_valid_type = True
        
        if not is_valid_type and image.filename:
            ext = image.filename.lower().split('.')[-1]
            if ext in ['jpg', 'jpeg', 'png', 'bmp', 'webp', 'tiff']:
                is_valid_type = True
        
        if not is_valid_type:
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
                pass
        
        # === PHASE 2: DUAL-STREAM PROCESSING ===
        logger.info("Phase 2 Dual-Stream Processing:")
        
        # Step 1: Extract signal from image (500Hz)
        logger.info("  1. Extracting ECG signal from image (500Hz, 5000 samples)...")
        try:
            signal = extract_signal_from_image(image_bytes, calibration_data)
        except Exception as e:
            logger.error(f"Signal extraction failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Failed to extract signal from image. Ensure it is a clear ECG image. Error: {str(e)}"
            )
        
        # Step 2: Preprocess image for EfficientNetB0
        logger.info("  2. Preprocessing image for EfficientNetB0 (224x224 RGB)...")
        try:
            image_preprocessed = preprocess_image_for_model(image_bytes)
        except Exception as e:
            logger.error(f"Image preprocessing failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Failed to preprocess image. Error: {str(e)}"
            )
        
        # Step 3: Assess signal quality
        logger.info("  3. Assessing signal quality...")
        signal_quality = assess_signal_quality(signal)
        
        # Step 4: Classify using dual-stream model
        logger.info("  4. Running dual-stream classifier (ResNet1D + EfficientNetB0)...")
        try:
            result = classify_ecg(signal=signal, image=image_preprocessed)
        except Exception as e:
            logger.error(f"Classification failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"AI Model Error: {str(e)}"
            )
        
        # Step 5: Return response
        logger.info(f"✅ Prediction complete: {result['diagnosis']} (confidence: {result['confidence']:.3f})")
        
        return PredictionResponse(
            diagnosis=result["diagnosis"],
            confidence=float(result["confidence"]),
            bpm=int(result["bpm"]),
            signal_quality=signal_quality,
            all_probabilities=result["all_probabilities"],
            input_mode="dual"  # Phase 2 always uses dual-stream
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Internal server error processing request"
        )


@app.post("/predict_signal_only", response_model=PredictionResponse, tags=["Prediction"])
async def predict_signal_only(
    signal_data: str = Form(..., description="JSON string of ECG signal data (5000, 12)")
):
    """
    **Experimental Endpoint:** Classify using signal data only (no image).
    
    The dual-stream model will generate a placeholder image from the signal.
    
    **Signal Format:**
    - JSON array of shape (5000, 12) @ 500Hz
    - Example: [[v1_lead1, v1_lead2, ..., v1_lead12], [v2_lead1, ...], ...]
    
    **Note:** Image-based input is recommended for best accuracy.
    """
    try:
        import numpy as np
        
        # Parse signal data
        try:
            signal = np.array(json.loads(signal_data), dtype=np.float32)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid signal data format. Expected JSON array of shape (5000, 12). Error: {str(e)}"
            )
        
        # Validate signal shape
        if signal.shape != (5000, 12):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Signal shape must be (5000, 12) @ 500Hz. Got: {signal.shape}"
            )
        
        logger.info("Received signal data for signal-only classification")
        
        # Assess signal quality
        signal_quality = assess_signal_quality(signal)
        
        # Classify (classifier will generate placeholder image internally)
        logger.info("Running classifier with signal-only input (placeholder image will be generated)...")
        try:
            result = classify_ecg(signal=signal, image=None)
        except Exception as e:
            logger.error(f"Classification failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"AI Model Error: {str(e)}"
            )
        
        logger.info(f"✅ Signal-only prediction complete: {result['diagnosis']} (confidence: {result['confidence']:.3f})")
        
        return PredictionResponse(
            diagnosis=result["diagnosis"],
            confidence=float(result["confidence"]),
            bpm=int(result["bpm"]),
            signal_quality=signal_quality,
            all_probabilities=result["all_probabilities"],
            input_mode="signal_only"
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
    
    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")
    
    print("\n" + "="*70)
    print("🚀 ECG Diagnosis API - Phase 2 Dual-Stream")
    print("="*70)
    print(f"Architecture: ResNet1D + EfficientNetB0 Fusion")
    print(f"Signal Processing: 500Hz (5000 samples)")
    print(f"Image Processing: 224x224 RGB")
    print(f"Starting server on http://{host}:{port}")
    print(f"API Docs: http://{host}:{port}/docs")
    print("="*70 + "\n")
    
    uvicorn.run("main:app", host=host, port=port, reload=True)
