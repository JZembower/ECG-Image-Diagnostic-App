import os
import sys
import numpy as np
import tensorflow as tf
from pathlib import Path

# Fix path to ensure we can import from services
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("="*60)
print("🔍 ECG MODEL DIAGNOSTIC TOOL (FIXED)")
print("="*60)
print(f"Python: {sys.version.split()[0]}")
print(f"TensorFlow: {tf.__version__}")

# Safe Keras version check
try:
    import keras
    print(f"Keras (Standalone): {keras.__version__}")
except ImportError:
    print("Keras: Embedded in TensorFlow")

# 1. SETUP PATHS
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "model.weights.h5"
CLASSES_PATH = MODELS_DIR / "classes.npy"

print(f"\nChecking paths:")
print(f"  [ {'OK' if MODEL_PATH.exists() else 'MISSING'} ] {MODEL_PATH}")
print(f"  [ {'OK' if CLASSES_PATH.exists() else 'MISSING'} ] {CLASSES_PATH}")

if not MODEL_PATH.exists() or not CLASSES_PATH.exists():
    print("\n❌ CRITICAL: Files missing. Cannot proceed.")
    sys.exit(1)

# 2. TRY LOADING CLASSES
print("\nTesting Class Loading...")
try:
    classes = np.load(CLASSES_PATH, allow_pickle=True)
    print(f"✅ Success! Found {len(classes)} classes: {classes}")
except Exception as e:
    print(f"❌ Failed to load classes: {e}")
    sys.exit(1)

# 3. IMPORT ARCHITECTURE
print("\nImporting Model Architecture from services.classifier...")
try:
    from services.classifier import build_resnet1d
    print("✅ Import successful.")
except ImportError as e:
    print(f"❌ Failed to import classifier service: {e}")
    print("   Make sure you are running this from the 'backend-api' folder.")
    sys.exit(1)

# 4. BUILD MODEL
print("\nBuilding Model...")
try:
    model = build_resnet1d(input_shape=(1000, 12), num_classes=len(classes))
    print("✅ Model built successfully.")
    # Verify input shape
    print(f"   Expected Input: {model.input_shape}")
    print(f"   Expected Output: {model.output_shape}")
except Exception as e:
    print(f"❌ Failed to build model architecture: {e}")
    sys.exit(1)

# 5. LOAD WEIGHTS
print(f"\nAttempting to load weights from {MODEL_PATH.name}...")
try:
    # Try standard load
    model.load_weights(MODEL_PATH)
    print("\n🎉 SUCCESS! The model loaded perfectly.")
    print("If you see this, the issue is likely simply that you need to RESTART the uvicorn server.")
except Exception as e:
    print("\n❌ LOAD WEIGHTS FAILED!")
    print("="*30)
    print(e)
    print("="*30)
    
    # Attempt fallback debugging
    print("\nTrying fallback: Loading with strict=False (skips mismatching layers)...")
    try:
        model.load_weights(MODEL_PATH, skip_mismatch=True, by_name=True)
        print("⚠️  WARNING: Loaded with skip_mismatch=True. This means layer names do not match.")
        print("   This confirms a mismatch between the Training Code and Classifier.py")
    except Exception as e2:
        print(f"  Fallback failed too: {e2}")