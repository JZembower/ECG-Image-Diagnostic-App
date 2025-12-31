"""
Setup Test Script for ECG Diagnosis Backend

Run this script after installing dependencies to verify the setup:
    python test_setup.py

Prerequisites:
    pip install -r requirements.txt
"""

import sys
import os

def test_imports():
    """Test that all required packages can be imported"""
    print("=" * 60)
    print("Testing Package Imports")
    print("=" * 60)
    
    packages = {
        'fastapi': 'FastAPI',
        'uvicorn': 'Uvicorn',
        'cv2': 'OpenCV',
        'numpy': 'NumPy',
        'scipy': 'SciPy',
        'pydantic': 'Pydantic',
        'PIL': 'Pillow',
    }
    
    failed = []
    
    for package, name in packages.items():
        try:
            __import__(package)
            print(f"✓ {name:20} imported successfully")
        except ImportError as e:
            print(f"✗ {name:20} import failed: {e}")
            failed.append(name)
    
    # Test TensorFlow separately (optional)
    try:
        import tensorflow as tf
        print(f"✓ {'TensorFlow':20} imported successfully (version {tf.__version__})")
    except ImportError as e:
        print(f"⚠ {'TensorFlow':20} not installed: {e}")
        print("  Install with: pip install tensorflow>=2.14.0")
        failed.append('TensorFlow')
    
    print()
    return failed


def test_module_structure():
    """Test that the module structure is correct"""
    print("=" * 60)
    print("Testing Module Structure")
    print("=" * 60)
    
    required_files = [
        'main.py',
        'requirements.txt',
        'Dockerfile',
        'docker-compose.yml',
        'services/__init__.py',
        'services/digitizer.py',
        'services/classifier.py',
        'models/.gitkeep',
        'README.md',
        '.env.example',
    ]
    
    failed = []
    
    for file_path in required_files:
        full_path = os.path.join(os.path.dirname(__file__), file_path)
        if os.path.exists(full_path):
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path} NOT FOUND")
            failed.append(file_path)
    
    print()
    return failed


def test_model_files():
    """Test that model files exist"""
    print("=" * 60)
    print("Testing Model Files")
    print("=" * 60)
    
    model_files = [
        'models/model.weights.h5',
        'models/classes.npy',
    ]
    
    missing = []
    
    for file_path in model_files:
        full_path = os.path.join(os.path.dirname(__file__), file_path)
        if os.path.exists(full_path):
            size = os.path.getsize(full_path)
            print(f"✓ {file_path} ({size:,} bytes)")
        else:
            print(f"⚠ {file_path} NOT FOUND")
            missing.append(file_path)
    
    if missing:
        print("\nNote: Model files are required for the API to work.")
        print("See README.md for instructions on obtaining model files.")
    
    print()
    return missing


def test_services_import():
    """Test that services can be imported"""
    print("=" * 60)
    print("Testing Services Import")
    print("=" * 60)
    
    try:
        # Add current directory to Python path
        sys.path.insert(0, os.path.dirname(__file__))
        
        from services import digitizer, classifier
        print("✓ services.digitizer imported successfully")
        print("✓ services.classifier imported successfully")
        
        # Check for key functions
        functions = [
            ('services.digitizer', 'extract_signal_from_image'),
            ('services.digitizer', 'assess_signal_quality'),
            ('services.classifier', 'classify_ecg'),
            ('services.classifier', 'load_model'),
            ('services.classifier', 'calculate_bpm'),
        ]
        
        for module_name, func_name in functions:
            module = sys.modules[module_name.split('.')[1]]
            if hasattr(module, func_name):
                print(f"✓ {module_name}.{func_name} exists")
            else:
                print(f"✗ {module_name}.{func_name} NOT FOUND")
        
        print()
        return True
    
    except Exception as e:
        print(f"✗ Failed to import services: {e}")
        print()
        return False


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("ECG Diagnosis Backend - Setup Test")
    print("=" * 60 + "\n")
    
    # Test imports
    import_failures = test_imports()
    
    # Test module structure
    structure_failures = test_module_structure()
    
    # Test model files
    model_missing = test_model_files()
    
    # Test services
    services_ok = test_services_import()
    
    # Summary
    print("=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    if import_failures:
        print(f"✗ {len(import_failures)} package(s) failed to import: {', '.join(import_failures)}")
        print("  Install dependencies: pip install -r requirements.txt")
    else:
        print("✓ All required packages imported successfully")
    
    if structure_failures:
        print(f"✗ {len(structure_failures)} file(s) missing: {', '.join(structure_failures)}")
    else:
        print("✓ All required files present")
    
    if model_missing:
        print(f"⚠ {len(model_missing)} model file(s) missing: {', '.join(model_missing)}")
        print("  See README.md for instructions on obtaining model files")
    else:
        print("✓ All model files present")
    
    if services_ok:
        print("✓ Services module structure correct")
    else:
        print("✗ Services module has issues")
    
    print()
    
    if not import_failures and not structure_failures and services_ok:
        print("🎉 Setup complete! You can now run the backend:")
        print("   uvicorn main:app --reload")
        print()
        if model_missing:
            print("⚠ Note: Model files are missing. The API will run but predictions will fail.")
            print("  See README.md for instructions on obtaining model files.")
        return 0
    else:
        print("❌ Setup incomplete. Please fix the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
