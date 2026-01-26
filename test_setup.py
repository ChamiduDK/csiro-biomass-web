"""
Test script to validate the CSIRO Biomass Web App setup
Checks dependencies, models, and configuration
"""

import sys
import os
from pathlib import Path

def print_header(text):
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60)

def check_python_version():
    """Check if Python version is 3.9 or higher"""
    print_header("Checking Python Version")
    version = sys.version_info
    print(f"Python {version.major}.{version.minor}.{version.micro}")
    
    if version.major >= 3 and version.minor >= 9:
        print("✓ Python version is compatible")
        return True
    else:
        print("✗ Python 3.9 or higher is required")
        return False

def check_dependencies():
    """Check if all required packages are installed"""
    print_header("Checking Dependencies")
    
    required_packages = {
        'flask': 'Flask',
        'numpy': 'numpy',
        'pandas': 'pandas',
        'sklearn': 'scikit-learn',
        'lightgbm': 'LightGBM',
        'catboost': 'CatBoost',
        'torch': 'PyTorch',
        'transformers': 'Transformers',
        'PIL': 'Pillow',
        'cv2': 'opencv-python',
        'dotenv': 'python-dotenv'
    }
    
    missing = []
    installed = []
    
    for module, package in required_packages.items():
        try:
            __import__(module)
            installed.append(package)
            print(f"✓ {package}")
        except ImportError:
            missing.append(package)
            print(f"✗ {package} - Not installed")
    
    print(f"\nInstalled: {len(installed)}/{len(required_packages)}")
    
    if missing:
        print("\nMissing packages:")
        for pkg in missing:
            print(f"  - {pkg}")
        print("\nRun: pip install -r requirements.txt")
        return False
    
    return True

def check_directory_structure():
    """Check if required directories exist"""
    print_header("Checking Directory Structure")
    
    required_dirs = [
        'models',
        'templates',
        'static',
        'static/css',
        'static/js',
        'static/results',
        'uploads'
    ]
    
    all_exist = True
    for dir_name in required_dirs:
        dir_path = Path(dir_name)
        if dir_path.exists():
            print(f"✓ {dir_name}/")
        else:
            print(f"✗ {dir_name}/ - Missing")
            all_exist = False
    
    return all_exist

def check_model_files():
    """Check if model files exist"""
    print_header("Checking Model Files")
    
    required_models = [
        'models/ensemble_models.pkl',
        'models/feature_engine.pkl',
        'models/model_metadata.pkl'
    ]
    
    optional_models = [
        'models/siglip-so400m-patch14-384'
    ]
    
    all_exist = True
    for model_file in required_models:
        model_path = Path(model_file)
        if model_path.exists():
            size = model_path.stat().st_size / (1024 * 1024)  # MB
            print(f"✓ {model_file} ({size:.1f} MB)")
        else:
            print(f"✗ {model_file} - Missing")
            all_exist = False
    
    print("\nOptional:")
    for model_file in optional_models:
        model_path = Path(model_file)
        if model_path.exists():
            print(f"✓ {model_file}")
        else:
            print(f"○ {model_file} - Not found (optional)")
    
    if not all_exist:
        print("\n⚠️  Some required model files are missing!")
        print("Please copy your trained models to the models/ directory")
    
    return all_exist

def check_configuration():
    """Check configuration files"""
    print_header("Checking Configuration")
    
    # Check .env file
    env_path = Path('.env')
    if env_path.exists():
        print("✓ .env file exists")
    else:
        print("○ .env file not found (optional)")
    
    # Check app.py
    app_path = Path('app.py')
    if app_path.exists():
        print("✓ app.py exists")
    else:
        print("✗ app.py missing")
        return False
    
    # Check templates
    template_path = Path('templates/index.html')
    if template_path.exists():
        print("✓ templates/index.html exists")
    else:
        print("✗ templates/index.html missing")
        return False
    
    return True

def test_import_app():
    """Try to import the Flask app"""
    print_header("Testing Flask App Import")
    
    try:
        # Add current directory to path
        sys.path.insert(0, os.getcwd())
        
        # Try to import app
        import app as flask_app
        print("✓ Flask app imports successfully")
        return True
    except Exception as e:
        print(f"✗ Error importing Flask app: {str(e)}")
        return False

def main():
    """Run all checks"""
    print("\n" + "="*60)
    print("  CSIRO Biomass Web App - Setup Validation")
    print("="*60)
    
    results = {
        'Python Version': check_python_version(),
        'Dependencies': check_dependencies(),
        'Directory Structure': check_directory_structure(),
        'Model Files': check_model_files(),
        'Configuration': check_configuration(),
        'App Import': test_import_app()
    }
    
    # Summary
    print_header("Summary")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for check, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:8} {check}")
    
    print(f"\nTotal: {passed}/{total} checks passed")
    
    if passed == total:
        print("\n🎉 All checks passed! Your setup is ready.")
        print("\nTo run the application:")
        print("  python app.py")
        print("\nThen open: http://localhost:5000")
    else:
        print("\n⚠️  Some checks failed. Please fix the issues above.")
        failed = [k for k, v in results.items() if not v]
        print("\nFailed checks:")
        for check in failed:
            print(f"  - {check}")
    
    return passed == total

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
