"""
Single Image Prediction Example
Demonstrates how to use the CSIRO Biomass API for single image predictions
"""

import requests
import json
from pathlib import Path

# Configuration
API_URL = "http://localhost:5000"
IMAGE_PATH = "test_image.jpg"  # Replace with your image path

def predict_single_image(image_path, models=None):
    """
    Predict biomass from a single image
    
    Args:
        image_path: Path to the image file
        models: List of model names (optional)
                Options: lightgbm, catboost, random_forest, extra_trees, 
                        mlp, histgbm, gradient_boosting
    
    Returns:
        dict: Prediction results
    """
    
    # Check if image exists
    if not Path(image_path).exists():
        print(f"❌ Error: Image not found at {image_path}")
        return None
    
    # Prepare the request
    url = f"{API_URL}/predict"
    
    # Open image file
    with open(image_path, 'rb') as image_file:
        files = {'image': image_file}
        
        # Optional: specify which models to use
        data = {}
        if models:
            data['models'] = ','.join(models)
        
        # Make the request
        print(f"📤 Sending request to {url}...")
        print(f"📸 Image: {image_path}")
        
        if models:
            print(f"🤖 Models: {', '.join(models)}")
        
        response = requests.post(url, files=files, data=data)
    
    # Check response
    if response.status_code == 200:
        result = response.json()
        
        if result.get('success'):
            print("\n✅ Prediction successful!")
            print(f"⏱️  Timestamp: {result.get('timestamp')}")
            print("\n📊 Predictions:")
            print("-" * 50)
            
            predictions = result.get('predictions', {})
            for target, value in predictions.items():
                print(f"  {target:15s}: {value:8.2f} grams")
            
            print("-" * 50)
            return result
        else:
            print(f"❌ Prediction failed: {result.get('error')}")
            return None
    else:
        print(f"❌ HTTP Error {response.status_code}")
        print(f"Response: {response.text}")
        return None

def check_server_health():
    """Check if the API server is running"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Server is {data.get('status')}")
            print(f"🤖 Models loaded: {data.get('models_loaded')}")
            return True
        else:
            print(f"⚠️  Server returned status code {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to server")
        print(f"Make sure the server is running at {API_URL}")
        return False
    except requests.exceptions.Timeout:
        print("❌ Connection timed out")
        return False

def get_model_info():
    """Get information about available models"""
    try:
        response = requests.get(f"{API_URL}/model-info")
        if response.status_code == 200:
            data = response.json()
            print("\n📋 Model Information:")
            print("-" * 50)
            print(f"Available models: {len(data.get('models', []))}")
            for model in data.get('models', []):
                print(f"  • {model}")
            print(f"Targets: {data.get('targets')}")
            print("-" * 50)
            return data
        return None
    except Exception as e:
        print(f"❌ Error getting model info: {e}")
        return None

if __name__ == "__main__":
    print("=" * 60)
    print("CSIRO Biomass Prediction API - Single Image Example")
    print("=" * 60)
    
    # Check server health
    print("\n1️⃣  Checking server health...")
    if not check_server_health():
        print("\n💡 Tip: Start the server with: python app.py")
        exit(1)
    
    # Get model information
    print("\n2️⃣  Getting model information...")
    get_model_info()
    
    # Make a prediction
    print("\n3️⃣  Making prediction...")
    
    # Example 1: Use all available models
    result = predict_single_image(IMAGE_PATH)
    
    # Example 2: Use specific models
    # result = predict_single_image(
    #     IMAGE_PATH, 
    #     models=['lightgbm', 'catboost', 'random_forest']
    # )
    
    if result:
        print("\n✨ Complete!")
        
        # You can access the predictions like this:
        predictions = result.get('predictions', {})
        
        # Example: Calculate total biomass
        total = predictions.get('Dry_Total_g', 0)
        print(f"\n🌿 Total Dry Biomass: {total:.2f} grams")
        
        # Example: Calculate green percentage
        green = predictions.get('Dry_Green_g', 0)
        if total > 0:
            green_pct = (green / total) * 100
            print(f"🌱 Green Matter: {green_pct:.1f}%")
