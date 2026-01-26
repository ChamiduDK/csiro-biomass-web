"""
Batch Prediction Example
Demonstrates how to use the CSIRO Biomass API for batch image predictions
"""

import requests
import json
from pathlib import Path
import time
import pandas as pd

# Configuration
API_URL = "http://localhost:5000"
IMAGES_FOLDER = "test_images"  # Folder containing images to process

def predict_batch(image_paths, models=None):
    """
    Predict biomass from multiple images
    
    Args:
        image_paths: List of paths to image files
        models: List of model names (optional)
    
    Returns:
        list: List of prediction results
    """
    
    url = f"{API_URL}/batch-predict"
    
    # Prepare files
    files = []
    valid_paths = []
    
    for path in image_paths:
        if Path(path).exists():
            files.append(('images', open(path, 'rb')))
            valid_paths.append(path)
        else:
            print(f"⚠️  Skipping missing file: {path}")
    
    if not files:
        print("❌ No valid images found")
        return None
    
    # Prepare data
    data = {}
    if models:
        data['models'] = ','.join(models)
    
    print(f"📤 Sending {len(files)} images for batch prediction...")
    start_time = time.time()
    
    try:
        # Make the request
        response = requests.post(url, files=files, data=data)
        
        # Close all file handles
        for _, file_handle in files:
            file_handle.close()
        
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            
            if result.get('success'):
                print(f"✅ Batch prediction successful!")
                print(f"⏱️  Total time: {elapsed:.2f} seconds")
                print(f"📊 Processed: {result.get('total_images')} images")
                print(f"⚡ Average: {elapsed/result.get('total_images'):.2f} sec/image")
                
                return result.get('results', [])
            else:
                print(f"❌ Batch prediction failed: {result.get('error')}")
                return None
        else:
            print(f"❌ HTTP Error {response.status_code}")
            return None
            
    except Exception as e:
        print(f"❌ Error: {e}")
        # Close file handles in case of error
        for _, file_handle in files:
            try:
                file_handle.close()
            except:
                pass
        return None

def find_images_in_folder(folder_path, extensions=None):
    """
    Find all image files in a folder
    
    Args:
        folder_path: Path to the folder
        extensions: List of extensions (default: jpg, jpeg, png)
    
    Returns:
        list: List of image paths
    """
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png']
    
    folder = Path(folder_path)
    
    if not folder.exists():
        print(f"❌ Folder not found: {folder_path}")
        return []
    
    images = []
    for ext in extensions:
        images.extend(folder.glob(f"*{ext}"))
        images.extend(folder.glob(f"*{ext.upper()}"))
    
    return [str(img) for img in images]

def save_results_to_csv(results, output_file="predictions.csv"):
    """
    Save batch prediction results to CSV
    
    Args:
        results: List of prediction results
        output_file: Output CSV file path
    """
    
    # Convert results to DataFrame
    rows = []
    for result in results:
        row = {
            'filename': result.get('filename'),
            'timestamp': result.get('timestamp')
        }
        
        # Add predictions
        predictions = result.get('predictions', {})
        for target, value in predictions.items():
            row[target] = value
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"💾 Results saved to: {output_file}")
    
    return df

def display_summary(results):
    """Display summary statistics of batch predictions"""
    
    if not results:
        return
    
    print("\n" + "=" * 60)
    print("📊 BATCH PREDICTION SUMMARY")
    print("=" * 60)
    
    # Extract all predictions
    all_predictions = {
        'Dry_Green_g': [],
        'Dry_Clover_g': [],
        'Dry_Dead_g': [],
        'GDM_g': [],
        'Dry_Total_g': []
    }
    
    for result in results:
        predictions = result.get('predictions', {})
        for target in all_predictions.keys():
            if target in predictions:
                all_predictions[target].append(predictions[target])
    
    # Calculate and display statistics
    for target, values in all_predictions.items():
        if values:
            avg = sum(values) / len(values)
            min_val = min(values)
            max_val = max(values)
            
            print(f"\n{target}:")
            print(f"  Average: {avg:8.2f} g")
            print(f"  Min:     {min_val:8.2f} g")
            print(f"  Max:     {max_val:8.2f} g")
    
    print("=" * 60)

if __name__ == "__main__":
    print("=" * 60)
    print("CSIRO Biomass Prediction API - Batch Prediction Example")
    print("=" * 60)
    
    # Find images in folder
    print(f"\n1️⃣  Searching for images in '{IMAGES_FOLDER}'...")
    image_paths = find_images_in_folder(IMAGES_FOLDER)
    
    if not image_paths:
        print("\n💡 Tips:")
        print(f"  • Create a folder named '{IMAGES_FOLDER}'")
        print(f"  • Add some pasture images (JPG/PNG)")
        print(f"  • Or modify IMAGES_FOLDER at the top of this script")
        exit(1)
    
    print(f"✅ Found {len(image_paths)} images")
    for path in image_paths[:5]:  # Show first 5
        print(f"  • {Path(path).name}")
    if len(image_paths) > 5:
        print(f"  ... and {len(image_paths) - 5} more")
    
    # Make batch predictions
    print(f"\n2️⃣  Making batch predictions...")
    results = predict_batch(image_paths)
    
    if results:
        # Display individual results
        print(f"\n3️⃣  Individual Results:")
        print("-" * 60)
        for i, result in enumerate(results, 1):
            filename = result.get('filename')
            predictions = result.get('predictions', {})
            total = predictions.get('Dry_Total_g', 0)
            print(f"{i}. {filename:30s} → {total:6.2f}g total")
        
        # Display summary statistics
        display_summary(results)
        
        # Save to CSV
        print(f"\n4️⃣  Saving results...")
        try:
            df = save_results_to_csv(results, "batch_predictions.csv")
            print("\n✅ Batch processing complete!")
            
            # Display DataFrame preview
            print("\n📋 Preview (first 5 rows):")
            print(df.head().to_string())
            
        except Exception as e:
            print(f"⚠️  Could not save CSV: {e}")
            print("Install pandas: pip install pandas")
    
    else:
        print("❌ Batch prediction failed")
        print("\n💡 Make sure:")
        print("  • The server is running (python app.py)")
        print("  • The images are valid")
        print("  • You have enough memory for batch processing")
