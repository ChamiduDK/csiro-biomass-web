# API Examples

This folder contains example scripts demonstrating how to use the CSIRO Biomass Prediction API.

## 📁 Files

- **`single_prediction.py`** - Example of predicting biomass from a single image
- **`batch_prediction.py`** - Example of processing multiple images in batch
- **`api_commands.sh`** - cURL examples for quick testing
- **`api_commands.ps1`** - PowerShell examples for Windows users

## 🚀 Quick Start

### Prerequisites

Make sure the server is running:

```bash
python app.py
```

The server should be accessible at `http://localhost:5000`

### Single Image Prediction

```bash
# Edit the IMAGE_PATH in the script
python examples/single_prediction.py
```

### Batch Prediction

```bash
# Create a folder with test images
mkdir test_images
# Copy your pasture images to test_images/

# Run batch prediction
python examples/batch_prediction.py
```

## 📝 Example Usage

### Python - Single Image

```python
import requests

url = "http://localhost:5000/predict"
files = {'image': open('pasture.jpg', 'rb')}

response = requests.post(url, files=files)
result = response.json()

if result['success']:
    print("Predictions:", result['predictions'])
```

### Python - Batch Processing

```python
import requests

url = "http://localhost:5000/batch-predict"
files = [
    ('images', open('image1.jpg', 'rb')),
    ('images', open('image2.jpg', 'rb')),
    ('images', open('image3.jpg', 'rb'))
]

response = requests.post(url, files=files)
result = response.json()

if result['success']:
    for item in result['results']:
        print(f"{item['filename']}: {item['predictions']}")
```

### cURL - Single Image (Linux/Mac)

```bash
curl -X POST http://localhost:5000/predict \
  -F "image=@pasture.jpg" \
  -F "models=lightgbm,catboost"
```

### PowerShell - Single Image (Windows)

```powershell
$uri = "http://localhost:5000/predict"
$imagePath = "pasture.jpg"

$form = @{
    image = Get-Item -Path $imagePath
    models = "lightgbm,catboost"
}

$response = Invoke-RestMethod -Uri $uri -Method Post -Form $form
$response | ConvertTo-Json -Depth 10
```

## 🎯 API Endpoints

### Health Check

```bash
curl http://localhost:5000/health
```

Response:
```json
{
  "status": "healthy",
  "models_loaded": true
}
```

### Model Information

```bash
curl http://localhost:5000/model-info
```

Response:
```json
{
  "models": ["lightgbm", "catboost", "random_forest", ...],
  "targets": ["Dry_Green_g", "Dry_Clover_g", "Dry_Dead_g", "GDM_g", "Dry_Total_g"]
}
```

### Single Prediction

**Endpoint:** `POST /predict`

**Parameters:**
- `image` (file, required) - The image file
- `models` (string, optional) - Comma-separated model names

**Example Response:**
```json
{
  "success": true,
  "predictions": {
    "Dry_Green_g": 45.32,
    "Dry_Clover_g": 2.15,
    "Dry_Dead_g": 12.87,
    "GDM_g": 47.47,
    "Dry_Total_g": 60.34
  },
  "timestamp": "20260126_182500"
}
```

### Batch Prediction

**Endpoint:** `POST /batch-predict`

**Parameters:**
- `images` (files, required) - Multiple image files
- `models` (string, optional) - Comma-separated model names

**Example Response:**
```json
{
  "success": true,
  "total_images": 3,
  "results": [
    {
      "filename": "image1.jpg",
      "predictions": {
        "Dry_Green_g": 45.32,
        "Dry_Clover_g": 2.15,
        "Dry_Dead_g": 12.87,
        "GDM_g": 47.47,
        "Dry_Total_g": 60.34
      },
      "timestamp": "20260126_182500"
    },
    ...
  ]
}
```

## 🔧 Customization

### Selecting Specific Models

You can specify which models to use for ensemble prediction:

```python
# Use only LightGBM and CatBoost
data = {'models': 'lightgbm,catboost'}

response = requests.post(url, files=files, data=data)
```

Available models:
- `lightgbm` - LightGBM (recommended)
- `catboost` - CatBoost (recommended)
- `random_forest` - Random Forest
- `extra_trees` - Extra Trees
- `mlp` - Multi-layer Perceptron
- `histgbm` - Histogram Gradient Boosting
- `gradient_boosting` - Gradient Boosting

### Error Handling

Always check for errors in the response:

```python
response = requests.post(url, files=files)
result = response.json()

if result.get('success'):
    predictions = result['predictions']
    # Process predictions
else:
    error = result.get('error', 'Unknown error')
    print(f"Error: {error}")
```

## 📊 Working with Results

### Saving to CSV

```python
import pandas as pd

# Convert results to DataFrame
df = pd.DataFrame([
    {
        'filename': r['filename'],
        **r['predictions']
    }
    for r in results
])

# Save to CSV
df.to_csv('predictions.csv', index=False)
```

### Calculating Statistics

```python
import numpy as np

# Extract total biomass values
totals = [r['predictions']['Dry_Total_g'] for r in results]

# Calculate statistics
print(f"Average: {np.mean(totals):.2f}g")
print(f"Std Dev: {np.std(totals):.2f}g")
print(f"Min: {np.min(totals):.2f}g")
print(f"Max: {np.max(totals):.2f}g")
```

## 🐛 Troubleshooting

### Connection Refused

**Problem:** `ConnectionRefusedError` or "Could not connect to server"

**Solution:** Make sure the Flask server is running:
```bash
python app.py
```

### Invalid Image

**Problem:** "Invalid image file"

**Solution:** 
- Ensure the image is a valid JPG or PNG file
- Check file size (max 16MB by default)
- Verify the file is not corrupted

### Out of Memory

**Problem:** Server crashes during batch processing

**Solution:**
- Process fewer images at once
- Reduce batch size in your script
- Close the server and restart it

### Model Not Found

**Problem:** "Model X not found"

**Solution:**
- Check available models with `/model-info` endpoint
- Use only models that are loaded
- Verify model files exist in `models/` directory

## 📚 Further Reading

- [Main README](../README.md) - Full documentation
- [Quick Start Guide](../QUICKSTART.md) - Getting started
- [API Reference](../README.md#api-endpoints) - Complete API documentation

---

**Need help?** Open an issue on GitHub or check the main documentation.
