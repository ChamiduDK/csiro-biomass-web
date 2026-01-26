# 🚀 Quick Start Guide

Get the CSIRO Biomass Prediction Web App running in just a few minutes!

## Prerequisites ✅

- **Python 3.9+** installed
- **8GB+ RAM** (recommended)
- **Trained model files** (ensemble_models.pkl, feature_engine.pkl, model_metadata.pkl)

---

## Option 1: Automated Setup (Windows) 🪟

Run the setup script:

```powershell
.\setup.ps1
```

This will:
- ✓ Check Python installation
- ✓ Create virtual environment
- ✓ Install all dependencies
- ✓ Set up project structure
- ✓ Check for model files

---

## Option 2: Manual Setup 🛠️

### Step 1: Create Virtual Environment

```bash
python -m venv venv
```

### Step 2: Activate Virtual Environment

**Windows:**
```powershell
venv\Scripts\activate
```

**Linux/Mac:**
```bash
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Add Your Model Files

Copy your trained models to the `models/` directory:

```
models/
├── ensemble_models.pkl
├── feature_engine.pkl
├── model_metadata.pkl
└── siglip-so400m-patch14-384/ (optional)
```

---

## Running the Application 🏃

### Development Mode

```bash
python app.py
```

The app will start on **http://localhost:5000**

### Production Mode

```bash
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

---

## Using the App 📱

### Single Image Prediction

1. Open **http://localhost:5000** in your browser
2. Click **"Choose Image"**
3. Select a pasture image (JPG/PNG)
4. Select model types (optional)
5. Click **"Analyze Biomass"**
6. View results with charts

### Batch Processing

1. Switch to **"Batch Upload"** tab
2. Select multiple images
3. Click **"Analyze All Images"**
4. Download results as CSV

### API Usage

```python
import requests

# Single prediction
url = "http://localhost:5000/predict"
files = {'image': open('pasture.jpg', 'rb')}
response = requests.post(url, files=files)
print(response.json())
```

---

## Docker Deployment 🐳

### Quick Start with Docker Compose

```bash
# Start the application
docker-compose up -d

# View logs
docker-compose logs -f

# Stop the application
docker-compose down
```

The app will be available at **http://localhost:5000**

---

## Troubleshooting 🔧

### Models Not Loading

**Problem:** "Model files not found"

**Solution:** 
1. Check that model files are in the `models/` directory
2. Verify file names match exactly:
   - `ensemble_models.pkl`
   - `feature_engine.pkl`
   - `model_metadata.pkl`

### Out of Memory

**Problem:** Application crashes due to memory

**Solution:**
1. Close other applications
2. Use CPU instead of GPU (edit `.env`: `DEVICE=cpu`)
3. Process fewer images at once in batch mode

### Port Already in Use

**Problem:** "Address already in use"

**Solution:**
```bash
# Find process using port 5000
netstat -ano | findstr :5000

# Kill the process (Windows)
taskkill /PID <process_id> /F
```

### Dependencies Installation Fails

**Problem:** pip install errors

**Solution:**
```bash
# Upgrade pip
python -m pip install --upgrade pip

# Install with verbose output
pip install -r requirements.txt -v
```

---

## Configuration ⚙️

### Environment Variables

Edit `.env` file:

```bash
# Flask settings
FLASK_ENV=production
SECRET_KEY=your-secret-key

# Model settings
MODEL_PATH=models/
DEVICE=cpu  # or 'cuda' for GPU

# Upload settings
MAX_FILE_SIZE=16777216  # 16MB
ALLOWED_EXTENSIONS=png,jpg,jpeg
```

---

## Next Steps 📚

- **Full Documentation:** See [README.md](README.md)
- **Deployment Guide:** See [COMPLETE_WEB_APP_GUIDE.md](COMPLETE_WEB_APP_GUIDE.md)
- **API Reference:** See [README.md#api-endpoints](README.md#api-endpoints)

---

## Getting Help 💬

- **GitHub Issues:** Report bugs or request features
- **Email:** [your-email@example.com]
- **Documentation:** Check README.md for detailed information

---

**Made with ❤️ for sustainable agriculture**
