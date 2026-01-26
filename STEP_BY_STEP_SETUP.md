# 🔧 Step-by-Step Setup Guide
## Complete Missing Parts Installation

**Last Updated:** January 26, 2026  
**Estimated Time:** 15-30 minutes  
**Difficulty:** Beginner-Friendly

---

## 📋 What You Need Before Starting

- [ ] Windows PC (or Linux/Mac - see alternative commands)
- [ ] Python 3.9 or higher installed
- [ ] Internet connection (for downloading packages)
- [ ] Your trained model files:
  - `ensemble_models.pkl`
  - `feature_engine.pkl`
  - `model_metadata.pkl`
- [ ] At least 8GB RAM
- [ ] At least 5GB free disk space

---

## 🎯 Overview

You'll complete these tasks:
1. ✅ Activate virtual environment
2. ✅ Install Python dependencies
3. ✅ Add your trained model files
4. ✅ Validate the setup
5. ✅ Run the application
6. ✅ Test with a sample image

**Total time:** About 15-30 minutes

---

## 📍 STEP 1: Open PowerShell/Terminal

### Windows:
1. Press `Windows + X`
2. Click **"Windows PowerShell"** or **"Terminal"**
3. Navigate to project directory:

```powershell
cd C:\Users\user\Desktop\csiro-biomass-web
```

### Linux/Mac:
```bash
cd ~/path/to/csiro-biomass-web
```

**Verify you're in the right place:**
```powershell
ls
```

You should see files like: `app.py`, `README.md`, `setup.ps1`

✅ **Checkpoint:** You're in the project directory

---

## 📍 STEP 2: Activate Virtual Environment

### Windows PowerShell:

```powershell
# Activate virtual environment
.\venv\Scripts\Activate.ps1
```

**If you get an error about execution policy:**

```powershell
# Allow script execution (run as Administrator if needed)
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Then try again
.\venv\Scripts\Activate.ps1
```

### Windows CMD:

```cmd
venv\Scripts\activate.bat
```

### Linux/Mac:

```bash
source venv/bin/activate
```

**Success indicators:**
- Your prompt should now show `(venv)` at the beginning
- Example: `(venv) PS C:\Users\user\Desktop\csiro-biomass-web>`

✅ **Checkpoint:** Virtual environment is active (you see `(venv)`)

---

## 📍 STEP 3: Upgrade pip (Recommended)

Before installing packages, upgrade pip to the latest version:

```powershell
python -m pip install --upgrade pip
```

**Expected output:**
```
Successfully installed pip-XX.X.X
```

✅ **Checkpoint:** pip is upgraded

---

## 📍 STEP 4: Install Python Dependencies

This is the most time-consuming step (5-15 minutes).

### Option A: Install All Dependencies (Recommended)

```powershell
pip install -r requirements.txt
```

**What you'll see:**
```
Collecting Flask==3.0.0
Downloading Flask-3.0.0-py3-none-any.whl (...)
Installing collected packages: ...
Successfully installed Flask-3.0.0 numpy-1.24.3 pandas-2.0.3 ...
```

### Option B: Install Core Packages First (If Option A Fails)

If you encounter errors, install packages in groups:

**Group 1: Web Framework**
```powershell
pip install Flask==3.0.0 Werkzeug==3.0.1
```

**Group 2: Core ML**
```powershell
pip install numpy==1.24.3 pandas==2.0.3 scikit-learn==1.3.0
```

**Group 3: Gradient Boosting**
```powershell
pip install lightgbm==4.1.0 xgboost==2.0.3
```

**Group 4: CatBoost (May fail on some systems - optional)**
```powershell
pip install catboost==1.2
```

**Group 5: Deep Learning**
```powershell
pip install torch==2.1.0 torchvision==0.16.0 transformers==4.35.0
```

**Group 6: Computer Vision**
```powershell
pip install opencv-python==4.8.1.78 Pillow==10.1.0
```

**Group 7: Utilities**
```powershell
pip install tqdm==4.66.1 python-dotenv==1.0.0
```

**Group 8: Production Server**
```powershell
pip install gunicorn==21.2.0
```

### Troubleshooting Installation Issues

**Issue: CatBoost fails to install**

```powershell
# Skip CatBoost for now (you can use other models)
pip install --no-deps catboost==1.2
# Or completely skip it
```

**Issue: PyTorch installation is slow**

```powershell
# Use CPU-only version (smaller, faster)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

**Issue: Out of memory during installation**

```powershell
# Close other applications
# Install one package at a time
pip install numpy
pip install pandas
# etc.
```

✅ **Checkpoint:** Dependencies installed (most packages should install successfully)

**Verify installation:**
```powershell
pip list
```

You should see Flask, numpy, pandas, scikit-learn, etc.

---

## 📍 STEP 5: Add Your Trained Model Files

This is **CRITICAL** - the app won't work without model files.

### 5.1: Locate Your Model Files

Find these files on your computer:
- `ensemble_models.pkl`
- `feature_engine.pkl`
- `model_metadata.pkl`

**Where they might be:**
- Your training notebook's output folder
- Kaggle downloads folder
- Project directory where you trained models
- Downloads folder

### 5.2: Copy Model Files

**Option A: Using File Explorer (Easiest)**

1. Open File Explorer
2. Navigate to `C:\Users\user\Desktop\csiro-biomass-web\models`
3. Copy your three `.pkl` files into this folder
4. Verify files are in the correct location

**Option B: Using PowerShell**

```powershell
# Example: if your models are in Downloads
Copy-Item "C:\Users\user\Downloads\ensemble_models.pkl" -Destination ".\models\"
Copy-Item "C:\Users\user\Downloads\feature_engine.pkl" -Destination ".\models\"
Copy-Item "C:\Users\user\Downloads\model_metadata.pkl" -Destination ".\models\"
```

**Option C: Using Command Line**

```powershell
# Navigate to where your models are
cd C:\path\to\your\models

# Copy to project
copy ensemble_models.pkl C:\Users\user\Desktop\csiro-biomass-web\models\
copy feature_engine.pkl C:\Users\user\Desktop\csiro-biomass-web\models\
copy model_metadata.pkl C:\Users\user\Desktop\csiro-biomass-web\models\
```

### 5.3: Verify Model Files

```powershell
# List files in models directory
ls models\
```

**Expected output:**
```
Mode                 LastWriteTime         Length Name
----                 -------------         ------ ----
-a----        1/26/2026   6:30 PM       12345678 ensemble_models.pkl
-a----        1/26/2026   6:30 PM        1234567 feature_engine.pkl
-a----        1/26/2026   6:30 PM          12345 model_metadata.pkl
```

✅ **Checkpoint:** All three model files are in `models/` directory

---

## 📍 STEP 6: Validate Your Setup

Run the validation script to check everything:

```powershell
python test_setup.py
```

### Expected Output:

```
============================================================
  CSIRO Biomass Web App - Setup Validation
============================================================

============================================================
  Checking Python Version
============================================================
Python 3.X.X
✓ Python version is compatible

============================================================
  Checking Dependencies
============================================================
✓ Flask
✓ numpy
✓ pandas
✓ scikit-learn
✓ LightGBM
✓ CatBoost
✓ PyTorch
✓ Transformers
✓ Pillow
✓ opencv-python
✓ python-dotenv

Installed: 11/11

============================================================
  Checking Directory Structure
============================================================
✓ models/
✓ templates/
✓ static/
✓ static/css/
✓ static/js/
✓ static/results/
✓ uploads/

============================================================
  Checking Model Files
============================================================
✓ models/ensemble_models.pkl (XX.X MB)
✓ models/feature_engine.pkl (XX.X MB)
✓ models/model_metadata.pkl (XX.X MB)

============================================================
  Checking Configuration
============================================================
✓ .env file exists
✓ app.py exists
✓ templates/index.html exists

============================================================
  Testing Flask App Import
============================================================
✓ Flask app imports successfully

============================================================
  Summary
============================================================
✓ PASS   Python Version
✓ PASS   Dependencies
✓ PASS   Directory Structure
✓ PASS   Model Files
✓ PASS   Configuration
✓ PASS   App Import

Total: 6/6 checks passed

🎉 All checks passed! Your setup is ready.

To run the application:
  python app.py

Then open: http://localhost:5000
```

### If Some Checks Fail:

**Failed: Dependencies**
- Run: `pip install -r requirements.txt` again
- Install missing packages individually

**Failed: Model Files**
- Verify files are in `models/` directory
- Check file names match exactly (case-sensitive)

**Failed: App Import**
- Check for Python errors in the output
- Ensure all dependencies are installed

✅ **Checkpoint:** All 6/6 checks pass

---

## 📍 STEP 7: Run the Application

### Method 1: Using Helper Script (Easiest)

**Windows:**
```powershell
.\run.bat
```

**Linux/Mac:**
```bash
chmod +x run.sh
./run.sh
```

### Method 2: Direct Python Command

```powershell
python app.py
```

### Expected Output:

```
================================================================================
CSIRO Biomass Prediction Web Application
================================================================================

Loading models...
✓ Loaded ensemble models
✓ Loaded feature engine
✓ Loaded model metadata

✓ All models loaded successfully!

Server Details:
  URL: http://localhost:5000
  Environment: development
  Debug mode: True

Ready to predict biomass! 🌿

 * Serving Flask app 'app'
 * Running on http://127.0.0.1:5000
Press CTRL+C to quit
```

### Troubleshooting Startup Issues:

**Issue: Port 5000 already in use**

```powershell
# Find what's using port 5000
netstat -ano | findstr :5000

# Kill the process (replace XXXX with PID from above)
taskkill /PID XXXX /F

# Or use a different port - edit app.py last line:
# app.run(debug=True, port=5001)
```

**Issue: ModuleNotFoundError**

```powershell
# Make sure virtual environment is active
# You should see (venv) in your prompt

# Reinstall the missing module
pip install <module-name>
```

**Issue: Model files not loading**

- Check files are in `models/` directory
- Verify file permissions (not read-only)
- Check file sizes (shouldn't be 0 bytes)

✅ **Checkpoint:** Application is running, you see "Running on http://127.0.0.1:5000"

---

## 📍 STEP 8: Test in Web Browser

### 8.1: Open Browser

1. Open your web browser (Chrome, Firefox, Edge)
2. Go to: **http://localhost:5000**

### 8.2: Expected Web Interface

You should see:
- **CSIRO Biomass Prediction** header
- **Upload image** button
- Model selection options
- Clean, modern interface

### 8.3: Test with an Image

**Option A: Use Your Own Pasture Image**

1. Click **"Choose Image"** or **"Upload Image"**
2. Select a pasture/grass image (JPG or PNG)
3. Click **"Analyze Biomass"** or **"Predict"**
4. Wait 2-5 seconds
5. View results with predictions

**Option B: Test with Any Image First**

If you don't have a pasture image:
1. Download a sample grass/field image from Google
2. Save it to your desktop
3. Upload and test

### 8.4: Verify Predictions Appear

You should see:
- **Dry_Green_g**: XXX grams
- **Dry_Clover_g**: XXX grams
- **Dry_Dead_g**: XXX grams
- **GDM_g**: XXX grams
- **Dry_Total_g**: XXX grams

Plus charts/visualizations

✅ **Checkpoint:** You can upload an image and get predictions

---

## 📍 STEP 9: Test the API (Optional)

### Using PowerShell

```powershell
# Health check
Invoke-RestMethod -Uri "http://localhost:5000/health"
```

**Expected output:**
```
status        : healthy
models_loaded : True
```

### Using Python

Open a **NEW** PowerShell window (keep the app running in the first one):

```powershell
cd C:\Users\user\Desktop\csiro-biomass-web

# Activate venv
.\venv\Scripts\Activate.ps1

# Run example
python examples\single_prediction.py
```

✅ **Checkpoint:** API is working

---

## 📍 STEP 10: Success! What's Next?

### 🎉 Congratulations! Your app is running!

You now have:
- ✅ Working web interface
- ✅ REST API
- ✅ Batch processing capability
- ✅ CSV export functionality

### Next Actions:

**1. Create Test Images Folder (for batch processing)**

```powershell
mkdir test_images
# Copy several pasture images here
```

**2. Test Batch Processing**

```powershell
python examples\batch_prediction.py
```

**3. Read the Documentation**

- `QUICKSTART.md` - Quick reference
- `README.md` - Full documentation
- `examples/README.md` - API examples

**4. Deploy (Optional)**

See `COMPLETE_WEB_APP_GUIDE.md` for production deployment

---

## 🐛 Common Issues & Solutions

### Issue 1: "Virtual environment not found"

**Solution:**
```powershell
# Create it
python -m venv venv

# Then activate
.\venv\Scripts\Activate.ps1
```

### Issue 2: "pip not recognized"

**Solution:**
```powershell
# Use full path
python -m pip install -r requirements.txt
```

### Issue 3: "Cannot activate virtual environment"

**Solution:**
```powershell
# Set execution policy
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Try again
.\venv\Scripts\Activate.ps1
```

### Issue 4: "torch is too large to install"

**Solution:**
```powershell
# Install CPU-only version
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Issue 5: "Models not loading"

**Solution:**
1. Check file names match exactly:
   - `ensemble_models.pkl` (not `ensemble_model.pkl`)
   - `feature_engine.pkl`
   - `model_metadata.pkl`
2. Check files are in `models/` directory
3. Verify files aren't corrupted (size > 0)

### Issue 6: "Connection refused" when testing API

**Solution:**
- Make sure app is running (`python app.py`)
- Check you're using correct URL: `http://localhost:5000`
- Try `http://127.0.0.1:5000` instead

---

## 📝 Quick Command Reference

```powershell
# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Validate setup
python test_setup.py

# Run application
python app.py
# Or: .\run.bat

# Test API
python examples\single_prediction.py

# Batch processing
python examples\batch_prediction.py

# Stop server
# Press Ctrl+C in the terminal running app.py
```

---

## ✅ Final Checklist

Before considering setup complete, verify:

- [ ] Virtual environment activates
- [ ] `pip list` shows all packages
- [ ] Three model files in `models/` directory
- [ ] `python test_setup.py` shows 6/6 passes
- [ ] `python app.py` starts without errors
- [ ] Browser loads http://localhost:5000
- [ ] Can upload image and get predictions
- [ ] Predictions show reasonable values

**If all checkboxes are checked: YOU'RE DONE! 🎉**

---

## 📞 Need More Help?

### Self-Help Resources:

1. **Run validation:** `python test_setup.py`
2. **Read docs:** `QUICKSTART.md`
3. **Check examples:** `examples/README.md`
4. **Review errors:** Read error messages carefully

### Documentation Files:

- `SETUP_COMPLETE.txt` - Simple overview
- `QUICKSTART.md` - Quick start
- `README.md` - Full reference
- `PROJECT_COMPLETION_REPORT.md` - Detailed info

---

**Setup guide complete! Now start predicting biomass! 🌿**

Last updated: January 26, 2026
