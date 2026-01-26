# 🌿 CSIRO Biomass Prediction Web Application

AI-powered web application for predicting pasture biomass from images using ensemble machine learning models and SigLIP embeddings.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-3.0-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## 📋 Features

- **High-Resolution Prediction** - Uses SigLIP (SO400M) for state-of-the-art vision features.
- **Single Image Prediction** - Upload and analyze individual pasture images.
- **Batch Processing** - Process multiple images simultaneously and export results.
- **Model Ensemble** - Combines LightGBM, CatBoost, and Random Forest for robust predictions.
- **Modern UI** - Interactive, responsive interface with real-time visualization.
- **CSV Export** - Download structured results for further scientific analysis.

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10 (Recommended for best compatibility)
- 8GB+ RAM
- NVIDIA GPU with CUDA (Optional, for faster processing)

### Installation

1. **Clone and Enter**
   ```bash
   cd csiro-biomass-web
   ```

2. **Run the Automatic Starter**
   Double-click `START_APP.bat` on Windows. This script will:
   - Activate the virtual environment.
   - Install all required dependencies.
   - Verify models and start the server.

3. **Open Browser**
   Navigate to: `http://localhost:5000`

---

## 📁 Project Structure

```
csiro-biomass-web/
├── app.py                # Main Flask application logic
├── train_pipeline.py     # End-to-end model training script
├── feature_engine.py      # Custom feature engineering (PCA/PLS/GMM)
├── requirements.txt      # Python dependencies
├── START_APP.bat         # One-click Windows starter
│
├── models/               # Trained models and metadata
│   ├── ensemble_models.pkl
│   ├── feature_engine.pkl
│   ├── model_metadata.pkl
│   └── siglip-so400m-patch14-384/  # High-res vision model
│
├── templates/            # Web interface (HTML)
├── static/               # Assets (CSS/JS) and generated results
└── uploads/              # Temporal storage for uploaded images
```

---

## 🛠️ Configuration

The application automatically adapts to the available hardware and models. It checks `models/model_metadata.pkl` to determine the correct feature dimensions and vision model to load (SigLIP Base vs. SO400M).

---

## 🧪 Scientific Methodology

The system follows a three-stage prediction pipeline:
1. **Vision Engine**: Extract embeddings using SigLIP with patch-based averaging.
2. **Concept Engine**: Generate semantic scores for pasture qualities (greenness, clover, etc.).
3. **Ensemble Engine**: Feed embeddings and semantic features through an ensemble of Gradient Boosting and Forest models.

---

## 📝 License

This project is licensed under the MIT License.

---

**Made with ❤️ for sustainable agriculture**
