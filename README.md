# 🌿 CSIRO Biomass Prediction Web Application

AI-powered web application for predicting pasture biomass from images using ensemble machine learning models.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-3.0-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## 📋 Features

- **Single Image Prediction** - Upload and analyze individual pasture images
- **Batch Processing** - Process multiple images simultaneously
- **Model Ensemble** - Combines LightGBM, CatBoost, Random Forest, Extra Trees, MLP, and more
- **Interactive UI** - Modern, responsive web interface with real-time results
- **Data Visualization** - Charts and graphs for prediction analysis
- **CSV Export** - Download results for further analysis
- **REST API** - Programmatic access to predictions
- **Docker Support** - Easy deployment with Docker and Docker Compose

---

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher
- 8GB+ RAM recommended
- GPU optional (CUDA support for faster inference)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/ChamiduDK/biomass-webapp.git
cd biomass-webapp
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or venv\Scripts\activate  # Windows
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Prepare models**
```bash
# Create models directory
mkdir models

# Copy your trained models to the models/ directory:
# - ensemble_models.pkl
# - feature_engine.pkl
# - model_metadata.pkl
# - siglip-so400m-patch14-384/ (optional, for embeddings)
```

5. **Run the application**
```bash
python app.py
```

6. **Open browser**
```
Navigate to: http://localhost:5000
```

---

## 🐳 Docker Deployment

### Using Docker Compose (Recommended)

```bash
# Build and start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Using Docker only

```bash
# Build image
docker build -t biomass-webapp .

# Run container
docker run -d \
  -p 5000:5000 \
  -v $(pwd)/models:/app/models:ro \
  -v $(pwd)/uploads:/app/uploads \
  biomass-webapp
```

---

## 📖 Usage

### Web Interface

1. **Single Image Prediction:**
   - Click "Choose Image"
   - Select a pasture image (JPG, PNG)
   - Select models for ensemble
   - Click "Analyze Biomass"
   - View predictions and charts

2. **Batch Processing:**
   - Switch to "Batch Upload" tab
   - Select multiple images
   - Click "Analyze All Images"
   - Export results to CSV

### API Usage

#### Python Client

```python
import requests

url = "http://localhost:5000/predict"
files = {'image': open('pasture.jpg', 'rb')}
data = {'models': 'lightgbm,catboost,random_forest'}

response = requests.post(url, files=files, data=data)
result = response.json()

if result['success']:
    print("Predictions:")
    for target, value in result['predictions'].items():
        print(f"  {target}: {value:.2f}g")
```

#### cURL

```bash
curl -X POST http://localhost:5000/predict \
  -F "image=@pasture.jpg" \
  -F "models=lightgbm,catboost"
```

---

## 📊 Prediction Targets

The model predicts five biomass components:

| Target | Description | Unit |
|--------|-------------|------|
| Dry_Green_g | Weight of dried green biomass | grams |
| Dry_Clover_g | Weight of dried clover biomass | grams |
| Dry_Dead_g | Weight of dried dead biomass | grams |
| GDM_g | Green Digestible Matter (Green + Clover) | grams |
| Dry_Total_g | Total dry matter (GDM + Dead) | grams |

---

## 🔌 API Endpoints

### `GET /`
Returns the web interface (HTML)

### `POST /predict`
Predict biomass from a single image

**Parameters:**
- `image` (file, required) - Image file
- `models` (string, optional) - Comma-separated model names

**Response:**
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
  "timestamp": "20240126_123045"
}
```

### `POST /batch-predict`
Predict biomass from multiple images

**Parameters:**
- `images` (files, required) - Multiple image files
- `models` (string, optional) - Comma-separated model names

### `POST /export-results`
Export predictions to CSV

### `GET /model-info`
Get information about loaded models

### `GET /health`
Health check endpoint

---

## 🛠️ Configuration

### Environment Variables

Create a `.env` file:

```bash
# Flask Configuration
FLASK_APP=app.py
FLASK_ENV=production
SECRET_KEY=your-secret-key-here

# Model Configuration
MODEL_PATH=models/
DEVICE=cuda  # or 'cpu'

# Upload Configuration
MAX_FILE_SIZE=16777216  # 16MB
ALLOWED_EXTENSIONS=png,jpg,jpeg
```

### Model Configuration

Available ensemble models:
- `lightgbm` - LightGBM (fast, accurate)
- `catboost` - CatBoost (robust)
- `random_forest` - Random Forest (stable)
- `extra_trees` - Extra Trees (fast)
- `mlp` - Neural Network (deep learning)
- `histgbm` - Histogram Gradient Boosting
- `gradient_boosting` - Gradient Boosting
- `tabnet` - TabNet (optional, requires installation)

---

## 📁 Project Structure

```
biomass-webapp/
├── app.py                  # Main Flask application
├── requirements.txt        # Python dependencies
├── Dockerfile             # Docker configuration
├── docker-compose.yml     # Docker Compose configuration
├── nginx.conf             # Nginx configuration
├── .gitignore             # Git ignore rules
├── README.md              # This file
│
├── models/                # Trained model files
│   ├── ensemble_models.pkl
│   ├── feature_engine.pkl
│   ├── model_metadata.pkl
│   └── siglip-so400m-patch14-384/
│
├── templates/             # HTML templates
│   └── index.html
│
├── static/                # Static files
│   ├── css/
│   ├── js/
│   └── results/
│
└── uploads/               # User uploaded images
```

---

## 🔧 Development

### Running Tests

```bash
python -m pytest tests/
```

### Code Formatting

```bash
black app.py
flake8 app.py
```

### Performance Benchmarking

```bash
python scripts/benchmark.py
```

---

## 🚦 Production Deployment

### Traditional Server

See [COMPLETE_WEB_APP_GUIDE.md](COMPLETE_WEB_APP_GUIDE.md) for detailed deployment instructions covering:
- Linux server setup
- Gunicorn configuration
- Nginx setup
- SSL certificates
- Systemd service
- Monitoring

### Cloud Platforms

- **AWS EC2**: Traditional deployment
- **AWS Elastic Beanstalk**: Managed deployment
- **Google Cloud Run**: Serverless containers
- **Heroku**: Simple deployment
- **DigitalOcean**: App platform

---

## 📈 Performance

- **Average prediction time**: 1-3 seconds per image
- **Batch processing**: 10-20 images per minute
- **Memory usage**: 2-4GB with all models loaded
- **GPU acceleration**: 3-5x faster with CUDA

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgments

- CSIRO for the biomass dataset
- Hugging Face for transformer models
- Flask team for the excellent web framework
- Contributors to scikit-learn, LightGBM, CatBoost, and other libraries

---

## 📧 Contact

For questions or support, please open an issue on GitHub or contact [your-email@example.com]

---

## 🔗 Links

- [Documentation](COMPLETE_WEB_APP_GUIDE.md)
- [Model Training Guide](MODEL_EXPORT_GUIDE.md)
- [Deployment Guide](COMPLETE_WEB_APP_GUIDE.md)

---

**Made with ❤️ for sustainable agriculture**
