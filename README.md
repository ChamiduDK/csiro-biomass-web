# CSIRO Biomass Prediction Web Application

AI-powered web application for predicting pasture biomass from images using ensemble machine learning models and SigLIP embeddings.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-3.0-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## Features

- High-resolution prediction using SigLIP vision features.
- Single image prediction for rapid sample analysis.
- Batch processing support with CSV export.
- Ensemble learning with LightGBM, CatBoost, and Random Forest.
- Responsive web UI with interactive charts.

## Quick Start

### Prerequisites

- Python 3.10 (recommended)
- 8 GB RAM or higher
- NVIDIA GPU with CUDA (optional)

### Installation and Run

1. Move to the project directory:

```bash
cd csiro-biomass-web
```

2. Run the Windows starter script:

```bash
START_APP.bat
```

The script activates the virtual environment, installs dependencies, verifies model assets, and starts the Flask server.

3. Open the application:

`http://localhost:5000`

## Project Structure

```text
csiro-biomass-web/
|-- app.py                # Main Flask application
|-- train_pipeline.py     # Model training pipeline
|-- feature_engine.py     # Feature engineering utilities
|-- requirements.txt      # Python dependencies
|-- START_APP.bat         # Windows startup script
|-- templates/            # HTML templates
|-- static/               # CSS, JS, and generated assets
|-- uploads/              # Uploaded image storage
`-- models/               # Trained model files and metadata
```

## Scientific Methodology

1. Feature extraction with SigLIP embeddings.
2. Feature engineering using PCA, PLS, and GMM transformations.
3. Ensemble inference from LightGBM, CatBoost, and Random Forest.
4. Post-processing constraints for biologically realistic biomass outputs.

## Author and Contact

Developed and maintained by Chamidu Dhilshan Kodithuwakkuarachchi.

- Email: KD-BSCSD-20-53@student.icbtcampus.edu.lk
- Kaggle: https://www.kaggle.com/chamidudhilshan
- LinkedIn: https://www.linkedin.com/in/chamidudhilshan
- GitHub: https://github.com/ChamiduDK

## Copyright and License

Copyright (c) 2026 Chamidu Dhilshan Kodithuwakkuarachchi.
All rights reserved.

Source code is distributed under the MIT License. See `LICENSE` for details.

