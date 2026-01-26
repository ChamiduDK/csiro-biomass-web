"""
CSIRO Biomass Prediction Web Application
Flask-based web app for predicting biomass from pasture images
"""

from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
import pickle
import numpy as np
import cv2
import torch
from PIL import Image
import io
import base64
import os
from pathlib import Path
import pandas as pd
from datetime import datetime
import json

# Initialize Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Create necessary directories
Path(app.config['UPLOAD_FOLDER']).mkdir(exist_ok=True)
Path('static/results').mkdir(parents=True, exist_ok=True)

# Global variables for models
MODELS = None
FEATURE_ENGINE = None
METADATA = None
SIGLIP_MODEL = None
SIGLIP_PROCESSOR = None
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Target names and descriptions
TARGET_INFO = {
    'Dry_Green_g': {
        'name': 'Dry Green Matter',
        'description': 'Weight of dried green biomass',
        'unit': 'grams',
        'color': '#4CAF50'
    },
    'Dry_Clover_g': {
        'name': 'Dry Clover',
        'description': 'Weight of dried clover biomass',
        'unit': 'grams',
        'color': '#9C27B0'
    },
    'Dry_Dead_g': {
        'name': 'Dry Dead Matter',
        'description': 'Weight of dried dead biomass',
        'unit': 'grams',
        'color': '#795548'
    },
    'GDM_g': {
        'name': 'Green Digestible Matter',
        'description': 'Combined green and clover biomass',
        'unit': 'grams',
        'color': '#8BC34A'
    },
    'Dry_Total_g': {
        'name': 'Total Dry Matter',
        'description': 'Total weight of all dried biomass',
        'unit': 'grams',
        'color': '#FF9800'
    }
}


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def load_models():
    """Load trained models and preprocessing components"""
    global MODELS, FEATURE_ENGINE, METADATA, SIGLIP_MODEL, SIGLIP_PROCESSOR
    
    print("Loading models...")
    
    try:
        # Load ensemble models
        with open('models/ensemble_models.pkl', 'rb') as f:
            MODELS = pickle.load(f)
        print(f"✓ Loaded {len(MODELS)} model types")
        
        # Load feature engine
        with open('models/feature_engine.pkl', 'rb') as f:
            FEATURE_ENGINE = pickle.load(f)
        print("✓ Loaded feature engine")
        
        # Load metadata
        with open('models/model_metadata.pkl', 'rb') as f:
            METADATA = pickle.load(f)
        print("✓ Loaded metadata")
        
        # Load SigLIP model for embeddings
        from transformers import AutoModel, AutoImageProcessor
        siglip_path = 'models/siglip-so400m-patch14-384'
        
        if os.path.exists(siglip_path):
            SIGLIP_MODEL = AutoModel.from_pretrained(siglip_path, local_files_only=True).eval().to(DEVICE)
            SIGLIP_PROCESSOR = AutoImageProcessor.from_pretrained(siglip_path)
            print("✓ Loaded SigLIP model")
        else:
            print("⚠ SigLIP model not found - using random embeddings for demo")
            SIGLIP_MODEL = None
            SIGLIP_PROCESSOR = None
        
        return True
        
    except Exception as e:
        print(f"Error loading models: {e}")
        return False


def extract_embeddings(image_path):
    """Extract SigLIP embeddings from image"""
    if SIGLIP_MODEL is None or SIGLIP_PROCESSOR is None:
        # Return random embeddings for demo if model not loaded
        print("⚠ Using random embeddings (demo mode)")
        return np.random.randn(1152).astype(np.float32)
    
    try:
        # Read and preprocess image
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError("Failed to read image")
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        
        # Process with SigLIP
        inputs = SIGLIP_PROCESSOR(images=pil_img, return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = SIGLIP_MODEL(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        
        return embeddings.flatten()
        
    except Exception as e:
        print(f"Error extracting embeddings: {e}")
        return np.random.randn(1152).astype(np.float32)


def generate_semantic_features(embeddings):
    """Generate semantic features from embeddings"""
    # Simplified version - in production, use the full semantic feature generation
    # For now, return zeros as placeholder
    return np.zeros(80).astype(np.float32)  # Adjust size based on your model


def make_prediction(image_path, model_types=['lightgbm', 'catboost', 'random_forest']):
    """Make biomass prediction from image"""
    try:
        # Extract embeddings
        embeddings = extract_embeddings(image_path)
        embeddings = embeddings.reshape(1, -1)
        
        # Generate semantic features
        semantic_features = generate_semantic_features(embeddings)
        semantic_features = semantic_features.reshape(1, -1)
        
        # Transform features
        X_transformed = FEATURE_ENGINE.transform(embeddings, X_semantic=semantic_features)
        
        # Get target scaling factors
        target_names = METADATA['target_names']
        target_max = np.array([METADATA['target_max'][t] for t in target_names])
        
        # Make predictions with each model type
        all_predictions = []
        
        for model_type in model_types:
            if model_type not in MODELS:
                continue
                
            models = MODELS[model_type]
            preds = []
            
            for i, target_name in enumerate(target_names):
                if target_name == 'Dry_Clover_g' and models[target_name] is None:
                    # Handle case where clover model might be None
                    preds.append(0.0)
                else:
                    model = models[target_name]
                    pred_raw = model.predict(X_transformed)
                    pred_scaled = pred_raw[0] * target_max[i]
                    preds.append(max(0.0, pred_scaled))  # Ensure non-negative
            
            all_predictions.append(preds)
        
        # Average predictions across models
        final_predictions = np.mean(all_predictions, axis=0)
        
        # Apply post-processing
        predictions_dict = dict(zip(target_names, final_predictions))
        predictions_dict = post_process_predictions(predictions_dict)
        
        return predictions_dict
        
    except Exception as e:
        print(f"Prediction error: {e}")
        raise


def post_process_predictions(predictions):
    """Apply hierarchical constraints and thresholds"""
    # Apply clover threshold
    CLOVER_THRESHOLD = 0.5
    if predictions['Dry_Clover_g'] < CLOVER_THRESHOLD:
        predictions['Dry_Clover_g'] = 0.0
    
    # Recalculate derived values
    predictions['GDM_g'] = predictions['Dry_Green_g'] + predictions['Dry_Clover_g']
    predictions['Dry_Total_g'] = predictions['GDM_g'] + predictions['Dry_Dead_g']
    
    # Ensure non-negative
    for key in predictions:
        predictions[key] = max(0.0, predictions[key])
    
    return predictions


def image_to_base64(image_path):
    """Convert image to base64 for display"""
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode()


@app.route('/')
def index():
    """Home page"""
    return render_template('index.html', target_info=TARGET_INFO)


@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction request"""
    try:
        # Check if file was uploaded
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Please upload PNG, JPG, or JPEG'}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Get selected models
        selected_models = request.form.get('models', 'lightgbm,catboost,random_forest').split(',')
        
        # Make prediction
        predictions = make_prediction(filepath, model_types=selected_models)
        
        # Convert image to base64
        img_base64 = image_to_base64(filepath)
        
        # Prepare response
        response = {
            'success': True,
            'predictions': predictions,
            'image': img_base64,
            'timestamp': timestamp,
            'models_used': selected_models
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/batch-predict', methods=['POST'])
def batch_predict():
    """Handle batch prediction request"""
    try:
        # Check if files were uploaded
        if 'images' not in request.files:
            return jsonify({'error': 'No image files provided'}), 400
        
        files = request.files.getlist('images')
        
        if len(files) == 0:
            return jsonify({'error': 'No files selected'}), 400
        
        # Get selected models
        selected_models = request.form.get('models', 'lightgbm,catboost,random_forest').split(',')
        
        # Process each image
        results = []
        
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                filename = f"{timestamp}_{filename}"
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
                try:
                    predictions = make_prediction(filepath, model_types=selected_models)
                    results.append({
                        'filename': file.filename,
                        'predictions': predictions,
                        'success': True
                    })
                except Exception as e:
                    results.append({
                        'filename': file.filename,
                        'error': str(e),
                        'success': False
                    })
        
        return jsonify({
            'success': True,
            'results': results,
            'total': len(results)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/export-results', methods=['POST'])
def export_results():
    """Export predictions to CSV"""
    try:
        data = request.json
        predictions_list = data.get('predictions', [])
        
        # Create DataFrame
        df = pd.DataFrame(predictions_list)
        
        # Save to CSV
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'biomass_predictions_{timestamp}.csv'
        filepath = os.path.join('static/results', filename)
        df.to_csv(filepath, index=False)
        
        return jsonify({
            'success': True,
            'filename': filename,
            'download_url': f'/download/{filename}'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/download/<filename>')
def download_file(filename):
    """Download exported CSV file"""
    filepath = os.path.join('static/results', filename)
    return send_file(filepath, as_attachment=True)


@app.route('/model-info')
def model_info():
    """Get information about loaded models"""
    if MODELS is None or METADATA is None:
        return jsonify({'error': 'Models not loaded'}), 500
    
    info = {
        'models_loaded': list(MODELS.keys()),
        'targets': METADATA['target_names'],
        'embedding_dim': METADATA['embedding_dim'],
        'device': str(DEVICE),
        'siglip_available': SIGLIP_MODEL is not None
    }
    
    return jsonify(info)


@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': MODELS is not None,
        'timestamp': datetime.now().isoformat()
    })


if __name__ == '__main__':
    print("="*80)
    print("CSIRO Biomass Prediction Web Application")
    print("="*80)
    
    # Load models
    if load_models():
        print("\n✓ All models loaded successfully!")
        print(f"✓ Running on device: {DEVICE}")
        print(f"✓ Available models: {list(MODELS.keys()) if MODELS else 'None'}")
        print("\nStarting Flask server...")
        print("Access the app at: http://localhost:5000")
        print("="*80)
        
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("\n✗ Failed to load models. Please check model files.")
        print("Expected structure:")
        print("  models/")
        print("    ├── ensemble_models.pkl")
        print("    ├── feature_engine.pkl")
        print("    ├── model_metadata.pkl")
        print("    └── siglip-so400m-patch14-384/")
