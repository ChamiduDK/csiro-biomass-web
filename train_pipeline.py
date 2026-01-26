import os
import sys
import gc
import random
import warnings
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
from PIL import Image
from tqdm.auto import tqdm
from pathlib import Path
from copy import deepcopy
from dataclasses import dataclass
import pickle
import argparse
import shutil

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from transformers import AutoModel, AutoImageProcessor, AutoTokenizer

# Import our Feature Engine
from feature_engine import SupervisedEmbeddingEngine

warnings.filterwarnings('ignore')
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- Configuration ---
@dataclass
class Config:
    DATA_PATH: Path = Path("data") # Updated to local data folder
    SPLIT_PATH: Path = Path("data/csiro_data_split.csv")
    SIGLIP_PATH: str = "models/siglip-so400m-patch14-384"
    CACHE_DIR: Path = Path("models/cache")
    MODELS_DIR: Path = Path("models")
    
    SEED: int = 42
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    PATCH_SIZE: int = 520
    OVERLAP: int = 16
    
    TARGET_NAMES = ['Dry_Clover_g', 'Dry_Dead_g', 'Dry_Green_g', 'Dry_Total_g', 'GDM_g']
    TARGET_MAX = {
        "Dry_Clover_g": 71.7865,
        "Dry_Dead_g": 83.8407,
        "Dry_Green_g": 157.9836,
        "Dry_Total_g": 185.70,
        "GDM_g": 157.9836,
    }

cfg = Config()
cfg.CACHE_DIR.mkdir(parents=True, exist_ok=True)
cfg.MODELS_DIR.mkdir(parents=True, exist_ok=True)

# --- Utilities ---
def seeding(SEED):
    np.random.seed(SEED)
    random.seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def split_image(image, patch_size=520, overlap=16):
    h, w, c = image.shape
    stride = patch_size - overlap
    patches = []
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            y2 = min(y + patch_size, h)
            x2 = min(x + patch_size, w)
            y1 = max(0, y2 - patch_size)
            x1 = max(0, x2 - patch_size)
            patch = image[y1:y2, x1:x2, :]
            patches.append(patch)
    return patches

def compute_embeddings(model_path, df, cache_file, desc="Computing"):
    # Skip cache loading to enforce re-computation for new model size
    # if cache_file.exists(): ...
    
    print(f"{desc} ({len(df)} images) - calculating embeddings...")
    
    # Determine Model Path (Local vs Cloud)
    real_model_path = model_path
    if not os.path.exists(model_path):
        print(f"Local model not found at {model_path}, downloading from Hugging Face Hub...")
        real_model_path = "google/siglip-base-patch16-224"
    
    try:
        model = AutoModel.from_pretrained(real_model_path).eval().to(cfg.DEVICE)
        processor = AutoImageProcessor.from_pretrained(real_model_path)
    except Exception as e:
        print(f"CRITICAL ERROR loading SigLIP model from {real_model_path}: {e}")
        print("Cannot proceed without a valid embedding model.")
        sys.exit(1)

    EMBEDDINGS = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=desc):
        try:
            # Handle image path (Fix Kaggle paths to local)
            filename = os.path.basename(row['image_path'])
            img_path = os.path.join(cfg.DATA_PATH, "train", filename)
            
            if not os.path.exists(img_path):
                 # Try test folder if not in train
                 test_path = os.path.join(cfg.DATA_PATH, "test", filename)
                 if os.path.exists(test_path):
                     img_path = test_path
            
            img = cv2.imread(img_path)
            if img is None: 
                 print(f"Warning: Image not found {img_path}")
                 EMBEDDINGS.append(np.zeros(1152))
                 continue
                 
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            patches = split_image(img, patch_size=cfg.PATCH_SIZE, overlap=cfg.OVERLAP)
            images = [Image.fromarray(p) for p in patches]
            
            inputs = processor(images=images, return_tensors="pt").to(cfg.DEVICE)
            with torch.no_grad():
                outputs = model.get_image_features(**inputs)
            
            # Handle different output types
            if isinstance(outputs, torch.Tensor):
                features = outputs
            elif hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                features = outputs.pooler_output
            elif hasattr(outputs, 'last_hidden_state'):
                features = outputs.last_hidden_state.mean(dim=1)
            else:
                features = outputs[0] # Tuple fallback
                
            avg_embed = features.mean(dim=0).cpu().numpy()
            EMBEDDINGS.append(avg_embed)
        except Exception as e:
            print(f"Error processing image {row.get('image_path', '?')}: {e}")
            EMBEDDINGS.append(np.zeros(1152))
    
    embeddings = np.stack(EMBEDDINGS)
    print(f"Caching to {cache_file.name}")
    with open(cache_file, 'wb') as f:
        pickle.dump(embeddings, f)
    
    del model, processor
    torch.cuda.empty_cache()
    gc.collect()
    return embeddings

def generate_semantic_features(image_embeddings_np, model_path):
    print("Generating Semantic Features...")
    try:
        model = AutoModel.from_pretrained(model_path, local_files_only=True).to(cfg.DEVICE)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    except Exception as e:
        print(f"Error loading model for semantic features: {e}")
        return np.zeros((len(image_embeddings_np), 11))

    concept_groups = {
        "bare": ["bare soil", "dirt ground", "sparse vegetation", "exposed earth"],
        "sparse": ["low density pasture", "thin grass", "short clipped grass"],
        "medium": ["average pasture cover", "medium height grass", "grazed pasture"],
        "dense": ["dense tall pasture", "thick grassy volume", "high biomass", "overgrown vegetation"],
        "green": ["lush green vibrant pasture", "photosynthesizing leaves", "fresh growth"],
        "dead": ["dry brown dead grass", "yellow straw", "senesced material", "standing hay"],
        "clover": ["white clover", "trifolium repens", "broadleaf legume", "clover flowers"],
        "grass": ["ryegrass", "blade-like leaves", "fescue", "grassy sward"]
    }
    
    concept_vectors = {}
    with torch.no_grad():
        for name, prompts in concept_groups.items():
            inputs = tokenizer(prompts, padding="max_length", return_tensors="pt").to(cfg.DEVICE)
            emb = model.get_text_features(**inputs)
            emb = emb / emb.norm(p=2, dim=-1, keepdim=True)
            concept_vectors[name] = emb.mean(dim=0, keepdim=True)
    
    img_tensor = torch.tensor(image_embeddings_np, dtype=torch.float32).to(cfg.DEVICE)
    img_tensor = img_tensor / img_tensor.norm(p=2, dim=-1, keepdim=True)
    
    scores = {}
    for name, vec in concept_vectors.items():
        scores[name] = torch.matmul(img_tensor, vec.T).cpu().numpy().flatten()
    
    df_scores = pd.DataFrame(scores)
    
    # Calculate Ratios (handling division by zero)
    df_scores['ratio_greenness'] = df_scores['green'] / (df_scores['green'] + df_scores['dead'] + 1e-6)
    df_scores['ratio_clover'] = df_scores['clover'] / (df_scores['clover'] + df_scores['grass'] + 1e-6)
    df_scores['ratio_cover'] = (df_scores['dense'] + df_scores['medium']) / (df_scores['bare'] + df_scores['sparse'] + 1e-6)
    
    torch.cuda.empty_cache()
    return df_scores.values

def train_final_models(model_cls, model_params, X_train, y_train, sem_train, feature_engine):
    target_max_arr = np.array([cfg.TARGET_MAX[t] for t in cfg.TARGET_NAMES], dtype=float)
    
    # Fit feature engine
    final_engine = deepcopy(feature_engine)
    final_engine.fit(X_train, y=y_train / target_max_arr, X_semantic=sem_train)
    
    X_transformed = final_engine.transform(X_train, X_semantic=sem_train)
    
    trained_models = {}
    for k, target_name in enumerate(cfg.TARGET_NAMES):
        print(f"  Training {target_name}...")
        model = model_cls(**model_params)
        model.fit(X_transformed, y_train[:, k] / target_max_arr[k])
        trained_models[target_name] = model
    
    return trained_models, final_engine

def main():
    seeding(cfg.SEED)
    print("="*80)
    print("CSIRO Biomass Prediction - Ultimate Pipeline Training")
    print("="*80)
    
    if not cfg.SPLIT_PATH.exists():
        print(f"Error: Data split file not found at {cfg.SPLIT_PATH}")
        print("Please place 'csiro_data_split.csv' in the data directory.")
        return

    # 1. Load Data
    print("Loading Data...")
    train_df = pd.read_csv(cfg.SPLIT_PATH)
    # Ensure image paths are absolute or correct relative to execution
    # Assuming 'image_path' in csv is relative to 'train/' or similar
    
    # Filter DataFrame to only existing images
    print("Verifying image existence...")
    valid_indices = []
    for idx, row in train_df.iterrows():
        filename = os.path.basename(row['image_path'])
        img_path = os.path.join(cfg.DATA_PATH, "train", filename)
        if os.path.exists(img_path):
            valid_indices.append(idx)
        else:
             # Check test
             test_path = os.path.join(cfg.DATA_PATH, "test", filename)
             if os.path.exists(test_path):
                 valid_indices.append(idx)
    
    print(f"Found {len(valid_indices)} images out of {len(train_df)} in CSV.")
    train_df = train_df.loc[valid_indices].reset_index(drop=True)
    
    if len(train_df) == 0:
        print("ERROR: No images found! Cannot train.")
        return

    # 2. Extract Embeddings
    train_cache = cfg.CACHE_DIR / "train_siglip.pkl"
    train_embeddings = compute_embeddings(cfg.SIGLIP_PATH, train_df, train_cache, "Train")
    
    # Filter out invalid embeddings (all zeros)
    print("Filtering invalid embeddings...")
    valid_mask = ~np.all(train_embeddings == 0, axis=1)
    
    if not np.any(valid_mask):
        print("CRITICAL ERROR: ALL embeddings are zeros. Check model loading and image paths.")
        sys.exit(1)
        
    n_dropped = np.sum(~valid_mask)
    if n_dropped > 0:
        print(f"Dropping {n_dropped} invalid/zero embeddings.")
        train_embeddings = train_embeddings[valid_mask]
        train_df = train_df.iloc[valid_mask].reset_index(drop=True)
    
    print(f"Final training set size: {len(train_df)}")
    
    emb_cols = [f"emb{i}" for i in range(train_embeddings.shape[1])]
    
    # 3. Semantic Features
    train_semantic = generate_semantic_features(train_embeddings, cfg.SIGLIP_PATH)
    
    # 4. Feature Engine Init
    feat_engine = SupervisedEmbeddingEngine(n_pca=0.80, n_pls=8, n_gmm=6)
    
    # 5. Train Models (Ultimate Ensemble)
    X_train = train_embeddings
    y_train = train_df[cfg.TARGET_NAMES].values.astype(np.float32)
    
    # Use LightGBM for the main engine fitting (can be any, but we save one)
    params_lgbm = {
        'n_estimators': 807, 'learning_rate': 0.014, 'num_leaves': 48, 
        'min_child_samples': 19, 'subsample': 0.745, 'colsample_bytree': 0.745, 
        'reg_alpha': 0.21, 'reg_lambda': 3.78, 'verbose': -1, 'random_state': 42
    }
    
    print("\nTraining Final LightGBM Models...")
    lgbm_models, final_engine = train_final_models(
        LGBMRegressor, params_lgbm, X_train, y_train, train_semantic, feat_engine
    )
    
    params_cat = {
        'iterations': 1900, 'learning_rate': 0.045, 'depth': 4, 
        'l2_leaf_reg': 0.56, 'random_strength': 0.045, 'bagging_temperature': 0.98, 
        'verbose': 0, 'random_state': 42, 'allow_writing_files': False
    }
    
    print("\nTraining Final CatBoost Models...")
    cat_models, _ = train_final_models(
        CatBoostRegressor, params_cat, X_train, y_train, train_semantic, feat_engine
    )

    params_rf = {
        'n_estimators': 500, 'max_depth': 20, 'min_samples_split': 5, 
        'min_samples_leaf': 2, 'max_features': 'sqrt', 'n_jobs': -1, 'random_state': 42
    }
    
    print("\nTraining Final Random Forest Models...")
    rf_models, _ = train_final_models(
        RandomForestRegressor, params_rf, X_train, y_train, train_semantic, feat_engine
    )
    
    # 6. Save Models
    print("\nSaving Models...")
    
    # Save Feature Engine
    with open(cfg.MODELS_DIR / 'feature_engine.pkl', 'wb') as f:
        pickle.dump(final_engine, f)
        
    # Save Ensemble Dict
    ensemble_models = {
        'lightgbm': lgbm_models,
        'catboost': cat_models,
        'random_forest': rf_models
    }
    
    with open(cfg.MODELS_DIR / 'ensemble_models.pkl', 'wb') as f:
        pickle.dump(ensemble_models, f)
        
    # Save Metadata
    metadata = {
        'target_names': cfg.TARGET_NAMES,
        'target_max': cfg.TARGET_MAX,
        'embedding_dim': train_embeddings.shape[1],
        'semantic_features': train_semantic.shape[1]
    }
    
    with open(cfg.MODELS_DIR / 'model_metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)
        
    print("\nTraining Complete! Models saved to 'models/' directory.")

if __name__ == "__main__":
    main()
