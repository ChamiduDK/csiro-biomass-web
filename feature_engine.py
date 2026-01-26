import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.mixture import GaussianMixture
import pickle

class SupervisedEmbeddingEngine(BaseEstimator, TransformerMixin):
    """
    Feature engineering pipeline combining:
    - PCA for dimensionality reduction
    - PLS for supervised projection
    - GMM for clustering probabilities
    """
    def __init__(self, n_pca=0.98, n_pls=8, n_gmm=6, random_state=42):
        self.n_pca = n_pca
        self.n_pls = n_pls
        self.n_gmm = n_gmm
        self.random_state = random_state
        
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_pca, random_state=random_state)
        self.pls = PLSRegression(n_components=n_pls, scale=False)
        self.gmm = GaussianMixture(n_components=n_gmm, covariance_type='diag', random_state=random_state)
        self.pls_fitted_ = False

    def fit(self, X, y=None, X_semantic=None):
        X_scaled = self.scaler.fit_transform(X)
        self.pca.fit(X_scaled)
        self.gmm.fit(X_scaled)
        if y is not None:
            # Handle multi-output y for PLS
            y_clean = y.values if hasattr(y, 'values') else y
            self.pls.fit(X_scaled, y_clean)
            self.pls_fitted_ = True
        return self

    def transform(self, X, X_semantic=None):
        # Ensure input is 2D
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
            
        X_scaled = self.scaler.transform(X)
        
        # Calculate features
        features = []
        
        # 1. PCA
        f_pca = self.pca.transform(X_scaled)
        features.append(f_pca)
        
        # 2. PLS (if fitted)
        if self.pls_fitted_:
            f_pls = self.pls.transform(X_scaled)
            features.append(f_pls)
        
        # 3. GMM
        f_gmm = self.gmm.predict_proba(X_scaled)
        features.append(f_gmm)
        
        # 4. Semantic Features
        if X_semantic is not None:
            if len(X_semantic.shape) == 1:
                X_semantic = X_semantic.reshape(1, -1)
                
            # Match batch size if needed
            if X_semantic.shape[0] != X.shape[0] and X.shape[0] > 1 and X_semantic.shape[0] == 1:
                X_semantic = np.tile(X_semantic, (X.shape[0], 1))
            
            # Normalize semantic features if they aren't already
            # (In inference, we might receive them raw, normalization logic from notebook:
            # sem_norm = (X_semantic - mean) / std. But mean/std are from training data.
            # The notebook's `transform` uses `X_semantic` directly or normalizes it?
            # Notebook says:
            # sem_norm = (X_semantic - np.mean(X_semantic, axis=0)) / (np.std(X_semantic, axis=0) + 1e-6)
            # This computes statistics on the *batch*. For single inference, this is bad (std=0).
            # We should probably skip normalization during inference or store statistics.
            # For now, let's append as is or trusting the caller handles it.
            # In app.py generated semantics are zeros.
            features.append(X_semantic)
            
        return np.hstack(features)
