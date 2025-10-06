"""
Reef health classifier using scikit-learn on SurfPerch embeddings
Multi-output classification for reef health and anthrophony detection
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
import joblib
import logging
from typing import Tuple, Dict, Any, Optional, List
from pathlib import Path
from src.utils import config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ReefClassifier:
    """
    Multi-output classifier for reef health assessment
    """
    
    def __init__(self, model_type: str = 'random_forest'):
        """
        Initialize reef classifier
        
        Args:
            model_type: Type of classifier ('random_forest', 'logistic_regression', 'svm')
        """
        self.model_type = model_type
        self.health_classifier = None
        self.anthro_classifier = None
        self.scaler = StandardScaler()
        self.is_trained = False
        
        # Initialize classifiers based on type
        self._initialize_classifiers()
    
    def _initialize_classifiers(self):
        """Initialize the classifiers based on model type"""
        if self.model_type == 'random_forest':
            self.health_classifier = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                max_depth=10,
                min_samples_split=5
            )
            self.anthro_classifier = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                max_depth=10,
                min_samples_split=5
            )
        elif self.model_type == 'logistic_regression':
            self.health_classifier = LogisticRegression(
                random_state=42,
                max_iter=1000
            )
            self.anthro_classifier = LogisticRegression(
                random_state=42,
                max_iter=1000
            )
        elif self.model_type == 'svm':
            self.health_classifier = SVC(
                random_state=42,
                probability=True
            )
            self.anthro_classifier = SVC(
                random_state=42,
                probability=True
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def train(self, embeddings: np.ndarray, health_labels: np.ndarray, 
              anthro_labels: np.ndarray, test_size: float = 0.2) -> Dict[str, Any]:
        """
        Train the multi-output classifier
        
        Args:
            embeddings: SurfPerch embeddings
            health_labels: Reef health labels (0=Degraded, 1=Healthy)
            anthro_labels: Anthrophony labels (0=Low, 1=High)
            test_size: Fraction of data to use for testing
            
        Returns:
            Dictionary with training results
        """
        try:
            logger.info(f"Training {self.model_type} classifier on {len(embeddings)} samples")
            
            # Split data
            X_train, X_test, y_health_train, y_health_test, y_anthro_train, y_anthro_test = train_test_split(
                embeddings, health_labels, anthro_labels, test_size=test_size, random_state=42
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train health classifier
            logger.info("Training reef health classifier...")
            self.health_classifier.fit(X_train_scaled, y_health_train)
            
            # Train anthrophony classifier
            logger.info("Training anthrophony classifier...")
            self.anthro_classifier.fit(X_train_scaled, y_anthro_train)
            
            # Evaluate models
            health_score = self.health_classifier.score(X_test_scaled, y_health_test)
            anthro_score = self.anthro_classifier.score(X_test_scaled, y_anthro_test)
            
            # Cross-validation scores
            health_cv_scores = cross_val_score(self.health_classifier, X_train_scaled, y_health_train, cv=5)
            anthro_cv_scores = cross_val_score(self.anthro_classifier, X_train_scaled, y_anthro_train, cv=5)
            
            self.is_trained = True
            
            results = {
                'health_accuracy': health_score,
                'anthro_accuracy': anthro_score,
                'health_cv_mean': health_cv_scores.mean(),
                'health_cv_std': health_cv_scores.std(),
                'anthro_cv_mean': anthro_cv_scores.mean(),
                'anthro_cv_std': anthro_cv_scores.std(),
                'test_size': test_size,
                'n_train_samples': len(X_train),
                'n_test_samples': len(X_test)
            }
            
            logger.info(f"Training completed. Health accuracy: {health_score:.3f}, Anthro accuracy: {anthro_score:.3f}")
            return results
            
        except Exception as e:
            logger.error(f"Error training classifier: {e}")
            raise
    
    def predict(self, embeddings: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict reef health and anthrophony from embeddings
        
        Args:
            embeddings: SurfPerch embeddings
            
        Returns:
            Tuple of (health_predictions, health_probabilities, anthro_predictions, anthro_probabilities)
        """
        if not self.is_trained:
            raise RuntimeError("Classifier not trained. Call train() first.")
        
        try:
            # Scale embeddings
            embeddings_scaled = self.scaler.transform(embeddings)
            
            # Predict health
            health_predictions = self.health_classifier.predict(embeddings_scaled)
            health_probabilities = self.health_classifier.predict_proba(embeddings_scaled)
            
            # Predict anthrophony
            anthro_predictions = self.anthro_classifier.predict(embeddings_scaled)
            anthro_probabilities = self.anthro_classifier.predict_proba(embeddings_scaled)
            
            return health_predictions, health_probabilities, anthro_predictions, anthro_probabilities
            
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            raise
    
    def predict_single(self, embedding: np.ndarray) -> Dict[str, Any]:
        """
        Predict for a single embedding
        
        Args:
            embedding: Single SurfPerch embedding
            
        Returns:
            Dictionary with prediction results
        """
        # Reshape for single prediction
        embedding = np.reshape(embedding, (1, -1))
        
        health_pred, health_prob, anthro_pred, anthro_prob = self.predict(embedding)
        
        # Convert predictions to labels
        health_label = "Healthy" if health_pred[0] == 1 else "Degraded"
        anthro_label = "High" if anthro_pred[0] == 1 else "Low"
        
        # Get confidence scores
        health_confidence = max(health_prob[0])
        anthro_confidence = max(anthro_prob[0])
        
        return {
            'health_status': health_label,
            'health_confidence': health_confidence,
            'anthro_status': anthro_label,
            'anthro_confidence': anthro_confidence,
            'health_probabilities': {
                'degraded': health_prob[0][0],
                'healthy': health_prob[0][1]
            },
            'anthro_probabilities': {
                'low': anthro_prob[0][0],
                'high': anthro_prob[0][1]
            }
        }
    
    def save_model(self, save_path: str):
        """
        Save trained model to disk
        
        Args:
            save_path: Path to save the model
        """
        if not self.is_trained:
            raise RuntimeError("No trained model to save")
        
        try:
            save_path = Path(save_path)
            save_path.mkdir(parents=True, exist_ok=True)
            
            # Save classifiers
            joblib.dump(self.health_classifier, save_path / 'health_classifier.pkl')
            joblib.dump(self.anthro_classifier, save_path / 'anthro_classifier.pkl')
            joblib.dump(self.scaler, save_path / 'scaler.pkl')
            
            # Save metadata
            metadata = {
                'model_type': self.model_type,
                'is_trained': self.is_trained,
                'embedding_dim': self.scaler.n_features_in_ if hasattr(self.scaler, 'n_features_in_') else None
            }
            
            joblib.dump(metadata, save_path / 'metadata.pkl')
            
            logger.info(f"Model saved to {save_path}")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise
    
    def load_model(self, load_path: str):
        """
        Load trained model from disk
        
        Args:
            load_path: Path to load the model from
        """
        try:
            load_path = Path(load_path)
            
            # Load classifiers
            self.health_classifier = joblib.load(load_path / 'health_classifier.pkl')
            self.anthro_classifier = joblib.load(load_path / 'anthro_classifier.pkl')
            self.scaler = joblib.load(load_path / 'scaler.pkl')
            
            # Load metadata
            metadata = joblib.load(load_path / 'metadata.pkl')
            self.model_type = metadata['model_type']
            self.is_trained = metadata['is_trained']
            
            logger.info(f"Model loaded from {load_path}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def get_feature_importance(self) -> Dict[str, np.ndarray]:
        """
        Get feature importance for both classifiers
        
        Returns:
            Dictionary with feature importance arrays
        """
        if not self.is_trained:
            raise RuntimeError("Classifier not trained")
        
        try:
            importance = {}
            
            if hasattr(self.health_classifier, 'feature_importances_'):
                importance['health'] = self.health_classifier.feature_importances_
            
            if hasattr(self.anthro_classifier, 'feature_importances_'):
                importance['anthro'] = self.anthro_classifier.feature_importances_
            
            return importance
            
        except Exception as e:
            logger.error(f"Error getting feature importance: {e}")
            raise


# -----------------------------
# Kaggle artifact loader helpers
# -----------------------------

def load_embeddings_from_csv(csv_path: Optional[Path] = None) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Load SurfPerch embeddings from a CSV file. Returns (X, df).

    - Numeric columns are treated as features
    - Non-numeric columns are preserved in the returned DataFrame for joins
    """
    path = Path(csv_path) if csv_path else Path(config.EMBEDDINGS_CSV)
    logger.info(f"Loading embeddings from {path}")
    df = pd.read_csv(path)
    feature_df = df.select_dtypes(include=[np.number])
    X = feature_df.to_numpy(dtype=np.float32)

    # Sanity check: embedding dimension
    expected_dim = config.SURFPERCH_SETTINGS.get("embedding_dim", X.shape[1])
    if X.shape[1] != expected_dim:
        logger.warning(
            f"Embedding dim mismatch: CSV has {X.shape[1]}, expected {expected_dim}. Proceeding anyway."
        )
    return X, df


def load_master_dataset(csv_path: Optional[Path] = None) -> pd.DataFrame:
    """Load the master dataset CSV with metadata and labels."""
    path = Path(csv_path) if csv_path else Path(config.MASTER_DATASET_CSV)
    logger.info(f"Loading master dataset from {path}")
    return pd.read_csv(path)


def align_embeddings_and_labels(
    embeddings_df: pd.DataFrame,
    dataset_df: pd.DataFrame,
    key_columns: Optional[List[str]] = None,
    label_columns: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Align embeddings with labels by joining on key columns.

    If key_columns are not provided, tries ['clip_id'] if present; otherwise falls back to positional alignment.
    Returns (X, y, merged_df). If multiple label columns are provided, y will be a 2D array.
    """
    # Determine join keys
    if key_columns is None:
        if "clip_id" in embeddings_df.columns and "clip_id" in dataset_df.columns:
            key_columns = ["clip_id"]
        else:
            key_columns = []

    # Determine label columns
    if label_columns is None:
        # Heuristic: prefer common label names
        candidate_labels = [
            "health_label",
            "reef_health",
            "label",
            "anthro_label",
            "anthrophony",
        ]
        label_columns = [c for c in candidate_labels if c in dataset_df.columns]
        if not label_columns:
            # Fall back to any non-numeric columns that look categorical
            label_columns = [
                c for c in dataset_df.columns
                if dataset_df[c].dtype == object and c not in key_columns
            ][:1]

    # Perform alignment
    if key_columns:
        merged = embeddings_df.merge(dataset_df, on=key_columns, how="inner")
        feature_df = merged.select_dtypes(include=[np.number])
        # Exclude label columns from features if they are numeric encoded
        feature_df = feature_df.drop(columns=[c for c in label_columns if c in feature_df.columns], errors="ignore")
        X = feature_df.to_numpy(dtype=np.float32)
        y = merged[label_columns].to_numpy() if label_columns else np.array([])
        return X, y, merged

    # Positional fallback
    logger.warning("No key columns for join; falling back to positional alignment.")
    min_len = min(len(embeddings_df), len(dataset_df))
    emb_feat = embeddings_df.select_dtypes(include=[np.number]).iloc[:min_len]
    X = emb_feat.to_numpy(dtype=np.float32)
    y = dataset_df.iloc[:min_len][label_columns].to_numpy() if label_columns else np.array([])
    merged = pd.concat([embeddings_df.iloc[:min_len], dataset_df.iloc[:min_len].reset_index(drop=True)], axis=1)
    return X, y, merged


def load_trained_rf_model(model_path: Optional[Path] = None):
    """Load the pre-trained RandomForest model (.joblib)."""
    path = Path(model_path) if model_path else Path(config.RF_MODEL_PATH)
    logger.info(f"Loading trained RF model from {path}")
    return joblib.load(path)


def predict_with_model(model, X: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Predict with a scikit-learn classifier, returning (preds, probs_or_none)."""
    preds = model.predict(X)
    probs = None
    if hasattr(model, "predict_proba"):
        try:
            probs = model.predict_proba(X)
        except Exception:
            probs = None
    return preds, probs
