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
