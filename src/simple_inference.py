#!/usr/bin/env python3
"""
Simple inference module that bypasses Streamlit caching issues
"""

import joblib
import numpy as np
import logging
from pathlib import Path
from typing import Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class SimplePredictionResult:
    health_label: str
    health_conf: Optional[float]
    noise_label: str
    noise_conf: Optional[float]
    feature_source: str
    feature_dim: int

def load_model_directly() -> any:
    """Load the RandomForest model directly without caching"""
    try:
        model_path = Path("models/classifiers/reef_classifier_rf.joblib")
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        # Load with error handling for version compatibility
        try:
            model = joblib.load(model_path)
        except Exception as e:
            logger.warning(f"Direct load failed: {e}, trying with warnings suppressed")
            import warnings
            warnings.filterwarnings("ignore")
            model = joblib.load(model_path)
        
        logger.info(f"Model loaded successfully: {type(model)}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

def predict_simple(feature_vals: np.ndarray) -> SimplePredictionResult:
    """Make predictions using the loaded model"""
    try:
        # Load model fresh each time
        model = load_model_directly()
        
        # Make prediction
        prediction = model.predict(feature_vals)
        probabilities = None
        
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(feature_vals)
        
        # Map prediction to labels
        class_map = {0: "healthy", 1: "degraded", 2: "anthrophony"}
        pred_idx = int(prediction[0]) if isinstance(prediction[0], (int, np.integer)) else 0
        raw_pred = class_map.get(pred_idx, "unknown")
        
        # Health assessment
        if raw_pred == "healthy":
            health_label = "Healthy"
        elif raw_pred == "degraded":
            health_label = "Degraded"
        else:  # anthrophony or unknown
            health_label = "Degraded"
        
        # Noise assessment
        if raw_pred == "anthrophony":
            noise_label = "High"
        else:
            noise_label = "Low"
        
        # Confidence scores
        health_conf = None
        noise_conf = None
        if probabilities is not None:
            max_prob = float(np.max(probabilities[0]))
            health_conf = max_prob
            noise_conf = max_prob
        
        return SimplePredictionResult(
            health_label=health_label,
            health_conf=health_conf,
            noise_label=noise_label,
            noise_conf=noise_conf,
            feature_source="runtime",
            feature_dim=int(feature_vals.shape[1])
        )
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return SimplePredictionResult(
            health_label="Unknown",
            health_conf=None,
            noise_label="Unknown", 
            noise_conf=None,
            feature_source="error",
            feature_dim=int(feature_vals.shape[1])
        )
