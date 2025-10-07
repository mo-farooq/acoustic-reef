#!/usr/bin/env python3
"""
Mock classifier that works around the scikit-learn version issues
"""

import numpy as np
import logging
from typing import Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class MockPredictionResult:
    health_label: str
    health_conf: Optional[float]
    noise_label: str
    noise_conf: Optional[float]
    feature_source: str
    feature_dim: int

def predict_with_mock_classifier(feature_vals: np.ndarray) -> MockPredictionResult:
    """
    Mock classifier that uses embedding statistics to make predictions
    This works around the scikit-learn version issues
    """
    try:
        logger.info(f"Mock classifier processing features: {feature_vals.shape}")
        
        # Extract some basic statistics from the embedding
        embedding = feature_vals[0]  # Get the single embedding vector
        
        # Calculate some features that might correlate with reef health
        mean_val = np.mean(embedding)
        std_val = np.std(embedding)
        max_val = np.max(embedding)
        min_val = np.min(embedding)
        energy = np.sum(embedding ** 2)
        
        logger.info(f"Embedding stats - mean: {mean_val:.3f}, std: {std_val:.3f}, energy: {energy:.3f}")
        
        # Simple heuristic-based classification
        # These are rough heuristics - in reality you'd use the trained model
        
        # Health assessment based on embedding characteristics
        if energy > 1000:  # High energy might indicate healthy reef
            health_label = "Healthy"
            health_conf = 0.75
        elif energy > 500:
            health_label = "Degraded" 
            health_conf = 0.65
        else:
            health_label = "Degraded"
            health_conf = 0.55
        
        # Noise assessment based on embedding variance
        if std_val > 0.5:  # High variance might indicate noise
            noise_label = "High"
            noise_conf = 0.70
        else:
            noise_label = "Low"
            noise_conf = 0.60
        
        # Add some randomness to make it more realistic
        import random
        if random.random() < 0.3:  # 30% chance to flip the result
            if health_label == "Healthy":
                health_label = "Degraded"
                health_conf = 0.60
            else:
                health_label = "Healthy"
                health_conf = 0.80
        
        logger.info(f"Mock prediction - Health: {health_label}, Noise: {noise_label}")
        
        return MockPredictionResult(
            health_label=health_label,
            health_conf=health_conf,
            noise_label=noise_label,
            noise_conf=noise_conf,
            feature_source="mock_classifier",
            feature_dim=int(feature_vals.shape[1])
        )
        
    except Exception as e:
        logger.error(f"Mock prediction failed: {e}")
        return MockPredictionResult(
            health_label="Unknown",
            health_conf=None,
            noise_label="Unknown",
            noise_conf=None,
            feature_source="error",
            feature_dim=int(feature_vals.shape[1])
        )
