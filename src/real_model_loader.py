#!/usr/bin/env python3
"""
Force the REAL trained model to work by bypassing version issues
"""

import numpy as np
import logging
import pickle
import io
from pathlib import Path
from typing import Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class RealPredictionResult:
    health_label: str
    health_conf: Optional[float]
    noise_label: str
    noise_conf: Optional[float]
    feature_source: str
    feature_dim: int

def force_load_real_model():
    """Force load the REAL trained model by bypassing all version issues"""
    try:
        model_path = Path("models/classifiers/reef_classifier_rf.joblib")
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        logger.info("🔧 FORCING REAL MODEL TO LOAD...")
        
        # Method 1: Try to load with compatibility mode
        try:
            import joblib
            import warnings
            
            # Suppress ALL warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                # Try to load with different protocols
                for protocol in [0, 1, 2, 3, 4, 5]:
                    try:
                        logger.info(f"Trying protocol {protocol}...")
                        model = joblib.load(model_path)
                        logger.info(f"✅ REAL MODEL LOADED with protocol {protocol}!")
                        return model
                    except Exception as e:
                        logger.warning(f"Protocol {protocol} failed: {e}")
                        continue
                        
        except Exception as e:
            logger.warning(f"Joblib methods failed: {e}")
        
        # Method 2: Try to reconstruct the model manually from the pickle
        try:
            logger.info("🔧 MANUAL MODEL RECONSTRUCTION...")
            
            # Load raw pickle data
            with open(model_path, 'rb') as f:
                raw_data = f.read()
            
            # Try to extract the model components manually
            import sklearn.ensemble
            from sklearn.ensemble import RandomForestClassifier
            
            # Create a new model with the same structure
            model = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                bootstrap=True,
                oob_score=False,
                n_jobs=1
            )
            
            # Try to manually extract the trained components
            try:
                # This is a hack to extract the trained trees
                import pickle
                import io
                
                # Create a custom unpickler that ignores version issues
                class CompatUnpickler(pickle.Unpickler):
                    def load_global(self):
                        module = self.readline()[:-1].decode('ascii')
                        name = self.readline()[:-1].decode('ascii')
                        # Map old module names to new ones
                        if module.startswith('sklearn.tree'):
                            module = 'sklearn.tree'
                        elif module.startswith('sklearn.ensemble'):
                            module = 'sklearn.ensemble'
                        elif module.startswith('sklearn.base'):
                            module = 'sklearn.base'
                        
                        try:
                            return self.find_class(module, name)
                        except:
                            # Fallback to current sklearn
                            return super().load_global()
                
                # Try to load with compatibility
                buffer = io.BytesIO(raw_data)
                unpickler = CompatUnpickler(buffer)
                model = unpickler.load()
                
                logger.info("✅ REAL MODEL RECONSTRUCTED!")
                return model
                
            except Exception as e:
                logger.warning(f"Manual reconstruction failed: {e}")
        except Exception as e:
            logger.warning(f"Method 2 failed: {e}")
        
        # Method 3: Try to use the model as-is with error handling
        try:
            logger.info("🔧 LOADING MODEL WITH ERROR HANDLING...")
            
            # Try to load and use the model even if it has warnings
            import joblib
            import warnings
            
            # Completely ignore all warnings
            warnings.filterwarnings("ignore")
            
            # Load the model
            model = joblib.load(model_path)
            
            # Test if it can make predictions
            test_input = np.random.randn(1, 1280).astype(np.float32)
            try:
                _ = model.predict(test_input)
                logger.info("✅ REAL MODEL WORKS WITH ERROR HANDLING!")
                return model
            except Exception as e:
                logger.warning(f"Model loaded but prediction failed: {e}")
                
        except Exception as e:
            logger.warning(f"Error handling method failed: {e}")
        
        # Method 4: Try to retrain a similar model with the same structure
        try:
            logger.info("🔧 RETRAINING SIMILAR MODEL...")
            
            # This is a last resort - create a model with similar structure
            # but we can't retrain it without the original data
            raise Exception("Cannot retrain without original data")
            
        except Exception as e:
            logger.warning(f"Retraining failed: {e}")
        
        # If all methods fail, raise an error
        raise Exception("ALL METHODS FAILED - REAL MODEL CANNOT BE LOADED")
        
    except Exception as e:
        logger.error(f"❌ REAL MODEL LOADING FAILED: {e}")
        raise

def predict_with_real_model(feature_vals: np.ndarray) -> RealPredictionResult:
    """Use the REAL trained model to make predictions"""
    try:
        logger.info(f"🎯 USING REAL MODEL for features: {feature_vals.shape}")
        
        # Force load the real model
        model = force_load_real_model()
        logger.info(f"✅ REAL MODEL LOADED: {type(model)}")
        logger.info(f"Model classes: {getattr(model, 'classes_', 'Unknown')}")
        
        # Make prediction with the REAL model
        prediction = model.predict(feature_vals)
        logger.info(f"🎯 REAL MODEL PREDICTION: {prediction}")
        
        # Get probabilities from the REAL model
        probabilities = None
        if hasattr(model, 'predict_proba'):
            try:
                probabilities = model.predict_proba(feature_vals)
                logger.info(f"🎯 REAL MODEL PROBABILITIES: {probabilities[0]}")
            except Exception as e:
                logger.warning(f"Real model probability prediction failed: {e}")
        
        # Determine raw_pred robustly for both string and numeric class labels
        classes = getattr(model, "classes_", None)
        if isinstance(prediction[0], (str, bytes)):
            raw_pred = str(prediction[0])
        elif isinstance(prediction[0], (int, np.integer)) and classes is not None:
            idx = int(prediction[0])
            if 0 <= idx < len(classes):
                raw_pred = str(classes[idx])
            else:
                raw_pred = "unknown"
        else:
            # Fallback: try to infer from argmax of probabilities
            if probabilities is not None and classes is not None:
                idx = int(np.argmax(probabilities[0]))
                raw_pred = str(classes[idx])
            else:
                raw_pred = "unknown"
        logger.info(f"🎯 MAPPED PREDICTION: {raw_pred}")
        
        # Health assessment
        if raw_pred == "healthy":
            health_label = "Healthy"
        elif raw_pred == "degraded":
            health_label = "Degraded"
        elif raw_pred == "anthrophony":
            health_label = "Degraded"  # Anthrophony implies degraded health
        else:
            health_label = "Unknown"
        
        # Noise assessment
        if raw_pred == "anthrophony":
            noise_label = "High"
        else:
            noise_label = "Low"
        
        # Confidence scores from REAL model (map to class probs if available)
        health_conf = None
        noise_conf = None
        if probabilities is not None:
            try:
                # If we can find the probability for the chosen raw_pred class, prefer that
                if classes is not None and raw_pred in set(map(str, classes)):
                    cls_to_prob = {str(c): float(p) for c, p in zip(classes, probabilities[0])}
                    chosen_prob = cls_to_prob.get(raw_pred)
                    if chosen_prob is not None:
                        health_conf = chosen_prob
                        noise_conf = chosen_prob
                    else:
                        max_prob = float(np.max(probabilities[0]))
                        health_conf = max_prob
                        noise_conf = max_prob
                else:
                    max_prob = float(np.max(probabilities[0]))
                    health_conf = max_prob
                    noise_conf = max_prob
            except Exception:
                max_prob = float(np.max(probabilities[0]))
                health_conf = max_prob
                noise_conf = max_prob
        
        logger.info(f"🎯 REAL MODEL RESULT - Health: {health_label}, Noise: {noise_label}")
        
        return RealPredictionResult(
            health_label=health_label,
            health_conf=health_conf,
            noise_label=noise_label,
            noise_conf=noise_conf,
            feature_source="REAL_TRAINED_MODEL",
            feature_dim=int(feature_vals.shape[1])
        )
        
    except Exception as e:
        logger.error(f"❌ REAL MODEL PREDICTION FAILED: {e}")
        # Don't fall back to anything else - this should work or fail
        raise Exception(f"REAL MODEL FAILED: {e}")
