"""
SurfPerch model integration for Acoustic Reef
Handles loading and using the Google SurfPerch model for audio embeddings
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from src.utils import config

# TensorFlow is optional at runtime. If unavailable, we fall back to a
# lightweight placeholder so the app can still run in demo mode.
try:
    import tensorflow as tf  # type: ignore
    TF_AVAILABLE = True
except Exception:  # pragma: no cover - environment-dependent
    tf = None  # type: ignore
    TF_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SurfPerchModel:
    """
    SurfPerch model wrapper for generating audio embeddings
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize SurfPerch model
        
        Args:
            model_path: Path to saved SurfPerch model (if None, will download from Kaggle)
        """
        self.model_path = model_path or str(config.SURFPERCH_SETTINGS.get("model_path"))
        self.model = None
        self.is_loaded = False
        
    def load_model(self) -> bool:
        """
        Load the SurfPerch model
        
        Returns:
            True if model loaded successfully, False otherwise
        """
        try:
            if self.model_path and os.path.exists(self.model_path) and TF_AVAILABLE:
                # Load from local path using TensorFlow SavedModel
                self.model = tf.saved_model.load(self.model_path)  # type: ignore[attr-defined]
                logger.info(f"SurfPerch model loaded from {self.model_path}")
            else:
                if self.model_path and not os.path.exists(self.model_path):
                    logger.warning(
                        f"SurfPerch model path does not exist: {self.model_path}. Falling back to placeholder model."
                    )
                if not TF_AVAILABLE:
                    logger.warning("TensorFlow not available. Using placeholder SurfPerch model.")
                # Create a placeholder model that returns random embeddings
                self.model = self._create_placeholder_model()

            self.is_loaded = True
            return True

        except Exception as e:
            logger.error(f"Error loading SurfPerch model: {e}")
            return False
    
    def _create_placeholder_model(self):
        """
        Create a placeholder model for development
        In production, this would be replaced with actual SurfPerch model
        """
        class PlaceholderModel:
            def __call__(self, audio_input: np.ndarray):
                # Accepts numpy arrays with shape [batch, time]
                if hasattr(audio_input, "shape"):
                    batch_size = audio_input.shape[0] if audio_input.ndim >= 2 else 1
                else:
                    batch_size = 1
                embedding_dim = int(config.SURFPERCH_SETTINGS.get("embedding_dim", 512))
                if TF_AVAILABLE:
                    # Produce a TensorFlow tensor for consistent downstream behavior
                    return tf.random.normal([batch_size, embedding_dim])  # type: ignore[attr-defined]
                # Pure numpy fallback
                return np.random.normal(size=(batch_size, embedding_dim)).astype(np.float32)
        
        return PlaceholderModel()
    
    def generate_embeddings(self, audio_data: np.ndarray, sample_rate: int = 22050) -> np.ndarray:
        """
        Generate embeddings from audio data using SurfPerch
        
        Args:
            audio_data: Audio array (1D or 2D)
            sample_rate: Sample rate of the audio
            
        Returns:
            Embeddings array
        """
        if not self.is_loaded:
            if not self.load_model():
                raise RuntimeError("SurfPerch model not loaded")
        
        try:
            # Ensure audio is [batch, time]
            if audio_data.ndim == 1:
                audio_data = np.expand_dims(audio_data, axis=0)

            if TF_AVAILABLE and self._is_tf_model(self.model):
                # SurfPerch expects exactly 160000 samples (about 7.3 seconds at 22kHz)
                target_length = 160000
                if len(audio_data[0]) != target_length:
                    # Pad or truncate to exact length
                    if len(audio_data[0]) < target_length:
                        # Pad with zeros
                        padding = target_length - len(audio_data[0])
                        audio_data = np.pad(audio_data, ((0, 0), (0, padding)), mode='constant')
                    else:
                        # Truncate
                        audio_data = audio_data[:, :target_length]
                
                audio_tensor = tf.constant(audio_data, dtype=tf.float32)  # type: ignore[name-defined]
                
                # Use the serving_default signature with keyword arguments
                try:
                    logger.info("Calling SurfPerch with serving_default signature")
                    result = self.model.signatures["serving_default"](inputs=audio_tensor)
                    
                    # Extract the embedding from the result dictionary
                    if isinstance(result, dict) and "embedding" in result:
                        embeddings = result["embedding"]
                        logger.info(f"Successfully extracted embedding with shape: {embeddings.shape}")
                    else:
                        raise RuntimeError(f"Unexpected result format: {type(result)}")
                        
                except Exception as e:
                    logger.error(f"SurfPerch model call failed: {e}")
                    raise RuntimeError(f"Could not call SurfPerch model: {e}")
                
                # Handle different output formats
                if isinstance(embeddings, dict):
                    # If it returns a dict, take the first tensor value
                    embeddings = next(iter(embeddings.values()))
                elif hasattr(embeddings, "numpy"):
                    # Convert TensorFlow tensor to numpy
                    embeddings = embeddings.numpy()
                elif not isinstance(embeddings, np.ndarray):
                    # Try to convert to numpy array
                    embeddings = np.array(embeddings)
            else:
                # Placeholder or numpy-compatible path
                embeddings = self.model(audio_data)

            logger.info(f"Generated embeddings: {np.shape(embeddings)}")
            return np.asarray(embeddings)

        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise

    @staticmethod
    def _is_tf_model(model: Any) -> bool:
        """Best-effort check if the loaded model is a TF callable."""
        if not TF_AVAILABLE:
            return False
        # SavedModel callables expose .signatures or are callable with tensors
        return hasattr(model, "signatures") or callable(model)
    
    def preprocess_audio(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Preprocess audio for SurfPerch model
        
        Args:
            audio_data: Raw audio data
            sample_rate: Original sample rate
            
        Returns:
            Preprocessed audio data
        """
        try:
            # Resample if necessary (SurfPerch typically expects 22.05kHz)
            target_sr = 22050
            if sample_rate != target_sr:
                try:
                    import librosa  # type: ignore
                    audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=target_sr)
                except Exception:
                    # If librosa is not available, skip resampling with a warning.
                    logger.warning(
                        "librosa not available for resampling; proceeding without resample. Results may vary."
                    )
                    target_sr = sample_rate

            # Normalize audio safely
            max_abs = np.max(np.abs(audio_data)) if np.size(audio_data) > 0 else 1.0
            if max_abs > 0:
                audio_data = audio_data / max_abs

            # Ensure minimum length (SurfPerch expects certain minimum duration)
            min_length = int(target_sr * 1.0)  # 1 second minimum
            if len(audio_data) < min_length:
                audio_data = np.pad(audio_data, (0, min_length - len(audio_data)), mode="constant")

            return audio_data

        except Exception as e:
            logger.error(f"Error preprocessing audio: {e}")
            raise
    
    def batch_generate_embeddings(self, audio_batch: np.ndarray, sample_rates: np.ndarray) -> np.ndarray:
        """
        Generate embeddings for a batch of audio files
        
        Args:
            audio_batch: Batch of audio arrays
            sample_rates: Corresponding sample rates
            
        Returns:
            Batch of embeddings
        """
        try:
            embeddings_list = []
            
            for i, (audio, sr) in enumerate(zip(audio_batch, sample_rates)):
                # Preprocess audio
                processed_audio = self.preprocess_audio(audio, sr)
                
                # Generate embeddings
                embeddings = self.generate_embeddings(processed_audio, 22050)
                embeddings_list.append(embeddings[0])  # Remove batch dimension
            
            return np.array(embeddings_list)
            
        except Exception as e:
            logger.error(f"Error in batch embedding generation: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model
        
        Returns:
            Dictionary with model information
        """
        return {
            'is_loaded': self.is_loaded,
            'model_path': self.model_path,
            'model_type': 'SurfPerch',
            'embedding_dim': int(config.SURFPERCH_SETTINGS.get("embedding_dim", 512)),
            'target_sample_rate': 22050
        }
