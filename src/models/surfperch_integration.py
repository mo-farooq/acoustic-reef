"""
SurfPerch model integration for Acoustic Reef
Handles loading and using the Google SurfPerch model for audio embeddings
"""

import tensorflow as tf
import numpy as np
import logging
from typing import Tuple, Optional, Dict, Any
import os
from pathlib import Path

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
        self.model_path = model_path
        self.model = None
        self.is_loaded = False
        
    def load_model(self) -> bool:
        """
        Load the SurfPerch model
        
        Returns:
            True if model loaded successfully, False otherwise
        """
        try:
            if self.model_path and os.path.exists(self.model_path):
                # Load from local path
                self.model = tf.saved_model.load(self.model_path)
                logger.info(f"SurfPerch model loaded from {self.model_path}")
            else:
                # For now, create a placeholder model
                # In production, this would load the actual SurfPerch model from Kaggle
                logger.warning("SurfPerch model not found. Using placeholder model.")
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
            def __call__(self, audio_input):
                # Placeholder: return random embeddings
                batch_size = tf.shape(audio_input)[0]
                embedding_dim = 512  # Typical SurfPerch embedding dimension
                return tf.random.normal([batch_size, embedding_dim])
        
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
            # Ensure audio is in the correct format
            if len(audio_data.shape) == 1:
                # Add batch dimension
                audio_data = np.expand_dims(audio_data, axis=0)
            
            # Convert to tensor
            audio_tensor = tf.constant(audio_data, dtype=tf.float32)
            
            # Generate embeddings
            embeddings = self.model(audio_tensor)
            
            # Convert back to numpy
            if isinstance(embeddings, tf.Tensor):
                embeddings = embeddings.numpy()
            
            logger.info(f"Generated embeddings: {embeddings.shape}")
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
    
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
                import librosa
                audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=target_sr)
            
            # Normalize audio
            audio_data = audio_data / np.max(np.abs(audio_data))
            
            # Ensure minimum length (SurfPerch expects certain minimum duration)
            min_length = int(target_sr * 1.0)  # 1 second minimum
            if len(audio_data) < min_length:
                # Pad with zeros
                audio_data = np.pad(audio_data, (0, min_length - len(audio_data)), mode='constant')
            
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
            'embedding_dim': 512,  # Typical SurfPerch embedding dimension
            'target_sample_rate': 22050
        }
