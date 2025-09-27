"""
Audio processing utilities for Acoustic Reef
Handles audio loading, preprocessing, and feature extraction
"""

import librosa
import numpy as np
import soundfile as sf
from typing import Tuple, Optional, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AudioProcessor:
    """Audio processing class for reef sound analysis"""
    
    def __init__(self, target_sr: int = 22050, max_duration: float = 60.0):
        """
        Initialize audio processor
        
        Args:
            target_sr: Target sample rate for audio processing
            max_duration: Maximum duration in seconds to process
        """
        self.target_sr = target_sr
        self.max_duration = max_duration
        
    def load_audio(self, file_path: str) -> Tuple[np.ndarray, int]:
        """
        Load audio file with librosa
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Tuple of (audio_array, sample_rate)
        """
        try:
            audio, sr = librosa.load(file_path, sr=self.target_sr)
            
            # Limit duration
            max_samples = int(self.max_duration * sr)
            if len(audio) > max_samples:
                audio = audio[:max_samples]
                logger.info(f"Audio truncated to {self.max_duration}s")
            
            logger.info(f"Loaded audio: {len(audio)} samples at {sr} Hz")
            return audio, sr
            
        except Exception as e:
            logger.error(f"Error loading audio: {e}")
            raise
    
    def extract_features(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """
        Extract audio features for analysis
        
        Args:
            audio: Audio array
            sr: Sample rate
            
        Returns:
            Dictionary of extracted features
        """
        features = {}
        
        try:
            # Basic audio properties
            features['duration'] = len(audio) / sr
            features['rms'] = np.sqrt(np.mean(audio**2))
            features['zero_crossing_rate'] = np.mean(librosa.feature.zero_crossing_rate(audio))
            
            # Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
            features['spectral_centroid_mean'] = np.mean(spectral_centroids)
            features['spectral_centroid_std'] = np.std(spectral_centroids)
            
            # Spectral rolloff
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
            features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
            
            # MFCC features
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            features['mfcc_mean'] = np.mean(mfccs, axis=1)
            features['mfcc_std'] = np.std(mfccs, axis=1)
            
            # Chroma features
            chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
            features['chroma_mean'] = np.mean(chroma, axis=1)
            
            # Tonnetz features
            tonnetz = librosa.feature.tonnetz(y=audio, sr=sr)
            features['tonnetz_mean'] = np.mean(tonnetz, axis=1)
            
            # Spectral contrast
            contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
            features['spectral_contrast_mean'] = np.mean(contrast, axis=1)
            
            logger.info("Audio features extracted successfully")
            return features
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            raise
    
    def create_spectrogram(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Create spectrogram for visualization
        
        Args:
            audio: Audio array
            sr: Sample rate
            
        Returns:
            Spectrogram array
        """
        try:
            # Compute short-time Fourier transform
            stft = librosa.stft(audio)
            spectrogram = np.abs(stft)
            
            # Convert to dB scale
            spectrogram_db = librosa.amplitude_to_db(spectrogram, ref=np.max)
            
            return spectrogram_db
            
        except Exception as e:
            logger.error(f"Error creating spectrogram: {e}")
            raise
    
    def detect_silence(self, audio: np.ndarray, threshold: float = 0.01) -> Dict[str, Any]:
        """
        Detect silence periods in audio
        
        Args:
            audio: Audio array
            threshold: Silence threshold
            
        Returns:
            Dictionary with silence information
        """
        try:
            # Calculate RMS energy
            frame_length = 1024
            hop_length = 512
            rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
            
            # Find silence frames
            silence_frames = rms < threshold
            silence_ratio = np.sum(silence_frames) / len(silence_frames)
            
            return {
                'silence_ratio': silence_ratio,
                'silence_frames': silence_frames,
                'rms_energy': rms
            }
            
        except Exception as e:
            logger.error(f"Error detecting silence: {e}")
            raise
    
    def analyze_frequency_content(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """
        Analyze frequency content of audio
        
        Args:
            audio: Audio array
            sr: Sample rate
            
        Returns:
            Dictionary with frequency analysis
        """
        try:
            # Get frequency spectrum
            fft = np.fft.fft(audio)
            freqs = np.fft.fftfreq(len(fft), 1/sr)
            
            # Get magnitude spectrum
            magnitude = np.abs(fft)
            
            # Find dominant frequencies
            positive_freqs = freqs[:len(freqs)//2]
            positive_magnitude = magnitude[:len(magnitude)//2]
            
            # Find peak frequencies
            peak_indices = np.argsort(positive_magnitude)[-10:]  # Top 10 peaks
            peak_frequencies = positive_freqs[peak_indices]
            peak_magnitudes = positive_magnitude[peak_indices]
            
            return {
                'dominant_frequencies': peak_frequencies,
                'peak_magnitudes': peak_magnitudes,
                'frequency_spectrum': positive_magnitude,
                'frequencies': positive_freqs
            }
            
        except Exception as e:
            logger.error(f"Error analyzing frequency content: {e}")
            raise
