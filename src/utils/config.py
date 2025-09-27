"""
Configuration settings for Acoustic Reef
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
DASHBOARD_DIR = PROJECT_ROOT / "dashboard"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Data paths
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"

# Model paths
SURFPERCH_MODEL_DIR = MODELS_DIR / "surfperch"
CLASSIFIER_MODEL_DIR = MODELS_DIR / "classifiers"

# Audio processing settings
AUDIO_SETTINGS = {
    "target_sample_rate": 22050,
    "max_duration": 60.0,  # seconds
    "min_duration": 1.0,   # seconds
    "hop_length": 512,
    "n_fft": 2048,
    "n_mels": 128
}

# SurfPerch model settings
SURFPERCH_SETTINGS = {
    "embedding_dim": 512,
    "target_sample_rate": 22050,
    "model_path": SURFPERCH_MODEL_DIR / "surfperch_model"
}

# Classifier settings
CLASSIFIER_SETTINGS = {
    "model_type": "random_forest",  # or "logistic_regression", "svm"
    "test_size": 0.2,
    "random_state": 42,
    "cv_folds": 5
}

# Dashboard settings
DASHBOARD_SETTINGS = {
    "page_title": "Acoustic Reef",
    "page_icon": "🌊",
    "layout": "wide",
    "max_file_size": 100 * 1024 * 1024,  # 100MB
    "supported_formats": [".wav", ".mp3", ".flac"]
}

# Logging settings
LOGGING_SETTINGS = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": PROJECT_ROOT / "logs" / "acoustic_reef.log"
}

# Create necessary directories
def create_directories():
    """Create necessary directories if they don't exist"""
    directories = [
        DATA_DIR,
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        EMBEDDINGS_DIR,
        MODELS_DIR,
        SURFPERCH_MODEL_DIR,
        CLASSIFIER_MODEL_DIR,
        PROJECT_ROOT / "logs"
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

# Initialize directories on import
create_directories()
