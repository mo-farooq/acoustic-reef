"""
Inference utilities for Acoustic Reef
 - Cached model/data loaders
 - Embedding resolution (precomputed vs runtime)
 - Dual predictions (health, noise)
 - UMAP transformation utilities
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
import joblib
import streamlit as st

from src.models.surfperch_integration import SurfPerchModel
from src.models.reef_classifier import (
    load_embeddings_from_csv,
)
from src.utils import config
from pathlib import Path

logger = logging.getLogger(__name__)


def load_health_classifier():
    """Load health classifier; fallback to legacy RF if dedicated model missing."""
    try:
        health_path = Path(config.HEALTH_CLASSIFIER_PATH)
        if health_path.exists():
            return joblib.load(health_path)
    except Exception as e:
        logger.warning(f"Could not load dedicated health classifier: {e}")
    
    # Fallback to legacy single model
    try:
        legacy = Path(config.RF_MODEL_PATH)
        if legacy.exists():
            return joblib.load(legacy)
    except Exception as e:
        logger.warning(f"Could not load legacy RF model: {e}")
    
    raise FileNotFoundError(
        f"No health classifier found. Checked {config.HEALTH_CLASSIFIER_PATH} and {config.RF_MODEL_PATH}"
    )


def load_noise_classifier():
    """Load noise classifier if available; return None if missing."""
    noise_path = Path(config.NOISE_CLASSIFIER_PATH)
    if noise_path.exists():
        return joblib.load(noise_path)
    return None


def load_umap_model():
    try:
        return joblib.load(config.UMAP_MODEL_PATH)
    except Exception as e:
        logger.warning(f"Could not load UMAP model: {e}")
        return None


def load_umap_coordinates() -> Optional[pd.DataFrame]:
    try:
        # Use the correct config attribute name
        path = getattr(config, "UMAP_COORDINATES_CSV", None)
        if path is None:
            raise FileNotFoundError("UMAP_COORDINATES_CSV not configured")
        return pd.read_csv(path)
    except Exception as e:
        logger.warning(f"Could not load UMAP coordinates: {e}")
        return None


def get_surfperch_model() -> SurfPerchModel:
    return SurfPerchModel(model_path=str(config.SURFPERCH_SETTINGS["model_path"]))


def load_precomputed_embeddings() -> Tuple[np.ndarray, pd.DataFrame]:
    return load_embeddings_from_csv()


def find_embedding_for_filename(emb_df: pd.DataFrame, filename: str) -> Optional[np.ndarray]:
    if not filename:
        return None
    try:
        # Look for filename in filepath column
        if 'filepath' in emb_df.columns:
            hits = emb_df[emb_df['filepath'].astype(str).str.contains(filename, case=False, na=False)]
            if len(hits) > 0:
                # Extract embedding columns (embedding_0, embedding_1, etc.)
                embedding_cols = [c for c in emb_df.columns if c.startswith('embedding_')]
                if embedding_cols:
                    return hits.iloc[0][embedding_cols].values.astype(np.float32)
    except Exception as e:
        logger.warning(f"Error finding embedding for {filename}: {e}")
        return None
    return None


def compute_embedding_from_audio(audio_np: np.ndarray, sample_rate: int) -> np.ndarray:
    sp = get_surfperch_model()
    processed = sp.preprocess_audio(audio_np, sample_rate)
    emb = sp.generate_embeddings(processed, 22050)
    return emb.reshape(1, -1) if emb.ndim == 2 else emb


@dataclass
class PredictionResult:
    health_label: str
    health_conf: Optional[float]
    noise_label: str
    noise_conf: Optional[float]
    feature_source: str
    feature_dim: int


def predict_vital_signs(feature_vals: np.ndarray) -> PredictionResult:
    """Predict vital signs using available models."""
    try:
        # Try to load dedicated classifiers first
        health_model = load_health_classifier()
        noise_model = load_noise_classifier()
        
        # Health prediction
        health_pred = health_model.predict(feature_vals)
        health_conf = None
        if hasattr(health_model, "predict_proba"):
            try:
                health_conf = float(np.max(health_model.predict_proba(feature_vals)[0]))
            except Exception:
                health_conf = None
        
        # The RF model predicts: 0=healthy, 1=degraded, 2=anthrophony
        # Map predictions to labels
        pred_map = {0: "healthy", 1: "degraded", 2: "anthrophony"}
        raw_pred = pred_map.get(int(health_pred[0]), "unknown")
        
        # Health assessment: healthy vs degraded (ignore anthrophony for health)
        if raw_pred == "healthy":
            health_label = "Healthy"
        elif raw_pred == "degraded":
            health_label = "Degraded"
        else:  # anthrophony or unknown
            health_label = "Degraded"  # Assume anthrophony indicates degraded health
        
        # Noise assessment: anthrophony vs others
        if raw_pred == "anthrophony":
            noise_label = "High"
            noise_conf = health_conf if health_conf is not None else 0.8
        else:
            noise_label = "Low"
            noise_conf = health_conf if health_conf is not None else 0.8

        return PredictionResult(
            health_label=health_label,
            health_conf=health_conf,
            noise_label=noise_label,
            noise_conf=noise_conf,
            feature_source="",
            feature_dim=int(feature_vals.shape[1]),
        )
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        # Return a safe fallback
        return PredictionResult(
            health_label="Unknown",
            health_conf=None,
            noise_label="Unknown", 
            noise_conf=None,
            feature_source="error",
            feature_dim=int(feature_vals.shape[1]),
        )


def resolve_features_for_file(file_basename: str, audio_np: np.ndarray, sample_rate: int) -> Tuple[np.ndarray, str]:
    """Prefer precomputed embeddings; if not found and TF runtime is unavailable, raise for accurate-only behavior."""
    X_pre, emb_df = load_precomputed_embeddings()
    # broaden filename matching: strip extension and lowercase
    base_no_ext = file_basename.rsplit('.', 1)[0].lower() if file_basename else ""
    row = None
    if file_basename:
        row = find_embedding_for_filename(emb_df, file_basename)
        if row is None and base_no_ext:
            row = find_embedding_for_filename(emb_df, base_no_ext)
    if row is not None:
        return row.reshape(1, -1), f"precomputed embeddings match for '{file_basename}'"

    # Try runtime embeddings; if TF is missing, insist on precomputed for accuracy
    try:
        import tensorflow as _tf  # noqa: F401
        tf_available = True
    except Exception:
        tf_available = False
    if not tf_available:
        raise RuntimeError(
            "No matching precomputed embedding found and TensorFlow is not available for real-time embeddings. "
            "To get accurate results, upload a file present in embeddings.csv or enable TensorFlow."
        )
    return compute_embedding_from_audio(audio_np, sample_rate), "runtime-generated embeddings"


def transform_with_umap(feature_vals: np.ndarray) -> Optional[np.ndarray]:
    try:
        umap_model = load_umap_model()
        if umap_model is None:
            logger.warning("UMAP model not available")
            return None
        coords = umap_model.transform(feature_vals)
        return coords.reshape(1, -1)
    except Exception as e:
        logger.warning(f"UMAP transform failed: {e}")
        return None


