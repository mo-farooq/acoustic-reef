"""
Inference utilities for Acoustic Reef
 - Data loaders
 - Embedding resolution (precomputed vs runtime SurfPerch)
 - Prediction (delegates to src.real_model_loader; no fallback classifiers)
 - UMAP transformation utilities
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
import joblib

from src.models.surfperch_integration import SurfPerchModel
from src.models.reef_classifier import (
    load_embeddings_from_csv,
)
from src.real_model_loader import RealPredictionResult, predict_with_real_model
from src.utils import config
from pathlib import Path

logger = logging.getLogger(__name__)


def load_umap_model():
    try:
        return joblib.load(config.UMAP_MODEL_PATH)
    except Exception as e:
        logger.warning(f"Could not load UMAP model: {e}")
        return None


def load_umap_coordinates() -> Optional[pd.DataFrame]:
    try:
        # Correct attribute from config
        path = config.UMAP_COORDS_CSV
        return pd.read_csv(path)
    except Exception as e:
        logger.warning(f"Could not load UMAP coordinates: {e}")
        return None


def get_surfperch_model() -> SurfPerchModel:
    return SurfPerchModel(model_path=str(config.SURFPERCH_SETTINGS["model_path"]))


def load_precomputed_embeddings() -> Tuple[np.ndarray, pd.DataFrame]:
    try:
        X, df = load_embeddings_from_csv()
        if X.size == 0:
            raise ValueError("Embeddings CSV contains no numeric feature columns.")
        return X, df
    except Exception as e:
        logger.error(f"Failed to load precomputed embeddings: {e}")
        raise


def find_embedding_for_filename(emb_df: pd.DataFrame, filename: str) -> Optional[np.ndarray]:
    if not filename:
        return None
    try:
        if 'filepath' not in emb_df.columns:
            return None
        # Normalize
        fname = str(filename).strip()
        base = fname.rsplit('.', 1)[0].lower()
        # Build exact-match candidates
        filepaths = emb_df['filepath'].astype(str)
        basenames = filepaths.apply(lambda p: p.rsplit('/', 1)[-1].rsplit('\\', 1)[-1])
        basenames_noext = basenames.str.rsplit('.', 1).str[0].str.lower()

        # 1) Exact basename match (case-sensitive)
        exact_hits = emb_df[basenames == fname]
        # 2) Exact basename (no extension, case-insensitive)
        if exact_hits.empty and base:
            exact_hits = emb_df[basenames_noext == base]
        # 3) Fallback: substring contains (case-insensitive)
        if exact_hits.empty:
            exact_hits = emb_df[filepaths.str.contains(fname, case=False, na=False)]

        if not exact_hits.empty:
            embedding_cols = [c for c in emb_df.columns if c.startswith('embedding_')]
            if embedding_cols:
                return exact_hits.iloc[0][embedding_cols].values.astype(np.float32)
    except Exception as e:
        logger.warning(f"Error finding embedding for {filename}: {e}")
        return None
    return None


def compute_embedding_from_audio(audio_np: np.ndarray, sample_rate: int) -> np.ndarray:
    """Compute a genuine SurfPerch embedding; raises SurfPerchError if impossible."""
    sp = get_surfperch_model()
    processed = sp.preprocess_audio(audio_np, sample_rate)
    emb = sp.generate_embeddings(processed, 32000)
    return emb.reshape(1, -1) if emb.ndim == 2 else emb


PredictionResult = RealPredictionResult


def predict_vital_signs(feature_vals: np.ndarray) -> RealPredictionResult:
    """Predict with the stored classifier. Raises on failure; never returns a fallback."""
    return predict_with_real_model(feature_vals)


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

    # No precomputed match: compute genuine SurfPerch embeddings. Raises a
    # SurfPerchError (TensorFlow/model missing, invalid output) if that is not possible.
    return compute_embedding_from_audio(audio_np, sample_rate), "runtime-generated SurfPerch embeddings"


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


