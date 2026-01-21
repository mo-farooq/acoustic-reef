#!/usr/bin/env python3
"""
Generate UMAP coordinates and model for Acoustic Map visualization
This script creates the missing UMAP files needed for the acoustic map feature.
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import logging
from typing import Tuple, Optional
import umap

from src.utils.config import (
    EMBEDDINGS_CSV, 
    MASTER_DATASET_CSV, 
    UMAP_COORDS_CSV, 
    UMAP_MODEL_PATH,
    PROCESSED_DATA_DIR
)
from src.models.reef_classifier import load_embeddings_from_csv, load_master_dataset, align_embeddings_and_labels

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_umap_visualization() -> Tuple[Optional[np.ndarray], Optional[umap.UMAP], Optional[pd.DataFrame]]:
    """
    Generate UMAP coordinates and model for acoustic map visualization.
    
    Returns:
        Tuple of (coordinates, umap_model, labels_df) or (None, None, None) if failed
    """
    try:
        logger.info("Loading embeddings...")
        
        # Load embeddings directly
        X_emb, emb_df = load_embeddings_from_csv()
        if X_emb.size == 0:
            raise ValueError("Embeddings are empty; cannot compute UMAP.")
        logger.info(f"Loaded {X_emb.shape[0]} samples with {X_emb.shape[1]} features")
        
        # Use actual category labels from embeddings data
        labels = []
        filenames = []
        for idx, row in emb_df.iterrows():
            # Extract filename from filepath
            filepath = row.get('filepath', f'sample_{idx}')
            filename = filepath.split('/')[-1] if '/' in filepath else filepath
            filenames.append(filename)
            
            # Use the actual category from the embeddings data
            category = row.get('category', 'healthy')
            labels.append(category)
        
        # Create labels DataFrame
        labels_df = pd.DataFrame({
            'filename': filenames,
            'health_status': labels
        })
        
        # Create UMAP reducer
        logger.info("Creating UMAP reducer...")
        umap_reducer = umap.UMAP(
            n_components=2,
            n_neighbors=15,
            min_dist=0.1,
            metric='cosine',
            random_state=42,
            verbose=True
        )
        
        # Fit and transform
        logger.info("Fitting UMAP...")
        coords = umap_reducer.fit_transform(X_emb)
        
        logger.info(f"UMAP coordinates shape: {coords.shape}")
        
        return coords, umap_reducer, labels_df
        
    except Exception as e:
        logger.error(f"Failed to generate UMAP: {e}")
        return None, None, None

def save_umap_artifacts(coords: np.ndarray, umap_model: umap.UMAP, labels_df: pd.DataFrame) -> bool:
    """
    Save UMAP coordinates and model to files.
    
    Args:
        coords: 2D coordinates from UMAP
        umap_model: Trained UMAP model
        labels_df: DataFrame with labels for coloring
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create coordinates DataFrame
        coords_df = pd.DataFrame({
            'x': coords[:, 0],
            'y': coords[:, 1],
            'label': labels_df.get('health_status', 'unknown'),
            'filename': labels_df.get('filename', 'unknown')
        })
        
        # Save coordinates
        logger.info(f"Saving coordinates to {UMAP_COORDS_CSV}")
        coords_df.to_csv(UMAP_COORDS_CSV, index=False)
        
        # Save UMAP model
        logger.info(f"Saving UMAP model to {UMAP_MODEL_PATH}")
        joblib.dump(umap_model, UMAP_MODEL_PATH)
        
        logger.info("UMAP artifacts saved successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Failed to save UMAP artifacts: {e}")
        return False

def generate_umap_visualization_3d() -> Tuple[Optional[np.ndarray], Optional[umap.UMAP], Optional[pd.DataFrame]]:
    """
    Generate 3D UMAP coordinates and model for acoustic map visualization.
    Returns:
        Tuple of (coordinates, umap_model, labels_df) or (None, None, None) if failed
    """
    try:
        logger.info("Loading embeddings for 3D UMAP...")
        X_emb, emb_df = load_embeddings_from_csv()
        if X_emb.size == 0:
            raise ValueError("Embeddings are empty; cannot compute 3D UMAP.")
        logger.info(f"Loaded {X_emb.shape[0]} samples with {X_emb.shape[1]} features (3D)")
        labels = []
        filenames = []
        for idx, row in emb_df.iterrows():
            filepath = row.get('filepath', f'sample_{idx}')
            filename = filepath.split('/')[-1] if '/' in filepath else filepath
            filenames.append(filename)
            category = row.get('category', 'healthy')
            labels.append(category)
        labels_df = pd.DataFrame({
            'filename': filenames,
            'health_status': labels
        })
        logger.info("Creating 3D UMAP reducer...")
        umap_reducer = umap.UMAP(
            n_components=3,
            n_neighbors=15,
            min_dist=0.1,
            metric='cosine',
            random_state=42,
            verbose=True
        )
        logger.info("Fitting 3D UMAP...")
        coords = umap_reducer.fit_transform(X_emb)
        logger.info(f"3D UMAP coordinates shape: {coords.shape}")
        return coords, umap_reducer, labels_df
    except Exception as e:
        logger.error(f"Failed to generate 3D UMAP: {e}")
        return None, None, None

def save_umap_artifacts_3d(coords: np.ndarray, umap_model: umap.UMAP, labels_df: pd.DataFrame) -> bool:
    """
    Save 3D UMAP coordinates and model to files.
    Args:
        coords: 3D coordinates from UMAP
        umap_model: Trained UMAP model
        labels_df: DataFrame with labels for coloring
    Returns:
        True if successful, False otherwise
    """
    try:
        # Save UMAP model
        logger.info(f"Saving 3D UMAP model to {UMAP_MODEL_PATH}")
        joblib.dump(umap_model, UMAP_MODEL_PATH)
        # Optionally, save coordinates
        coords_3d_df = pd.DataFrame({
            'x': coords[:, 0],
            'y': coords[:, 1],
            'z': coords[:, 2],
            'label': labels_df.get('health_status', 'unknown'),
            'filename': labels_df.get('filename', 'unknown')
        })
        out_csv = UMAP_COORDS_CSV
        logger.info(f"Saving 3D coordinates to {out_csv}")
        coords_3d_df.to_csv(out_csv, index=False)
        logger.info("3D UMAP artifacts saved successfully!")
        return True
    except Exception as e:
        logger.error(f"Failed to save 3D UMAP artifacts: {e}")
        return False

def main():
    """Main function to generate UMAP visualization files."""
    logger.info("Starting UMAP generation for Acoustic Map...")
    # 2D UMAP Generation
    if UMAP_COORDS_CSV.exists() and UMAP_MODEL_PATH.exists():
        logger.info("UMAP files already exist. Skipping 2D generation.")
    else:
        coords, umap_model, labels_df = generate_umap_visualization()
        if coords is not None and umap_model is not None and labels_df is not None:
            save_umap_artifacts(coords, umap_model, labels_df)
    # 3D UMAP Generation
    from src.utils import config
    if not UMAP_MODEL_PATH.exists():
        coords3d, umap_model3d, labels_df3d = generate_umap_visualization_3d()
        if coords3d is not None and umap_model3d is not None and labels_df3d is not None:
            save_umap_artifacts_3d(coords3d, umap_model3d, labels_df3d)
        else:
            logger.error("Failed to generate 3D UMAP visualization")
    else:
        logger.info("3D UMAP model already exists. Skipping 3D generation.")

if __name__ == "__main__":
    main()
