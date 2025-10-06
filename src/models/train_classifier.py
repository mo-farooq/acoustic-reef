"""
Train a reef health + anthrophony classifier using SurfPerch embeddings.

Expected input labels CSV (default: data/raw/labels.csv):
    filepath,health_label,anthro_label
    path/to/audio1.wav,1,0
    path/to/audio2.wav,0,1

The script will:
1) Load audio files
2) Generate SurfPerch embeddings (uses local model if available, else placeholder)
3) Train the classifier and save under models/classifiers/

Usage (Windows):
    python -m src.models.train_classifier --labels data\\raw\\labels.csv

Optional flags:
    --output_dir models\\classifiers
    --model_type random_forest|logistic_regression|svm
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np

from .reef_classifier import ReefClassifier
from .surfperch_integration import SurfPerchModel
from ..utils.config import (
    CLASSIFIER_MODEL_DIR,
    CLASSIFIER_SETTINGS,
    SURFPERCH_SETTINGS,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("train_classifier")


@dataclass
class LabeledItem:
    audio_path: Path
    health_label: int
    anthro_label: int


def read_labels_csv(csv_path: Path, project_root: Path) -> List[LabeledItem]:
    items: List[LabeledItem] = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rel_or_abs = row["filepath"].strip()
            audio_path = Path(rel_or_abs)
            if not audio_path.is_absolute():
                audio_path = (project_root / rel_or_abs).resolve()
            health_label = int(row["health_label"])  # 0/1
            anthro_label = int(row["anthro_label"])  # 0/1
            if not audio_path.exists():
                logger.warning(f"Audio file not found, skipping: {audio_path}")
                continue
            items.append(LabeledItem(audio_path=audio_path, health_label=health_label, anthro_label=anthro_label))
    if not items:
        raise RuntimeError("No valid labeled items found. Check your CSV paths.")
    return items


def load_wav_as_float(path: Path) -> Tuple[np.ndarray, int]:
    import wave
    import contextlib

    with contextlib.closing(wave.open(str(path), "rb")) as wf:
        n_channels = wf.getnchannels()
        sr = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)
        sample_width = wf.getsampwidth()
        dtype = {1: np.int8, 2: np.int16, 3: np.int32, 4: np.int32}.get(sample_width, np.int16)
        audio = np.frombuffer(raw, dtype=dtype)
        if n_channels > 1:
            audio = audio.reshape(-1, n_channels).mean(axis=1)
        max_val = np.max(np.abs(audio)) or 1
        audio = (audio.astype(np.float32) / max_val).astype(np.float32)
        return audio, sr


def build_embeddings(items: List[LabeledItem], model_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model = SurfPerchModel(model_path=str(model_path))

    embeddings_list: List[np.ndarray] = []
    health_labels: List[int] = []
    anthro_labels: List[int] = []

    for idx, item in enumerate(items):
        try:
            audio, sr = load_wav_as_float(item.audio_path)
            processed = model.preprocess_audio(audio, sr)
            emb = model.generate_embeddings(processed, 22050)
            # emb is [1, D] typically
            embeddings_list.append(emb[0] if emb.ndim == 2 else np.reshape(emb, (-1,)))
            health_labels.append(item.health_label)
            anthro_labels.append(item.anthro_label)
        except Exception as e:
            logger.warning(f"Failed to embed {item.audio_path}: {e}")
            continue

    if not embeddings_list:
        raise RuntimeError("No embeddings generated. Check audio files and dependencies.")

    X = np.vstack(embeddings_list)
    y_health = np.asarray(health_labels, dtype=int)
    y_anthro = np.asarray(anthro_labels, dtype=int)
    logger.info(f"Embeddings shape: {X.shape}; labels: health={y_health.shape}, anthro={y_anthro.shape}")
    return X, y_health, y_anthro


def main() -> None:
    parser = argparse.ArgumentParser(description="Train reef classifier using SurfPerch embeddings.")
    parser.add_argument("--labels", type=str, default=str(Path("data") / "raw" / "labels.csv"))
    parser.add_argument("--output_dir", type=str, default=str(CLASSIFIER_MODEL_DIR))
    parser.add_argument("--model_type", type=str, default=CLASSIFIER_SETTINGS["model_type"],
                        choices=["random_forest", "logistic_regression", "svm"])
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    labels_csv = Path(args.labels)
    output_dir = Path(args.output_dir)
    model_path = Path(SURFPERCH_SETTINGS["model_path"]).resolve()

    logger.info(f"Labels CSV: {labels_csv}")
    logger.info(f"SurfPerch model path: {model_path}")
    logger.info(f"Output dir: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    items = read_labels_csv(labels_csv, project_root)
    X, y_health, y_anthro = build_embeddings(items, model_path)

    clf = ReefClassifier(model_type=args.model_type)
    results = clf.train(X, y_health, y_anthro, test_size=CLASSIFIER_SETTINGS["test_size"])
    logger.info(f"Training results: {results}")

    clf.save_model(str(output_dir))
    logger.info("Classifier saved successfully.")


if __name__ == "__main__":
    main()


