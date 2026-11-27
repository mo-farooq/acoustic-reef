"""
Train a reef health + anthrophony classifier using SurfPerch embeddings.

Expected input labels CSV (default: data/raw/labels.csv):
    filepath,health_label,anthro_label
    path/to/audio1.wav,1,0
    path/to/audio2.wav,0,1
The script will:
1) Validate the labels CSV and that every listed audio file exists
2) Generate genuine SurfPerch embeddings (fails if TensorFlow or the model is unavailable)
3) Train the classifier and save under models/classifiers/ only after training succeeded

The pipeline fails explicitly on: missing labels CSV, malformed/duplicate rows,
missing audio files, an empty dataset, any embedding failure. It never skips
rows silently, never uses placeholder or random embeddings and never saves a
model after a failed or incomplete run.

NOTE: ReefClassifier.train uses a random row-level split. Its scores are NOT
estimates of generalisation to unseen sites, recorders or periods (P1 work).

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

REQUIRED_COLUMNS = ("filepath", "health_label", "anthro_label")


class TrainingDataError(ValueError):
    """The labelled dataset is missing, empty, malformed or inconsistent."""


@dataclass
class LabeledItem:
    audio_path: Path
    health_label: int
    anthro_label: int


def _parse_binary(value: str, column: str, line: int) -> int:
    try:
        parsed = int(str(value).strip())
    except ValueError:
        raise TrainingDataError(f"Line {line}: {column}={value!r} is not an integer") from None
    if parsed not in (0, 1):
        raise TrainingDataError(f"Line {line}: {column}={parsed} must be 0 or 1")
    return parsed


def read_labels_csv(csv_path: Path, project_root: Path) -> List[LabeledItem]:
    """Read and validate the labels CSV. Raises instead of skipping anything."""
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Labels CSV not found: {csv_path}")
    items: List[LabeledItem] = []
    missing_audio: List[Path] = []
    seen = set()
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        absent = [c for c in REQUIRED_COLUMNS if c not in (reader.fieldnames or [])]
        if absent:
            raise TrainingDataError(f"Labels CSV is missing required column(s): {absent}")
        for line, row in enumerate(reader, start=2):
            rel_or_abs = (row["filepath"] or "").strip()
            if not rel_or_abs:
                raise TrainingDataError(f"Line {line}: empty filepath")
            audio_path = Path(rel_or_abs)
            if not audio_path.is_absolute():
                audio_path = (project_root / rel_or_abs).resolve()
            if audio_path in seen:
                raise TrainingDataError(f"Line {line}: duplicate filepath {rel_or_abs}")
            seen.add(audio_path)
            health = _parse_binary(row["health_label"], "health_label", line)
            anthro = _parse_binary(row["anthro_label"], "anthro_label", line)
            if not audio_path.is_file():
                missing_audio.append(audio_path)
                continue
            items.append(LabeledItem(audio_path=audio_path, health_label=health, anthro_label=anthro))
    if missing_audio:
        shown = ", ".join(str(p) for p in missing_audio[:10])
        raise FileNotFoundError(
            f"{len(missing_audio)} audio file(s) listed in {csv_path} do not exist "
            f"(first {min(10, len(missing_audio))}: {shown}). Training aborted."
        )
    if not items:
        raise TrainingDataError(f"No labelled rows found in {csv_path}; refusing to train on an empty dataset.")
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


def build_embeddings(
    items: List[LabeledItem],
    model_path: Path,
    model: SurfPerchModel | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Embed every item with the genuine SurfPerch model.

    Raises if the model cannot be loaded or if any item fails to embed.
    ``model`` may be injected (used by tests); it must be a SurfPerchModel-like object.
    """
    if not items:
        raise TrainingDataError("No items to embed; refusing to train on an empty dataset.")
    model = model if model is not None else SurfPerchModel(model_path=str(model_path))
    if not model.is_loaded:
        model.load_model()  # raises if TensorFlow or the SavedModel is unavailable
    embeddings_list: List[np.ndarray] = []
    for item in items:
        try:
            audio, sr = load_wav_as_float(item.audio_path)
            processed = model.preprocess_audio(audio, sr)
            emb = model.generate_embeddings(processed, 32000)
        except Exception as exc:
            raise RuntimeError(f"Embedding failed for {item.audio_path}: {type(exc).__name__}: {exc}") from exc
        embeddings_list.append(emb[0] if emb.ndim == 2 else np.reshape(emb, (-1,)))
    X = np.vstack(embeddings_list)
    y_health = np.asarray([i.health_label for i in items], dtype=int)
    y_anthro = np.asarray([i.anthro_label for i in items], dtype=int)
    if not (len(X) == len(y_health) == len(y_anthro) == len(items)):
        raise TrainingDataError("Embeddings and labels have different lengths")
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

    items = read_labels_csv(labels_csv, project_root)
    X, y_health, y_anthro = build_embeddings(items, model_path)

    clf = ReefClassifier(model_type=args.model_type)
    results = clf.train(X, y_health, y_anthro, test_size=CLASSIFIER_SETTINGS["test_size"])
    logger.info(f"Training results: {results}")

    output_dir.mkdir(parents=True, exist_ok=True)
    clf.save_model(str(output_dir))
    logger.info("Classifier saved successfully.")


if __name__ == "__main__":
    main()
