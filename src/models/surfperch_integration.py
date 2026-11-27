"""
SurfPerch model integration for Acoustic Reef.

Loads the genuine Google SurfPerch TensorFlow SavedModel and produces audio
embeddings. This module never fabricates embeddings: if TensorFlow or the
model files are unavailable, loading raises a descriptive exception, and
every embedding returned by ``generate_embeddings`` has been validated
(2D, finite, expected dimension).
"""
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from src.utils import config

logger = logging.getLogger(__name__)

TARGET_SAMPLE_RATE = 32000
# SurfPerch SavedModel input length (samples at 32 kHz).
MODEL_INPUT_SAMPLES = 160000


class SurfPerchError(RuntimeError):
    """Base class for SurfPerch loading and inference errors."""


class SurfPerchUnavailableError(SurfPerchError):
    """TensorFlow is not importable, so the SavedModel cannot be used."""


class SurfPerchModelNotFoundError(SurfPerchError):
    """The SurfPerch SavedModel directory is missing or incomplete."""


class SurfPerchLoadError(SurfPerchError):
    """TensorFlow could not load the SavedModel (e.g. version mismatch)."""


class InvalidEmbeddingError(SurfPerchError):
    """The model output is not a valid embedding array."""


def _import_tensorflow() -> Any:
    """Import TensorFlow lazily. Raises SurfPerchUnavailableError if absent."""
    try:
        import tensorflow as tf  # type: ignore
    except Exception as exc:
        raise SurfPerchUnavailableError(
            "TensorFlow could not be imported "
            f"({type(exc).__name__}: {exc}). Genuine SurfPerch embeddings "
            "cannot be produced without TensorFlow (requirements.txt pins "
            "tensorflow==2.20.0). No placeholder embeddings are generated."
        ) from exc
    return tf


def _expected_embedding_dim() -> int:
    return int(config.SURFPERCH_SETTINGS.get("embedding_dim", 1280))


def validate_embeddings(embeddings: Any, batch_size: Optional[int] = None,
                        embedding_dim: Optional[int] = None) -> np.ndarray:
    """Return embeddings as a float32 array or raise InvalidEmbeddingError."""
    try:
        arr = np.asarray(embeddings, dtype=np.float32)
    except Exception as exc:
        raise InvalidEmbeddingError(f"Model output is not numeric: {exc}") from exc
    dim = _expected_embedding_dim() if embedding_dim is None else embedding_dim
    if arr.ndim != 2:
        raise InvalidEmbeddingError(f"Embedding must be 2D [batch, dim], got shape {arr.shape}")
    if arr.shape[0] == 0 or arr.shape[1] != dim:
        raise InvalidEmbeddingError(f"Expected embedding shape [n>0, {dim}], got {arr.shape}")
    if batch_size is not None and arr.shape[0] != batch_size:
        raise InvalidEmbeddingError(f"Expected batch size {batch_size}, got {arr.shape[0]}")
    if not np.all(np.isfinite(arr)):
        raise InvalidEmbeddingError("Embedding contains NaN or infinite values")
    return arr


class SurfPerchModel:
    """Wrapper around the genuine SurfPerch SavedModel."""

    def __init__(self, model_path: Optional[str] = None):
        """
        Args:
            model_path: Directory of the SurfPerch SavedModel. Defaults to
                config.SURFPERCH_SETTINGS["model_path"].
        """
        self.model_path = model_path or str(config.SURFPERCH_SETTINGS.get("model_path"))
        self.model = None
        self.is_loaded = False

    def load_model(self) -> bool:
        """
        Load the SavedModel.

        Returns True on success. Raises SurfPerchUnavailableError,
        SurfPerchModelNotFoundError or SurfPerchLoadError otherwise; it never
        substitutes a placeholder.
        """
        path = Path(self.model_path) if self.model_path else None
        if path is None or not path.is_dir() or not (path / "saved_model.pb").is_file():
            raise SurfPerchModelNotFoundError(
                f"SurfPerch SavedModel not found at '{self.model_path}' "
                "(expected a directory containing saved_model.pb and variables/)."
            )
        tf = _import_tensorflow()
        try:
            model = tf.saved_model.load(str(path))
        except Exception as exc:
            raise SurfPerchLoadError(
                f"tf.saved_model.load failed for '{path}' with TensorFlow "
                f"{getattr(tf, '__version__', 'unknown')}: {type(exc).__name__}: {exc}"
            ) from exc
        signatures = getattr(model, "signatures", None)
        if not signatures or "serving_default" not in signatures:
            raise SurfPerchLoadError(
                f"SavedModel at '{path}' has no 'serving_default' signature; "
                "it does not look like the expected SurfPerch export."
            )
        self.model = model
        self.is_loaded = True
        logger.info("SurfPerch model loaded from %s", path)
        return True

    def generate_embeddings(self, audio_data: np.ndarray, sample_rate: int = TARGET_SAMPLE_RATE) -> np.ndarray:
        """
        Generate validated embeddings from audio sampled at 32 kHz.

        Args:
            audio_data: 1D [time] or 2D [batch, time] float audio.
            sample_rate: Must be 32000 (use preprocess_audio to resample).

        Raises:
            SurfPerchError subclasses on load or output problems.
        """
        if sample_rate != TARGET_SAMPLE_RATE:
            raise ValueError(f"SurfPerch requires {TARGET_SAMPLE_RATE} Hz audio, got {sample_rate}")
        if not self.is_loaded:
            self.load_model()
        audio = np.asarray(audio_data, dtype=np.float32)
        if audio.size == 0:
            raise ValueError("audio_data is empty")
        if audio.ndim == 1:
            audio = audio[np.newaxis, :]
        if audio.ndim != 2:
            raise ValueError(f"audio_data must be 1D or 2D, got shape {audio.shape}")
        if audio.shape[1] < MODEL_INPUT_SAMPLES:
            audio = np.pad(audio, ((0, 0), (0, MODEL_INPUT_SAMPLES - audio.shape[1])), mode="constant")
        elif audio.shape[1] > MODEL_INPUT_SAMPLES:
            audio = audio[:, :MODEL_INPUT_SAMPLES]

        tf = _import_tensorflow()
        try:
            result = self.model.signatures["serving_default"](inputs=tf.constant(audio, dtype=tf.float32))
        except Exception as exc:
            raise SurfPerchError(f"SurfPerch inference call failed: {type(exc).__name__}: {exc}") from exc
        if not isinstance(result, dict) or "embedding" not in result:
            keys = list(result.keys()) if isinstance(result, dict) else type(result).__name__
            raise InvalidEmbeddingError(f"SurfPerch output has no 'embedding' entry (got {keys})")
        embedding = result["embedding"]
        if hasattr(embedding, "numpy"):
            embedding = embedding.numpy()
        return validate_embeddings(embedding, batch_size=audio.shape[0])

    def preprocess_audio(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        """Resample to 32 kHz, peak-normalise and pad to at least one second.

        Raises RuntimeError if resampling is required but librosa is missing.
        """
        audio_data = np.asarray(audio_data, dtype=np.float32)
        if audio_data.size == 0:
            raise ValueError("audio_data is empty")
        if sample_rate != TARGET_SAMPLE_RATE:
            try:
                import librosa  # type: ignore
            except Exception as exc:
                raise RuntimeError(
                    f"librosa is required to resample {sample_rate} Hz audio to {TARGET_SAMPLE_RATE} Hz"
                ) from exc
            audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=TARGET_SAMPLE_RATE)
        max_abs = np.max(np.abs(audio_data))
        if max_abs > 0:
            audio_data = audio_data / max_abs
        min_length = TARGET_SAMPLE_RATE  # 1 second
        if len(audio_data) < min_length:
            audio_data = np.pad(audio_data, (0, min_length - len(audio_data)), mode="constant")
        return audio_data

    def batch_generate_embeddings(self, audio_batch: np.ndarray, sample_rates: np.ndarray) -> np.ndarray:
        """Generate one validated embedding per audio array."""
        embeddings_list = []
        for audio, sr in zip(audio_batch, sample_rates):
            processed = self.preprocess_audio(audio, sr)
            embeddings = self.generate_embeddings(processed, TARGET_SAMPLE_RATE)
            embeddings_list.append(embeddings[0])
        return np.array(embeddings_list)

    def get_model_info(self) -> Dict[str, Any]:
        """Describe the wrapper state (is_loaded is True only for a real SavedModel)."""
        return {
            "is_loaded": self.is_loaded,
            "model_path": self.model_path,
            "model_type": "SurfPerch (TensorFlow SavedModel)",
            "embedding_dim": _expected_embedding_dim(),
            "target_sample_rate": TARGET_SAMPLE_RATE,
        }
