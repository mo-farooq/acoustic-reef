"""
Single loading and prediction pathway for the stored reef classifier.

The stored artifact (models/classifiers/reef_classifier_rf.joblib) is a
single-output, three-class scikit-learn RandomForest with classes
``anthrophony``, ``degraded`` and ``healthy`` on 1280-d SurfPerch embeddings.

Rules enforced here:
- no warning suppression: a scikit-learn version-mismatch warning while
  unpickling makes the model "incompatible" and loading fails;
- no reconstruction hacks and no fallback classifier: loading either returns
  the genuine stored estimator or raises ClassifierUnavailableError;
- no random vectors are used to "prove" validity; only structural checks;
- anthrophony is a separate acoustic class. It is never converted into a
  "Degraded" reef-health status; health is "Not Assessed" for it.
"""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

EXPECTED_CLASSES = frozenset({"anthrophony", "degraded", "healthy"})
HEALTH_NOT_ASSESSED = "Not Assessed"
NOISE_NOT_ASSESSED = "Not Assessed"
FEATURE_SOURCE = "stored_random_forest"


class ClassifierUnavailableError(RuntimeError):
    """The genuine classifier cannot be used. Never substitute another model."""


class ClassifierNotFoundError(ClassifierUnavailableError):
    """The classifier file does not exist."""


class ClassifierIncompatibleError(ClassifierUnavailableError):
    """The classifier loaded with warnings or has an unexpected structure."""


@dataclass(frozen=True)
class RealPredictionResult:
    """Output of the stored classifier mapped to dashboard fields.

    ``*_conf`` values are RandomForest class-vote fractions. They are NOT
    calibrated probabilities.
    """
    health_label: str
    health_conf: Optional[float]
    noise_label: str
    noise_conf: Optional[float]
    feature_source: str
    feature_dim: int
    predicted_class: str = ""
    class_probabilities: Dict[str, float] = field(default_factory=dict)


def default_model_path() -> Path:
    from src.utils.config import RF_MODEL_PATH
    return Path(RF_MODEL_PATH)


def _is_version_warning(w: warnings.WarningMessage) -> bool:
    text = str(w.message).lower()
    return (
        type(w.message).__name__ == "InconsistentVersionWarning"
        or ("version" in text and ("sklearn" in text or "scikit-learn" in text))
    )


def load_classifier(model_path: Optional[Path] = None) -> Any:
    """Load the stored classifier or raise ClassifierUnavailableError.

    Raises:
        ClassifierNotFoundError: file missing.
        ClassifierIncompatibleError: unpickling failed, produced a version
            warning, or the estimator lacks the expected structure.
    """
    path = Path(model_path) if model_path is not None else default_model_path()
    if not path.is_file():
        raise ClassifierNotFoundError(f"Classifier file not found: {path}")
    try:
        import joblib
    except Exception as exc:
        raise ClassifierUnavailableError(f"joblib is not importable: {exc}") from exc

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            model = joblib.load(path)
        except Exception as exc:
            raise ClassifierIncompatibleError(
                f"Could not load classifier {path}: {type(exc).__name__}: {exc}"
            ) from exc
    version_warnings = [str(w.message) for w in caught if _is_version_warning(w)]
    if version_warnings:
        raise ClassifierIncompatibleError(
            f"Classifier {path} was serialised with a different scikit-learn version "
            f"than the one installed; refusing to use it: {version_warnings[0]}"
        )
    for w in caught:  # other warnings are not hidden
        logger.warning("Warning while loading %s: %s", path, w.message)
    _check_structure(model, path)
    return model


def _check_structure(model: Any, path: Path) -> None:
    for attr in ("predict", "predict_proba", "classes_", "n_features_in_"):
        if not hasattr(model, attr):
            raise ClassifierIncompatibleError(f"Classifier {path} has no '{attr}'")
    classes = {str(c) for c in model.classes_}
    if classes != EXPECTED_CLASSES:
        raise ClassifierIncompatibleError(
            f"Classifier {path} has classes {sorted(classes)}; expected {sorted(EXPECTED_CLASSES)}. "
            "Class semantics were not verified, so the model is not used."
        )
    if int(getattr(model, "n_outputs_", 1)) != 1:
        raise ClassifierIncompatibleError(f"Classifier {path} is multi-output; expected single-output")


def _normalize_feature_vals(feature_vals: np.ndarray) -> np.ndarray:
    if feature_vals is None or np.size(feature_vals) == 0:
        raise ValueError("feature_vals is empty; cannot run prediction.")
    arr = np.asarray(feature_vals, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2 or arr.shape[0] != 1:
        raise ValueError(f"feature_vals must be one embedding [1, D], got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("feature_vals contains NaN or infinite values")
    return arr


def map_class_to_outputs(predicted_class: str, class_probabilities: Dict[str, float]) -> Dict[str, Any]:
    """Map the single predicted class to separate health and noise outputs.

    healthy -> health Healthy; degraded -> health Degraded;
    anthrophony -> health "Not Assessed" (the model cannot assess reef health
    for an anthrophony-dominated clip). Noise is High only for anthrophony;
    otherwise "Low" means "not predicted as anthrophony".
    """
    p_anthro = class_probabilities.get("anthrophony")
    if predicted_class == "healthy":
        health, health_conf = "Healthy", class_probabilities.get("healthy")
    elif predicted_class == "degraded":
        health, health_conf = "Degraded", class_probabilities.get("degraded")
    else:
        health, health_conf = HEALTH_NOT_ASSESSED, None
    if predicted_class == "anthrophony":
        noise, noise_conf = "High", p_anthro
    elif predicted_class in ("healthy", "degraded"):
        noise, noise_conf = "Low", (None if p_anthro is None else 1.0 - p_anthro)
    else:
        noise, noise_conf = NOISE_NOT_ASSESSED, None
    return {"health_label": health, "health_conf": health_conf,
            "noise_label": noise, "noise_conf": noise_conf}


def predict_with_real_model(feature_vals: np.ndarray, model: Any = None,
                            model_path: Optional[Path] = None) -> RealPredictionResult:
    """Run the stored classifier on one embedding.

    Raises ClassifierUnavailableError if the model cannot be used and
    ValueError for invalid input. Never returns a fallback result.
    """
    features = _normalize_feature_vals(feature_vals)
    clf = model if model is not None else load_classifier(model_path)
    n_expected = int(clf.n_features_in_)
    if features.shape[1] != n_expected:
        raise ValueError(f"Classifier expects {n_expected} features, got {features.shape[1]}")
    proba = np.asarray(clf.predict_proba(features))[0]
    classes = [str(c) for c in clf.classes_]
    class_probabilities = {c: float(p) for c, p in zip(classes, proba)}
    predicted_class = classes[int(np.argmax(proba))]
    outputs = map_class_to_outputs(predicted_class, class_probabilities)
    return RealPredictionResult(
        feature_source=FEATURE_SOURCE,
        feature_dim=int(features.shape[1]),
        predicted_class=predicted_class,
        class_probabilities=class_probabilities,
        **outputs,
    )
