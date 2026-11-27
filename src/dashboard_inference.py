"""
Streamlit-free inference and presentation logic for the dashboard.

dashboard/app.py calls these functions and only renders their output, so the
behaviour can be unit tested without Streamlit, TensorFlow or SurfPerch.

An outcome is "genuine" only when the stored classifier loaded and produced a
prediction. Otherwise the outcome is "unavailable" (model cannot be used) or
"error" (the request failed) and carries no prediction at all. There is no
heuristic or mock fallback.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Dict, Optional

import numpy as np

from src.real_model_loader import (
    ClassifierUnavailableError,
    RealPredictionResult,
    predict_with_real_model,
)

logger = logging.getLogger(__name__)

STATUS_GENUINE = "genuine"
STATUS_UNAVAILABLE = "unavailable"
STATUS_ERROR = "error"

DEMO_LABEL = "DEMO - static example, not a result for your recording"
CONFIDENCE_NOTE = (
    "Score = share of random-forest trees voting for the class. "
    "It is not a calibrated probability."
)


@dataclass(frozen=True)
class InferenceOutcome:
    """Result of one inference attempt."""
    status: str
    message: str
    result: Optional[RealPredictionResult] = None
    embedding_source: str = ""

    @property
    def is_genuine(self) -> bool:
        return self.status == STATUS_GENUINE and self.result is not None


def run_genuine_inference(
    feature_vals: np.ndarray,
    embedding_source: str = "",
    predictor: Callable[[np.ndarray], RealPredictionResult] = predict_with_real_model,
) -> InferenceOutcome:
    """Classify one embedding with the stored model; never fabricate a result."""
    try:
        result = predictor(feature_vals)
    except ClassifierUnavailableError as exc:
        logger.error("Classifier unavailable: %s", exc)
        return InferenceOutcome(STATUS_UNAVAILABLE, f"Trained classifier unavailable: {exc}",
                                embedding_source=embedding_source)
    except Exception as exc:  # reported to the user as an error state, never as a prediction
        logger.exception("Inference failed")
        return InferenceOutcome(STATUS_ERROR, f"Inference failed: {type(exc).__name__}: {exc}",
                                embedding_source=embedding_source)
    return InferenceOutcome(STATUS_GENUINE, "Prediction from the stored trained classifier.",
                            result=result, embedding_source=embedding_source)


def method_banner(outcome: InferenceOutcome) -> Dict[str, str]:
    """Banner (kind in success/error) describing where the output comes from."""
    if outcome.is_genuine:
        text = "Stored trained RandomForest classifier on SurfPerch-style embeddings."
        if outcome.embedding_source:
            text += f" Embedding source: {outcome.embedding_source}."
        text += " Prototype output; not validated for ecological use."
        return {"kind": "success", "text": text}
    return {
        "kind": "error",
        "text": f"No prediction was produced. {outcome.message} "
                "This is a model/inference failure, not an ecological result.",
    }


def _health_style(label: str) -> str:
    if label == "Healthy":
        return "success"
    if label == "Degraded":
        return "error"
    return "info"


def _noise_style(label: str) -> str:
    if label == "High":
        return "error"
    if label == "Low":
        return "success"
    return "info"


def _score_text(score: Optional[float]) -> Optional[str]:
    return None if score is None else f"{score:.0%}"


def present_outcome(outcome: InferenceOutcome) -> Optional[Dict[str, Optional[str]]]:
    """Display fields for the vital-signs panel, or None when there is no prediction."""
    if not outcome.is_genuine:
        return None
    r = outcome.result
    return {
        "health_label": r.health_label,
        "health_style": _health_style(r.health_label),
        "health_score": _score_text(r.health_conf),
        "noise_label": r.noise_label,
        "noise_style": _noise_style(r.noise_label),
        "noise_score": _score_text(r.noise_conf),
        "predicted_class": r.predicted_class,
        "note": CONFIDENCE_NOTE,
    }
