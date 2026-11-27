"""Dashboard inference/presentation logic (no Streamlit rendering)."""
from pathlib import Path

import numpy as np
import pytest

from src import dashboard_inference as di
from src.real_model_loader import (
    ClassifierNotFoundError,
    RealPredictionResult,
    map_class_to_outputs,
)

REPO = Path(__file__).resolve().parents[1]


def _result(cls):
    probs = {"anthrophony": 0.1, "degraded": 0.1, "healthy": 0.1}
    probs[cls] = 0.8
    return RealPredictionResult(feature_source="stored_random_forest", feature_dim=3, predicted_class=cls,
                                class_probabilities=probs, **map_class_to_outputs(cls, probs))


def test_real_success_is_identified_as_genuine():
    out = di.run_genuine_inference(np.zeros((1, 3)), "precomputed", predictor=lambda f: _result("healthy"))
    assert out.is_genuine and out.status == di.STATUS_GENUINE
    view = di.present_outcome(out)
    assert view["health_label"] == "Healthy" and view["health_score"] == "80%"
    assert "not a calibrated probability" in view["note"]
    assert di.method_banner(out)["kind"] == "success"


def test_load_failure_gives_unavailable_state_without_prediction():
    def boom(_):
        raise ClassifierNotFoundError("missing")

    out = di.run_genuine_inference(np.zeros((1, 3)), predictor=boom)
    assert out.status == di.STATUS_UNAVAILABLE and out.result is None and not out.is_genuine
    assert di.present_outcome(out) is None
    banner = di.method_banner(out)
    assert banner["kind"] == "error" and "not an ecological result" in banner["text"]


def test_other_failure_gives_error_state():
    def boom(_):
        raise ValueError("bad input")

    out = di.run_genuine_inference(np.zeros((1, 3)), predictor=boom)
    assert out.status == di.STATUS_ERROR and out.result is None
    assert di.present_outcome(out) is None


def test_anthrophony_health_is_not_assessed_not_degraded():
    out = di.run_genuine_inference(np.zeros((1, 3)), predictor=lambda f: _result("anthrophony"))
    view = di.present_outcome(out)
    assert view["health_label"] == "Not Assessed"
    assert view["health_label"] != "Degraded"
    assert view["health_score"] is None and view["noise_label"] == "High"


def test_missing_health_output_never_defaults_to_degraded():
    r = RealPredictionResult(health_label="Not Assessed", health_conf=None, noise_label="Not Assessed",
                             noise_conf=None, feature_source="x", feature_dim=1)
    view = di.present_outcome(di.InferenceOutcome(di.STATUS_GENUINE, "ok", result=r))
    assert view["health_label"] == "Not Assessed" and view["health_style"] == "info"


def test_no_mock_path_in_dashboard_sources():
    for rel in ("dashboard/app.py", "src/dashboard_inference.py", "src/inference.py"):
        text = (REPO / rel).read_text(encoding="utf-8")
        for banned in ("mock_classifier", "predict_with_mock", "force_real_classifier", "simple_inference",
                       "Heuristic Analysis"):
            assert banned not in text, (rel, banned)


def test_demo_label_is_explicit():
    assert di.DEMO_LABEL.startswith("DEMO")
    assert "DEMO" in (REPO / "dashboard/app.py").read_text(encoding="utf-8")


def test_real_inference_never_uses_random(monkeypatch):
    """Same input, same output: the genuine path is deterministic."""
    monkeypatch.setattr(np.random, "random", lambda *a, **k: (_ for _ in ()).throw(AssertionError("random used")))
    out1 = di.run_genuine_inference(np.zeros((1, 3)), predictor=lambda f: _result("degraded"))
    out2 = di.run_genuine_inference(np.zeros((1, 3)), predictor=lambda f: _result("degraded"))
    assert out1.result == out2.result
