"""Classifier loading is strict; no fake classifier, no suppressed warnings, no random validity probes."""
import importlib
import warnings
from pathlib import Path

import joblib
import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from src import real_model_loader as rml

REPO = Path(__file__).resolve().parents[1]
STORED = REPO / "models" / "classifiers" / "reef_classifier_rf.joblib"


def _rf(classes=("anthrophony", "degraded", "healthy"), n_features=8):
    rng = np.random.default_rng(0)  # tiny synthetic software fixture only
    X = rng.normal(size=(30, n_features))
    y = np.array([classes[i % len(classes)] for i in range(30)])
    return RandomForestClassifier(n_estimators=5, random_state=0).fit(X, y)


def test_legacy_fallback_modules_removed():
    for name in ("src.mock_classifier", "src.force_real_classifier", "src.simple_inference"):
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(name)


def test_missing_model_fails_safely(tmp_path):
    with pytest.raises(rml.ClassifierNotFoundError):
        rml.load_classifier(tmp_path / "missing.joblib")
    with pytest.raises(rml.ClassifierUnavailableError):
        rml.predict_with_real_model(np.zeros((1, 8)), model_path=tmp_path / "missing.joblib")


def test_corrupt_model_fails_safely(tmp_path):
    p = tmp_path / "bad.joblib"
    p.write_bytes(b"not a pickle")
    with pytest.raises(rml.ClassifierIncompatibleError):
        rml.load_classifier(p)


def test_version_warning_makes_model_incompatible(tmp_path, monkeypatch):
    p = tmp_path / "m.joblib"
    joblib.dump(_rf(), p)
    real_load = joblib.load

    def noisy_load(path):
        warnings.warn("Trying to unpickle estimator from version 0.24 when using version 1.2.2 (sklearn)")
        return real_load(path)

    monkeypatch.setattr(joblib, "load", noisy_load)
    with pytest.raises(rml.ClassifierIncompatibleError, match="different scikit-learn version"):
        rml.load_classifier(p)


def test_wrong_classes_rejected(tmp_path):
    p = tmp_path / "m.joblib"
    joblib.dump(_rf(classes=("a", "b")), p)
    with pytest.raises(rml.ClassifierIncompatibleError, match="classes"):
        rml.load_classifier(p)


def test_non_estimator_rejected(tmp_path):
    p = tmp_path / "m.joblib"
    joblib.dump({"not": "a model"}, p)
    with pytest.raises(rml.ClassifierIncompatibleError):
        rml.load_classifier(p)


def test_no_fake_classifier_or_random_probe_in_source():
    src = Path(rml.__file__).read_text(encoding="utf-8")
    for banned in ("FallbackClassifier", "np.random", "random.", "simplefilter(\"ignore\")", "filterwarnings"):
        assert banned not in src


def test_wrong_feature_count_rejected():
    with pytest.raises(ValueError, match="expects 8"):
        rml.predict_with_real_model(np.zeros((1, 5)), model=_rf())


def test_empty_and_nonfinite_input_rejected():
    with pytest.raises(ValueError):
        rml.predict_with_real_model(np.array([]), model=_rf())
    with pytest.raises(ValueError):
        rml.predict_with_real_model(np.full((1, 8), np.nan), model=_rf())


def test_prediction_uses_model_classes_not_hardcoded_map():
    class Stub:
        classes_ = np.array(["anthrophony", "degraded", "healthy"])
        n_features_in_ = 3
        n_outputs_ = 1

        def predict_proba(self, X):
            return np.array([[0.1, 0.2, 0.7]])

    r = rml.predict_with_real_model(np.zeros((1, 3)), model=Stub())
    assert (r.predicted_class, r.health_label, r.noise_label) == ("healthy", "Healthy", "Low")
    assert r.health_conf == pytest.approx(0.7) and r.noise_conf == pytest.approx(0.9)


@pytest.mark.parametrize("cls,health,noise", [("degraded", "Degraded", "Low"), ("healthy", "Healthy", "Low"),
                                               ("anthrophony", "Not Assessed", "High")])
def test_anthrophony_is_never_degraded_health(cls, health, noise):
    probs = {"anthrophony": 0.1, "degraded": 0.1, "healthy": 0.1}
    probs[cls] = 0.8
    out = rml.map_class_to_outputs(cls, probs)
    assert out["health_label"] == health and out["noise_label"] == noise
    if cls == "anthrophony":
        assert out["health_conf"] is None


@pytest.mark.skipif(not STORED.exists(), reason="stored classifier not present")
def test_stored_classifier_loads_or_reports_incompatibility():
    """With the pinned scikit-learn it loads; with another version it must raise, never degrade."""
    try:
        model = rml.load_classifier(STORED)
    except rml.ClassifierIncompatibleError:
        return  # honest incompatibility report
    assert list(model.classes_) == ["anthrophony", "degraded", "healthy"]
    assert model.n_features_in_ == 1280
    r = rml.predict_with_real_model(np.zeros((1, 1280), dtype=np.float32), model=model)  # structural call only
    assert r.predicted_class in {"anthrophony", "degraded", "healthy"}
