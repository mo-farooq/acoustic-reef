"""SurfPerch loading must fail loudly; never return placeholder embeddings."""
import builtins
import sys

import numpy as np
import pytest

from src.models import surfperch_integration as sp


def _model_dir(tmp_path):
    d = tmp_path / "surfperch"
    d.mkdir()
    (d / "saved_model.pb").write_bytes(b"x")
    return d


def test_module_has_no_placeholder_model():
    assert not hasattr(sp.SurfPerchModel, "_create_placeholder_model")
    assert not hasattr(sp, "TF_AVAILABLE")


def test_missing_model_path_raises(tmp_path):
    model = sp.SurfPerchModel(model_path=str(tmp_path / "nope"))
    with pytest.raises(sp.SurfPerchModelNotFoundError, match="not found"):
        model.load_model()
    assert model.is_loaded is False
    with pytest.raises(sp.SurfPerchModelNotFoundError):
        model.generate_embeddings(np.zeros(32000, dtype=np.float32))


def test_directory_without_saved_model_pb_raises(tmp_path):
    with pytest.raises(sp.SurfPerchModelNotFoundError):
        sp.SurfPerchModel(model_path=str(tmp_path)).load_model()


def test_tensorflow_unavailable_raises_without_embeddings(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "tensorflow", None)  # import raises ImportError
    model = sp.SurfPerchModel(model_path=str(_model_dir(tmp_path)))
    with pytest.raises(sp.SurfPerchUnavailableError, match="TensorFlow"):
        model.generate_embeddings(np.zeros(32000, dtype=np.float32))
    assert model.is_loaded is False and model.model is None


def test_initialisation_failure_raises_load_error(tmp_path, monkeypatch):
    class FakeTF:
        __version__ = "0.0-test"

        class saved_model:
            @staticmethod
            def load(path):
                raise ValueError("incompatible SavedModel")

    monkeypatch.setitem(sys.modules, "tensorflow", FakeTF)
    model = sp.SurfPerchModel(model_path=str(_model_dir(tmp_path)))
    with pytest.raises(sp.SurfPerchLoadError, match="incompatible SavedModel"):
        model.load_model()
    assert model.model is None and model.is_loaded is False


class _LoadedStub(sp.SurfPerchModel):
    """SurfPerchModel whose SavedModel call is replaced by a fixed output (software fixture)."""

    def __init__(self, output):
        super().__init__(model_path="unused")
        self._output = output
        self.is_loaded = True

        class Sig:
            def __call__(inner, inputs):
                return output

        class M:
            signatures = {"serving_default": Sig()}

        self.model = M()


@pytest.fixture
def fake_tf(monkeypatch):
    class FakeTF:
        float32 = np.float32

        @staticmethod
        def constant(a, dtype=None):
            return a

    monkeypatch.setitem(sys.modules, "tensorflow", FakeTF)


@pytest.mark.parametrize(
    "output",
    [
        {"other": np.zeros((1, 1280), dtype=np.float32)},
        {"embedding": np.zeros((1, 512), dtype=np.float32)},
        {"embedding": np.zeros((1280,), dtype=np.float32)},
        {"embedding": np.full((1, 1280), np.nan, dtype=np.float32)},
        {"embedding": np.zeros((2, 1280), dtype=np.float32)},
        "not a dict",
    ],
)
def test_invalid_model_output_rejected(fake_tf, output):
    model = _LoadedStub(output)
    with pytest.raises(sp.InvalidEmbeddingError):
        model.generate_embeddings(np.zeros(32000, dtype=np.float32))


def test_valid_output_passes_validation(fake_tf):
    out = {"embedding": np.ones((1, 1280), dtype=np.float32)}
    emb = _LoadedStub(out).generate_embeddings(np.zeros(32000, dtype=np.float32))
    assert emb.shape == (1, 1280)


def test_wrong_sample_rate_rejected():
    with pytest.raises(ValueError):
        _LoadedStub({}).generate_embeddings(np.zeros(100), sample_rate=16000)
