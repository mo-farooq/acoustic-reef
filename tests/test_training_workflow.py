"""Training pipeline guards. Tiny synthetic arrays are software fixtures only."""
import json
import re
import wave
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.models import train_classifier as tc
from src.models.reef_classifier import LabelAlignmentError, ReefClassifier, align_embeddings_and_labels
from src.models.surfperch_integration import SurfPerchUnavailableError

REPO = Path(__file__).resolve().parents[1]


def _wav(path):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(32000)
        w.writeframes(np.zeros(3200, dtype=np.int16).tobytes())


def _csv(path, rows):
    path.write_text("filepath,health_label,anthro_label\n" + "".join(f"{r}\n" for r in rows), encoding="utf-8")


def test_missing_labels_file_fails(tmp_path):
    with pytest.raises(FileNotFoundError):
        tc.read_labels_csv(tmp_path / "labels.csv", tmp_path)


def test_empty_dataset_fails(tmp_path):
    _csv(tmp_path / "labels.csv", [])
    with pytest.raises(tc.TrainingDataError, match="empty dataset"):
        tc.read_labels_csv(tmp_path / "labels.csv", tmp_path)
    with pytest.raises(tc.TrainingDataError):
        tc.build_embeddings([], tmp_path)


def test_missing_audio_fails_and_is_reported(tmp_path):
    _wav(tmp_path / "a.wav")
    _csv(tmp_path / "labels.csv", ["a.wav,1,0", "gone.wav,0,1"])
    with pytest.raises(FileNotFoundError, match="gone.wav"):
        tc.read_labels_csv(tmp_path / "labels.csv", tmp_path)


def test_malformed_rows_fail(tmp_path):
    _wav(tmp_path / "a.wav")
    _csv(tmp_path / "labels.csv", ["a.wav,2,0"])
    with pytest.raises(tc.TrainingDataError):
        tc.read_labels_csv(tmp_path / "labels.csv", tmp_path)
    _csv(tmp_path / "labels.csv", ["a.wav,1,0", "a.wav,0,0"])
    with pytest.raises(tc.TrainingDataError, match="duplicate"):
        tc.read_labels_csv(tmp_path / "labels.csv", tmp_path)


def test_training_without_surfperch_fails_instead_of_placeholder(tmp_path, monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "tensorflow", None)
    (tmp_path / "sp").mkdir()
    (tmp_path / "sp" / "saved_model.pb").write_bytes(b"x")
    _wav(tmp_path / "a.wav")
    items = [tc.LabeledItem(tmp_path / "a.wav", 1, 0)]
    with pytest.raises(SurfPerchUnavailableError):
        tc.build_embeddings(items, tmp_path / "sp")


def test_embedding_failure_aborts_not_skipped(tmp_path):
    class Broken:
        is_loaded = True

        def preprocess_audio(self, a, sr):
            return a

        def generate_embeddings(self, a, sr):
            raise RuntimeError("boom")

    _wav(tmp_path / "a.wav")
    with pytest.raises(RuntimeError, match="Embedding failed"):
        tc.build_embeddings([tc.LabeledItem(tmp_path / "a.wav", 1, 0)], tmp_path, model=Broken())


def test_trainer_rejects_nonfinite_embeddings():
    X = np.full((10, 4), np.nan)
    with pytest.raises(ValueError, match="NaN"):
        ReefClassifier().train(X, np.zeros(10, dtype=int), np.zeros(10, dtype=int))
    with pytest.raises(ValueError):
        ReefClassifier().train(np.zeros((0, 4)), np.zeros(0), np.zeros(0))


def test_alignment_rejects_missing_key_instead_of_positional():
    emb = pd.DataFrame({"embedding_0": [0.1, 0.2], "embedding_1": [0.3, 0.4]})
    lab = pd.DataFrame({"label": ["healthy", "degraded"]})
    with pytest.raises(LabelAlignmentError, match="Positional alignment"):
        align_embeddings_and_labels(emb, lab)


def test_alignment_rejects_incomplete_or_duplicate_join():
    emb = pd.DataFrame({"filepath": ["a", "b", "c"], "embedding_0": [1.0, 2.0, 3.0]})
    lab = pd.DataFrame({"filepath": ["a", "b"], "label": ["x", "y"]})
    with pytest.raises(LabelAlignmentError, match="incomplete"):
        align_embeddings_and_labels(emb, lab, key_columns=["filepath"])
    dup = pd.DataFrame({"filepath": ["a", "a", "b"], "label": ["x", "x", "y"]})
    with pytest.raises(LabelAlignmentError, match="not unique"):
        align_embeddings_and_labels(emb, dup, key_columns=["filepath"])


def test_alignment_with_key_uses_embedding_columns_only():
    emb = pd.DataFrame({"filepath": ["b", "a"], "category": ["q", "p"], "embedding_0": [2.0, 1.0],
                        "embedding_1": [4.0, 3.0]})
    lab = pd.DataFrame({"filepath": ["a", "b"], "category": ["healthy", "degraded"], "is_healthy": [1, 0]})
    X, y, merged = align_embeddings_and_labels(emb, lab, key_columns=["filepath"], label_columns=["category"])
    assert X.shape == (2, 2) and list(merged["filepath"]) == ["b", "a"]
    assert list(y[:, 0]) == ["degraded", "healthy"]  # label follows the key, not the row position
    assert X[0].tolist() == [2.0, 4.0]


def test_generator_and_notebook_have_no_synthetic_data():
    gen = (REPO / "generate_kaggle_notebook.py").read_text(encoding="utf-8")
    nb = (REPO / "notebooks" / "kaggle_surfperch_training.ipynb").read_text(encoding="utf-8")
    for text in (gen, nb):
        assert "np.random" not in text
        assert not re.search(r"dummy_(healthy|degraded)", text)
        assert "Creating dummy data" not in text
        assert "FileNotFoundError" in text and "empty dataset" in text
    # the committed notebook is exactly what the generator emits
    src = [c["source"] for c in json.loads(nb)["cells"] if c["cell_type"] == "code"]
    assert any("raise FileNotFoundError" in "".join(s) for s in src)


def test_training_source_has_no_placeholder_reference():
    for rel in ("src/models/train_classifier.py", "src/models/surfperch_integration.py"):
        text = (REPO / rel).read_text(encoding="utf-8")
        assert "np.random" not in text and "tf.random" not in text
        assert "PlaceholderModel" not in text
