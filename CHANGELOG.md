# Changelog

## Unreleased: inference reliability and research safety (P0)

### Removed
- `src/mock_classifier.py`, `src/force_real_classifier.py`, `src/simple_inference.py`: heuristic/random classifiers, a fabricated fallback classifier and warning-suppressing loader hacks.
- SurfPerch placeholder model that returned random embeddings.
- Dummy-row / random-embedding branch in `generate_kaggle_notebook.py` and the generated notebook.
- Positional label alignment fallback in `align_embeddings_and_labels`.

### Changed
- `src/models/surfperch_integration.py`: loading raises typed errors (`SurfPerchUnavailableError`, `SurfPerchModelNotFoundError`, `SurfPerchLoadError`, `InvalidEmbeddingError`); output is validated.
- `src/real_model_loader.py`: single strict classifier loader and predictor; scikit-learn version warnings make the model incompatible instead of being suppressed.
- `src/inference.py`: delegates to the single loader; no fallback result on failure.
- `dashboard/app.py`: uses `src/dashboard_inference.py`; shows an unavailable/error state instead of mock predictions; anthrophony yields health "Not Assessed" instead of "Degraded"; landing-page sample numbers are labelled DEMO; scores are labelled uncalibrated vote shares; Dataset view joins on `filepath`.
- `src/models/train_classifier.py`, `generate_kaggle_notebook.py`, `notebooks/kaggle_surfperch_training.ipynb`: fail on missing labels, missing audio, empty data, SurfPerch failure; save only after successful training.
- `load_embeddings_from_csv`: embedding-dimension mismatch is an error.

### Added
- `src/dashboard_inference.py` (testable inference/presentation logic), `tests/`, `pytest.ini`, `requirements-dev.txt`, `docs/inference-safety.md`.

### Not changed
- Model assets, embeddings, UMAP artifacts.
- Random row-level split and ordinary CV in `ReefClassifier.train` (P1).
