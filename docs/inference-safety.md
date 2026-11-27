# Inference safety (P0)

This document describes what the code does after the `fix/inference-safety` changes. It is engineering documentation for the prototype. It reports no research results, and prototype outputs are not validated ecological predictions.

## Modes

| Mode | What it is | Where |
|---|---|---|
| Genuine inference | The stored RandomForest (`models/classifiers/reef_classifier_rf.joblib`) classifies an embedding. The embedding is either a precomputed row from `data/embeddings/embeddings.csv` or a genuine SurfPerch embedding computed at runtime. | Upload and batch tabs of the dashboard |
| Unavailable / error | The classifier or the SurfPerch encoder cannot be used. The dashboard shows an error and **no prediction**. | Same tabs |
| DEMO | Static, hard-coded example numbers on the landing page, labelled DEMO. They are not results for any recording. | Landing page "Sample Results (DEMO)" |

There is no heuristic, mock or random fallback mode. `src/mock_classifier.py`, `src/force_real_classifier.py` and `src/simple_inference.py` were removed (still in git history). `src/real_model_loader.py` is the only classifier loading and prediction path.

## Requirements for genuine inference

- `models/classifiers/reef_classifier_rf.joblib` with `scikit-learn==1.2.2` (the version recorded inside the file and pinned in `requirements.txt`).
- For recordings not present in `embeddings.csv`: TensorFlow (`tensorflow==2.20.0` pinned) and the SavedModel in `models/surfperch/surfperch-tensorflow2-1-v1/`. Whether that SavedModel loads under the pinned TensorFlow has not been verified in this environment (TensorFlow was not installed when this change was written).

## What happens when something cannot load

- Missing or corrupt classifier, class set other than `{anthrophony, degraded, healthy}`, multi-output model, or **any scikit-learn version warning during unpickling** raises `ClassifierUnavailableError` subclasses. Warnings are not suppressed and nothing is reconstructed.
- Missing TensorFlow, missing SavedModel files, a failing `tf.saved_model.load`, or invalid model output (wrong shape, NaN/inf, no `embedding` key) raise `SurfPerchError` subclasses with diagnostics.
- The dashboard turns these into an "unavailable / inference failed" panel that states this is a model failure, not an ecological result.

## Supported outputs

The stored model is single-output with three classes. The mapping is:

| Predicted class | Reef health | Anthrophony |
|---|---|---|
| healthy | Healthy | Low (not predicted as anthrophony) |
| degraded | Degraded | Low (not predicted as anthrophony) |
| anthrophony | **Not Assessed** | High |

Anthrophony is never converted into degraded health. No separate health model exists, so for anthrophony clips reef health is unknown. Scores shown are random-forest vote shares, not calibrated probabilities. The earlier floor of 0.1 on the non-anthrophony score was removed.

## Training safeguards

`src/models/train_classifier.py`, `generate_kaggle_notebook.py` and the generated `notebooks/kaggle_surfperch_training.ipynb` now fail on: missing labels CSV, missing required columns, duplicate or malformed rows, missing audio files, empty dataset, SurfPerch load failure, any embedding failure. No dummy rows, random embeddings or placeholder encoder are used, and the model is saved only after training completes. `align_embeddings_and_labels` requires an explicit unique key (`clip_id` or `key_columns`), requires a complete one-to-one join and uses only embedding columns as features. Positional alignment was removed.

The dashboard Dataset view now passes `key_columns=['filepath']`. With the shipped CSVs both tables have 2194 unique filepaths and the join is complete, and the aligned features equal the old positional ones; the join is now verified instead of assumed.

## Running tests

```
pip install -r requirements-dev.txt
python -m pytest -q
```

Tests need neither TensorFlow nor the SurfPerch download. Small synthetic arrays are software fixtures only.

## Known issues and remaining work

- **Evaluation split (P1):** `ReefClassifier.train` and the notebook still use a random row-level split and ordinary cross-validation. Their scores do not estimate generalisation to unseen sites, recorders or time periods.
- **Provenance:** the training data, date and labels behind the stored classifier and `embeddings.csv` are not documented in the repo. Whether the embeddings are genuine SurfPerch output is unverified here.
- **Label definitions:** how `healthy`, `degraded` and `anthrophony` were assigned is undocumented; they may encode recording location rather than ecological state.
- **Other dashboard text:** insight text generated in `src/utils/acoustic_map_enhancements.py` and static explanatory text in `dashboard/app.py` still use "healthy/degraded reef" wording; it was not audited line by line.
- **Docs from PR #1** (`docs/prototype-audit.md`, `docs/architecture-and-scope.md`, `docs/research-status.md`, `README.md`) are not on `main`. After PR #1 merges they need a follow-up edit to describe the changes above.
- **Not done:** Lizard Island training pipeline, grouped (site/recorder/time) evaluation and calibration are out of scope.
