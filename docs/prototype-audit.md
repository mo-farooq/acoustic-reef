# Prototype code audit

**Audit date:** 2026-11-24  
**Scope:** Static review of the public `main` branch source and tracked repository structure.  
**Execution status:** No dependencies were installed; the Streamlit app, notebooks and training scripts were not executed. No models were retrained and no audio was accessed.

This document reviews the original hackathon/prototype implementation. It is not a review of E0/E1 audit scripts in the private research workspace.

## Executive assessment

The repository contains a substantial prototype: a Streamlit dashboard, SurfPerch integration, classifier utilities, notebooks, precomputed embeddings, serialized classifiers and UMAP artifacts. The most important improvement is not a cosmetic rewrite. It is to make the boundary between **prototype/demo inference** and **scientifically validated research** explicit, and to prevent synthetic fallback paths from being mistaken for model predictions.

The original prototype is preserved. Findings below are recommendations for subsequent reviewed changes; no prototype source code was altered during this audit.

## Findings

### P0 — Synthetic fallback paths can produce biological-looking predictions

**Evidence inspected**
- `src/models/surfperch_integration.py`: when the SavedModel path is missing or TensorFlow is unavailable, `load_model()` creates a placeholder model. The placeholder generates random embeddings.
- `src/mock_classifier.py`: classifies using hand-written embedding-statistic thresholds, adds a 30% random label-flip chance, and returns hard-coded confidence values.
- `dashboard/app.py`: if the primary model inference raises an exception, the dashboard falls back to the mock classifier and displays reef-health/noise labels and confidence metrics.
- `src/models/train_classifier.py`: instantiates the SurfPerch wrapper without a strict assertion that the genuine pretrained model was loaded. The wrapper can therefore enter its placeholder path if model assets/dependencies are unavailable.

**Risk**  
A user can see plausible-looking health labels and confidence values that do not come from a verified SurfPerch model and validated classifier. The same placeholder path could contaminate a training run if not blocked.

**Recommendation**  
Introduce an explicit demo mode with persistent UI labelling and no export/report wording that implies diagnosis. Research and production inference should fail closed if the real encoder or classifier cannot be loaded. Training must assert model provenance and reject placeholder embeddings. Never report mock confidence as calibrated probability.

### P0 — The prototype notebook can create synthetic training examples

**Evidence inspected**
- `generate_kaggle_notebook.py` emits a Kaggle notebook that creates dummy rows if the labels CSV is absent and generates random embeddings for filenames marked as dummy. It then proceeds into classifier training and evaluation cells.

**Risk**  
Running the notebook without a real labelled dataset can produce meaningless model metrics that look like experimental results.

**Recommendation**  
Remove the synthetic-data-to-training path from any research notebook. If a demo dataset is needed, label it synthetic in every cell/output and prevent it from being used for model evaluation. Require an existing labels file and verified audio paths for research runs.

### P1 — Existing classifier evaluation is not suitable for the new research question

**Evidence inspected**
- `src/models/reef_classifier.py`: `train()` uses a random `train_test_split` and ordinary five-fold `cross_val_score`.
- `src/models/train_classifier.py`: the training path calls this method and passes no site, recorder, deployment or time grouping metadata.
- The generated Kaggle notebook also uses a random row-level split.

**Risk**  
Adjacent clips or recordings from the same site/deployment can occur on both sides of a random split. This does not estimate generalisation to unseen sites, recorders or recording periods.

**Recommendation**  
Do not reuse these prototype scores for the research paper. Build a separate research evaluation pipeline with explicit groups and frozen split manifests only after E1 data-integrity questions and supervisor decisions are resolved. Keep site as the inferential unit; clips are not independent site replicates.

### P1 — Feature/label alignment has an unsafe positional fallback

**Evidence inspected**
- `align_embeddings_and_labels()` in `src/models/reef_classifier.py` uses `clip_id` if available, but otherwise falls back to joining rows by position.

**Risk**  
Different row ordering, missing rows or independently filtered tables can silently pair embeddings with incorrect labels.

**Recommendation**  
Research pipelines should require a stable unique recording/clip key, validate one-to-one joins, report unmatched and duplicate keys, and fail rather than silently align by row position.

### P1 — Prototype inference includes assumptions that should not become ecological claims

**Evidence inspected**
- `src/inference.py` and `src/real_model_loader.py` map an `anthrophony` classifier output to a `Degraded` health label.
- The dashboard labels the output as reef “vital signs”.

**Risk**  
Anthropogenic noise presence and reef habitat condition are different target concepts. The mapping is a prototype assumption, not evidence that a noisy recording means a degraded reef.

**Recommendation**  
Keep anthrophony as a separate acoustic target. Do not infer reef health from anthrophony without an independently justified label definition and ecological validation. Present the existing dashboard as an exploratory prototype, not a validated diagnostic.

### P1 — Data/model artifacts are tracked in the application repository

**Evidence inspected**
- `data/embeddings/embeddings.csv` is approximately 34.3 MB.
- The tracked SurfPerch SavedModel variable file is approximately 97.0 MB.
- Two tracked UMAP model artifacts are approximately 11.7 MB each.
- GitHub reports a repository size field of 803,270 KB.

**Risk**  
Large generated artifacts make clones and maintenance heavier, and their provenance/licensing/version compatibility are not self-evident from filenames.

**Recommendation**  
Add an artifact inventory with source, licence, generation command, model/library versions and checksum. Decide which artifacts genuinely need to remain in Git; consider external model/data hosting or Git LFS only after checking licences, repository history impact and quota/cost. Do not delete existing tracked assets during this documentation pass.

### P2 — Dashboard is a large monolithic module

**Evidence inspected**
- `dashboard/app.py` is approximately 124 KB and handles UI, upload flow, feature resolution, prediction presentation, environmental lookups and visualisations.

**Recommendation**  
Refactor incrementally into UI pages/components and service modules, with tests around boundaries. Avoid a high-risk rewrite before baseline functionality has been captured.

### P2 — Test coverage is not yet a reproducible automated suite

**Evidence inspected**
- Root contains `test_inference.py`, which is a command-line smoke-check script that loads the first embedding row and performs a prediction.
- No top-level `tests/` directory or pytest-style `test_*` functions were visible in the inspected repository tree.
- README currently describes running `pytest`.

**Recommendation**  
Add a proper test suite for filename parsing, archive inventory validation, unique-key joins, split-manifest integrity, model-loading fail-closed behavior and dashboard service functions. Keep integration tests requiring large model assets opt-in.

### P2 — Dependency and artifact compatibility needs an explicit environment record

**Evidence inspected**
- `requirements.txt` pins TensorFlow 2.20.0 and scikit-learn 1.2.2 while several other dependencies use lower bounds/ranges.
- Serialized scikit-learn models are committed.
- `check_sklearn.py` only prints the installed scikit-learn version; it does not verify compatibility against the serialized model's training environment.

**Recommendation**  
Record the environment used to produce each serialized artifact, add a lockfile or tested constraints file, and validate load/predict in CI. Treat model deserialization as trusted-artifact-only; do not load arbitrary user-supplied pickle/joblib files.

## Recommended implementation order

1. **Safety and honesty:** clearly label the existing Streamlit application as a prototype; prevent mock/random fallback output from being mistaken for validated prediction.
2. **Training guardrails:** assert real SurfPerch model loading and reject placeholder embeddings in any training path.
3. **Research separation:** implement a new `experiments/` pipeline; do not reuse random-split prototype evaluation.
4. **Data integrity:** require stable IDs and strict one-to-one embedding/label joins; preserve duplicate evidence.
5. **Tests:** add unit tests for metadata parsing, duplicate handling, data joins, grouping and model-loading failures.
6. **Reproducibility:** document artifact provenance, dependencies, commands, checksums and limitations.
7. **Refactor UI:** split the large dashboard module only after tests protect existing behaviour.

## What this audit does not establish

- It does not establish the performance or validity of the existing classifier.
- It does not verify the training provenance of committed model/embedding artifacts.
- It does not establish that a model was trained on the Williams et al. public dataset.
- It does not validate the E0/E1 metadata audit implementation; that requires the corresponding scripts, logs and final reports to be reviewed.
- It does not include a runtime test, vulnerability scan, installation test or full history/binary audit.

These limits should remain visible until those checks are performed.
