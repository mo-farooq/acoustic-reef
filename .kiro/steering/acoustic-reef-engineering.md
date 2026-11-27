---
inclusion: auto
name: acoustic-reef-engineering
description: Engineering guidance for changes to Acoustic Reef Python code, the Streamlit dashboard, model wrappers, notebooks, research pipeline, data tooling, tests, or GitHub automation.
---

# Acoustic Reef engineering conventions

## Keep prototype and research paths separate

The legacy dashboard, prototype model wrappers, stored embeddings, and serialized classifier are historical prototype assets. Do not silently repurpose them as the research pipeline. New research code should have an explicit research namespace and documented inputs/outputs.

## Fail closed for model provenance

- Do not silently substitute random embeddings, mock classifiers, placeholder models, or synthetic samples when real assets or inputs are missing.
- A fallback intended for a demo must be visibly labelled as a demo in the UI and output metadata.
- Research and evaluation paths must stop with a clear error when the expected real model, data, label mapping, or provenance record is unavailable.
- Never call a mock confidence value a calibrated probability.

## Data and identity handling

- Preserve source filenames, archive identifiers, CRCs, and original metadata.
- Join records only using explicit, verified identifiers. Never fall back to positional row matching.
- Do not silently deduplicate possible cross-site matches.
- Keep site, deployment, recorder-token, timestamp, and publication-derived labels as separate fields until their relationships are verified.
- Store source, transformation, software version, and validation outcome with derived artifacts.

## Evaluation and leakage prevention

- Avoid random clip-level train/test splits for claims about unseen sites, recorders, or time periods.
- Require explicit group IDs and a documented split rationale.
- Put scaling, imputation, feature selection, and every other learned preprocessing step inside the training fold (for example, a scikit-learn Pipeline).
- Do not tune thresholds or select models using the final held-out test group.
- Add tests that assert groups do not overlap across train/validation/test partitions and that split construction fails on unresolved or missing identities.

## Implementation and testing

- Prefer small modules with typed interfaces, deterministic outputs, explicit errors, and concise docstrings.
- Keep notebooks exploratory; move reusable logic into tested Python modules.
- Add focused tests for edge cases and failure paths. Synthetic fixtures are acceptable for unit tests only when clearly named and never written into empirical-results folders.
- Do not download TensorFlow/SoundTrap models, datasets, or large checkpoints as part of routine tests.
- Avoid installing all of `requirements.txt` for a small code change; it includes heavyweight machine-learning and notebook dependencies. Identify the minimum dependency set needed.
- Do not modify large binary artifacts or data files unless a task explicitly requires it and the change has been reviewed.

## Pull requests

- Work on a feature branch from the current default branch.
- Include a concise problem statement, the rationale, files changed, exact verification commands and their outcomes, and explicit limitations.
- Never report tests as passed when they were not executed.
- Do not merge or enable auto-merge. Leave final approval and merge decisions to the repository maintainers.
