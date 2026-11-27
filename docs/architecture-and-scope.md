# Architecture and scope

## Existing prototype

The current codebase contains a Streamlit application, SurfPerch integration, scikit-learn classifier utilities, exploratory UMAP/acoustic-map utilities, notebooks and stored artifacts. These were developed in the original hackathon/prototype phase and are retained for continuity.

## Research workspace separation

The research workflow is distinct from the prototype dashboard. Prototype inference and stored embeddings/models must not be presented as results from the E0/E1 public-dataset audit or as validation of the new research question.

## Current research pipeline (planned)

1. Archive inventory and provenance verification (E0/E1 completed).
2. Resolve duplicate recording identities and site labels.
3. Obtain authoritative mapping of filename token, site, deployment/block and timezone where possible.
4. Supervisor-approved limited audio-header validation.
5. Freeze grouping rules, split design, nuisance baselines and statistical testing protocol.
6. Extract features using a reproducible, versioned pipeline.
7. Conduct grouped site/time evaluation only after data-integrity gates are met.
8. Report uncertainty, limitations and ecological interpretation boundaries.

No model training or confirmatory research experiment is represented as completed by this documentation.
