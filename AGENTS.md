# Repository-wide agent instructions

Before making a substantive change, read:

- `docs/research-status.md`
- `docs/data-provenance.md`
- `docs/architecture-and-scope.md`
- `docs/prototype-audit.md`
- `docs/kiro-web-workflow.md`

## Non-negotiable research integrity

1. Keep the original hackathon prototype distinct from the new research workflow. Prototype predictions, stored embeddings, existing model artifacts, and dashboard screenshots are not evidence for the new study.
2. E0 and E1 are metadata-only audits. No audio payload has been decoded, no research features have been extracted, no research model has been trained, and no confirmatory research test has been run.
3. Never invent, reconstruct, or silently alter raw data, labels, archive counts, CRCs, recorder identities, site assignments, deployment mappings, experiment outputs, or statistical results.
4. Treat the 217 potential SiteA/SiteB duplicate pairs, archive/publication count discrepancies, missing site-by-block cells, filename-token identity, deployment mapping, timezone, and authoritative class labels as unresolved until evidence is added and reviewed.
5. Do not infer that a filename token is a physical SoundTrap serial number. Preserve the original token as an opaque identifier unless a traceable source verifies its identity.
6. Do not interpret acoustic class discrimination as direct proof of coral cover, biodiversity, reef health, ecological condition, or restoration success.
7. Do not download or decode audio, extract research features, train research models, or run confirmatory analyses unless the user explicitly requests the approved next stage and the documented supervisor/data-integrity gates have been met.

## Engineering workflow

- Make small, reviewable changes on a feature branch. Do not push directly to `main`, merge a PR, or enable auto-merge.
- Preserve the original prototype and large tracked artifacts unless the task explicitly calls for a reviewed change.
- Never route placeholder, random, mock, or synthetic predictions into research results. Demo/test data must be unmistakably labelled and must not be treated as empirical evidence.
- Do not introduce a new dependency or a heavyweight model/data download without explaining why it is needed and how it will be validated.
- Keep research code and outputs separate from the legacy dashboard and prototype model paths.
- For new research code, add tests for identity matching, provenance checks, grouping/split construction, and failure cases. Tests using synthetic fixtures must be labelled as synthetic and must not be reported as research results.
- Prevent leakage: fit preprocessing within each training fold, and use the independently justified grouping unit for all splits and uncertainty calculations. Do not use random clip-level splits as evidence of generalisation to unseen sites, recorders, or periods.
- Never commit credentials, private correspondence, unpublished personal information, restricted data, or unapproved raw audio.

## Reporting

State exactly what was inspected, executed, changed, and not executed. Separate verified observations from assumptions and recommendations. If a required fact is unresolved, preserve it as unresolved rather than choosing a convenient value.
