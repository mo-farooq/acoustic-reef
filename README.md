# Acoustic Reef 🌊🎧
**From an ocean-soundscape prototype to an evidence-driven reef acoustics research project**

Acoustic Reef began as a hackathon prototype exploring whether underwater soundscapes could support coral-reef monitoring. The project has since evolved into a research and engineering effort focused on the reliability and generalisation of machine-learning analyses of reef soundscapes.

> **Research status:** Dataset metadata audits E0 and E1 are complete. Audio-level validation, feature extraction for the research study, model training, and confirmatory statistical experiments have **not** yet been performed.

## Project evolution

| Phase | Work | Status |
|---|---|---|
| Prototype | Streamlit application, SurfPerch integration and exploratory classifier workflow | Existing prototype retained |
| Literature review | Review of machine-learning approaches to reef soundscape analysis and their evaluation limitations | Completed in research workspace |
| E0 — Metadata feasibility | Metadata-only feasibility audit of candidate public datasets | Completed; independently reviewed |
| E1 — Archive inventory | Remote ZIP archive inventory using HTTP byte-range requests; filename, file-size and CRC-32 indexing | Completed; independently reviewed |
| E2 — Audio validation | Inspect a supervisor-approved subset of archive audio and verify WAV metadata | Pending |
| Experiments | Leakage-safe site/time evaluation, model comparison and statistical analysis | Not started |

## Current research direction

The current candidate question is:

**How much of the AI-detected difference between coral reef soundscapes survives when site, recorder and time are held out?**

This is an evaluation-focused research direction, not a claim that an AI model can directly diagnose coral-reef health. Classification of soundscape labels does not, by itself, establish coral cover, biodiversity, ecological condition or restoration success.

The candidate primary dataset is the Lizard Island task described by Williams et al. (2025a). Dataset choice and final statistical protocol remain subject to supervisor approval.

## E0 and E1: verified metadata-audit progress

The following figures are reported by the completed E1 archive-inventory audit. They describe archive entries and metadata; they are **not** audio-analysis or model-performance results.

- All 50 Lizard Island ZIP archives were indexed using HTTP Range requests.
- 8,975 archive entries were enumerated with file sizes and CRC-32 checksums.
- Approximately 12.9 MiB was transferred for the Lizard Island inventory; no indexing failures were reported.
- 50 French Polynesia archives were indexed for a count cross-check, yielding 8,027 entries.
- An independent reviewer re-read three Lizard Island archives and confirmed that all 540 entries matched.
- 217 entries appear under both SiteA and SiteB with matching filename metadata (token, timestamp, size and CRC-32). This is a potential cross-site duplication/data-leakage issue; it has **not** been resolved or silently corrected.
- Published and archive inventory counts differ. The reasons remain unresolved.
- Three leading numeric filename tokens were observed. Their identity as physical recorder serial numbers has not been authoritatively confirmed.

### Important limitations

- No audio payload was downloaded or decoded during E0/E1.
- No audio features were extracted.
- No model was trained or evaluated as part of E0/E1.
- No confirmatory statistical tests were run.
- The token-to-recorder identity, deployment-to-published-block mapping, timezone, duplicate-site labels, missing site/block cells, published-count discrepancy and authoritative class-label table remain unresolved.
- Recorder-held-out evaluation remains **OPEN** and must not be presented as established feasibility.
- The manifest from E1 is an archive inventory, not evidence that audio contents, WAV headers or ecological labels have been validated.

See the detailed E0/E1 reports in the research workspace. Only audited reports and scripts should be added to this public repository; do not publish unpublished correspondence, supervisor comments, private notes or restricted datasets.

## Repository map

```text
.
├── dashboard/              # Original Streamlit prototype
├── data/                   # Existing project data artifacts; check provenance before reuse
├── models/                 # Existing prototype model artifacts
├── notebooks/              # Exploratory/model-development notebooks
├── src/
│   ├── audio/              # Audio utilities
│   ├── models/             # SurfPerch and classifier code
│   └── utils/              # Configuration and utilities
├── experiments/            # Research audit reports and reproducible research scripts
├── docs/                   # Architecture, provenance, limitations and research status
├── tests/                  # Automated tests
├── requirements.txt
└── README.md
```

The original prototype directories and files are retained. The research work is being separated from exploratory application code so that prototype predictions are not confused with confirmatory research results.

## Prototype setup

The existing dashboard can be explored with the repository's current dependencies:

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS / Linux
# source .venv/bin/activate

pip install -r requirements.txt
streamlit run dashboard/app.py
```

Model assets and data artifacts in this repository were created during the earlier prototype workflow. Their provenance, training split, label definitions and compatibility must be checked before treating their predictions or reported metrics as scientifically validated.

## Research integrity and data policy

- Never commit raw third-party audio or large public dataset archives.
- Check dataset licences and citation requirements before publishing derived artifacts.
- Keep source metadata, inferred metadata and unresolved fields distinct.
- Preserve original filenames and checksums in archive inventories.
- Do not silently reassign, remove or deduplicate disputed recordings.
- Keep all site-level evaluation splits grouped by site; minutes or clips from the same site must not be treated as independent site replicates.
- Do not claim model accuracy, ecological validity, or generalisation until the corresponding analysis has actually been run and reviewed.
- Do not expose private email correspondence, credentials, API tokens, local paths or personal data.

## Team

- **Onkar Damale** — Project Lead & AI/ML Lead
- **Om Niungare** — Research & API Integration Lead
- **Farooq** — Data Curation & Management Lead
- **Shyamsundar Gitte** — UI/UX & Dashboard Design Lead

## Citation

The research repository should cite the original datasets and papers when their materials are used. A formal citation file will be added after the project authorship, dataset references and software citation details have been checked.

## Disclaimer

Acoustic Reef is a research prototype and research-in-progress. It is not a validated reef-health diagnostic system and should not be used as a substitute for ecological surveys or expert conservation assessments.
