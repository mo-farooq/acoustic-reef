---
inclusion: always
---

# Acoustic Reef — research context and scientific guardrails

## Project identity

Acoustic Reef began as a hackathon prototype described as “The Stethoscope for the Ocean.” The repository is being repositioned as a transparent, evidence-driven undergraduate research project while preserving the original prototype for historical continuity and portfolio demonstration.

The current candidate research question is:

> How much of the AI-detected difference between coral reef soundscapes survives when site, recorder and time are held out?

This is a question about robustness of soundscape discrimination under grouped generalisation tests. It is not, by itself, a direct test of coral cover, biodiversity, reef health, ecological condition, or restoration success.

## Verified project status

As of the repository's 2026-11-24 status audit:

- E0 metadata feasibility audit: completed and independently reviewed.
- E1 archive inventory: completed and independently reviewed.
- E1 enumerated 50 Lizard Island ZIP archives using HTTP Range requests; 8,975 entries were indexed with file sizes and CRC-32 checksums.
- No audio payload was decoded, and WAV headers were not inspected.
- No research features were extracted.
- No research model was trained or evaluated.
- No confirmatory statistical tests were run.
- E2/audio-header validation and all model experiments remain not started.

Do not describe E0/E1 as audio analysis, model validation, or completed experiments.

## Unresolved data-integrity issues — do not silently repair

- 217 entries appear under both SiteA and SiteB with matching token, timestamp, size, and CRC-32. Treat these as potential cross-site duplicates and a possible site-label/leakage problem. Do not select one site, delete entries, or deduplicate them without traceable evidence and an approved rule.
- Archive totals differ from published counts: 8,975 enumerated versus 8,127 published total; the habitat subset is 6,106 enumerated versus 6,289 published. After removing the potential duplicate pairs, the inventory totals also differ. Causes are unresolved.
- 70 of 72 site-by-block cells are present in the current inventory.
- The archive's latest timestamp is Dec 9 while the publication reports data through Dec 14.
- Filename grammar observed in E1 is `Site<L>deployment<N>.<token>.<YYMMDDHHMMSS>.wav`. Preserve the actual filename; do not rename files to match an earlier assumed grammar.
- Numeric filename tokens are candidate recorder-like identifiers only. They have not been verified as physical SoundTrap serial numbers. Keep them opaque.
- Deployment index has not been verified as the same thing as the publication's recording-block index.
- Class labels were derived from a publication figure image, not an authoritative per-site table.
- Timezone is unresolved. Lunar phase is a possible covariate only; tidal height is unverified.
- Recorder-held-out feasibility remains OPEN. Three candidate tokens are not enough to claim robust recorder-level generalisation, and recorder composition differs between classes.

Use the repository's provenance/status documents as the current source of truth. When new evidence arrives, update provenance documents with source, date, method, and review status before changing analysis code.

## Study-design constraints

- Do not pool Lizard Island and the UCL MARRS restoration archive as if they were one dataset or one shared label system. Any use of MARRS must be described as a separate external-validation dataset and evaluated independently.
- Do not use French Polynesia pooled p-values to strengthen the primary Lizard Island task.
- Do not treat thousands of clips as thousands of independent ecological replicates. Site-level inference and grouping must reflect the sampling design.
- Do not reuse the legacy prototype's random row-level split, ordinary clip-level cross-validation, or pre-split scaler fit as valid evidence for the proposed research question.
- Fit all learned preprocessing within the relevant training fold. Freeze and version split manifests and grouping decisions before confirmatory evaluation.
- Preserve uncertainty and limitations. A technically executable split is not automatically a scientifically defensible split.

## Work boundaries

Until the user explicitly authorizes a new research stage and its prerequisite gates are met:

- Do not fetch raw audio or bulk-download external archives.
- Do not decode WAV files or run feature extraction.
- Do not train, fine-tune, or evaluate a research model.
- Do not generate p-values, confidence intervals, performance metrics, or figures that look like empirical research results.
- Do not fabricate missing E0/E1 reports, CSVs, logs, or audit outputs from summaries.

Safe work includes documentation, code review, static analysis, provenance tooling, unit tests with clearly synthetic fixtures, and implementation plans that do not access research audio or represent results as empirical.

## Required communication style

For every task, distinguish:
1. source-backed observations;
2. implementation choices;
3. assumptions or unresolved questions;
4. work executed;
5. work not executed;
6. remaining decisions requiring the user or supervisor.

Never claim that a command, test, model, dataset, or workflow was run unless it actually was.
