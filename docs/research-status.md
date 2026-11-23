# Research status

**Last repository modernization audit:** 2026-11-24

## Completed
- E0 metadata feasibility audit (metadata-only; independently reviewed).
- E1 archive inventory (metadata-only; independently reviewed).
- All 50 Lizard Island archives indexed with HTTP Range requests; 8,975 entries, file sizes and CRC-32 checksums; no indexing failures reported.
- 217 potential SiteA/SiteB cross-label duplicate entries flagged and left unresolved.
- Archive-versus-publication count discrepancies and unverified recorder/deployment mappings documented.

## Not performed in E0/E1
- Audio payload download or decoding.
- WAV-header inspection.
- Feature extraction.
- Research-model training or evaluation.
- Confirmatory statistical tests.

## Open data-integrity questions
- Do filename numeric tokens identify physical SoundTrap serial numbers?
- Does filename deployment index map directly to published recording blocks?
- Which site label is correct for the 217 potential duplicate pairs?
- What explains the archive count differences and absent site/block cells?
- What timezone applies to timestamp interpretation?
- What is the authoritative site-level class-label table?

## Feasibility status
- Site-held-out: computationally constructible after duplicate and label resolution; ecological interpretation remains constrained until recorder/site confounding is addressed.
- Time-held-out: computationally constructible using the observed deployment index, but mapping to published blocks remains unverified.
- Recorder-held-out: OPEN; token identity and mapping are unresolved, and three candidate tokens alone would not support strong recorder-level inference.
- Lunar covariate: candidate only; not added to confirmatory plan.
- Tidal covariate: unverified; not in confirmatory plan.

## Next gate
Supervisor-approved protocol decisions and a limited, justified audio-header validation step before any research feature extraction or model training.
