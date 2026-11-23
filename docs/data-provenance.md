# Data provenance and handling

## Public archive inventory (E1)

The E1 audit used HTTP Range requests to inspect ZIP central-directory metadata for 50 Lizard Island archives. It enumerated 8,975 entries and captured filenames, sizes and CRC-32 checksums. Approximately 12.9 MiB was transferred. No audio payload was decoded. Three archives were independently re-read by a reviewer, covering 540 entries, and matched the original inventory.

The French Polynesia archive set was indexed as a count cross-check only (50 archives; 8,027 entries).

## Evidence levels

Keep these fields distinct in all research outputs:

1. **Archive-observed:** ZIP entry path/name, uncompressed size and CRC-32 recorded from archive index.
2. **Filename-parsed:** fields parsed according to the observed filename grammar; parser result must retain the original filename.
3. **Publication-derived:** site classes, recording blocks and sampling context copied from cited publication/supplementary material.
4. **Inferred:** candidate interpretations such as whether a numeric token is a physical recorder serial or whether deployment index maps to a published block.
5. **Unresolved:** fields for which authoritative evidence is unavailable.

Never promote inferred or unresolved fields to verified metadata without new evidence.

## Potential duplicates

E1 identified 217 entries with matching token, timestamp, size and CRC-32 across SiteA and SiteB. Two representative pairs were additionally checked at compressed-stream level. This is a potential cross-site duplication and label-integrity problem. The public repository must not automatically delete, relabel or silently drop these entries. Preserve source evidence and record any later adjudication with provenance.

## Data restrictions

Do not commit raw audio, full third-party archive files, unpublished correspondence, credentials, local paths, or private supervisor materials. Check source dataset licences before distributing derived manifests or metadata.
