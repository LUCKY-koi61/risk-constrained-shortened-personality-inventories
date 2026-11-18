# Manuscript and analysis alignment

## Version record

| Item | Version |
| --- | --- |
| Manuscript title | *Risk-Constrained Evaluation of Shortened Personality Inventories for Fixed-Capacity Follow-Up* |
| Current submission manuscript | r10, 18 November 2026 |
| Reproducibility release | Frozen r7 analysis package |
| Repository purpose | Public computational reproduction record |

## What this release supports

The scripts and frozen, non-identifying outputs in this repository support the
main manuscript's fixed-capacity screening analyses and simulations:

- P0: repeated-split risk-budget feasibility;
- P1: calibration-to-known-truth simulation;
- P2: latent-tail IRT simulation;
- P3: quota-matched oracle-CAT simulation; and
- P4: calibration-rule comparison and failure-boundary analysis.

In the r10 manuscript, these analyses support the conservative-calibration,
parent-score-versus-latent-target, and IRT/CAT benchmark results. The lower
confidence bounds are calibration selection criteria whose held-out attainment
is evaluated empirically; they are not simultaneous, post-selection, or
distribution-free risk guarantees.

## Version boundary

The r10 revision reorganizes the manuscript argument and updates the abstract
and keywords. It does not change the frozen r7 scripts, derived results,
random seeds, or reported numerical findings. No new analysis is required to
reproduce the r10 manuscript's quantitative claims.

## Deliberate exclusions

This public repository does not contain the submitted manuscript, title page,
cover letter, participant-level raw data, or other submission-system files.
Raw data can be obtained according to [`DATA_SOURCES.md`](../DATA_SOURCES.md).
