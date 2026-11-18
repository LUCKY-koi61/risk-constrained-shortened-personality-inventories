# Risk-Constrained Evaluation of Shortened Personality Inventories

This repository provides the reproducibility materials for *Risk-Constrained
Evaluation of Shortened Personality Inventories for Fixed-Capacity Follow-Up*.
It implements and documents a fixed-capacity, two-stage screening workflow for
shortened personality inventories. The repository contains executable analysis
code, frozen non-identifying results, and concise protocols; it does not
redistribute respondent-level data.

## What is included

- `code/` — P0–P4 analysis scripts and the shared evaluation utility.
- `results/` — frozen JSON and CSV outputs used to construct the reported
  empirical and simulation results.
- `docs/` — protocols for risk-budget feasibility, quota-matched oracle-CAT,
  and calibration-rule comparisons.
- `DATA_SOURCES.md` — public-data source and reuse information.
- `requirements.txt` — pinned package versions used for the audited reruns.

## Reproduce the analyses

Create a Python 3.10+ environment and install the pinned dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The P1–P3 simulations can run without respondent-level data. P0 and P4 use
the four public Open-Source Psychometrics Project datasets. Download the
archives described in [`DATA_SOURCES.md`](DATA_SOURCES.md), then arrange the
extracted files as follows under the repository root:

```text
data/HEXACO/HEXACO/data.csv
data/BIG5/BIG5/data.csv
data/NPI/NPI/data.csv
data/SD3/SD3/data.csv
```

Run the analyses from the repository root:

```bash
python code/brm_p0_risk_budget_feasibility.py
python code/brm_p4_rule_comparison_failure_boundary.py
python code/brm_p1_calibration_truth_simulation.py
python code/brm_p2_tail_irt_simulation.py
python code/brm_p3_quota_cat_simulation.py
```

The scripts overwrite the corresponding derived files in `results/`. P2 and
P3 deliberately share the same seed and data-generating sequence so that the
two-stage results can be checked exactly; P3 adds only the quota-matched
oracle-CAT comparator.

## Interpretation boundary

P0 and P4 evaluate agreement with each inventory's full-form top-decile
reference, not clinical diagnosis or an external criterion. Their lower
confidence bounds are conservative selection rules in a repeated-split design,
not distribution-free risk guarantees. The IRT/CAT analyses use known 2PL
parameters and are simulations rather than operational CAT evaluations.

## Data and privacy

Raw respondent data are not included. They remain available from the
[Open-Source Psychometrics Project raw-data archive](https://openpsychometrics.org/_rawdata/).
Please follow the source provider's terms and retain the accompanying
codebooks when working with the data.

## Release status

This repository is the public reproducibility record for the r7 analysis
package. It contains no submission files, author title page, cover letter, or
participant-level data.
