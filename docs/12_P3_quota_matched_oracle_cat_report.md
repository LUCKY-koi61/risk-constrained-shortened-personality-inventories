# P3: quota- and burden-matched oracle-CAT check

## Purpose

P2 compared the selected two-stage policy with fixed-budget oracle IRT/CAT
benchmarks. Those comparators distributed the expected item budget across all
respondents, whereas the focal policy concentrated later items in a candidate
quota. P3 tests whether the apparent difference remains when an oracle-CAT
comparator is given the same initial item count, the same stage-2 candidate
quota, and therefore the same expected item burden.

## Design

For each 2PL data-generating condition and each full-score capture floor, the
calibration sample selected the least-cost P1 policy. On an independent test
sample, the comparator administered that policy's first-stage item count to
everyone by maximum Fisher information at their current EAP estimate. It then
advanced the same proportion of respondents and adaptively administered every
remaining item to that group. Final selection was the top latent-score capacity
within that group. Item parameters are known: this is an oracle simulation,
not an empirical CAT analysis.

## Results

Twenty repetitions per cell produced the following mean latent-tail capture:

| Condition | Floor | Cost | Policy | Quota-matched oracle CAT | Policy minus CAT [95% paired bootstrap interval] |
|:--|--:|--:|--:|--:|:--|
| Lower information | .80 | .408 | .685 | .677 | .008 [-.002, .017] |
| Lower information | .90 | .482 | .697 | .703 | -.005 [-.012, .001] |
| Higher information | .80 | .336 | .776 | .764 | .012 [.001, .024] |
| Higher information | .90 | .412 | .786 | .790 | -.004 [-.011, .002] |

## Interpretation boundary

The observed differences are small. The paired intervals are descriptive of
the 20 simulation repetitions, not formal equivalence tests. P3 does not support a general claim that
the fixed-form risk-constrained item rule exceeds an oracle sequential IRT
rule. It supports the narrower claim that a prespecified, fixed-capacity
two-stage design can be evaluated through explicit burden, candidate-capacity,
and internal-capture constraints. Any applied CAT comparison would additionally
need estimated parameters, model checks, and operational exposure/stopping
constraints.

## Reproduction

Run `src/brm_p3_quota_cat_simulation.py` with the project runtime. The machine
readable results are `results/brm_p3_quota_cat_simulation.json` and
`results/brm_p3_quota_cat_simulation_summary.csv`.
