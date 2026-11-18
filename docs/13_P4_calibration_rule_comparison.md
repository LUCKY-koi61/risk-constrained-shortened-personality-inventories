# P4: calibration-rule comparison and failure-boundary analysis

## Question

Does selecting a low-cost two-stage policy with a lower confidence-bound rule
provide a useful operational difference from selecting on observed calibration
capture alone? When strict replication fails, is the failure primarily a loss
of point capture or a lower-bound precision problem?

## Matched design

P4 reused P0's four inventories, 10 disjoint 50%/25%/25%
development--calibration--test splits, development-derived item orders,
candidate-policy grid, full-score top-decile reference, and deterministic tie
protocol. On each calibration set, each rule selected the least-cost eligible
policy:

1. point: every unit's observed capture met the floor;
2. Wilson: every unit's two-sided 95% Wilson lower bound met the floor;
3. exact lower quantile: every unit's one-sided 99% exact-binomial lower
   quantile met the floor.

Untouched test rows evaluated all-unit point and Wilson-bound attainment. The
reported 50% item-cost threshold is a descriptive reporting benchmark, not an
externally specified budget.

## Main repeated-split comparison

The table pools four inventories by 10 splits (40 inventory-by-split
evaluations per rule and floor). These are descriptive repeated-split
frequencies, not independent replications.

| Floor | Rule | Mean cost | Cost > .50 | Test point attainment | Test Wilson attainment |
|:--|:--|--:|--:|--:|--:|
| .90 | Point | .421 | .225 | .800 | .375 |
| .90 | Wilson 95% | .471 | .675 | 1.000 | .875 |
| .90 | Exact 99% | .479 | .750 | 1.000 | .975 |
| .85 | Point | .372 | .000 | .725 | .325 |
| .85 | Wilson 95% | .400 | .050 | .975 | .675 |
| .85 | Exact 99% | .403 | .050 | 1.000 | .750 |
| .80 | Point | .342 | .000 | .775 | .425 |
| .80 | Wilson 95% | .365 | .000 | .975 | .775 |
| .80 | Exact 99% | .373 | .000 | 1.000 | .850 |

## Failure boundary

For the focal Wilson rule, 119/120 inventory-by-split evaluations met their
point-capture floor, whereas 93/120 met the strict Wilson criterion. Passing
policies had mean minimum point-capture and Wilson margins of .069 and .043;
failed policies had corresponding margins of .015 and -.019. This supports a
narrow interpretation: in this resampling design, strict failures accumulated
near the constraint boundary. It does not identify the causal contributions of
scale length, sample size, precision, or dimensionality.

## Reproduction

Run `src/brm_p4_rule_comparison_failure_boundary.py`. The machine-readable
record is `results/brm_p4_rule_comparison_failure_boundary.json`; tabular
outputs are `results/brm_p4_rule_comparison_summary.csv` and
`results/brm_p4_wilson_failure_profile.csv`.
