# P0 feasibility protocol: risk-constrained two-stage measurement

## Status and decision purpose

This is a feasibility gate for the BRM route, not a preregistered
confirmation and not a manuscript result. It asks whether the existing public
web-data examples can support a much narrower methods claim than the original
submission:

> When a full-form top-decile list is used as the internal operational
> reference, can a fixed short first stage plus full-form follow-up for an
> advance quota reduce item burden while meeting a prespecified capture-floor
> rule under repeated disjoint development, calibration, and test splits?

The target is **not** an external criterion, a diagnosis, a clinical risk, or
latent truth. “Missed-target risk” in this file means one minus capture of the
full-form top decile only.

## Why this is needed

The previous v19--v21 analyses are useful precursors: they established
two-stage operating curves, one holdout split, and fixed-form comparisons.
They do not yet establish a decision rule over multiple risk levels or assess
whether a lower-bound criterion selected on calibration rows reproduces in
unseen rows. This P0 fills that narrow gap; it does not repair the lack of an
external outcome.

## Frozen v1 design

| Component | Decision |
| --- | --- |
| Public examples | HEXACO, IPIP50, NPI40, and SD3; all analyses are within inventory |
| Target capacity | Fixed top 10% of the full-form score |
| Repeats | 10 deterministic respondent splits per inventory |
| Data roles | 50% development / 25% calibration / 25% untouched test |
| Risk levels | 10%, 15%, and 20% tolerated non-capture, hence target floors .90, .85, and .80 |
| Form development | Nested item order optimized for top-decile capture, separately for each advance quota, using development data only |
| Two-stage choice | Lowest item-cost ​(k, advance quota)​ whose **minimum per-unit** two-sided 95% Wilson lower bound on calibration capture meets the floor |
| Direct baseline | Fixed short form at the same 10% allocation capacity; no second stage; its own lowest-cost calibrated choice |
| Cost | `k/m + a(1-k/m)` for two stage, and `k/m` for direct short form |
| Ties | A shared deterministic random key resolves exact-score boundary ties; identical scores therefore give identical capacity sets |
| Test report | Minimum and median capture across units; share of splits where every unit reaches the point floor and the Wilson-lower-bound floor |

## Interpretation boundary

The calibration Wilson bound is a deliberately conservative **selection
criterion** for this resampling exercise. It is neither a confidence guarantee
for new people nor a false-negative-rate guarantee against true pathology. A
low test pass share is useful negative evidence: it means this route has not
yet earned a deployment-oriented claim.

An IRT/CAT comparator is listed as a required later benchmark but is not
fabricated here. These data have no fitted and checked IRT model, calibrated
item bank, item-selection rule, stopping rule, or CAT validation. A genuine
comparison must add those components before it is reported.

## Outputs and checks

- Script: `../../src/brm_p0_risk_budget_feasibility.py`
- Machine-readable split results: `results/brm_p0_risk_budget_feasibility_v1.json`
- Compact summary: `results/brm_p0_risk_budget_feasibility_v1_summary.csv`
- Minimum validity checks: every selected policy uses no test rows in item or
  configuration selection; selected forms are fixed within each test split;
  cost lies in [0, 1]; capture and lower bounds lie in [0, 1].

## P0 decision rule after the run

Proceed to a fully specified BRM methods paper only if the repeated-split
results show a reproducible burden--capture trade-off, disclose the failure
regions, and retain a contribution beyond existing short-form/CAT literature.
Otherwise, keep the work as an exploratory methodological supplement and do
not force a BRM submission.
