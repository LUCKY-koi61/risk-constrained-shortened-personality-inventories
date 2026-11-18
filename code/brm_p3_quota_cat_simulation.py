#!/usr/bin/env python3
"""P3: quota-matched sequential oracle-CAT comparison for the BRM manuscript.

The comparison deliberately gives the CAT baseline exactly the same stage-2
candidate quota and expected item burden as the risk-constrained two-stage
policy selected on the calibration set.  It is still an oracle simulation:
the 2PL parameters are known, not estimated.
"""
from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
P1_PATH = ROOT / "code" / "brm_p1_calibration_truth_simulation.py"
spec = importlib.util.spec_from_file_location("p1", P1_PATH)
p1 = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(p1)

# P3 intentionally reuses P2's random stream and data-generating sequence.
# Consequently Table 12 and Table 13 report the same selected policies; P3
# adds only the quota-matched comparator, rather than a new simulation batch.
SEED = p1.SEED + 1


def eap(log_posterior: np.ndarray) -> float:
    weights = np.exp(log_posterior - log_posterior.max())
    return float(weights @ p1.GRID / weights.sum())


def quota_matched_cat(x: np.ndarray, a: np.ndarray, b: np.ndarray, k: int,
                      advance: float, key: np.ndarray) -> np.ndarray:
    """Two-stage oracle CAT with the policy's initial length and candidate quota.

    Everyone receives *k* adaptively selected items.  The top ``advance``
    proportion by the stage-1 EAP score then receives every remaining item.
    Thus its expected burden is precisely k/M + advance(1-k/M), matching the
    selected two-stage policy, and its stage-2 capacity is also identical.
    """
    n, m = x.shape
    posteriors = [p1.LOG_PRIOR.copy() for _ in range(n)]
    remaining = [list(range(m)) for _ in range(n)]

    def administer(person: int) -> None:
        lp, candidates = posteriors[person], remaining[person]
        estimate = eap(lp)
        js = np.asarray(candidates)
        probability = 1 / (1 + np.exp(-a[js] * (estimate - b[js])))
        information = a[js] ** 2 * probability * (1 - probability)
        j = candidates.pop(int(np.argmax(information)))
        curve = 1 / (1 + np.exp(-a[j] * (p1.GRID - b[j])))
        posteriors[person] = lp + np.log(curve if x[person, j] else 1 - curve)

    for person in range(n):
        for _ in range(k):
            administer(person)
    stage1 = np.asarray([eap(lp) for lp in posteriors])
    passed = p1.top(stage1, advance, key)
    for person in np.flatnonzero(passed):
        while remaining[person]:
            administer(int(person))
    # Non-candidates cannot be final positives, exactly as in the focal policy.
    scores = np.full(n, -np.inf)
    scores[passed] = [eap(posteriors[i]) for i in np.flatnonzero(passed)]
    return scores


def run_one(rng: np.random.Generator, quality: str, floor: float) -> dict:
    _, _, a, b = p1.simulate(rng, quality, p1.N)
    _, x_cal, _, _ = p1.simulate(rng, quality, p1.N // 2)
    theta_test, x_test, _, _ = p1.simulate(rng, quality, p1.N // 2)
    cal_key, test_key = rng.random(len(x_cal)), rng.random(len(x_test))
    choice = p1.choose(None, x_cal, a, cal_key, floor)
    if choice is None:
        return {"quality": quality, "floor": floor, "feasible": False}
    cost, k, advance, cal_bound = choice
    order = np.argsort(-a)
    short = x_test[:, order[:k]].sum(axis=1)
    passed = p1.top(short, advance, test_key)
    full_reference = p1.top(x_test.sum(axis=1), p1.Q, test_key)
    latent_reference = p1.top(theta_test, p1.Q, test_key)
    policy_final = p1.top(x_test.sum(axis=1), p1.Q, test_key, passed)
    cat_scores = quota_matched_cat(x_test, a, b, k, advance, test_key)
    cat_final = p1.top(cat_scores, p1.Q, test_key, np.isfinite(cat_scores))
    capture = lambda selected, reference: float((selected & reference).sum() / reference.sum())
    return {
        "quality": quality, "floor": floor, "feasible": True,
        "cost_share": cost, "initial_items": k, "candidate_quota": advance,
        "calibration_lower": cal_bound,
        "test_full_reference_capture_policy": capture(policy_final, full_reference),
        "test_latent_capture_policy": capture(policy_final, latent_reference),
        "test_latent_capture_quota_matched_oracle_cat": capture(cat_final, latent_reference),
        "test_policy_minus_quota_cat_latent_capture": capture(policy_final, latent_reference) - capture(cat_final, latent_reference),
    }


def main() -> None:
    rng = np.random.default_rng(SEED)
    rows = []
    for quality in ("low", "high"):
        for floor in p1.FLOORS:
            for replicate in range(p1.N_REPS):
                row = run_one(rng, quality, floor)
                row["replicate"] = replicate
                rows.append(row)
            print(quality, floor, "complete", flush=True)
    fields = ("cost_share", "test_full_reference_capture_policy", "test_latent_capture_policy",
              "test_latent_capture_quota_matched_oracle_cat", "test_policy_minus_quota_cat_latent_capture")
    summary = []
    interval_rng = np.random.default_rng(SEED + 99)
    for quality in ("low", "high"):
        for floor in p1.FLOORS:
            matching = [r for r in rows if r["quality"] == quality and r["floor"] == floor and r["feasible"]]
            differences = np.asarray([r["test_policy_minus_quota_cat_latent_capture"] for r in matching])
            boot = np.mean(interval_rng.choice(differences, size=(10_000, len(differences)), replace=True), axis=1)
            summary.append({"quality": quality, "full_score_capture_floor": floor,
                            "n_reps": p1.N_REPS, "feasible_reps": len(matching),
                            **{"mean_" + f: float(np.mean([r[f] for r in matching])) for f in fields},
                            "paired_difference_bootstrap_ci_95_low": float(np.quantile(boot, .025)),
                            "paired_difference_bootstrap_ci_95_high": float(np.quantile(boot, .975))})
    OUT.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "simulation only; quota-matched CAT uses true 2PL parameters",
        "simulation_series": "P2/P3 matched random stream and selected policies; P3 adds the quota-matched CAT comparator",
        "design": "same initial length, candidate quota, and expected item burden as the selected policy",
        "seed": SEED, "n_reps": p1.N_REPS, "n_calibration": p1.N // 2, "n_test": p1.N // 2,
        "full_items": p1.M, "top_capacity": p1.Q, "rows": rows, "summary": summary,
    }
    (OUT / "brm_p3_quota_cat_simulation.json").write_text(json.dumps(payload, indent=2))
    with (OUT / "brm_p3_quota_cat_simulation_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0])); writer.writeheader(); writer.writerows(summary)
    assert all(-1 <= r["test_policy_minus_quota_cat_latent_capture"] <= 1 for r in rows if r["feasible"])


if __name__ == "__main__":
    main()
