#!/usr/bin/env python3
"""P2: add a threshold-targeted oracle IRT baseline to the P1 simulation."""
from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
P1 = ROOT / "code" / "brm_p1_calibration_truth_simulation.py"
spec = importlib.util.spec_from_file_location("p1", P1)
p1 = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(p1)

SEED = 20261118 + 3
THETA_CUTOFF = 1.281551565545  # standard-normal 90th percentile


def tail_irt_scores(x: np.ndarray, a: np.ndarray, b: np.ndarray, budget: int) -> np.ndarray:
    """Oracle-parameter fixed IRT form optimized for information at the cutoff.

    This is a decision-targeted IRT baseline, not an adaptive test: a fixed
    short form is selected by true Fisher information at the latent 90th
    percentile, then EAP scores are ranked at the same item budget as P1.
    """
    p_cut = 1 / (1 + np.exp(-a * (THETA_CUTOFF - b)))
    info = a**2 * p_cut * (1 - p_cut)
    chosen = np.argsort(-info)[:budget]
    probs = 1 / (1 + np.exp(-np.outer(p1.GRID, a[chosen]) + a[chosen] * b[chosen]))
    loglike = x[:, chosen] @ np.log(probs).T + (1 - x[:, chosen]) @ np.log(1 - probs).T
    logpost = loglike + p1.LOG_PRIOR
    weights = np.exp(logpost - logpost.max(axis=1, keepdims=True))
    return weights @ p1.GRID / weights.sum(axis=1)


def run_one(rng, quality, floor):
    theta_d, x_d, a, b = p1.simulate(rng, quality, p1.N)
    theta_c, x_c, _, _ = p1.simulate(rng, quality, p1.N // 2)
    theta_t, x_t, _, _ = p1.simulate(rng, quality, p1.N // 2)
    _ = (theta_d, x_d)
    cal_key, test_key = rng.random(len(x_c)), rng.random(len(x_t))
    choice = p1.choose(theta_c, x_c, a, cal_key, floor)
    if choice is None:
        return {"quality": quality, "floor": floor, "feasible": False}
    cost, k, advance, cal_bound = choice
    order = np.argsort(-a)
    full, short = x_t.sum(1), x_t[:, order[:k]].sum(1)
    full_ref, theta_ref = p1.top(full, p1.Q, test_key), p1.top(theta_t, p1.Q, test_key)
    passed = p1.top(short, advance, test_key)
    final = p1.top(full, p1.Q, test_key, passed)
    budget = max(1, min(p1.M, int(round(cost * p1.M))))
    generic_cat = p1.top(p1.cat_scores(x_t, a, b, budget), p1.Q, test_key)
    tail_irt = p1.top(tail_irt_scores(x_t, a, b, budget), p1.Q, test_key)
    cap = lambda selected: float((theta_ref & selected).sum() / theta_ref.sum())
    return {
        "quality": quality, "floor": floor, "feasible": True, "cost_share": cost,
        "k": k, "advance": advance, "calibration_lower": cal_bound, "budget": budget,
        "test_full_reference_capture": float((full_ref & final).sum() / full_ref.sum()),
        "test_latent_capture_two_stage": cap(final),
        "test_latent_capture_full_form": cap(full_ref),
        "test_latent_capture_generic_oracle_cat": cap(generic_cat),
        "test_latent_capture_threshold_irt": cap(tail_irt),
    }


def main():
    rng = np.random.default_rng(SEED); rows = []
    for quality in ("low", "high"):
        for floor in p1.FLOORS:
            for rep in range(p1.N_REPS):
                row = run_one(rng, quality, floor); row["replicate"] = rep; rows.append(row)
            print(quality, floor, "complete", flush=True)
    summary = []
    fields = ("cost_share", "test_full_reference_capture", "test_latent_capture_two_stage",
              "test_latent_capture_full_form", "test_latent_capture_generic_oracle_cat",
              "test_latent_capture_threshold_irt")
    for quality in ("low", "high"):
        for floor in p1.FLOORS:
            rs = [r for r in rows if r["quality"] == quality and r["floor"] == floor and r["feasible"]]
            summary.append({"quality": quality, "full_score_capture_floor": floor,
                            "n_reps": p1.N_REPS, "feasible_reps": len(rs),
                            **{"mean_" + f: float(np.mean([r[f] for r in rs])) for f in fields}})
    OUT.mkdir(parents=True, exist_ok=True)
    payload = {"status": "simulation only; all IRT/CAT comparators use true 2PL parameters",
               "simulation_series": "P2/P3 matched random stream and selected policies; P3 adds only the quota-matched CAT comparator",
               "seed": SEED, "theta_cutoff": THETA_CUTOFF, "rows": rows, "summary": summary}
    (OUT / "brm_p2_tail_irt_simulation.json").write_text(json.dumps(payload, indent=2))
    with (OUT / "brm_p2_tail_irt_simulation_summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0])); w.writeheader(); w.writerows(summary)
    assert all(0 <= r[f] <= 1 for r in rows if r["feasible"] for f in fields)


if __name__ == "__main__":
    main()
