#!/usr/bin/env python3
"""P4: compare calibration rules and map internal risk-constraint failures.

The analysis uses the P0 data split and policy grid.  Three rules choose the
least-cost policy: (a) a customary point-estimate threshold, (b) the focal
two-sided 95% Wilson lower-bound threshold, and (c) a conservative one-sided
99% exact-binomial lower-quantile threshold.  All rules use the same
development-derived item orders and untouched test rows.
"""
from __future__ import annotations

import csv
import importlib.util
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import beta

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
P0_PATH = ROOT / "code" / "brm_p0_risk_budget_feasibility.py"
spec = importlib.util.spec_from_file_location("p0", P0_PATH)
p0 = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(p0)

RULES = ("point", "wilson_95", "exact_lower_quantile_99")
BUDGET_SHARE = .50


def exact_lower_quantile(hits: int, total: int, alpha: float = .01) -> float:
    """One-sided exact binomial lower endpoint (the alpha lower quantile)."""
    return 0.0 if hits == 0 else float(beta.ppf(alpha, hits, total - hits + 1))


def candidates(calibration_units, orders_by_advance, grid, ties):
    m = next(iter(calibration_units.values())).shape[1]
    out = []
    for k in grid:
        for advance in p0.ADVANCE_SHARES:
            metrics = p0.calibration_metrics(calibration_units, orders_by_advance[advance], k, advance, ties)
            exact = min(exact_lower_quantile(v["hits"], v["target_count"])
                        for v in metrics["per_unit"].values())
            out.append({"items_stage_1": k, "advance_share": advance,
                        "cost_share": k / m + advance * (1 - k / m),
                        "metrics": metrics, "min_exact_lower_quantile_99": exact})
    return out


def choose(candidates_, floor, rule):
    if rule == "point":
        qualified = [x for x in candidates_ if x["metrics"]["min_capture"] >= floor]
    elif rule == "wilson_95":
        qualified = [x for x in candidates_ if x["metrics"]["min_wilson_lower_95"] >= floor]
    elif rule == "exact_lower_quantile_99":
        qualified = [x for x in candidates_ if x["min_exact_lower_quantile_99"] >= floor]
    else:
        raise ValueError(rule)
    return min(qualified, key=lambda x: (x["cost_share"], x["items_stage_1"], x["advance_share"])) if qualified else None


def one_split(inventory, raw, split):
    n = next(iter(raw["units"].values())).shape[0]
    rng = np.random.default_rng(p0.SEED + 1000 * list(p0.GRID).index(inventory) + split)
    perm = rng.permutation(n); a, b = n // 2, (3 * n) // 4
    development, calibration, test = perm[:a], perm[a:b], perm[b:]
    dev, cal, tst = p0.orient_split(raw, development, calibration, test)
    dev_ties, cal_ties, test_ties = rng.random(a), rng.random(b-a), rng.random(n-b)
    orders_by_advance = p0.make_orders(dev, dev_ties)
    candidates_ = candidates(cal, orders_by_advance, p0.GRID[inventory], cal_ties)
    rows = []
    for risk in p0.RISK_LEVELS:
        floor = 1 - risk
        for rule in RULES:
            policy = choose(candidates_, floor, rule)
            row = {"inventory": inventory, "split": split, "rule": rule, "miss_risk": risk,
                   "capture_floor": floor, "n_development": a, "n_calibration": b-a, "n_test": n-b,
                   "n_units": len(tst), "selection_feasible": policy is not None}
            if policy is not None:
                orders = orders_by_advance[policy["advance_share"]]
                test_metrics = p0.test_policy(tst, policy, orders, test_ties, floor)
                row.update({"items_stage_1": policy["items_stage_1"], "advance_share": policy["advance_share"],
                            "cost_share": policy["cost_share"], "over_half_budget": policy["cost_share"] > BUDGET_SHARE,
                            "calibration_min_capture": policy["metrics"]["min_capture"],
                            "calibration_min_wilson_lower_95": policy["metrics"]["min_wilson_lower_95"],
                            "calibration_min_exact_lower_quantile_99": policy["min_exact_lower_quantile_99"],
                            "test_min_capture": test_metrics["min_capture"],
                            "test_min_wilson_lower_95": test_metrics["min_wilson_lower_95"],
                            "test_all_units_point_floor": test_metrics["all_units_point_capture_at_floor"],
                            "test_all_units_wilson_floor": test_metrics["all_units_wilson_lower_at_floor"],
                            "test_min_capture_margin": test_metrics["min_capture"] - floor,
                            "test_min_wilson_margin": test_metrics["min_wilson_lower_95"] - floor})
            rows.append(row)
    return rows


def summarize(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[(r["inventory"], r["miss_risk"], r["rule"])].append(r)
    summary = []
    for (inventory, risk, rule), group in sorted(groups.items()):
        usable = [r for r in group if r["selection_feasible"]]
        rec = {"inventory": inventory, "miss_risk": risk, "capture_floor": 1-risk, "rule": rule,
               "n_splits": len(group), "feasible_splits": len(usable),
               "feasible_share": len(usable) / len(group)}
        if usable:
            for field in ("cost_share", "over_half_budget", "test_min_capture", "test_min_wilson_lower_95",
                          "test_all_units_point_floor", "test_all_units_wilson_floor", "test_min_capture_margin",
                          "test_min_wilson_margin"):
                rec["mean_" + field] = float(np.mean([r[field] for r in usable]))
        summary.append(rec)
    return summary


def failure_profile(rows):
    """Descriptive margins for the focal Wilson rule, stratified by inventory."""
    groups = defaultdict(list)
    for r in rows:
        if r["rule"] == "wilson_95" and r["selection_feasible"]:
            groups[r["inventory"]].append(r)
    out = []
    for inventory, group in sorted(groups.items()):
        passed = [r for r in group if r["test_all_units_wilson_floor"]]
        failed = [r for r in group if not r["test_all_units_wilson_floor"]]
        out.append({
            "inventory": inventory, "n_policy_evaluations": len(group),
            "wilson_pass_share": len(passed) / len(group),
            "mean_cost_pass": float(np.mean([r["cost_share"] for r in passed])) if passed else None,
            "mean_cost_fail": float(np.mean([r["cost_share"] for r in failed])) if failed else None,
            "mean_test_min_capture_margin_pass": float(np.mean([r["test_min_capture_margin"] for r in passed])) if passed else None,
            "mean_test_min_capture_margin_fail": float(np.mean([r["test_min_capture_margin"] for r in failed])) if failed else None,
            "mean_test_min_wilson_margin_pass": float(np.mean([r["test_min_wilson_margin"] for r in passed])) if passed else None,
            "mean_test_min_wilson_margin_fail": float(np.mean([r["test_min_wilson_margin"] for r in failed])) if failed else None,
            "mean_stage1_items_pass": float(np.mean([r["items_stage_1"] for r in passed])) if passed else None,
            "mean_stage1_items_fail": float(np.mean([r["items_stage_1"] for r in failed])) if failed else None,
        })
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    raw = p0.v21.load_raw(); rows = []
    for inventory, data in raw.items():
        for split in range(p0.N_SPLITS):
            rows.extend(one_split(inventory, data, split))
            print(f"{inventory}: split {split + 1}/{p0.N_SPLITS}", flush=True)
    summary, profile = summarize(rows), failure_profile(rows)
    payload = {"status": "internal full-form-referenced repeated-split comparison; no clinical or transportable guarantee",
               "rules": {"point": "minimum calibration point capture meets floor",
                         "wilson_95": "minimum calibration two-sided 95% Wilson lower bound meets floor",
                         "exact_lower_quantile_99": "minimum calibration one-sided 99% exact binomial lower quantile meets floor"},
               "budget_definition": f"selected policy cost share > {BUDGET_SHARE:.2f} of full-form administration",
               "rows": rows, "summary": summary, "focal_wilson_failure_profile": profile}
    (OUT / "brm_p4_rule_comparison_failure_boundary.json").write_text(json.dumps(payload, indent=2))
    for name, data in (("brm_p4_rule_comparison_summary.csv", summary), ("brm_p4_wilson_failure_profile.csv", profile)):
        with (OUT / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data[0])); writer.writeheader(); writer.writerows(data)
    assert all(0 <= r["cost_share"] <= 1 for r in rows if r["selection_feasible"])


if __name__ == "__main__":
    main()
