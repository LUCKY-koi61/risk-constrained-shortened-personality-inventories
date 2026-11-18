#!/usr/bin/env python3
"""P0 feasibility study: risk-constrained two-stage short measurement.

This is deliberately a *within-inventory, full-form-referenced* feasibility
study.  The target is the fixed top decile of the full score, not a diagnosis,
latent-trait truth, or an external criterion.  It asks a narrower question:

    Across repeated development/calibration/test splits, can a fixed short
    first stage and fixed advance quota reduce item burden while satisfying a
    prespecified lower-bound criterion for capture of the full-form top decile?

For each split and requested missed-target risk r, development rows select
fixed item orders, calibration rows select the least-cost policy whose Wilson
lower bound for capture is at least 1-r in *every* scoring unit, and untouched
test rows evaluate the frozen policy.  The lower bound is a selection rule
under this resampling design, not a guarantee for a new population.

The fixed-form direct-screen baseline uses the same development procedure but
does not administer a second stage.  An IRT/CAT benchmark is intentionally not
reported: no calibrated IRT model or CAT design has been fit and validated for
these public web samples.  Treating a post-hoc score subset as CAT would be
misleading.
"""
from __future__ import annotations

import csv
import importlib.util
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


PROJ = Path(__file__).resolve().parents[1]
OUT_DIR = PROJ / "results"
OUT_JSON = OUT_DIR / "brm_p0_risk_budget_feasibility_v1.json"
OUT_CSV = OUT_DIR / "brm_p0_risk_budget_feasibility_v1_summary.csv"
V21 = PROJ / "code" / "v21_fixed_form_holdout_validation_20261107.py"
SPEC = importlib.util.spec_from_file_location("v21_fixed", V21)
v21 = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(v21)

SEED = 20261118
N_SPLITS = 10
TOP_SHARE = 0.10
RISK_LEVELS = (0.10, 0.15, 0.20)  # missed full-form-target cases tolerated
ADVANCE_SHARES = (0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50)
GRID = v21.GRID
Z_95 = 1.959963984540054


def fixed_top(score: np.ndarray, share: float, tie_key: np.ndarray) -> np.ndarray:
    """Select exactly round(share*n) rows with a shared random tie protocol.

    ``tie_key`` is generated once per split and reused for full and short
    scores.  Thus equal scores are resolved reproducibly and a short score
    identical to the full score produces the identical decision set.
    """
    count = int(round(share * len(score)))
    order = np.lexsort((tie_key, -score))
    out = np.zeros(len(score), dtype=bool)
    out[order[:count]] = True
    return out


def recall(target: np.ndarray, selected: np.ndarray) -> tuple[int, int, float]:
    total = int(target.sum())
    hits = int((target & selected).sum())
    return hits, total, hits / total


def wilson_lower(hits: int, total: int) -> float:
    """Two-sided 95% Wilson lower endpoint for a binomial proportion."""
    if total == 0:
        return float("nan")
    p = hits / total
    denom = 1.0 + Z_95**2 / total
    centre = p + Z_95**2 / (2.0 * total)
    radius = Z_95 * math.sqrt(p * (1.0 - p) / total + Z_95**2 / (4.0 * total**2))
    return (centre - radius) / denom


def orient_split(raw: dict, development: np.ndarray, calibration: np.ndarray,
                 test: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Orient/standardize with development rows only, as in v21."""
    dev, cal, tst = {}, {}, {}
    for name, x in raw["units"].items():
        npi_score = raw["npi_score"][development] if raw["npi_score"] is not None else None
        dev[name], cal[name] = v21.orient_train_test(
            x[development], x[calibration], npi_train_score=npi_score
        )
        # Apply precisely the development-derived orientation to the test set.
        train_raw, test_raw = x[development].copy(), x[test].copy()
        if npi_score is not None:
            keys = np.array([
                1 if npi_score[train_raw[:, j] == 1].mean() > npi_score[train_raw[:, j] == 2].mean() else 2
                for j in range(train_raw.shape[1])
            ])
            train_raw = (train_raw == keys).astype(float)
            test_raw = (test_raw == keys).astype(float)
        mu, sd = train_raw.mean(axis=0), train_raw.std(axis=0, ddof=0)
        sd[sd == 0] = 1.0
        train_z, test_z = (train_raw - mu) / sd, (test_raw - mu) / sd
        signs = v21.mc.pc_signs(train_z)
        signs[signs == 0] = 1.0
        tst[name] = test_z * signs
    return dev, cal, tst


def greedy_order(development: np.ndarray, advance_share: float, tie_key: np.ndarray) -> list[int]:
    """Build one fixed nested item order optimized for top-decile capture."""
    full_target = fixed_top(development.mean(axis=1), TOP_SHARE, tie_key)
    selected: list[int] = []
    remaining = list(range(development.shape[1]))
    for _ in range(development.shape[1]):
        best_item, best_value = None, -1.0
        for item in remaining:
            score = development[:, selected + [item]].mean(axis=1)
            value = recall(full_target, fixed_top(score, advance_share, tie_key))[2]
            if value > best_value:
                best_item, best_value = item, value
        assert best_item is not None
        selected.append(best_item)
        remaining.remove(best_item)
    return selected


def calibration_metrics(units: dict[str, np.ndarray], orders: dict[str, list[int]],
                        k: int, advance_share: float, tie_key: np.ndarray) -> dict:
    """Per-unit internal capture and Wilson lower bounds for a frozen policy."""
    per_unit = {}
    for name, x in units.items():
        target = fixed_top(x.mean(axis=1), TOP_SHARE, tie_key)
        selected = fixed_top(x[:, orders[name][:k]].mean(axis=1), advance_share, tie_key)
        hits, total, point = recall(target, selected)
        per_unit[name] = {
            "hits": hits, "target_count": total,
            "capture": point, "capture_wilson_lower_95": wilson_lower(hits, total),
        }
    lower = [row["capture_wilson_lower_95"] for row in per_unit.values()]
    point = [row["capture"] for row in per_unit.values()]
    return {
        "per_unit": per_unit,
        "min_capture": min(point), "median_capture": float(np.median(point)),
        "min_wilson_lower_95": min(lower),
    }


def make_orders(development_units: dict[str, np.ndarray], tie_key: np.ndarray) -> dict[float, dict[str, list[int]]]:
    return {
        advance: {name: greedy_order(x, advance, tie_key) for name, x in development_units.items()}
        for advance in ADVANCE_SHARES
    }


def choose_two_stage(calibration_units: dict[str, np.ndarray], orders_by_advance: dict,
                     grid: list[int], calibration_ties: np.ndarray, recall_floor: float) -> dict | None:
    candidates = []
    m = next(iter(calibration_units.values())).shape[1]
    for k in grid:
        for advance in ADVANCE_SHARES:
            metrics = calibration_metrics(calibration_units, orders_by_advance[advance], k, advance, calibration_ties)
            if metrics["min_wilson_lower_95"] >= recall_floor:
                cost = k / m + advance * (1.0 - k / m)
                candidates.append({
                    "policy": "two_stage", "items_stage_1": k, "advance_share": advance,
                    "cost_share": cost, "item_saving_share": 1.0 - cost,
                    "calibration": metrics,
                })
    if not candidates:
        return None
    return min(candidates, key=lambda x: (x["cost_share"], x["items_stage_1"], x["advance_share"]))


def choose_direct(calibration_units: dict[str, np.ndarray], orders: dict[str, list[int]],
                  grid: list[int], calibration_ties: np.ndarray, recall_floor: float) -> dict | None:
    candidates = []
    m = next(iter(calibration_units.values())).shape[1]
    for k in grid:
        metrics = calibration_metrics(calibration_units, orders, k, TOP_SHARE, calibration_ties)
        if metrics["min_wilson_lower_95"] >= recall_floor:
            candidates.append({
                "policy": "direct_fixed_short_form", "items_stage_1": k,
                "advance_share": TOP_SHARE, "cost_share": k / m,
                "item_saving_share": 1.0 - k / m, "calibration": metrics,
            })
    if not candidates:
        return None
    return min(candidates, key=lambda x: x["cost_share"])


def test_policy(test_units: dict[str, np.ndarray], policy: dict | None, orders: dict[str, list[int]],
                test_ties: np.ndarray, recall_floor: float) -> dict | None:
    if policy is None:
        return None
    result = calibration_metrics(test_units, orders, policy["items_stage_1"], policy["advance_share"], test_ties)
    result["all_units_point_capture_at_floor"] = all(
        row["capture"] >= recall_floor for row in result["per_unit"].values()
    )
    result["all_units_wilson_lower_at_floor"] = all(
        row["capture_wilson_lower_95"] >= recall_floor for row in result["per_unit"].values()
    )
    return result


def one_split(inventory: str, raw: dict, split: int) -> list[dict]:
    n = next(iter(raw["units"].values())).shape[0]
    rng = np.random.default_rng(SEED + 1000 * list(GRID).index(inventory) + split)
    perm = rng.permutation(n)
    a, b = n // 2, (3 * n) // 4
    development, calibration, test = perm[:a], perm[a:b], perm[b:]
    dev, cal, tst = orient_split(raw, development, calibration, test)
    dev_ties = rng.random(len(development))
    cal_ties = rng.random(len(calibration))
    test_ties = rng.random(len(test))
    orders_by_advance = make_orders(dev, dev_ties)
    direct_orders = orders_by_advance[TOP_SHARE]
    rows = []
    for risk in RISK_LEVELS:
        floor = 1.0 - risk
        two = choose_two_stage(cal, orders_by_advance, GRID[inventory], cal_ties, floor)
        direct = choose_direct(cal, direct_orders, GRID[inventory], cal_ties, floor)
        for policy_name, policy, orders in (
            ("two_stage", two, orders_by_advance[two["advance_share"]] if two else direct_orders),
            ("direct_fixed_short_form", direct, direct_orders),
        ):
            row = {
                "inventory": inventory, "split": split, "miss_risk": risk,
                "capture_floor": floor, "policy": policy["policy"] if policy else policy_name,
                "n_development": len(development), "n_calibration": len(calibration), "n_test": len(test),
                "selection_feasible": policy is not None,
            }
            if policy is not None:
                row.update({key: policy[key] for key in ("items_stage_1", "advance_share", "cost_share", "item_saving_share")})
                row["calibration"] = policy["calibration"]
                row["test"] = test_policy(tst, policy, orders, test_ties, floor)
            rows.append(row)
    return rows


def summary(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["inventory"], row["miss_risk"], row["policy"])].append(row)
    out = []
    for (inventory, risk, policy), group in sorted(grouped.items()):
        feasible = [r for r in group if r["selection_feasible"]]
        record = {
            "inventory": inventory, "miss_risk": risk, "capture_floor": 1 - risk,
            "policy": policy, "n_splits": len(group), "selection_feasible_splits": len(feasible),
            "selection_feasibility_share": len(feasible) / len(group),
        }
        if feasible:
            record.update({
                "median_cost_share": float(np.median([r["cost_share"] for r in feasible])),
                "median_item_saving_share": float(np.median([r["item_saving_share"] for r in feasible])),
                "median_test_min_capture": float(np.median([r["test"]["min_capture"] for r in feasible])),
                "median_test_median_capture": float(np.median([r["test"]["median_capture"] for r in feasible])),
                "test_all_units_point_floor_share": float(np.mean([r["test"]["all_units_point_capture_at_floor"] for r in feasible])),
                "test_all_units_wilson_floor_share": float(np.mean([r["test"]["all_units_wilson_lower_at_floor"] for r in feasible])),
            })
        out.append(record)
    return out


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = v21.load_raw()
    rows = []
    for inventory, spec in raw.items():
        for split in range(N_SPLITS):
            rows.extend(one_split(inventory, spec, split))
            print(f"{inventory}: split {split + 1}/{N_SPLITS}", flush=True)
    summaries = summary(rows)
    payload = {
        "status": "P0 feasibility only; not a submission result or external validation",
        "seed": SEED, "n_repeated_splits": N_SPLITS,
        "split": "50% development / 25% calibration / 25% untouched test, disjoint within inventory",
        "target": "full-form fixed-size top decile under a shared deterministic random tie key",
        "risk_rule": "policy must have every unit's two-sided 95% Wilson lower capture bound on calibration rows at least 1 - missed-target risk",
        "risk_levels": list(RISK_LEVELS), "advance_shares": list(ADVANCE_SHARES),
        "item_grids": {key: list(value) for key, value in GRID.items()},
        "scope": "internal full-form-referenced feasibility in four public web inventories; no external outcome, clinical claim, latent-truth claim, or transportable risk guarantee",
        "comparators": {
            "two_stage": "fixed recall-greedy first-stage form; advanced respondents complete remaining parent-form items",
            "direct_fixed_short_form": "fixed recall-greedy short form makes the capacity decision with no second stage",
            "irt_cat": "not estimated because an IRT model, item-bank calibration, CAT stopping rule, and their validation are absent",
        },
        "split_results": rows, "summary": summaries,
    }
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    fieldnames = list(dict.fromkeys(key for row in summaries for key in row))
    with OUT_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(summaries)
    print(f"wrote {OUT_JSON}")
    print(f"wrote {OUT_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
