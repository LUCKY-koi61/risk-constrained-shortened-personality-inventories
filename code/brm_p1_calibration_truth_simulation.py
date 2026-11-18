#!/usr/bin/env python3
"""P1: does full-score calibrated two-stage capture protect latent-tail capture?

This is a simulation-only extension of P0.  It uses known 2PL parameters and
known theta, while the two-stage policy is calibrated only against the
full-score top decile.  A fixed-length oracle-parameter CAT is an explicitly
optimistic benchmark, not an empirical CAT analysis.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
SEED, N_REPS, N, M = 20261118 + 2, 20, 3_000, 40
Q, FLOORS = .10, (.80, .90)
KS, ADVANCES, Z = (4, 8, 12), (.15, .25, .35, .50), 1.959963984540054
GRID = np.linspace(-4, 4, 81)
LOG_PRIOR = -.5 * GRID**2


def top(score, q, key, allowed=None):
    idx = np.arange(len(score)) if allowed is None else np.flatnonzero(allowed)
    n = int(round(q * len(score)))
    order = idx[np.lexsort((key[idx], -score[idx]))]
    out = np.zeros(len(score), bool); out[order[:n]] = True
    return out


def lower(hits, total):
    p = hits / total; d = 1 + Z**2 / total
    return (p + Z**2/(2*total) - Z*math.sqrt(p*(1-p)/total + Z**2/(4*total**2))) / d


def simulate(rng, quality, n):
    theta = rng.standard_normal(n)
    a = np.linspace(.55, .95, M) if quality == "low" else np.linspace(1.0, 1.6, M)
    b = np.linspace(-1.8, 1.8, M)
    p = 1 / (1 + np.exp(-np.outer(theta, a) + a*b))
    return theta, rng.binomial(1, p).astype(np.int8), a, b


def choose(cal_theta, cal_x, a, key, floor):
    full = cal_x.sum(1); reference = top(full, Q, key)
    orders = np.argsort(-a)
    options = []
    for k in KS:
        short = cal_x[:, orders[:k]].sum(1)
        for advance in ADVANCES:
            passed = top(short, advance, key)
            hits = int((reference & passed).sum()); bound = lower(hits, int(reference.sum()))
            if bound >= floor:
                cost = k/M + advance*(1-k/M)
                options.append((cost, k, advance, bound))
    return min(options) if options else None


def cat_scores(x, a, b, budget):
    """EAP CAT scores using true 2PL parameters; no calibration uncertainty."""
    out = np.empty(len(x))
    for person, row in enumerate(x):
        lp, remain = LOG_PRIOR.copy(), list(range(M))
        for _ in range(budget):
            w = np.exp(lp - lp.max()); estimate = float((w*GRID).sum()/w.sum())
            info = [a[j]**2*(1/(1+np.exp(-a[j]*(estimate-b[j]))))*(1-1/(1+np.exp(-a[j]*(estimate-b[j])))) for j in remain]
            j = remain.pop(int(np.argmax(info)))
            p = 1/(1+np.exp(-a[j]*(GRID-b[j])))
            lp += np.log(p if row[j] else 1-p)
        w = np.exp(lp-lp.max()); out[person] = (w*GRID).sum()/w.sum()
    return out


def run_one(rng, quality, floor):
    theta_d, x_d, a, b = simulate(rng, quality, N)
    theta_c, x_c, _, _ = simulate(rng, quality, N//2)
    theta_t, x_t, _, _ = simulate(rng, quality, N//2)
    # Development is intentionally unused for configuration choice: item order
    # is prespecified by known parameters in this oracle DGP.
    _ = (theta_d, x_d)
    cal_key, test_key = rng.random(len(x_c)), rng.random(len(x_t))
    choice = choose(theta_c, x_c, a, cal_key, floor)
    if choice is None:
        return {"quality": quality, "floor": floor, "feasible": False}
    cost, k, advance, cal_bound = choice
    order = np.argsort(-a)
    full, short = x_t.sum(1), x_t[:, order[:k]].sum(1)
    full_ref, theta_ref, passed = top(full,Q,test_key), top(theta_t,Q,test_key), top(short,advance,test_key)
    final = top(full,Q,test_key,passed)
    def cap(ref): return float((ref & final).sum()/ref.sum())
    budget = max(1, min(M, int(round(cost*M))))
    cat = top(cat_scores(x_t,a,b,budget),Q,test_key)
    return {
        "quality": quality, "floor": floor, "feasible": True,
        "cost_share": cost, "k": k, "advance": advance, "calibration_lower": cal_bound,
        "test_full_reference_capture": cap(full_ref),
        "test_latent_capture_two_stage": cap(theta_ref),
        "test_latent_capture_full_form": float((theta_ref & full_ref).sum()/theta_ref.sum()),
        "cat_budget_items": budget,
        "test_latent_capture_oracle_cat": float((theta_ref & cat).sum()/theta_ref.sum()),
    }


def main():
    rng=np.random.default_rng(SEED); rows=[]
    for quality in ("low","high"):
        for floor in FLOORS:
            for rep in range(N_REPS):
                row=run_one(rng,quality,floor); row["replicate"]=rep; rows.append(row)
            print(quality, floor, "complete", flush=True)
    keys=["quality","floor"]
    summary=[]
    for quality in ("low","high"):
        for floor in FLOORS:
            rs=[r for r in rows if r["quality"]==quality and r["floor"]==floor and r["feasible"]]
            rec={"quality":quality,"full_score_capture_floor":floor,"n_reps":N_REPS,"feasible_reps":len(rs)}
            for field in ("cost_share","test_full_reference_capture","test_latent_capture_two_stage","test_latent_capture_full_form","test_latent_capture_oracle_cat"):
                rec["mean_"+field]=float(np.mean([r[field] for r in rs])) if rs else None
            summary.append(rec)
    OUT.mkdir(parents=True,exist_ok=True)
    payload={"status":"simulation only; oracle CAT uses true 2PL parameters","seed":SEED,"n_reps":N_REPS,"n_development":N,"n_calibration":N//2,"n_test":N//2,"full_items":M,"top_capacity":Q,"rows":rows,"summary":summary}
    (OUT/"brm_p1_calibration_truth_simulation.json").write_text(json.dumps(payload,indent=2))
    with (OUT/"brm_p1_calibration_truth_simulation_summary.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0])); w.writeheader();w.writerows(summary)
    assert all(0<=r[f]<=1 for r in rows if r["feasible"] for f in ("cost_share","test_full_reference_capture","test_latent_capture_two_stage","test_latent_capture_full_form","test_latent_capture_oracle_cat"))

if __name__=="__main__": main()
