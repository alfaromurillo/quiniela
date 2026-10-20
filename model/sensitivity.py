"""
Uncertainty and sensitivity of the model's parameters (article, Results).

  - α: posterior on a fine grid with all Kalshi matches, under the
    Normal(1.10, 0.2²) prior and under a flat prior; 95% equal-tailed
    interval and P(α > 1).
  - γ: log-likelihood of each forward-validation fold.  The 2018 fold
    trains on 2014 alone, whose weight γ⁰ = 1 does not depend on γ, so
    only the 2022 fold identifies γ.
  - κ: the pseudo-prospective evaluation of the final method rerun with
    other amounts of Poisson smoothing (γ and δ re-estimated with each).
  - γ: the same evaluation with γ fixed at the ends of its 95% interval,
    at the maximum of the 2022 fold, at the estimate and at 1.

Run from repo root:
    python model/sensitivity.py              # prints a summary
    python model/sensitivity.py --json F     # also writes it to F
"""
import json
import math
import statistics as st
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from model import historical, learn, prospective

KAPPAS = (1.0, 2.5, 5.0, 10.0, 20.0)
GAMMAS = (0.41, 0.62, 0.84, 0.99, 1.0)


# ── α ────────────────────────────────────────────────────────────────────────

def alpha_data() -> list[tuple]:
    _, matches, results, entries = prospective.load()
    data = []
    for m in matches:
        e = entries[m["id"]]
        if e.get("source") != "kalshi":
            continue
        r = results[str(m["id"])]
        data.append((e["home_win"], e["draw"], e["away_win"],
                     prospective._outcome(r["home_score"], r["away_score"])))
    return data


def _interval(grid: np.ndarray, logp: np.ndarray) -> dict:
    w = np.exp(logp - logp.max())
    w /= w.sum()
    cdf = np.cumsum(w)
    return {"map": float(grid[w.argmax()]),
            "mean": float((grid * w).sum()),
            "lo": float(grid[np.searchsorted(cdf, 0.025)]),
            "hi": float(grid[np.searchsorted(cdf, 0.975)]),
            "p_gt_1": float(w[grid > 1].sum())}


def alpha_posterior(data=None) -> dict:
    data = data or alpha_data()
    grid = np.linspace(0.5, 2.5, 4001)
    ll = np.zeros_like(grid)
    for ph, pd, pa, outcome in data:
        p = np.array([ph, pd, pa])
        i = ["home_win", "draw", "away_win"].index(outcome)
        q = p[None, :] ** grid[:, None]
        ll += np.log(q[:, i] / q.sum(axis=1))
    prior = -(grid - learn.ALPHA_0) ** 2 / (2 * learn.SIGMA_ALPHA ** 2)
    # likelihood-ratio statistic for α = 1 against the MLE
    lr = 2 * (ll.max() - ll[np.argmin(abs(grid - 1))])
    return {"n": len(data), "prior": _interval(grid, ll + prior),
            "flat": _interval(grid, ll), "lr_alpha1": float(lr)}


# ── γ ────────────────────────────────────────────────────────────────────────

def gamma_folds() -> dict:
    """Predictive log-likelihood of each fold over the γ grid."""
    tournaments = historical.get_tournament_counts()
    grid = np.linspace(learn.GAMMA_MIN, 1.0, 91)
    folds = {2018: [], 2022: []}
    for gamma in grid:
        for test_idx, year in ((1, 2018), (2, 2022)):
            train_w = {"group": {}, "knockout": {}}
            train_d = {"group": {}, "knockout": {}}
            for train_idx in range(test_idx):
                weight = gamma ** ((test_idx - 1) - train_idx)
                wc, dc = tournaments[train_idx]
                for ph in ("group", "knockout"):
                    for s, c in wc[ph].items():
                        train_w[ph][s] = train_w[ph].get(s, 0) + weight * c
                    for s, c in dc[ph].items():
                        train_d[ph][s] = train_d[ph].get(s, 0) + weight * c
            pw, pd = learn._build_priors(train_w, train_d)
            folds[year].append(learn._log_likelihood_counts(
                *tournaments[test_idx], train_w, train_d, pw, pd,
                historical.SMOOTH_KAPPA))
    f18, f22 = np.array(folds[2018]), np.array(folds[2022])
    prior = -(grid - 1.0) ** 2 / (2 * learn.SIGMA_GAMMA ** 2)
    return {"range_2018": float(f18.max() - f18.min()),
            "range_2022": float(f22.max() - f22.min()),
            "ll22_at_1": float(f22[-1]),
            "ll22_max": float(f22.max()),
            "gamma_mle_2022": float(grid[f22.argmax()]),
            "posterior": _interval(grid, f18 + f22 + prior)}


# ── κ ────────────────────────────────────────────────────────────────────────

def kappa_sensitivity(kappas=KAPPAS) -> list[dict]:
    base = historical.SMOOTH_KAPPA
    out, ref = [], None
    try:
        for kappa in kappas:
            historical.SMOOTH_KAPPA = kappa
            prospective._ROWS = None
            rows = prospective.run()
            ks = [r for r in rows if r["kalshi"]]
            picks = [r["pick"]["final"] for r in rows]
            if kappa == base:
                ref = picks
            out.append({
                "kappa": kappa,
                "gamma": learn.estimate_gamma(),
                "delta_final": rows[-1]["delta"],
                "points": sum(r["pts"]["final"] for r in rows),
                "points_hist": sum(r["pts"]["hist"] for r in rows),
                "draw_picks": sum(p[0] == p[1] for p in picks),
                "log_final": st.mean(r["log"]["final"] for r in ks),
                "picks": picks})
    finally:
        historical.SMOOTH_KAPPA = base
        prospective._ROWS = None
    for o in out:
        o["differ_vs_base"] = sum(p != q for p, q in zip(o.pop("picks"), ref))
    return out


def gamma_sensitivity(gammas=GAMMAS) -> list[dict]:
    est = prospective.estimate_gamma
    base = est()
    out, ref = [], None
    try:
        for gamma in (base,) + tuple(gammas):
            prospective.estimate_gamma = lambda g=gamma: g
            prospective._ROWS = None
            rows = prospective.run()
            ks = [r for r in rows if r["kalshi"]]
            picks = [r["pick"]["final"] for r in rows]
            if ref is None:
                ref = picks
                continue
            out.append({
                "gamma": gamma,
                "points": sum(r["pts"]["final"] for r in rows),
                "log_final": st.mean(r["log"]["final"] for r in ks),
                "differ_vs_base": sum(p != q for p, q in zip(picks, ref))})
    finally:
        prospective.estimate_gamma = est
        prospective._ROWS = None
    return out


def main():
    stats = {"alpha": alpha_posterior(), "gamma": gamma_folds(),
             "kappa": kappa_sensitivity(), "gamma_eval": gamma_sensitivity()}
    print(json.dumps(stats, indent=1))
    if "--json" in sys.argv:
        Path(sys.argv[sys.argv.index("--json") + 1]).write_text(
            json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
