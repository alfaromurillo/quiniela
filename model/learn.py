"""
Adaptive learning: incorporate WC 2026 results into historical model.

Delta (δ) weights current-tournament data: each WC 2026 result counts
as (1+δ) × a WC 2022 result in both the Poisson MLE and smoothing.

δ is estimated by leave-one-jornada-out MAP with a HalfNormal(σ=1)
prior. For jornada k, the model is trained on all jornadas < k and
evaluated on jornada k. The total objective is the sum of these
log-likelihoods across all evaluable jornadas, minus the prior term.

Jornada sequence: Matchday 1→2→3 (groups), Round of 32, Round of 16,
Quarter-finals, Semi-finals, Third place / Final.

The Poisson λ parameters update after every new result (each run of
predict.py recomputes λ from the combined data for the current δ).
"""
import json
import math
from pathlib import Path

ROOT = Path(__file__).parent.parent

SIGMA_DELTA = 1.0    # HalfNormal prior scale — see modelo.html for rationale
DELTA_MAX   = 8.0    # search upper bound for δ
DELTA_STEPS = 160    # grid resolution: 0.0, 0.05, ..., 8.0

JORNADA_ORDER = [
    "Matchday 1", "Matchday 2", "Matchday 3",
    "Round of 32", "Round of 16",
    "Quarter-finals", "Semi-finals",
    "Third place", "Final",
]


def _jornada_idx(round_name: str) -> int:
    rn = round_name.lower()
    for i, name in enumerate(JORNADA_ORDER):
        if name.lower() in rn:
            return i
    return -1


def _canonical(home: int, away: int) -> tuple:
    """Return (result_type, canonical_score) in (winner, loser) form."""
    if home > away:
        return "win", (home, away)
    if home < away:
        return "win", (away, home)
    return "draw", (home, home)


def load_canonical(results_json: dict, schedule_json: dict) -> list[dict]:
    """
    Parse results.json + schedule.json into canonical game records.
    Only includes matches with status="final".
    Each record: {id, jornada_idx, phase, result_type, canonical, in_grid}
    """
    from model.historical import MAX_GOALS
    by_id = {str(m["id"]): m for m in schedule_json["matches"]}
    out = []
    for mid_str, r in results_json.get("matches", {}).items():
        if r.get("status") != "final":
            continue
        m = by_id.get(mid_str)
        if not m:
            continue
        hs = r.get("home_score")
        as_ = r.get("away_score")
        if hs is None or as_ is None:
            continue
        rt, canon = _canonical(int(hs), int(as_))
        out.append({
            "id":          int(mid_str),
            "jornada_idx": _jornada_idx(m["round"]),
            "phase":       m["phase"],
            "result_type": rt,
            "canonical":   canon,
            "in_grid":     canon[0] <= MAX_GOALS,
        })
    return out


def count_2026(canonical_games: list[dict]) -> tuple[dict, dict]:
    """
    Return (wins_by_phase, draws_by_phase) canonical counts.
    Includes out-of-grid scores so λ MLE uses all observed goals.
    """
    wins  = {"group": {}, "knockout": {}}
    draws = {"group": {}, "knockout": {}}
    for g in canonical_games:
        ph, s = g["phase"], g["canonical"]
        if g["result_type"] == "win":
            wins[ph][s] = wins[ph].get(s, 0) + 1
        else:
            draws[ph][s] = draws[ph].get(s, 0) + 1
    return wins, draws


def _compute_priors(n2022_w: dict, n2022_d: dict) -> tuple[dict, dict]:
    """Normalised Poisson priors from WC 2022 MLE λ (fixed, used in log-posterior)."""
    from model.historical import MAX_GOALS, _poisson_prior
    all_win_cells  = [(w, l) for w in range(MAX_GOALS+1) for l in range(MAX_GOALS+1) if w > l]
    all_draw_cells = [(g, g) for g in range(MAX_GOALS+1)]

    prior_w = {}
    prior_d = {}
    for phase in ["group", "knockout"]:
        wc = n2022_w.get(phase, {})
        dc = n2022_d.get(phase, {})
        n_wins  = sum(wc.values())
        n_draws = sum(dc.values())
        lam_w = max(sum(w*c for (w,l),c in wc.items()) / n_wins,  0.5) if n_wins  else 2.0
        lam_l = max(sum(l*c for (w,l),c in wc.items()) / n_wins,  0.1) if n_wins  else 0.6
        lam_d = max(sum(g*c for (g,_),c in dc.items()) / n_draws, 0.1) if n_draws else 0.6
        prior_w[phase] = _poisson_prior(all_win_cells,  lam_w, lam_l)
        prior_d[phase] = _poisson_prior(all_draw_cells, lam_d)
    return prior_w, prior_d


def _log_likelihood(
    delta: float,
    n2022_w: dict, n2022_d: dict,
    n2026_w: dict, n2026_d: dict,
    test_games: list[dict],
    prior_w: dict, prior_d: dict,
    kappa: float,
) -> float:
    """
    Sum of log P(score_i | delta) over in-grid test games.
    P(s | delta, phase, rt) ∝ n2022[ph][s] + (1+delta)·n2026[ph][s] + κ·prior[ph][s]
    """
    from model.historical import MAX_GOALS
    all_win_cells  = [(w, l) for w in range(MAX_GOALS+1) for l in range(MAX_GOALS+1) if w > l]
    all_draw_cells = [(g, g) for g in range(MAX_GOALS+1)]

    z_cache: dict = {}

    def get_z(phase, rt):
        key = (phase, rt)
        if key not in z_cache:
            n22 = n2022_w.get(phase, {}) if rt == "win" else n2022_d.get(phase, {})
            n26 = n2026_w.get(phase, {}) if rt == "win" else n2026_d.get(phase, {})
            pr  = prior_w.get(phase, {}) if rt == "win" else prior_d.get(phase, {})
            cells = all_win_cells if rt == "win" else all_draw_cells
            z_cache[key] = sum(
                n22.get(s, 0) + (1+delta)*n26.get(s, 0) + kappa*pr.get(s, 0)
                for s in cells
            )
        return z_cache[key]

    ll = 0.0
    for g in test_games:
        if not g["in_grid"]:
            continue
        ph, rt, s = g["phase"], g["result_type"], g["canonical"]
        n22 = n2022_w if rt == "win" else n2022_d
        n26 = n2026_w if rt == "win" else n2026_d
        pr  = prior_w  if rt == "win" else prior_d
        num = (n22.get(ph, {}).get(s, 0) +
               (1+delta) * n26.get(ph, {}).get(s, 0) +
               kappa * pr.get(ph, {}).get(s, 0))
        z = get_z(ph, rt)
        ll += math.log(max(num / z if z > 1e-12 else 0.0, 1e-12))
    return ll


def estimate_delta(canonical: list[dict], sigma: float = SIGMA_DELTA) -> float:
    """
    MAP estimate of δ by maximising the sum of leave-one-jornada-out
    log-likelihoods plus a HalfNormal(sigma) log-prior.

    Returns 0.0 when no jornada pair (train + test) is available.
    """
    from model.historical import get_wc2022_counts, SMOOTH_KAPPA

    if not canonical:
        return 0.0

    n2022_w, n2022_d = get_wc2022_counts()
    prior_w, prior_d = _compute_priors(n2022_w, n2022_d)

    all_jornadas = sorted({g["jornada_idx"] for g in canonical if g["jornada_idx"] >= 0})

    # Precompute train/test data per jornada
    jornada_data = {}
    for test_j in all_jornadas:
        train = [g for g in canonical if g["jornada_idx"] < test_j]
        test  = [g for g in canonical if g["jornada_idx"] == test_j]
        if not train or not any(g["in_grid"] for g in test):
            continue
        jornada_data[test_j] = (count_2026(train), test)

    if not jornada_data:
        return 0.0

    # Grid search for MAP δ
    delta_grid = [i * DELTA_MAX / DELTA_STEPS for i in range(DELTA_STEPS + 1)]
    best_delta, best_lp = 0.0, -1e18

    for delta in delta_grid:
        lp = 0.0
        for (n26_w, n26_d), test in jornada_data.values():
            lp += _log_likelihood(delta, n2022_w, n2022_d, n26_w, n26_d,
                                   test, prior_w, prior_d, SMOOTH_KAPPA)
        lp -= delta**2 / (2 * sigma**2)   # HalfNormal log-prior
        if lp > best_lp:
            best_lp = lp
            best_delta = delta

    return best_delta
