"""
Adaptive learning: γ decay for historical WCs, δ weight for WC 2026.

γ (historical decay):
  WC 2022 = weight 1.0, WC 2018 = γ, WC 2014 = γ².
  Estimated by leave-one-tournament-out MLE. When predicting WC 2022
  from 2018×1.0 + 2014×γ, find γ that maximises log P(WC 2022 results).
  Prior: Normal(1, 0.3) truncated at [0, 1] — weak push toward equal weights.

δ (current-tournament weight):
  Each WC 2026 result counts as (1+δ) × a historical result.
  Estimated by leave-one-jornada-out MAP with HalfNormal(σ=1) prior.

Jornada sequence: Matchday 1→2→3 (groups), Round of 32, Round of 16,
Quarter-finals, Semi-finals, Third place / Final.
"""
import json
import math
from pathlib import Path

ROOT = Path(__file__).parent.parent

SIGMA_DELTA  = 1.0    # HalfNormal prior scale for δ
SIGMA_GAMMA  = 0.3    # Normal(1, σ) truncated prior scale for γ
GAMMA_MIN    = 0.10   # search lower bound
GAMMA_STEPS  = 90     # grid: 0.10, 0.11, …, 1.00
DELTA_MAX    = 8.0
DELTA_STEPS  = 160    # grid: 0.00, 0.05, …, 8.00

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
        hs  = r.get("home_score")
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
    """Return (wins_by_phase, draws_by_phase) counts for given WC 2026 games."""
    wins  = {"group": {}, "knockout": {}}
    draws = {"group": {}, "knockout": {}}
    for g in canonical_games:
        ph, s = g["phase"], g["canonical"]
        if g["result_type"] == "win":
            wins[ph][s] = wins[ph].get(s, 0) + 1
        else:
            draws[ph][s] = draws[ph].get(s, 0) + 1
    return wins, draws


# ── shared helpers ────────────────────────────────────────────────────────────

def _mle_lambdas(wc: dict, dc: dict) -> tuple[float, float, float]:
    """MLE Poisson parameters from win/draw count dicts (may be float counts)."""
    n_wins  = sum(wc.values())
    n_draws = sum(dc.values())
    lam_w = max(sum(w*c for (w,l),c in wc.items()) / n_wins,  0.5) if n_wins  else 2.0
    lam_l = max(sum(l*c for (w,l),c in wc.items()) / n_wins,  0.1) if n_wins  else 0.6
    lam_d = max(sum(g*c for (g,_),c in dc.items()) / n_draws, 0.1) if n_draws else 0.6
    return lam_w, lam_l, lam_d


def _build_priors(wc_by_phase: dict, dc_by_phase: dict) -> tuple[dict, dict]:
    """Normalised Poisson priors for each phase, derived from MLE λ."""
    from model.historical import MAX_GOALS, _poisson_prior
    all_win_cells  = [(w, l) for w in range(MAX_GOALS+1) for l in range(MAX_GOALS+1) if w > l]
    all_draw_cells = [(g, g) for g in range(MAX_GOALS+1)]
    prior_w, prior_d = {}, {}
    for phase in ["group", "knockout"]:
        lw, ll, ld = _mle_lambdas(wc_by_phase.get(phase, {}), dc_by_phase.get(phase, {}))
        prior_w[phase] = _poisson_prior(all_win_cells,  lw, ll)
        prior_d[phase] = _poisson_prior(all_draw_cells, ld)
    return prior_w, prior_d


def _log_likelihood_counts(
    test_w: dict, test_d: dict,
    train_w: dict, train_d: dict,
    prior_w: dict, prior_d: dict,
    kappa: float,
) -> float:
    """
    Log-likelihood of test score counts under the smoothed model trained on
    train_w/d + kappa × prior.  Out-of-grid scores are skipped.
    Accepts float counts (γ-weighted).
    """
    from model.historical import MAX_GOALS
    all_win_cells  = [(w, l) for w in range(MAX_GOALS+1) for l in range(MAX_GOALS+1) if w > l]
    all_draw_cells = [(g, g) for g in range(MAX_GOALS+1)]
    win_cell_set  = set(all_win_cells)
    draw_cell_set = set(all_draw_cells)

    ll = 0.0
    for phase in ["group", "knockout"]:
        tw = train_w.get(phase, {})
        td = train_d.get(phase, {})
        pw = prior_w.get(phase, {})
        pd = prior_d.get(phase, {})

        z_w = sum(tw.get(s, 0) + kappa*pw.get(s, 0) for s in all_win_cells)
        z_d = sum(td.get(s, 0) + kappa*pd.get(s, 0) for s in all_draw_cells)

        for s, cnt in test_w.get(phase, {}).items():
            if s not in win_cell_set or cnt <= 0:
                continue
            p = (tw.get(s, 0) + kappa*pw.get(s, 0)) / z_w if z_w > 1e-12 else 0.0
            ll += cnt * math.log(max(p, 1e-12))

        for s, cnt in test_d.get(phase, {}).items():
            if s not in draw_cell_set or cnt <= 0:
                continue
            p = (td.get(s, 0) + kappa*pd.get(s, 0)) / z_d if z_d > 1e-12 else 0.0
            ll += cnt * math.log(max(p, 1e-12))

    return ll


def _log_likelihood_games(
    delta: float,
    hist_w: dict, hist_d: dict,
    extra_w: dict, extra_d: dict,
    test_games: list[dict],
    prior_w: dict, prior_d: dict,
    kappa: float,
) -> float:
    """
    Sum of log P(score_i | delta) over in-grid test_games.
    P(s | delta, phase, rt) ∝ hist[ph][s] + (1+delta)·extra[ph][s] + κ·prior[ph][s]
    """
    from model.historical import MAX_GOALS
    all_win_cells  = [(w, l) for w in range(MAX_GOALS+1) for l in range(MAX_GOALS+1) if w > l]
    all_draw_cells = [(g, g) for g in range(MAX_GOALS+1)]
    z_cache: dict = {}

    def get_z(phase, rt):
        key = (phase, rt)
        if key not in z_cache:
            h = hist_w.get(phase, {}) if rt == "win" else hist_d.get(phase, {})
            e = extra_w.get(phase, {}) if rt == "win" else extra_d.get(phase, {})
            p = prior_w.get(phase, {}) if rt == "win" else prior_d.get(phase, {})
            cells = all_win_cells if rt == "win" else all_draw_cells
            z_cache[key] = sum(
                h.get(s, 0) + (1+delta)*e.get(s, 0) + kappa*p.get(s, 0)
                for s in cells
            )
        return z_cache[key]

    ll = 0.0
    for g in test_games:
        if not g["in_grid"]:
            continue
        ph, rt, s = g["phase"], g["result_type"], g["canonical"]
        h = hist_w if rt == "win" else hist_d
        e = extra_w if rt == "win" else extra_d
        p = prior_w if rt == "win" else prior_d
        num = (h.get(ph, {}).get(s, 0) +
               (1+delta) * e.get(ph, {}).get(s, 0) +
               kappa * p.get(ph, {}).get(s, 0))
        z = get_z(ph, rt)
        ll += math.log(max(num / z if z > 1e-12 else 0.0, 1e-12))
    return ll


# ── γ estimation ──────────────────────────────────────────────────────────────

def estimate_gamma() -> float:
    """
    Leave-one-tournament-out MLE for the historical decay γ.

    For each test tournament k (0-indexed, chronological):
      - Training data = tournaments 0..k-1, with most recent at weight 1.0
        and each older one multiplied by γ.
      - Evaluate log P(WC_k | trained model).
    Objective = sum of leave-one-out log-likelihoods + Normal(1, σ_γ) log-prior.

    Returns γ ∈ [GAMMA_MIN, 1.0].  Defaults to 1.0 when insufficient data.
    """
    from model.historical import get_tournament_counts, SMOOTH_KAPPA

    tournaments = get_tournament_counts()
    if len(tournaments) < 2:
        return 1.0

    gamma_grid = [GAMMA_MIN + i * (1.0 - GAMMA_MIN) / GAMMA_STEPS
                  for i in range(GAMMA_STEPS + 1)]
    best_gamma, best_obj = 1.0, -1e18

    for gamma in gamma_grid:
        obj = 0.0

        for test_idx in range(1, len(tournaments)):
            # Build training data: most recent training WC has weight 1.0,
            # each older training WC is multiplied by γ.
            train_w = {"group": {}, "knockout": {}}
            train_d = {"group": {}, "knockout": {}}
            for train_idx in range(test_idx):
                age    = (test_idx - 1) - train_idx   # 0 = most recent training WC
                weight = gamma ** age
                wc, dc = tournaments[train_idx]
                for phase in ["group", "knockout"]:
                    for s, cnt in wc[phase].items():
                        train_w[phase][s] = train_w[phase].get(s, 0) + weight * cnt
                    for s, cnt in dc[phase].items():
                        train_d[phase][s] = train_d[phase].get(s, 0) + weight * cnt

            prior_w, prior_d = _build_priors(train_w, train_d)

            test_wc, test_dc = tournaments[test_idx]
            obj += _log_likelihood_counts(
                test_wc, test_dc, train_w, train_d, prior_w, prior_d, SMOOTH_KAPPA,
            )

        # Weak Normal(1, σ_γ) prior: penalise distance from γ=1.0
        obj -= (gamma - 1.0)**2 / (2 * SIGMA_GAMMA**2)

        if obj > best_obj:
            best_obj   = obj
            best_gamma = gamma

    return best_gamma


# ── δ estimation ──────────────────────────────────────────────────────────────

def estimate_delta(
    canonical: list[dict],
    gamma:     float = 1.0,
    sigma:     float = SIGMA_DELTA,
) -> float:
    """
    MAP estimate of δ via leave-one-jornada-out likelihood.

    Uses γ-weighted historical baseline so δ is always relative to
    the same combined-historical distribution used for predictions.
    Returns 0.0 when no jornada pair (train + test) is available.
    """
    from model.historical import get_historical_counts, SMOOTH_KAPPA

    if not canonical:
        return 0.0

    hist_w, hist_d = get_historical_counts(gamma)
    prior_w, prior_d = _build_priors(hist_w, hist_d)

    all_jornadas = sorted({g["jornada_idx"] for g in canonical if g["jornada_idx"] >= 0})

    # Precompute train/test splits per jornada
    jornada_data = {}
    for test_j in all_jornadas:
        train = [g for g in canonical if g["jornada_idx"] < test_j]
        test  = [g for g in canonical if g["jornada_idx"] == test_j]
        if not train or not any(g["in_grid"] for g in test):
            continue
        jornada_data[test_j] = (count_2026(train), test)

    if not jornada_data:
        return 0.0

    delta_grid = [i * DELTA_MAX / DELTA_STEPS for i in range(DELTA_STEPS + 1)]
    best_delta, best_lp = 0.0, -1e18

    for delta in delta_grid:
        lp = 0.0
        for (n26_w, n26_d), test in jornada_data.values():
            lp += _log_likelihood_games(
                delta, hist_w, hist_d, n26_w, n26_d,
                test, prior_w, prior_d, SMOOTH_KAPPA,
            )
        lp -= delta**2 / (2 * sigma**2)
        if lp > best_lp:
            best_lp    = lp
            best_delta = delta

    return best_delta
