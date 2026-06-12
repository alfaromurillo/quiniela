"""
Build scoreline probability distributions from WC historical data.

Data: WC 2014, 2018, 2022 — combined with exponential-decay weights.
  WC 2022 weight = 1.0, WC 2018 = γ, WC 2014 = γ².
  γ is estimated from the data by leave-one-tournament-out; see learn.py.

Score convention: for knockout games the quiniela counts goals in 90+30 min,
so we use the 'et' (extra time) score when available, else 'ft' (90 min).

Smoothing: Poisson product prior instead of uniform Laplace.
  For wins:  P_prior(w, l) ∝ λ_w^w/w! · λ_l^l/l!
  For draws: P_prior(g, g) ∝ (λ_d^g/g!)²
  Parameters λ are estimated by MLE from the γ-weighted combined data.
  SMOOTH_KAPPA total pseudo-observations are spread proportionally.

Adaptive WC 2026 update: pass extra_wins, extra_draws, delta to
build_distributions() to blend in-progress tournament results with
weight (1+delta) relative to the historical baseline.
"""
import json
import math
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
MAX_GOALS    = 5
SMOOTH_KAPPA = 5.0

# Chronological order (oldest first). Weight = γ^age, age=0 for most recent.
HISTORICAL_FILES = [
    DATA_DIR / "wc2014.json",
    DATA_DIR / "wc2018.json",
    DATA_DIR / "wc2022.json",
]


def _pois(k: int, lam: float) -> float:
    """Unnormalised Poisson PMF: lam^k / k!"""
    return lam**k / math.factorial(k)


def _poisson_prior(cells: list[tuple], *lams: float) -> dict:
    """
    Normalised Poisson product prior over a list of cells.
    For win cells (w, l): lams = (λ_w, λ_l).
    For draw cells (g, g): pass a single lam = λ_d (squared internally).
    """
    raw = {}
    for cell in cells:
        if len(lams) == 1:
            raw[cell] = _pois(cell[0], lams[0]) ** 2
        else:
            raw[cell] = _pois(cell[0], lams[0]) * _pois(cell[1], lams[1])
    total = sum(raw.values())
    return {s: v / total for s, v in raw.items()}


def _load_tournament(path) -> tuple[dict, dict]:
    """
    Load canonical win/draw counts from a single WC tournament file.
    Knockout scores use 'et' (120 min) when available, else 'ft' (90 min).
    Returns (win_counts, draw_counts) keyed by phase → canonical_score → int.
    """
    raw = json.loads(Path(path).read_text())
    win_counts  = {"group": {}, "knockout": {}}
    draw_counts = {"group": {}, "knockout": {}}

    for m in raw["matches"]:
        sc_raw = m.get("score", {})
        ft = sc_raw.get("ft")
        if not ft or len(ft) != 2:
            continue
        rnd = m.get("round", "").lower()
        phase = "knockout" if any(k in rnd for k in
                  ["round of", "quarter", "semi", "final", "third"]) else "group"
        # Quiniela counts 90+30 min goals; use et score for knockout when present
        if phase == "knockout":
            et = sc_raw.get("et")
            sc = et if (et and len(et) == 2) else ft
        else:
            sc = ft
        h, a = int(sc[0]), int(sc[1])
        if h > a:
            s = (h, a)
            win_counts[phase][s] = win_counts[phase].get(s, 0) + 1
        elif h < a:
            s = (a, h)
            win_counts[phase][s] = win_counts[phase].get(s, 0) + 1
        else:
            s = (h, a)
            draw_counts[phase][s] = draw_counts[phase].get(s, 0) + 1

    return win_counts, draw_counts


def get_tournament_counts() -> list[tuple[dict, dict]]:
    """
    Return list of (win_counts, draw_counts) for each available historical
    tournament, in chronological order (oldest first).
    """
    return [_load_tournament(p) for p in HISTORICAL_FILES if p.exists()]


def get_historical_counts(gamma: float = 1.0) -> tuple[dict, dict]:
    """
    γ-weighted combination of all historical tournaments.
    Most recent WC has weight 1.0; each older WC is multiplied by γ.
    """
    tournaments = get_tournament_counts()
    n = len(tournaments)
    combined_w = {"group": {}, "knockout": {}}
    combined_d = {"group": {}, "knockout": {}}
    for k, (wc, dc) in enumerate(tournaments):
        age   = (n - 1) - k          # 0 = most recent, 1 = one before, …
        weight = gamma ** age
        for phase in ["group", "knockout"]:
            for s, cnt in wc[phase].items():
                combined_w[phase][s] = combined_w[phase].get(s, 0) + weight * cnt
            for s, cnt in dc[phase].items():
                combined_d[phase][s] = combined_d[phase].get(s, 0) + weight * cnt
    return combined_w, combined_d


def build_distributions(
    gamma:       float      = 1.0,
    extra_wins:  dict | None = None,
    extra_draws: dict | None = None,
    delta:       float      = 0.0,
) -> dict:
    """
    Build scoreline distributions for all phases.
    Returns dist[phase]["neutral_win"|"draw"][canonical_score] = probability.

    gamma: historical decay (WC 2022=1.0, WC 2018=γ, WC 2014=γ²)
    extra_wins / extra_draws: WC 2026 in-progress results
    delta: WC 2026 weight = (1 + delta) relative to historical baseline
    """
    wc_raw, dc_raw = get_historical_counts(gamma)
    all_win_cells  = [(w, l) for w in range(MAX_GOALS + 1)
                               for l in range(MAX_GOALS + 1) if w > l]
    all_draw_cells = [(g, g) for g in range(MAX_GOALS + 1)]

    dist = {}
    for phase in ["group", "knockout"]:
        wc = wc_raw[phase]
        dc = dc_raw[phase]
        ew = (extra_wins  or {}).get(phase, {})
        ed = (extra_draws or {}).get(phase, {})

        # All observed scores (may include out-of-grid) for MLE of λ
        all_win_obs  = set(wc.keys()) | set(ew.keys())
        all_draw_obs = set(dc.keys()) | set(ed.keys())

        def wc_eff(s):
            return wc.get(s, 0) + (1 + delta) * ew.get(s, 0)

        def dc_eff(s):
            return dc.get(s, 0) + (1 + delta) * ed.get(s, 0)

        n_wins  = sum(wc_eff(s) for s in all_win_obs)
        n_draws = sum(dc_eff(s) for s in all_draw_obs)
        lam_w = max(sum(s[0] * wc_eff(s) for s in all_win_obs)  / n_wins,  0.5) if n_wins  else 2.0
        lam_l = max(sum(s[1] * wc_eff(s) for s in all_win_obs)  / n_wins,  0.1) if n_wins  else 0.6
        lam_d = max(sum(s[0] * dc_eff(s) for s in all_draw_obs) / n_draws, 0.1) if n_draws else 0.6

        wp = _poisson_prior(all_win_cells,  lam_w, lam_l)
        dp = _poisson_prior(all_draw_cells, lam_d)

        nw = {s: wc_eff(s) + SMOOTH_KAPPA * wp[s] for s in all_win_cells}
        nd = {s: dc_eff(s) + SMOOTH_KAPPA * dp[s] for s in all_draw_cells}
        tw, td = sum(nw.values()), sum(nd.values())
        dist[phase] = {
            "neutral_win": {s: v / tw for s, v in nw.items()},
            "draw":        {s: v / td for s, v in nd.items()},
        }
    return dist


_DIST = None

def get_distributions():
    """Cached baseline distribution (gamma=1.0, no WC 2026 data)."""
    global _DIST
    if _DIST is None:
        _DIST = build_distributions()
    return _DIST


def scoreline_probs(
    p_home_win: float, p_draw: float, p_away_win: float,
    phase: str,
    total_goals_probs: dict | None = None,
    spread_home: dict | None = None,
    spread_away: dict | None = None,
    dist: dict | None = None,
) -> dict:
    """
    Return P(home=a, away=b) for all (a,b) in 0..MAX_GOALS grid.

    dist: pre-built distribution from build_distributions(); if None, uses
      the cached baseline (gamma=1.0, delta=0).
    """
    if dist is None:
        dist = get_distributions()
    all_scores = [(a, b) for a in range(MAX_GOALS + 1) for b in range(MAX_GOALS + 1)]

    # ── Step 1: base distribution ──
    probs = {}
    for a, b in all_scores:
        if a > b:
            probs[(a, b)] = p_home_win * dist[phase]["neutral_win"].get((a, b), 0.0)
        elif a < b:
            probs[(a, b)] = p_away_win * dist[phase]["neutral_win"].get((b, a), 0.0)
        else:
            probs[(a, b)] = p_draw * dist[phase]["draw"].get((a, b), 0.0)

    # ── Step 2: reweight by Kalshi total-goals distribution ──
    if total_goals_probs:
        from collections import defaultdict
        total_wt = defaultdict(float)
        for (a, b), p in probs.items():
            total_wt[a + b] += p
        adjusted = {}
        for (a, b), p in probs.items():
            t = a + b
            kt = total_goals_probs.get(t, 0.0)
            ht = total_wt[t]
            adjusted[(a, b)] = p * (kt / ht) if (ht > 1e-12 and kt > 1e-12) else 0.0
        probs = adjusted

    # ── Step 3: reweight by Kalshi spread (goal margin) ──
    for spread, is_home in [(spread_home, True), (spread_away, False)]:
        if not spread:
            continue
        from collections import defaultdict
        margin_wt = defaultdict(float)
        for (a, b), p in probs.items():
            if is_home and a > b:
                margin_wt[a - b] += p
            elif not is_home and b > a:
                margin_wt[b - a] += p
        MAX_MULT = 4.0
        adjusted = dict(probs)
        for (a, b), p in probs.items():
            margin = (a - b) if is_home else (b - a)
            if margin <= 0:
                continue
            k = min(margin, 4)
            ks = spread.get(k, 0.0)
            hs = margin_wt[margin]
            if hs > 1e-12 and ks > 1e-12:
                group_hs = sum(margin_wt[m] for m in margin_wt if min(m, 4) == k)
                mult = min(ks / group_hs, MAX_MULT) if group_hs > 1e-12 else 0.0
                adjusted[(a, b)] = p * mult
        probs = adjusted

    # ── Normalise ──
    total = sum(probs.values())
    if total > 1e-12:
        probs = {s: v / total for s, v in probs.items()}
    return probs
