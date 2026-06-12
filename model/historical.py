"""
Build scoreline probability distributions from WC 2022 historical data.

Key design: WC 2022 (Qatar) was fully neutral-venue, so we collapse
home_win/away_win into a single "winner" distribution expressed as
(winner_goals, loser_goals). This removes spurious home/away bias.
The three true host nations of WC 2026 (USA, Mexico, Canada) have any
home advantage already priced into the Kalshi market probabilities.

Smoothing: Poisson product prior instead of uniform Laplace.
  For wins:  P_prior(w, l) ∝ λ_w^w/w! · λ_l^l/l!
  For draws: P_prior(g, g) ∝ (λ_d^g/g!)²
  Parameters λ estimated by MLE from combined WC 2022 + WC 2026 data.
  SMOOTH_KAPPA total pseudo-observations are spread over cells
  proportionally to the prior, so 2-2 >> 5-4 >> 5-5 for zero-count
  cells, while observed cells remain close to their empirical rate.

Adaptive update: pass extra_wins, extra_draws, delta to
build_distributions() to blend WC 2026 results into the model.
Each WC 2026 result counts as (1+delta) × a WC 2022 result.
"""
import json
import math
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
MAX_GOALS    = 5    # prediction grid: 0..MAX_GOALS goals per team
SMOOTH_KAPPA = 5.0  # total pseudo-observations added via the prior


def _pois(k: int, lam: float) -> float:
    """Unnormalised Poisson PMF: lam^k / k!"""
    return lam**k / math.factorial(k)


def _poisson_prior(cells: list[tuple], *lams: float) -> dict:
    """
    Normalised Poisson product prior over a list of cells.
    Each cell is a tuple of goals; lams[i] is the rate for dimension i.
    E.g. for win cells (w, l): lams = (λ_w, λ_l).
    For draw cells (g, g): pass g once with lam = λ_d and square it.
    """
    raw = {}
    for cell in cells:
        if len(lams) == 1:            # draw: (g,g), use λ_d twice
            raw[cell] = _pois(cell[0], lams[0]) ** 2
        else:                         # win: (w,l)
            raw[cell] = _pois(cell[0], lams[0]) * _pois(cell[1], lams[1])
    total = sum(raw.values())
    return {s: v / total for s, v in raw.items()}


def _load_wc2022_counts():
    """Load WC 2022 canonical counts by phase. Returns regular dicts."""
    raw = json.loads((DATA_DIR / "wc2022.json").read_text())
    win_counts  = {"group": {}, "knockout": {}}
    draw_counts = {"group": {}, "knockout": {}}

    for m in raw["matches"]:
        score = m.get("score", {}).get("ft")
        if not score or len(score) != 2:
            continue
        home, away = int(score[0]), int(score[1])
        rnd = m.get("round", "").lower()
        phase = "knockout" if any(k in rnd for k in
                                  ["round of", "quarter", "semi", "final", "third"]) else "group"
        if home > away:
            s = (home, away)
            win_counts[phase][s] = win_counts[phase].get(s, 0) + 1
        elif home < away:
            s = (away, home)
            win_counts[phase][s] = win_counts[phase].get(s, 0) + 1
        else:
            s = (home, away)
            draw_counts[phase][s] = draw_counts[phase].get(s, 0) + 1

    return win_counts, draw_counts


def get_wc2022_counts():
    """Return (win_counts, draw_counts) for WC 2022 keyed by phase → canonical → int."""
    return _load_wc2022_counts()


def build_distributions(
    extra_wins:  dict | None = None,
    extra_draws: dict | None = None,
    delta: float = 0.0,
) -> dict:
    """
    Returns dist[phase][key] where:
      key = "neutral_win"  → P(winner_goals=w, loser_goals=l)  w > l
      key = "draw"         → P(goals=g, goals=g)
    Indices are integer tuples (a, b).

    extra_wins:  {phase: {(w,l): count}} — WC 2026 win results
    extra_draws: {phase: {(g,g): count}} — WC 2026 draw results
    delta: WC 2026 data counts as (1+delta) × a WC 2022 observation.
    """
    wc_raw, dc_raw = _load_wc2022_counts()
    all_win_cells  = [(w, l) for w in range(MAX_GOALS + 1)
                               for l in range(MAX_GOALS + 1) if w > l]
    all_draw_cells = [(g, g) for g in range(MAX_GOALS + 1)]

    dist = {}
    for phase in ["group", "knockout"]:
        wc = wc_raw[phase]
        dc = dc_raw[phase]
        ew = (extra_wins  or {}).get(phase, {})
        ed = (extra_draws or {}).get(phase, {})

        # All observed canonical scores for λ MLE (includes out-of-grid)
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
    """Cached WC-2022-only distribution (delta=0, no 2026 data)."""
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

    p_home_win/draw/away_win: from Kalshi game market (should sum to ~1).
    total_goals_probs: {n: P(total==n)} from Kalshi over/under chain.
    spread_home / spread_away: {k: P(team wins by exactly k goals)} for k=1..4+
      derived from KXWCSPREAD markets.  k=4 means "4 or more".
    dist: pre-built distribution from build_distributions(); if None, uses
      the cached WC-2022-only baseline.
    """
    if dist is None:
        dist = get_distributions()
    all_scores = [(a, b) for a in range(MAX_GOALS + 1) for b in range(MAX_GOALS + 1)]

    # ── Step 1: base distribution (neutral-venue model) ──
    probs = {}
    for a, b in all_scores:
        if a > b:   # home team wins
            probs[(a, b)] = p_home_win * dist[phase]["neutral_win"].get((a, b), 0.0)
        elif a < b:  # away team wins — flip coordinates to (winner, loser)
            probs[(a, b)] = p_away_win * dist[phase]["neutral_win"].get((b, a), 0.0)
        else:        # draw
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
