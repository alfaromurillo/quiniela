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
    Also populates 'knockout_reg' phase using ft (90 min) scores for knockout games.
    Returns (win_counts, draw_counts) keyed by phase → canonical_score → int.
    """
    raw = json.loads(Path(path).read_text())
    win_counts  = {"group": {}, "knockout": {}, "knockout_reg": {}}
    draw_counts = {"group": {}, "knockout": {}, "knockout_reg": {}}

    for m in raw["matches"]:
        sc_raw = m.get("score", {})
        ft = sc_raw.get("ft")
        if not ft or len(ft) != 2:
            continue
        rnd = m.get("round", "").lower()
        phase = "knockout" if any(k in rnd for k in
                  ["round of", "quarter", "semi", "final", "third"]) else "group"

        if phase == "knockout":
            et = sc_raw.get("et")
            sc_120 = et if (et and len(et) == 2) else ft
            scores_to_record = [(sc_120, "knockout"), (ft, "knockout_reg")]
        else:
            scores_to_record = [(ft, "group")]

        for sc, ph in scores_to_record:
            h, a = int(sc[0]), int(sc[1])
            if h > a:
                s = (h, a)
                win_counts[ph][s] = win_counts[ph].get(s, 0) + 1
            elif h < a:
                s = (a, h)
                win_counts[ph][s] = win_counts[ph].get(s, 0) + 1
            else:
                s = (h, a)
                draw_counts[ph][s] = draw_counts[ph].get(s, 0) + 1

    return win_counts, draw_counts


def _load_et_data(path) -> list[tuple[int, int]]:
    """
    Extract ET goal deltas (delta_home, delta_away) for knockout games that went to ET.
    delta = et_score - ft_score for each team.
    """
    raw = json.loads(Path(path).read_text())
    deltas = []
    for m in raw["matches"]:
        sc_raw = m.get("score", {})
        ft = sc_raw.get("ft")
        et = sc_raw.get("et")
        if not ft or len(ft) != 2 or not et or len(et) != 2:
            continue
        rnd = m.get("round", "").lower()
        if not any(k in rnd for k in ["round of", "quarter", "semi", "final", "third"]):
            continue
        deltas.append((int(et[0]) - int(ft[0]), int(et[1]) - int(ft[1])))
    return deltas


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
    combined_w = {"group": {}, "knockout": {}, "knockout_reg": {}}
    combined_d = {"group": {}, "knockout": {}, "knockout_reg": {}}
    for k, (wc, dc) in enumerate(tournaments):
        age   = (n - 1) - k
        weight = gamma ** age
        for phase in ["group", "knockout", "knockout_reg"]:
            for s, cnt in wc[phase].items():
                combined_w[phase][s] = combined_w[phase].get(s, 0) + weight * cnt
            for s, cnt in dc[phase].items():
                combined_d[phase][s] = combined_d[phase].get(s, 0) + weight * cnt
    return combined_w, combined_d


def knockout_draw_rate(gamma: float = 1.0) -> float:
    """
    Fraction of historical knockout games that ended as ET draws (→ penalties).
    Used as fallback when KXWCGAME has no TIE market.
    """
    wc, dc = get_historical_counts(gamma)
    n_d = sum(dc["knockout"].values())
    n_w = sum(wc["knockout"].values())
    return n_d / (n_w + n_d) if (n_w + n_d) > 0 else 0.25


def knockout_et_draw_rate(gamma: float = 1.0) -> float:
    """
    P(ET draw → penalties | match went to ET).
    Used to convert P(reg_draw) from KXWCGAME-TIE into P(quiniela_draw).
    """
    files = [f for f in HISTORICAL_FILES if f.exists()]
    n = len(files)
    et_total = 0.0
    et_draws = 0.0
    for k, path in enumerate(files):
        age = (n - 1) - k
        weight = gamma ** age
        for dh, da in _load_et_data(path):
            et_total += weight
            if dh == da:
                et_draws += weight
    return et_draws / et_total if et_total > 0 else 0.72


def build_et_kernel(gamma: float = 1.0,
                    extra_et: dict | None = None,
                    delta: float = 0.0) -> dict:
    """
    Build gamma-weighted ET transition kernel.
    Returns {(delta_home, delta_away): probability}.

    Symmetrized: each observed home-wins-ET case is split 50/50 with its
    mirror (away-wins-ET), correcting for small-sample home bias.
    extra_et: {(dh, da): count} of WC 2026 knockout ET games; weight = (1+delta).
    """
    files = [f for f in HISTORICAL_FILES if f.exists()]
    n = len(files)
    kernel_raw: dict[tuple, float] = {}

    def _add(dh: int, da: int, w: float) -> None:
        if dh > da:
            kernel_raw[(dh, da)] = kernel_raw.get((dh, da), 0.0) + w * 0.5
            kernel_raw[(da, dh)] = kernel_raw.get((da, dh), 0.0) + w * 0.5
        elif da > dh:
            kernel_raw[(dh, da)] = kernel_raw.get((dh, da), 0.0) + w * 0.5
            kernel_raw[(da, dh)] = kernel_raw.get((da, dh), 0.0) + w * 0.5
        else:
            kernel_raw[(dh, da)] = kernel_raw.get((dh, da), 0.0) + w

    for k, path in enumerate(files):
        age = (n - 1) - k
        weight = gamma ** age
        for dh, da in _load_et_data(path):
            _add(dh, da, weight)

    if extra_et:
        for (dh, da), cnt in extra_et.items():
            _add(dh, da, (1.0 + delta) * cnt)

    total = sum(kernel_raw.values())
    if total < 1e-9:
        return {(0, 0): 1.0}
    return {k: v / total for k, v in kernel_raw.items()}


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
    for phase in ["group", "knockout", "knockout_reg"]:
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

    # ── Restore outcome marginals ──────────────────────────────────────────
    # Total-goals and spread reweighting shift mass between cells without
    # respecting the outcome boundaries, so the draw region shrinks
    # significantly after global normalisation.  This one-pass rescale
    # forces the three regions to sum to exactly p_home_win / p_draw /
    # p_away_win, while preserving all within-region relative weights.
    # After rescaling the grid still sums to 1 (the three targets sum to 1).
    home_cells = [(a, b) for a, b in probs if a > b]
    draw_cells = [(a, b) for a, b in probs if a == b]
    away_cells = [(a, b) for a, b in probs if a < b]
    for cells, p_target in ((home_cells, p_home_win),
                             (draw_cells, p_draw),
                             (away_cells, p_away_win)):
        s = sum(probs[c] for c in cells)
        if s > 1e-12 and abs(s - p_target) > 1e-9:
            scale = p_target / s
            for c in cells:
                probs[c] *= scale

    return probs


def _teamtotal_to_exact(over_probs: dict) -> dict | None:
    """Convert P(goals >= n) dict to P(goals = k) for k = 0..MAX_GOALS."""
    if not over_probs:
        return None
    p_over = {}
    for n in range(1, MAX_GOALS + 1):
        if n in over_probs:
            p_over[n] = over_probs[n]
    if not p_over:
        return None
    # Fill missing thresholds: monotone decreasing
    for n in range(1, MAX_GOALS + 1):
        if n not in p_over:
            prev = p_over.get(n - 1, 1.0)
            nxt  = p_over.get(n + 1, 0.0)
            p_over[n] = max(nxt, min(prev, prev * 0.3))
    # Enforce monotone decreasing
    for n in range(2, MAX_GOALS + 1):
        p_over[n] = min(p_over[n], p_over[n - 1])

    result = {0: max(0.0, 1.0 - p_over.get(1, 0.0))}
    for k in range(1, MAX_GOALS):
        result[k] = max(0.0, p_over.get(k, 0.0) - p_over.get(k + 1, 0.0))
    result[MAX_GOALS] = max(0.0, p_over.get(MAX_GOALS, 0.0))
    total = sum(result.values())
    if total < 0.01:
        return None
    return {k: v / total for k, v in result.items()}


def scoreline_probs_ipf(
    p_home_win: float, p_draw: float, p_away_win: float,
    phase: str,
    total_goals_probs: dict | None = None,
    spread_home: dict | None = None,
    spread_away: dict | None = None,
    dist: dict | None = None,
    n_iter: int = 20,
    tol: float = 1e-8,
    team_totals: dict | None = None,
) -> dict:
    """
    IPF (iterative proportional fitting) version of scoreline_probs.

    Uses the historical distribution as a prior and iteratively projects
    onto three sets of Kalshi marginal constraints until convergence:
      1. Outcome regions  → sum to p_home_win / p_draw / p_away_win
      2. Total-goals diagonals → sum to total_goals_probs[T]
      3. Spread anti-diagonals → sum to p_side * spread[m]

    Each projection preserves the relative weights within the constrained
    subspace (multiplicative scaling), so the result is the minimum KL-
    divergence distribution from the historical prior that simultaneously
    satisfies all three Kalshi constraint sets.

    Falls back to scoreline_probs() when total_goals_probs and both spread
    dicts are None (no Kalshi constraints to fit).
    """
    if total_goals_probs is None and spread_home is None and spread_away is None:
        return scoreline_probs(p_home_win, p_draw, p_away_win, phase,
                               None, None, None, dist=dist)

    if dist is None:
        dist = get_distributions()

    all_scores = [(a, b) for a in range(MAX_GOALS + 1) for b in range(MAX_GOALS + 1)]

    # ── Initialise from historical prior weighted by outcome probs ──
    probs: dict[tuple, float] = {}
    for a, b in all_scores:
        if a > b:
            probs[(a, b)] = p_home_win * dist[phase]["neutral_win"].get((a, b), 0.0)
        elif a < b:
            probs[(a, b)] = p_away_win * dist[phase]["neutral_win"].get((b, a), 0.0)
        else:
            probs[(a, b)] = p_draw * dist[phase]["draw"].get((a, b), 0.0)

    # Normalise initial distribution (sums to 1 by construction, but guard)
    s0 = sum(probs.values())
    if s0 > 1e-12:
        probs = {c: v / s0 for c, v in probs.items()}

    # Pre-compute cell groups (constant across iterations)
    home_cells = [c for c in all_scores if c[0] > c[1]]
    draw_cells = [c for c in all_scores if c[0] == c[1]]
    away_cells = [c for c in all_scores if c[0] < c[1]]

    # Total-goals: cells bucketed by min(h+a, 6)
    tg_cells: dict[int, list] = {t: [] for t in range(7)}
    for a, b in all_scores:
        tg_cells[min(a + b, 6)].append((a, b))

    # Spread home: cells bucketed by min(h-a, 4) for h>a
    sh_cells: dict[int, list] = {m: [] for m in range(1, 5)}
    for a, b in all_scores:
        if a > b:
            sh_cells[min(a - b, 4)].append((a, b))

    # Spread away: cells bucketed by min(b-a, 4) for b>a
    sa_cells: dict[int, list] = {m: [] for m in range(1, 5)}
    for a, b in all_scores:
        if b > a:
            sa_cells[min(b - a, 4)].append((a, b))

    def _scale_group(cells: list, target: float) -> None:
        s = sum(probs[c] for c in cells)
        if s > 1e-12 and abs(s - target) > 1e-12:
            f = target / s
            for c in cells:
                probs[c] *= f

    # ── IPF iterations ──
    for _ in range(n_iter):
        old = dict(probs)

        # Projection 1: outcome regions
        _scale_group(home_cells, p_home_win)
        _scale_group(draw_cells, p_draw)
        _scale_group(away_cells, p_away_win)

        # Projection 2: total-goals diagonals
        if total_goals_probs:
            s_tg = sum(total_goals_probs.values())
            for t, cells in tg_cells.items():
                target = total_goals_probs.get(t, 0.0) / s_tg if s_tg > 1e-12 else 0.0
                _scale_group(cells, target)

        # Projection 3a: spread home (absolute target = p_home * sh[m])
        if spread_home:
            s_sh = sum(spread_home.values())
            for m, cells in sh_cells.items():
                target = p_home_win * spread_home.get(m, 0.0) / s_sh if s_sh > 1e-12 else 0.0
                _scale_group(cells, target)

        # Projection 3b: spread away (absolute target = p_away * sa[m])
        if spread_away:
            s_sa = sum(spread_away.values())
            for m, cells in sa_cells.items():
                target = p_away_win * spread_away.get(m, 0.0) / s_sa if s_sa > 1e-12 else 0.0
                _scale_group(cells, target)

        # Projections 4-5: per-team goal marginals from KXWCTEAMTOTAL
        if team_totals:
            home_exact = _teamtotal_to_exact(team_totals.get("home") or {})
            away_exact = _teamtotal_to_exact(team_totals.get("away") or {})
            if home_exact:
                for k, target in home_exact.items():
                    if k < MAX_GOALS:
                        cells = [c for c in all_scores if c[0] == k]
                    else:
                        cells = [c for c in all_scores if c[0] >= k]
                    _scale_group(cells, target)
            if away_exact:
                for k, target in away_exact.items():
                    if k < MAX_GOALS:
                        cells = [c for c in all_scores if c[1] == k]
                    else:
                        cells = [c for c in all_scores if c[1] >= k]
                    _scale_group(cells, target)

        # Convergence check
        if max(abs(probs[c] - old[c]) for c in probs) < tol:
            break

    return probs


def scoreline_probs_knockout(
    p_q_home: float, p_q_draw: float, p_q_away: float,
    p_reg_home: float, p_reg_draw: float, p_reg_away: float,
    total_goals_probs: dict | None = None,
    spread_home: dict | None = None,
    spread_away: dict | None = None,
    team_totals: dict | None = None,
    dist: dict | None = None,
    et_kernel: dict | None = None,
) -> dict:
    """
    Two-stage knockout scoreline distribution (90+30 min).

    Stage 1: Build regulation-time (90 min) scoreline using "knockout_reg"
      historical phase with Kalshi reg-time constraints (KXWCTOTAL, KXWCSPREAD,
      KXWCTEAMTOTAL).
    Stage 2: For reg-time wins, final score = reg score.
      For reg-time draws, convolve with ET kernel:
        P(final = k+dh, k+da) += P_reg(k,k) × et_kernel[(dh,da)]
      ET draws (dh==da) remain draws; ET winners shift to a win cell.
    Stage 3: Recalibrate to quiniela outcome probs (p_q_home/draw/away).

    p_q_*: quiniela outcome probs (90+30 min, penalties→draw)
    p_reg_*: regulation-time outcome probs from KXWCGAME
    et_kernel: from build_et_kernel(); fallback used if None
    """
    if dist is None:
        dist = get_distributions()
    if et_kernel is None:
        et_kernel = {(0, 0): 0.72, (1, 0): 0.07, (0, 1): 0.07, (1, 1): 0.14}

    # Stage 1: reg-time scoreline distribution
    reg_dist = scoreline_probs_ipf(
        p_reg_home, p_reg_draw, p_reg_away, "knockout_reg",
        total_goals_probs, spread_home, spread_away,
        dist=dist, team_totals=team_totals,
    )

    # Stage 2: fold ET transitions into draw cells
    final_dist: dict[tuple, float] = {}
    for (rh, ra), p_reg in reg_dist.items():
        if p_reg < 1e-12:
            continue
        if rh != ra:
            # No ET: final score = reg score
            s = (rh, ra)
            final_dist[s] = final_dist.get(s, 0.0) + p_reg
        else:
            # ET: convolve with kernel
            for (dh, da), p_et in et_kernel.items():
                fh = min(rh + dh, MAX_GOALS)
                fa = min(ra + da, MAX_GOALS)
                s = (fh, fa)
                final_dist[s] = final_dist.get(s, 0.0) + p_reg * p_et

    # Fill missing cells
    for a in range(MAX_GOALS + 1):
        for b in range(MAX_GOALS + 1):
            final_dist.setdefault((a, b), 0.0)

    # Normalize
    total = sum(final_dist.values())
    if total > 1e-12:
        final_dist = {s: v / total for s, v in final_dist.items()}

    # Stage 3: recalibrate to quiniela outcome probs
    home_cells = [(a, b) for a, b in final_dist if a > b]
    draw_cells = [(a, b) for a, b in final_dist if a == b]
    away_cells = [(a, b) for a, b in final_dist if a < b]
    for cells, p_target in ((home_cells, p_q_home),
                             (draw_cells, p_q_draw),
                             (away_cells, p_q_away)):
        s = sum(final_dist[c] for c in cells)
        if s > 1e-12 and abs(s - p_target) > 1e-9:
            scale = p_target / s
            for c in cells:
                final_dist[c] *= scale

    return final_dist
