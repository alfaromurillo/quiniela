"""
Build scoreline probability distributions from WC 2022 historical data.
Returns P(home_goals=a, away_goals=b | result, phase) with Laplace smoothing.
"""
import json
from collections import defaultdict
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
MAX_GOALS = 5  # grid 0..MAX_GOALS for each team


def _result(home, away):
    if home > away:
        return "home_win"
    if home < away:
        return "away_win"
    return "draw"


def build_distributions():
    """
    Returns dict:
      dist[phase][result][(home_goals, away_goals)] = probability
      dist[phase]["any"][(home_goals, away_goals)] = probability
    where phase in {"group", "knockout"}.
    """
    raw = json.loads((DATA_DIR / "wc2022.json").read_text())
    matches = raw["matches"]

    counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    for m in matches:
        score = m.get("score", {}).get("ft")
        if not score or len(score) != 2:
            continue
        home, away = int(score[0]), int(score[1])

        round_name = m.get("round", "").lower()
        if any(kw in round_name for kw in ["round of", "quarter", "semi", "final", "third"]):
            phase = "knockout"
        else:
            phase = "group"

        res = _result(home, away)
        counts[phase][res][(home, away)] += 1
        counts[phase]["any"][(home, away)] += 1
        counts["all"]["any"][(home, away)] += 1

    # Build normalised distributions with Laplace smoothing (+1 to every cell in 0-MAX_GOALS grid)
    all_scores = [(a, b) for a in range(MAX_GOALS + 1) for b in range(MAX_GOALS + 1)]
    dist = {}
    for phase in ["group", "knockout"]:
        dist[phase] = {}
        for result in ["home_win", "draw", "away_win", "any"]:
            raw_counts = counts[phase][result]
            smoothed = {s: raw_counts.get(s, 0) + 1 for s in all_scores}
            # For "home_win" cells, zero-out impossible results (draws, away wins)
            if result == "home_win":
                smoothed = {(a, b): v for (a, b), v in smoothed.items() if a > b}
            elif result == "draw":
                smoothed = {(a, b): v for (a, b), v in smoothed.items() if a == b}
            elif result == "away_win":
                smoothed = {(a, b): v for (a, b), v in smoothed.items() if a < b}
            total = sum(smoothed.values())
            dist[phase][result] = {s: v / total for s, v in smoothed.items()}
    return dist


# Singleton — loaded once on import
_DIST = None


def get_distributions():
    global _DIST
    if _DIST is None:
        _DIST = build_distributions()
    return _DIST


def scoreline_probs(p_home_win: float, p_draw: float, p_away_win: float,
                   phase: str, total_goals_probs: dict | None = None,
                   spread_probs: dict | None = None) -> dict:
    """
    Return P(home=a, away=b) for all (a,b) in 0..MAX_GOALS grid.

    p_home_win, p_draw, p_away_win: Kalshi probabilities (should sum to 1)
    phase: "group" or "knockout"
    total_goals_probs: optional dict {n: P(total_goals == n)} from Kalshi over/under chain
    spread_probs: optional dict {"home": {k: P(home wins by exactly k)}, "away": {...}}
    """
    dist = get_distributions()
    all_scores = [(a, b) for a in range(MAX_GOALS + 1) for b in range(MAX_GOALS + 1)]

    # Base: weighted mix of conditional distributions
    probs = {}
    for s in all_scores:
        a, b = s
        res = _result(a, b)
        p_result = {"home_win": p_home_win, "draw": p_draw, "away_win": p_away_win}[res]
        probs[s] = p_result * dist[phase][res].get(s, 0.0)

    # Reweight by Kalshi total goals distribution if available
    if total_goals_probs:
        total_weight = defaultdict(float)
        for (a, b), p in probs.items():
            total_weight[a + b] += p
        adjusted = {}
        for (a, b), p in probs.items():
            t = a + b
            kalshi_t = total_goals_probs.get(t, 0.0)
            hist_t = total_weight[t]
            if hist_t > 1e-12 and kalshi_t > 1e-12:
                adjusted[(a, b)] = p * (kalshi_t / hist_t)
            else:
                adjusted[(a, b)] = 0.0
        probs = adjusted

    # Normalise
    total = sum(probs.values())
    if total > 1e-12:
        probs = {s: v / total for s, v in probs.items()}

    return probs
