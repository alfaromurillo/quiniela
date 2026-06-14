"""
Find the scoreline prediction that maximises expected quiniela points.

Scoring rules:
  GROUP STAGE:
    5 pts  - exact score
    3 pts  - correct winner/draw AND correct goals of exactly one team
    2 pts  - correct winner/draw, no goals correct
    1 pt   - wrong winner/draw, but correct goals of one team
    0 pts  - nothing correct

  KNOCKOUT (goals in 90+30 min; if penalties → result = draw):
    3 pts  - exact score
    1 pt   - correct winner/draw (wrong goals)
    0 pts  - wrong winner/draw
"""

MAX_GOALS = 5


def _winner(a: int, b: int) -> str:
    if a > b:
        return "home"
    if a < b:
        return "away"
    return "draw"


def _points_group(pred_a: int, pred_b: int, act_a: int, act_b: int) -> int:
    exact = (pred_a == act_a) and (pred_b == act_b)
    if exact:
        return 5
    correct_winner = _winner(pred_a, pred_b) == _winner(act_a, act_b)
    correct_home = pred_a == act_a
    correct_away = pred_b == act_b
    if correct_winner and (correct_home or correct_away):
        return 3
    if correct_winner:
        return 2
    if correct_home or correct_away:
        return 1
    return 0


def _points_knockout(pred_a: int, pred_b: int, act_a: int, act_b: int) -> int:
    if pred_a == act_a and pred_b == act_b:
        return 3
    if _winner(pred_a, pred_b) == _winner(act_a, act_b):
        return 1
    return 0


def expected_points(pred_a: int, pred_b: int, score_probs: dict, phase: str) -> float:
    pts_fn = _points_group if phase == "group" else _points_knockout
    total = 0.0
    for (act_a, act_b), p in score_probs.items():
        if p > 0:
            total += p * pts_fn(pred_a, pred_b, act_a, act_b)
    return total


def modal_prediction(score_probs: dict) -> dict:
    """
    Return the highest-probability scoreline argmax P(h,k).
    Ties broken in favour of lower-scoring scoreline.
    """
    best_p, best_h, best_k = -1.0, 1, 0
    for (h, k), p in score_probs.items():
        if p > best_p or (p == best_p and h + k < best_h + best_k):
            best_p, best_h, best_k = p, h, k
    return {"home": best_h, "away": best_k, "prob": round(best_p, 4)}


def best_prediction(score_probs: dict, phase: str) -> dict:
    """
    Return the prediction (a, b) maximising E[points], along with top-3 alternatives.

    Returns:
      {
        "home": int, "away": int,
        "expected_pts": float,
        "top3": [{"home": int, "away": int, "expected_pts": float}, ...]
      }
    """
    candidates = []
    for pred_a in range(MAX_GOALS + 1):
        for pred_b in range(MAX_GOALS + 1):
            ep = expected_points(pred_a, pred_b, score_probs, phase)
            candidates.append((ep, pred_a, pred_b))

    candidates.sort(reverse=True)
    best_ep, best_a, best_b = candidates[0]

    top3 = [
        {"home": a, "away": b, "expected_pts": round(ep, 4)}
        for ep, a, b in candidates[:3]
    ]

    return {
        "home": best_a,
        "away": best_b,
        "expected_pts": round(best_ep, 4),
        "top3": top3,
    }
