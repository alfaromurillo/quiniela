"""
Main pipeline: fetch Kalshi probabilities → compute scoreline distributions
→ optimize predictions → write site/data/predictions.json
"""
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from model.kalshi import fetch_match_probs, _fallback_probs
from model.historical import scoreline_probs
from model.optimizer import best_prediction

SCHEDULE_PATH = ROOT / "data" / "schedule.json"
OUT_PATH = ROOT / "site" / "data" / "predictions.json"


def _is_tbd(name: str) -> bool:
    return not name or name[0].isdigit() or name.startswith("W")


def run():
    schedule = json.loads(SCHEDULE_PATH.read_text())["matches"]
    predictions = []
    total = len(schedule)

    for i, match in enumerate(schedule, 1):
        mid = match["id"]
        phase = match["phase"]
        home = match["home"]
        away = match["away"]

        print(f"[{i}/{total}] {home} vs {away} ({match['date']})", end=" ... ", flush=True)

        # Skip TBD knockout teams (bracket not yet decided)
        if phase == "knockout" and (_is_tbd(home) or _is_tbd(away)):
            print("TBD — skipped")
            predictions.append({
                "id": mid,
                "round": match["round"],
                "date": match["date"],
                "time_utc": match["time_utc"],
                "time_local": match["time_local"],
                "home": home,
                "away": away,
                "venue": match.get("venue", ""),
                "phase": phase,
                "group": match.get("group"),
                "tbd": True,
            })
            continue

        # Fetch Kalshi probabilities
        try:
            kalshi = fetch_match_probs(match)
        except Exception as e:
            print(f"Kalshi error: {e} — using fallback")
            kalshi = _fallback_probs()

        time.sleep(0.05)

        p_home = kalshi["home_win"]
        p_draw = kalshi["draw"]
        p_away = kalshi["away_win"]
        total_goals  = kalshi.get("total_goals")
        spread_home  = kalshi.get("spread_home")
        spread_away  = kalshi.get("spread_away")
        source = kalshi.get("source", "unknown")

        # Build joint scoreline distribution
        score_dist = scoreline_probs(
            p_home, p_draw, p_away, phase,
            total_goals, spread_home, spread_away,
        )

        # Find best prediction
        best = best_prediction(score_dist, phase)

        entry = {
            "id": mid,
            "round": match["round"],
            "date": match["date"],
            "time_utc": match["time_utc"],
            "time_local": match["time_local"],
            "home": home,
            "away": away,
            "venue": match.get("venue", ""),
            "phase": phase,
            "group": match.get("group"),
            "prediction": {
                "home": best["home"],
                "away": best["away"],
                "expected_pts": best["expected_pts"],
                "top3": best["top3"],
            },
            "probabilities": {
                "home_win": round(p_home, 4),
                "draw": round(p_draw, 4),
                "away_win": round(p_away, 4),
                "total_goals": {str(k): round(v, 4) for k, v in total_goals.items()} if total_goals else None,
            },
            "source": source,
            "tbd": False,
        }
        predictions.append(entry)
        print(f"{best['home']}-{best['away']} (E[pts]={best['expected_pts']:.2f}, src={source})")

    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "matches": predictions,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\nSaved {len(predictions)} matches → {OUT_PATH}")


if __name__ == "__main__":
    run()
