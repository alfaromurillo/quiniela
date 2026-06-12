"""
Sanity checks on predictions.json. Run after predict.py, before committing.
Exits with code 1 if any hard check fails — this aborts the workflow commit
so bad predictions never reach GitHub Pages.
"""
import json
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT          = Path(__file__).parent.parent
PRED_PATH     = ROOT / "site" / "data" / "predictions.json"
LOCK_PATH     = ROOT / "site" / "data" / "locked_predictions.json"
SCHEDULE_PATH = ROOT / "data" / "schedule.json"
RESULTS_PATH  = ROOT / "site" / "data" / "results.json"

DEGENERATE_SCORE      = (5, 5)
MIN_EPTS              = 0.5   # any upcoming match below this is suspicious
MAX_EPTS              = 5.1   # group stage max is 5
RESULT_DELAY_GROUP    = 120   # minutes — must match results.py
RESULT_DELAY_KNOCKOUT = 210


def check_missing_results() -> list[str]:
    """
    Warn about matches past their result window with no recorded result.
    Symptom of an ESPN alias bug: results.py printed "No result yet"
    for a finished game, silently leaving the result missing.
    """
    warnings = []
    if not SCHEDULE_PATH.exists():
        return warnings

    schedule = json.loads(SCHEDULE_PATH.read_text()).get("matches", [])
    results  = {}
    if RESULTS_PATH.exists():
        results = json.loads(RESULTS_PATH.read_text()).get("matches", {})

    now = datetime.now(timezone.utc)
    for m in schedule:
        time_utc = m.get("time_utc", "")
        if not time_utc or "TBD" in time_utc:
            continue
        if results.get(str(m["id"]), {}).get("status") == "final":
            continue
        kickoff = datetime.strptime(time_utc, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        delay   = RESULT_DELAY_KNOCKOUT if m["phase"] == "knockout" else RESULT_DELAY_GROUP
        if now >= kickoff + timedelta(minutes=delay):
            warnings.append(
                f"Match {m['id']} ({m['home']} vs {m['away']}, {m['date']}): "
                "past result window but no result — possible ESPN alias mismatch"
            )
    return warnings


def check(pred_path: Path, lock_path: Path) -> list[str]:
    errors   = []
    warnings = []

    data    = json.loads(pred_path.read_text())
    matches = data.get("matches", [])
    locked  = json.loads(lock_path.read_text()).get("matches", {}) if lock_path.exists() else {}

    upcoming = [m for m in matches if not m.get("tbd") and m.get("prediction")]

    if not upcoming:
        warnings.append("No upcoming non-TBD matches found — nothing to check.")

    for m in upcoming:
        mid  = str(m["id"])
        pred = m["prediction"]
        home, away = pred["home"], pred["away"]
        ep   = pred.get("expected_pts", 0.0)
        label = f"Match {mid} ({m['home']} vs {m['away']})"

        # Hard: degenerate prediction
        if (home, away) == DEGENERATE_SCORE:
            errors.append(f"{label}: prediction is {home}-{away} (degenerate)")

        # Hard: zero or negative expected pts
        if ep < 1e-6:
            errors.append(f"{label}: expected_pts={ep:.4f} (near zero — distribution collapsed)")

        # Soft: suspiciously low
        if 1e-6 < ep < MIN_EPTS:
            warnings.append(f"{label}: low expected_pts={ep:.4f}")

        # Hard: expected pts out of range
        if ep > MAX_EPTS:
            errors.append(f"{label}: expected_pts={ep:.4f} exceeds maximum")

        # Hard: probabilities don't sum to ~1
        probs = m.get("probabilities", {})
        if probs:
            s = probs.get("home_win", 0) + probs.get("draw", 0) + probs.get("away_win", 0)
            if not (0.97 < s < 1.03):
                errors.append(f"{label}: outcome probs sum to {s:.4f} (not ~1)")

        # Soft: locked prediction changed vs current
        if mid in locked:
            lk = locked[mid]
            if lk.get("home") != home or lk.get("away") != away:
                warnings.append(
                    f"{label}: prediction changed since lock "
                    f"({lk['home']}-{lk['away']} → {home}-{away})"
                )

    return errors, warnings


def main():
    if not PRED_PATH.exists():
        print("SANITY ERROR: predictions.json not found", file=sys.stderr)
        sys.exit(1)

    errors, warnings = check(PRED_PATH, LOCK_PATH)
    warnings += check_missing_results()

    for w in warnings:
        print(f"SANITY WARNING: {w}")

    if errors:
        print(file=sys.stderr)
        for e in errors:
            print(f"SANITY ERROR: {e}", file=sys.stderr)
        print(
            f"\n{len(errors)} error(s) found — predictions NOT committed.",
            file=sys.stderr,
        )
        sys.exit(1)

    n = sum(1 for m in json.loads(PRED_PATH.read_text())["matches"]
            if not m.get("tbd") and m.get("prediction"))
    print(f"Sanity OK — {n} predictions validated, {len(warnings)} warning(s).")


if __name__ == "__main__":
    main()
