"""
Resolve Round of 32 bracket placeholders in data/schedule.json.

Kalshi sets close_time = match_time_utc + 14 days for every R32 market,
which allows unambiguous matching without team names. For each knockout
entry that still has a placeholder (e.g. "1E", "3A/B/C/D/F", "W74"),
this script finds the corresponding Kalshi market and fills in the real
team names.
"""
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).parent.parent
SCHEDULE_PATH = ROOT / "data" / "schedule.json"
API_URL = "https://api.elections.kalshi.com/trade-api/v2/markets"

from model.kalshi import TEAM_CODES
REVERSE_CODES = {v: k for k, v in TEAM_CODES.items()}


def _is_tbd(name: str) -> bool:
    return not name or name[0].isdigit() or name.startswith("W") or name.startswith("L")


def fetch_kalshi_r32() -> dict:
    """Return {close_time_str: {home, away}} for all open KXWCGAME markets."""
    resp = requests.get(API_URL, params={
        "series_ticker": "KXWCGAME",
        "limit": 200,
        "status": "open",
    }, timeout=15)
    resp.raise_for_status()

    result = {}
    for m in resp.json().get("markets", []):
        ticker = m["ticker"]
        if not ticker.endswith("-TIE"):
            continue
        close_time = m["close_time"]  # "2026-07-12T19:00:00Z"
        if close_time in result:
            continue
        # Ticker: KXWCGAME-{7-char date}{3-char code1}{3-char code2}-TIE
        # Knockout uses team codes as suffixes (e.g. -RSA, -CAN) not -HOME/-AWAY
        inner = ticker[len("KXWCGAME-"):-len("-TIE")]  # e.g. "26JUN28RSACAN"
        team_part = inner[7:]                            # e.g. "RSACAN"
        home_code, away_code = team_part[:3], team_part[3:]
        home = REVERSE_CODES.get(home_code)
        away = REVERSE_CODES.get(away_code)
        if not home or not away:
            print(f"  WARNING: unknown codes {home_code!r}/{away_code!r} in {ticker}",
                  file=sys.stderr)
            continue
        result[close_time] = {"home": home, "away": away}

    return result


def main():
    schedule_data = json.loads(SCHEDULE_PATH.read_text())
    matches = schedule_data["matches"]

    r32_markets = fetch_kalshi_r32()
    print(f"Found {len(r32_markets)} open R32 markets on Kalshi")

    updated = 0
    for match in matches:
        if match["phase"] != "knockout":
            continue
        home, away = match["home"], match["away"]
        if not (_is_tbd(home) or _is_tbd(away)):
            continue

        t = datetime.strptime(
            match["time_utc"], "%Y-%m-%dT%H:%M:%SZ"
        ).replace(tzinfo=timezone.utc)
        expected_close = (t + timedelta(days=14)).strftime("%Y-%m-%dT%H:%M:%SZ")

        if expected_close not in r32_markets:
            print(f"  No market yet for match {match['id']} "
                  f"({home} vs {away}, expected close {expected_close})")
            continue

        info = r32_markets[expected_close]
        print(f"  Match {match['id']}: {home} vs {away} "
              f"→ {info['home']} vs {info['away']}")
        match["home"] = info["home"]
        match["away"] = info["away"]
        updated += 1

    SCHEDULE_PATH.write_text(json.dumps(schedule_data, indent=2, ensure_ascii=False))
    print(f"\nUpdated {updated} matches in schedule.json")
    return updated


if __name__ == "__main__":
    main()
