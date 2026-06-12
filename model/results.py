"""
Fetch WC 2026 final match results from ESPN public API.
Writes to site/data/results.json.

Result windows (time after kickoff before we attempt to fetch):
  Group stage:  120 min (2 h)
  Knockout:     210 min (3 h 30 min) — covers 90 min regulation +
                ~25 min added time + 30 min ET + 15 min ET added +
                ~20 min penalties
"""
import json
import time
import requests
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT          = Path(__file__).parent.parent
SCHEDULE_PATH = ROOT / "data" / "schedule.json"
RESULTS_PATH  = ROOT / "site" / "data" / "results.json"

ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/scoreboard"

RESULT_DELAY_GROUP    = 120   # minutes
RESULT_DELAY_KNOCKOUT = 210   # minutes

# Map our schedule names → possible ESPN displayName values
ESPN_ALIASES: dict[str, list[str]] = {
    "USA":                  ["United States", "USA"],
    "Czech Republic":       ["Czech Republic", "Czechia"],
    "South Korea":          ["South Korea", "Republic of Korea"],
    "DR Congo":             ["DR Congo", "Congo DR", "Democratic Republic of Congo"],
    "Bosnia & Herzegovina": ["Bosnia and Herzegovina", "Bosnia & Herzegovina", "Bosnia-Herzegovina"],
    "Ivory Coast":          ["Ivory Coast", "Cote d'Ivoire", "Côte d'Ivoire"],
    "Turkey":               ["Turkey", "Türkiye"],
    "Iran":                 ["Iran", "Islamic Republic of Iran"],
    "Scotland":             ["Scotland"],
    "Saudi Arabia":         ["Saudi Arabia", "KSA"],
    "New Zealand":          ["New Zealand", "New Zealand All Whites"],
    "Cape Verde":           ["Cape Verde", "Cabo Verde"],
    "Curaçao":              ["Curaçao", "Curacao"],
}


def _aliases(name: str) -> list[str]:
    return ESPN_ALIASES.get(name, [name])


def _teams_match(our_name: str, espn_name: str) -> bool:
    espn_lc = espn_name.lower().strip()
    return any(a.lower() == espn_lc for a in _aliases(our_name))


def _fetch_espn(date_str: str) -> list[dict]:
    """Return ESPN event list for date_str (YYYY-MM-DD)."""
    compact = date_str.replace("-", "")
    try:
        r = requests.get(ESPN_BASE, params={"dates": compact}, timeout=15)
        if r.status_code == 200:
            return r.json().get("events", [])
        print(f"  ESPN returned HTTP {r.status_code} for {date_str}")
    except Exception as exc:
        print(f"  ESPN fetch error ({date_str}): {exc}")
    return []


def _parse_event(event: dict) -> dict | None:
    """
    Extract final score from an ESPN event.
    Returns {home_name, away_name, home_score, away_score} or None.
    """
    comps = event.get("competitions", [])
    if not comps:
        return None
    comp = comps[0]
    status = comp.get("status", {}).get("type", {})
    if not status.get("completed"):
        return None
    competitors = comp.get("competitors", [])
    if len(competitors) != 2:
        return None
    home = next((c for c in competitors if c.get("homeAway") == "home"), None)
    away = next((c for c in competitors if c.get("homeAway") == "away"), None)
    if not home or not away:
        return None
    try:
        hs = int(home["score"])
        as_ = int(away["score"])
    except (KeyError, ValueError, TypeError):
        return None
    return {
        "home_name":  home.get("team", {}).get("displayName", ""),
        "away_name":  away.get("team", {}).get("displayName", ""),
        "home_score": hs,
        "away_score": as_,
    }


def update_results() -> None:
    schedule = json.loads(SCHEDULE_PATH.read_text())["matches"]

    # Load existing results
    if RESULTS_PATH.exists():
        stored = json.loads(RESULTS_PATH.read_text())
        results: dict = stored.get("matches", {})
    else:
        results = {}

    now = datetime.now(timezone.utc)

    # Find matches that need a result
    need: list[dict] = []
    for m in schedule:
        mid = str(m["id"])
        if results.get(mid, {}).get("status") == "final":
            continue
        time_utc = m.get("time_utc", "")
        if not time_utc or "TBD" in time_utc:
            continue
        kickoff = datetime.strptime(time_utc, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        delay = RESULT_DELAY_KNOCKOUT if m["phase"] == "knockout" else RESULT_DELAY_GROUP
        if now >= kickoff + timedelta(minutes=delay):
            need.append(m)

    if not need:
        print("Results: nothing to update.")
        return

    # Group by date (minimise ESPN requests)
    by_date: dict[str, list] = {}
    for m in need:
        by_date.setdefault(m["date"], []).append(m)

    updated = 0
    for date_str, matches in sorted(by_date.items()):
        events = _fetch_espn(date_str)
        time.sleep(0.3)
        for m in matches:
            mid = str(m["id"])
            matched = None
            for ev in events:
                parsed = _parse_event(ev)
                if not parsed:
                    continue
                if (_teams_match(m["home"], parsed["home_name"]) and
                        _teams_match(m["away"], parsed["away_name"])):
                    matched = parsed
                    break
            if matched:
                results[mid] = {
                    "home_score": matched["home_score"],
                    "away_score": matched["away_score"],
                    "status":     "final",
                    "fetched_at": now.isoformat(),
                }
                updated += 1
                print(f"  {m['home']} {matched['home_score']}-{matched['away_score']} {m['away']}")
            else:
                print(f"  No result yet: {m['home']} vs {m['away']} ({m['date']})")

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(
        {"generated_at": now.isoformat(), "matches": results},
        indent=2, ensure_ascii=False,
    ))
    print(f"Results: updated {updated} → {RESULTS_PATH}")


if __name__ == "__main__":
    update_results()
