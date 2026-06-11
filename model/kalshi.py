"""
Fetch WC 2026 match probabilities from Kalshi public API.

Market types per match:
  KXWCGAME-{date}{team1}{team2}  → win/draw/loss (-HOME, -AWAY, -TIE)
  KXWCTOTAL-{date}{team1}{team2} → total goals over N.5 (-1 to -6)
  KXWCSPREAD-{date}{team1}{team2}→ spread markets (-HOME2/-HOME3/-HOME4/-AWAY2/-AWAY3)

Date format: 26JUN11 (year=26, month abbreviated, day 2-digit zero-padded)
"""
import re
import json
import time
import requests
from pathlib import Path
from datetime import datetime

API_BASE = "https://api.elections.kalshi.com/trade-api/v2"
CACHE_PATH = Path(__file__).parent.parent / "data" / "kalshi_cache.json"
CACHE_TTL = 3600  # seconds

# Mapping from schedule team names to Kalshi 3-letter codes
TEAM_CODES = {
    "Algeria": "ALG",
    "Argentina": "ARG",
    "Australia": "AUS",
    "Austria": "AUT",
    "Belgium": "BEL",
    "Bosnia & Herzegovina": "BIH",
    "Brazil": "BRA",
    "Canada": "CAN",
    "Cape Verde": "CPV",
    "Colombia": "COL",
    "Croatia": "CRO",
    "Curaçao": "CUW",
    "Czech Republic": "CZE",
    "DR Congo": "COD",
    "Ecuador": "ECU",
    "Egypt": "EGY",
    "England": "ENG",
    "France": "FRA",
    "Germany": "GER",
    "Ghana": "GHA",
    "Haiti": "HAI",
    "Iran": "IRN",
    "Iraq": "IRQ",
    "Ivory Coast": "CIV",
    "Japan": "JPN",
    "Jordan": "JOR",
    "Mexico": "MEX",
    "Morocco": "MAR",
    "Netherlands": "NED",
    "New Zealand": "NZL",
    "Norway": "NOR",
    "Panama": "PAN",
    "Paraguay": "PAR",
    "Portugal": "POR",
    "Qatar": "QAT",
    "Saudi Arabia": "KSA",
    "Scotland": "SCO",
    "Senegal": "SEN",
    "South Africa": "RSA",
    "South Korea": "KOR",
    "Spain": "ESP",
    "Sweden": "SWE",
    "Switzerland": "SUI",
    "Tunisia": "TUN",
    "Turkey": "TUR",
    "USA": "USA",
    "Uruguay": "URU",
    "Uzbekistan": "UZB",
}

MONTHS = {1:"JAN",2:"FEB",3:"MAR",4:"APR",5:"MAY",6:"JUN",7:"JUL",8:"AUG",9:"SEP",10:"OCT",11:"NOV",12:"DEC"}


def _date_code(date_str: str) -> str:
    """'2026-06-11' → '26JUN11'"""
    d = datetime.strptime(date_str, "%Y-%m-%d")
    return f"{str(d.year)[2:]}{MONTHS[d.month]}{d.day:02d}"


def _midprice(market: dict) -> float | None:
    """Return midpoint of bid/ask, or None if unavailable."""
    bid = market.get("yes_bid_dollars")
    ask = market.get("yes_ask_dollars")
    if bid is None or ask is None:
        return None
    b, a = float(bid), float(ask)
    if a <= 0 or b >= 1:
        return None
    if a - b > 0.30:
        return None  # too wide spread → unreliable
    return (b + a) / 2


def _fetch_event(event_ticker: str) -> list[dict]:
    """Fetch markets for a Kalshi event. Returns list of market dicts."""
    url = f"{API_BASE}/events/{event_ticker}?with_nested_markets=true"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            return r.json().get("event", {}).get("markets", [])
    except Exception:
        pass
    return []


def _parse_game(markets: list[dict], home_code: str, away_code: str) -> dict | None:
    """Extract normalised win/draw/loss probs from KXWCGAME markets."""
    probs = {}
    for m in markets:
        ticker = m["ticker"]
        if ticker.endswith(f"-{home_code}"):
            p = _midprice(m)
            if p is not None:
                probs["home_win"] = p
        elif ticker.endswith(f"-{away_code}"):
            p = _midprice(m)
            if p is not None:
                probs["away_win"] = p
        elif ticker.endswith("-TIE"):
            p = _midprice(m)
            if p is not None:
                probs["draw"] = p

    if len(probs) != 3:
        return None
    total = sum(probs.values())
    if total < 0.5:
        return None
    return {k: v / total for k, v in probs.items()}


def _parse_totals(markets: list[dict]) -> dict | None:
    """
    Extract P(total_goals == n) for n in 0..6+ from KXWCTOTAL markets.
    Markets: -1=over0.5, -2=over1.5, ..., -6=over5.5
    """
    thresholds = {}  # n_minus_half → P(total > n-0.5)
    for m in markets:
        ticker = m["ticker"]
        match = re.search(r"-(\d+)$", ticker)
        if match:
            idx = int(match.group(1))  # 1-6 → over 0.5 to over 5.5
            p = _midprice(m)
            if p is not None:
                thresholds[idx] = p

    if not thresholds:
        return None

    # P(total == n) = P(over n-0.5) - P(over n+0.5)
    # For n=0: 1 - P(over 0.5)
    result = {}
    p_over = {i: thresholds.get(i, 0.0) for i in range(1, 7)}
    result[0] = 1.0 - p_over.get(1, 0.95)
    for n in range(1, 6):
        result[n] = p_over.get(n, 0.0) - p_over.get(n + 1, 0.0)
    result[6] = p_over.get(6, 0.0)  # 6+ goals

    # Clamp negatives and normalise
    result = {k: max(0.0, v) for k, v in result.items()}
    total = sum(result.values())
    if total < 0.1:
        return None
    return {k: v / total for k, v in result.items()}


def _load_cache() -> dict:
    if CACHE_PATH.exists():
        data = json.loads(CACHE_PATH.read_text())
        if time.time() - data.get("_ts", 0) < CACHE_TTL:
            return data
    return {}


def _save_cache(data: dict):
    data["_ts"] = time.time()
    CACHE_PATH.write_text(json.dumps(data, indent=2))


def fetch_match_probs(match: dict) -> dict:
    """
    Fetch Kalshi probabilities for a single match dict (from schedule.json).
    Returns:
      {
        "home_win": float, "draw": float, "away_win": float,
        "total_goals": {0: p, 1: p, ..., 6: p},  # optional
        "source": "kalshi" | "historical_fallback"
      }
    Falls back to None values if market unavailable.
    """
    home = match["home"]
    away = match["away"]
    date_str = match["date"]
    home_code = TEAM_CODES.get(home)
    away_code = TEAM_CODES.get(away)

    if not home_code or not away_code:
        return _fallback_probs()

    date_code = _date_code(date_str)
    game_ticker = f"KXWCGAME-{date_code}{home_code}{away_code}"
    total_ticker = f"KXWCTOTAL-{date_code}{home_code}{away_code}"

    cache = _load_cache()
    cache_key = game_ticker

    if cache_key in cache and "_ts" in cache:
        cached = cache[cache_key]
        return cached

    game_markets = _fetch_event(game_ticker)
    game_probs = _parse_game(game_markets, home_code, away_code)

    total_markets = _fetch_event(total_ticker)
    total_goals = _parse_totals(total_markets)

    if game_probs is None:
        result = _fallback_probs()
    else:
        result = {
            "home_win": game_probs["home_win"],
            "draw": game_probs["draw"],
            "away_win": game_probs["away_win"],
            "total_goals": total_goals,
            "source": "kalshi",
        }

    cache[cache_key] = result
    _save_cache(cache)
    return result


def _fallback_probs() -> dict:
    """Historical WC base rates when Kalshi data unavailable."""
    return {
        "home_win": 0.40,
        "draw": 0.25,
        "away_win": 0.35,
        "total_goals": {0: 0.08, 1: 0.20, 2: 0.27, 3: 0.24, 4: 0.13, 5: 0.05, 6: 0.03},
        "source": "historical_fallback",
    }


def fetch_all_probs(schedule: list[dict]) -> dict:
    """
    Fetch probabilities for all group-stage matches (knockout TBD teams skipped).
    Returns dict keyed by match id.
    """
    results = {}
    for match in schedule:
        mid = match["id"]
        # Skip knockout matches with TBD teams
        if match["phase"] == "knockout" and (
            match["home"].startswith("W") or match["home"][0].isdigit()
        ):
            results[mid] = _fallback_probs()
            continue
        results[mid] = fetch_match_probs(match)
        time.sleep(0.1)  # be polite to the API
    return results
