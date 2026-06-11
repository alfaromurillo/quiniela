"""
Fetch WC 2026 match probabilities from Kalshi public API.

Market types per match:
  KXWCGAME-{date}{team1}{team2}  → win/draw/loss (-HOME, -AWAY, -TIE)
  KXWCTOTAL-{date}{team1}{team2} → total goals over N.5 (-1 to -6)
  KXWCSPREAD-{date}{team1}{team2}→ spread markets (-HOME2/-HOME3/-HOME4/-AWAY2/-AWAY3)

Date code rule: Kalshi uses Eastern Time (UTC-4) date. Derived from time_utc in schedule.json.
"""
import re
import json
import time
import requests
from pathlib import Path
from datetime import datetime, timezone, timedelta

API_BASE = "https://api.elections.kalshi.com/trade-api/v2"
CACHE_PATH = Path(__file__).parent.parent / "data" / "kalshi_cache.json"
CACHE_TTL = 3600  # seconds

# Schedule → Kalshi 3-letter codes (several differ from FIFA/ISO)
TEAM_CODES = {
    "Algeria": "DZA",
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
    "Haiti": "HTI",
    "Iran": "IRI",
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

MONTHS = {1:"JAN",2:"FEB",3:"MAR",4:"APR",5:"MAY",6:"JUN",
          7:"JUL",8:"AUG",9:"SEP",10:"OCT",11:"NOV",12:"DEC"}


def _date_code(time_utc: str) -> str:
    """'2026-06-12T02:00:00Z' → '26JUN11'  (converts UTC → Eastern Time UTC-4)"""
    d = datetime.strptime(time_utc, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    et = d - timedelta(hours=4)
    return f"{str(et.year)[2:]}{MONTHS[et.month]}{et.day:02d}"


def _midprice(market: dict, use_last: bool = False) -> float | None:
    """Bid/ask midpoint; falls back to last_price for settled/live markets when use_last=True."""
    bid = market.get("yes_bid_dollars")
    ask = market.get("yes_ask_dollars")
    if bid is None or ask is None:
        return None
    b, a = float(bid), float(ask)
    if a - b > 0.30:
        if use_last:
            last = market.get("last_price_dollars")
            if last is not None:
                lp = float(last)
                if 0.02 < lp < 0.98:
                    return lp
        return None
    if a <= 0 or b >= 1:
        return None
    return (b + a) / 2


def _fetch_event(event_ticker: str) -> list[dict]:
    url = f"{API_BASE}/events/{event_ticker}?with_nested_markets=true"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            return r.json().get("event", {}).get("markets", [])
    except Exception:
        pass
    return []


def _parse_game(markets: list[dict], home_code: str, away_code: str) -> dict | None:
    """Extract normalised P(home_win), P(draw), P(away_win)."""
    probs = {}
    for m in markets:
        t = m["ticker"]
        if t.endswith(f"-{home_code}"):
            p = _midprice(m)
            if p is not None:
                probs["home_win"] = p
        elif t.endswith(f"-{away_code}"):
            p = _midprice(m)
            if p is not None:
                probs["away_win"] = p
        elif t.endswith("-TIE"):
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
    Derive P(total_goals==n) from KXWCTOTAL markets.
    -1=over0.5, -2=over1.5, …, -6=over5.5
    """
    thresholds = {}
    for m in markets:
        match = re.search(r"-(\d+)$", m["ticker"])
        if match:
            idx = int(match.group(1))
            p = _midprice(m, use_last=True)
            if p is not None:
                thresholds[idx] = p

    if not thresholds:
        return None

    # Fill missing lower thresholds (monotone: P(over n) >= P(over n+1))
    for i in range(5, 0, -1):
        if i not in thresholds and (i + 1) in thresholds:
            thresholds[i] = thresholds[i + 1]
    # Fill missing upper thresholds with decay
    for i in range(2, 7):
        if i not in thresholds and (i - 1) in thresholds:
            thresholds[i] = thresholds[i - 1] * 0.4

    p_over = {i: thresholds.get(i, 0.0) for i in range(1, 7)}
    result = {
        0: max(0.0, 1.0 - p_over[1]),
        **{n: max(0.0, p_over[n] - p_over.get(n + 1, 0.0)) for n in range(1, 6)},
        6: max(0.0, p_over[6]),
    }
    total = sum(result.values())
    if total < 0.05:
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
    Fetch Kalshi probabilities for one match from schedule.json.
    Returns {home_win, draw, away_win, total_goals, source}.
    """
    home_code = TEAM_CODES.get(match["home"])
    away_code = TEAM_CODES.get(match["away"])
    if not home_code or not away_code:
        return _fallback_probs()

    date_code = _date_code(match["time_utc"])
    game_ticker = f"KXWCGAME-{date_code}{home_code}{away_code}"
    total_ticker = f"KXWCTOTAL-{date_code}{home_code}{away_code}"

    cache = _load_cache()
    if game_ticker in cache:
        return cache[game_ticker]

    game_probs = _parse_game(_fetch_event(game_ticker), home_code, away_code)
    total_goals = _parse_totals(_fetch_event(total_ticker))

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

    cache[game_ticker] = result
    _save_cache(cache)
    return result


def _fallback_probs() -> dict:
    return {
        "home_win": 0.40,
        "draw": 0.25,
        "away_win": 0.35,
        "total_goals": {0: 0.08, 1: 0.20, 2: 0.27, 3: 0.24, 4: 0.13, 5: 0.05, 6: 0.03},
        "source": "historical_fallback",
    }
