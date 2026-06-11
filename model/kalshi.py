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
# Pre-game entries are considered stale after 90 min; once kickoff passes they are
# locked permanently so in-game/post-game prices never overwrite the prediction.
CACHE_TTL = 5400  # 90 minutes

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


def _parse_spread(markets: list[dict], home_code: str, away_code: str,
                  p_home_win: float, p_away_win: float) -> tuple[dict | None, dict | None]:
    """
    Parse KXWCSPREAD markets into margin-probability dicts.
    Returns (spread_home, spread_away) where each is {k: P(team wins by exactly k goals)}
    with k=4 meaning "4 or more goals".
    """
    # Collect raw spread thresholds: P(team wins by > k.5)
    raw = {"home": {}, "away": {}}
    for m in markets:
        t = m["ticker"]
        p = _midprice(m, use_last=True)
        if p is None:
            continue
        # HOME markets: end in {HOME_CODE}2 / {HOME_CODE}3 / {HOME_CODE}4
        for k in (2, 3, 4):
            if t.endswith(f"-{home_code}{k}"):
                raw["home"][k] = p
            elif t.endswith(f"-{away_code}{k}"):
                raw["away"][k] = p

    def to_exact(thresholds: dict, p_win: float) -> dict | None:
        if not thresholds or p_win < 0.01:
            return None
        over = {1: p_win, 2: thresholds.get(2, 0.0),
                3: thresholds.get(3, 0.0), 4: thresholds.get(4, 0.0)}
        # Fill missing lower thresholds monotonically (upward)
        for k in (3, 2):
            if over[k] == 0.0 and over[k + 1] > 0.0:
                over[k] = over[k + 1]
        # Clip to enforce monotone decreasing (noisy markets can invert)
        over[4] = min(over[4], over[3])
        over[3] = min(over[3], over[2])
        over[2] = min(over[2], over[1])
        exact = {
            1: max(0.0, over[1] - over[2]),
            2: max(0.0, over[2] - over[3]),
            3: max(0.0, over[3] - over[4]),
            4: max(0.0, over[4]),
        }
        total = sum(exact.values())
        if total < 0.01:
            return None
        return {k: v / total for k, v in exact.items()}

    sh = to_exact(raw["home"], p_home_win)
    sa = to_exact(raw["away"], p_away_win)
    return sh, sa


def _load_cache() -> dict:
    if CACHE_PATH.exists():
        data = json.loads(CACHE_PATH.read_text())
        # Discard old single-timestamp format
        if "_ts" in data and isinstance(data["_ts"], (int, float)):
            return {}
        return data
    return {}


def _save_cache(data: dict):
    CACHE_PATH.write_text(json.dumps(data, indent=2))


def _clean_entry(entry: dict) -> dict:
    """Strip internal fields and restore int keys for spread dicts."""
    result = {k: v for k, v in entry.items() if k != "_ts"}
    for key in ("spread_home", "spread_away"):
        if result.get(key):
            result[key] = {int(k): v for k, v in result[key].items()}
    return result


def fetch_match_probs(match: dict) -> dict:
    """
    Fetch Kalshi probabilities for one match from schedule.json.
    Returns {home_win, draw, away_win, total_goals, spread_home, spread_away, source}.

    Cache policy:
    - Once kickoff time has passed, the cached entry is locked permanently so
      in-game/post-game prices never alter the prediction.
    - Before kickoff, a cached entry is reused if it is less than 90 minutes old,
      meaning the last pre-kickoff run (scheduled 1.5 h before the match) is the
      one that sets the final prediction.
    """
    home_code = TEAM_CODES.get(match["home"])
    away_code = TEAM_CODES.get(match["away"])
    if not home_code or not away_code:
        return _fallback_probs()

    date_code     = _date_code(match["time_utc"])
    game_ticker   = f"KXWCGAME-{date_code}{home_code}{away_code}"
    total_ticker  = f"KXWCTOTAL-{date_code}{home_code}{away_code}"
    spread_ticker = f"KXWCSPREAD-{date_code}{home_code}{away_code}"

    kickoff = datetime.strptime(
        match["time_utc"], "%Y-%m-%dT%H:%M:%SZ"
    ).replace(tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)

    cache = _load_cache()
    if game_ticker in cache:
        entry = cache[game_ticker]
        # Kickoff passed → prices locked, never re-fetch
        if now >= kickoff:
            return _clean_entry(entry)
        # Pre-game → use cached prices if still within 90-minute window
        if time.time() - entry.get("_ts", 0) < CACHE_TTL:
            return _clean_entry(entry)

    game_probs  = _parse_game(_fetch_event(game_ticker), home_code, away_code)
    total_goals = _parse_totals(_fetch_event(total_ticker))

    if game_probs is None:
        result = _fallback_probs()
    else:
        spread_home, spread_away = _parse_spread(
            _fetch_event(spread_ticker), home_code, away_code,
            game_probs["home_win"], game_probs["away_win"],
        )
        result = {
            "home_win":    game_probs["home_win"],
            "draw":        game_probs["draw"],
            "away_win":    game_probs["away_win"],
            "total_goals": total_goals,
            "spread_home": spread_home,
            "spread_away": spread_away,
            "source": "kalshi",
        }

    entry = dict(result)
    entry["_ts"] = time.time()
    cache[game_ticker] = entry
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
