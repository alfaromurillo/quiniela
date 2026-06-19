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
CACHE_TTL = 5400       # 90 min — covers the 1.5h-before-kickoff → kickoff window
EARLY_SNAPSHOT_H = 24  # stamp early_* once when match is this many hours away
SNAPSHOT_INTERVAL = 3600  # seconds between history snapshots (one per hour)

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


def _parse_game(markets: list[dict], home_code: str, away_code: str,
                phase: str = "group", p_draw_ko: float = 0.25) -> dict | None:
    """
    Extract normalised P(home_win), P(draw), P(away_win).

    For group stage: requires HOME, AWAY and TIE markets.

    For knockout: Kalshi has no TIE market (match always has a winner via
    penalties). Requires HOME and AWAY only. Translates binary Kalshi match-
    winner probabilities into 90+30 min ternary probabilities using the
    historical knockout draw rate p_draw_ko:
        p_h = r_h - 0.5*p_d,  p_a = r_a - 0.5*p_d,  p_d = p_draw_ko
    (assumes penalty shootouts are 50/50 between the two teams).
    """
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

    if phase == "knockout":
        if "home_win" not in probs or "away_win" not in probs:
            return None
        total = probs["home_win"] + probs["away_win"]
        if total < 0.5:
            return None
        r_h = probs["home_win"] / total
        r_a = probs["away_win"] / total
        p_d = p_draw_ko
        p_h = max(0.0, r_h - 0.5 * p_d)
        p_a = max(0.0, r_a - 0.5 * p_d)
        t2 = p_h + p_d + p_a
        return {"home_win": p_h / t2, "draw": p_d / t2, "away_win": p_a / t2}

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
    """Strip internal/history fields; restore int keys for numeric-keyed dicts."""
    _strip = {"_ts", "_early_ts", "snapshots",
              "early_home_win", "early_draw", "early_away_win",
              "early_total_goals", "early_spread_home", "early_spread_away",
              "early_source"}
    result = {k: v for k, v in entry.items() if k not in _strip}
    for key in ("spread_home", "spread_away", "total_goals",
                "early_spread_home", "early_spread_away", "early_total_goals"):
        if result.get(key):
            result[key] = {int(k): v for k, v in result[key].items()}
    return result


def get_early_snapshot(match: dict) -> dict | None:
    """
    Return the early (≥24h-before-kickoff) Kalshi snapshot for a match, or
    None if no early snapshot was recorded.  Keys are the same as fetch_match_probs
    but prefixed with 'early_'.  Restores int keys for numeric-keyed dicts.
    """
    home_code = TEAM_CODES.get(match["home"])
    away_code = TEAM_CODES.get(match["away"])
    if not home_code or not away_code:
        return None
    game_ticker = f"KXWCGAME-{_date_code(match['time_utc'])}{home_code}{away_code}"
    cache = _load_cache()
    entry = cache.get(game_ticker, {})
    if "early_home_win" not in entry:
        return None
    snap = {
        "home_win":   entry["early_home_win"],
        "draw":       entry["early_draw"],
        "away_win":   entry["early_away_win"],
        "total_goals": entry.get("early_total_goals"),
        "spread_home": entry.get("early_spread_home"),
        "spread_away": entry.get("early_spread_away"),
        "source":      entry.get("early_source", "kalshi"),
        "early_ts":    entry.get("_early_ts"),
    }
    for key in ("total_goals", "spread_home", "spread_away"):
        if snap.get(key):
            snap[key] = {int(k): v for k, v in snap[key].items()}
    return snap


def fetch_match_probs(match: dict, p_draw_knockout: float = 0.25) -> dict:
    """
    Fetch Kalshi probabilities for one match from schedule.json.
    Returns {home_win, draw, away_win, total_goals, spread_home, spread_away, source}.

    p_draw_knockout: historical knockout ET-draw rate, used to convert Kalshi
      binary (home/away only, no TIE market) into 90+30 min ternary probs.
      Pass the value from historical.knockout_draw_rate(gamma).

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

    phase = match.get("phase", "group")
    game_probs  = _parse_game(_fetch_event(game_ticker), home_code, away_code,
                               phase=phase, p_draw_ko=p_draw_knockout)
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

    # Preserve early snapshot: stamp once when match is >EARLY_SNAPSHOT_H away.
    # Keep existing early data if already present.
    existing_early = cache.get(game_ticker, {})
    if "early_home_win" in existing_early:
        for k in ("early_home_win", "early_draw", "early_away_win",
                  "early_total_goals", "early_spread_home", "early_spread_away",
                  "early_source", "_early_ts"):
            if k in existing_early:
                entry[k] = existing_early[k]
    elif result.get("source") == "kalshi":
        hours_to_kickoff = (kickoff - now).total_seconds() / 3600
        if hours_to_kickoff >= EARLY_SNAPSHOT_H:
            entry["early_home_win"]   = result["home_win"]
            entry["early_draw"]       = result["draw"]
            entry["early_away_win"]   = result["away_win"]
            entry["early_total_goals"]= result.get("total_goals")
            entry["early_spread_home"]= result.get("spread_home")
            entry["early_spread_away"]= result.get("spread_away")
            entry["early_source"]     = "kalshi"
            entry["_early_ts"]        = time.time()

    # Append to time-series snapshot history (one entry per hour, pre-kickoff only).
    # Each snapshot: {ts, home_win, draw, away_win, total_goals, spread_home, spread_away}
    # Never modified once written — allows later analysis of which fetch time is best.
    if result.get("source") == "kalshi":
        snapshots: list = existing_early.get("snapshots", [])
        last_ts = snapshots[-1]["ts"] if snapshots else 0
        if time.time() - last_ts >= SNAPSHOT_INTERVAL:
            snap = {
                "ts":          time.time(),
                "home_win":    result["home_win"],
                "draw":        result["draw"],
                "away_win":    result["away_win"],
                "total_goals": result.get("total_goals"),
                "spread_home": result.get("spread_home"),
                "spread_away": result.get("spread_away"),
            }
            snapshots.append(snap)
        entry["snapshots"] = snapshots

    cache[game_ticker] = entry
    _save_cache(cache)
    return result


def _correct_bias(home_win: float, draw: float, away_win: float,
                  alpha: float) -> tuple[float, float, float]:
    """
    Power transformation to correct favourite-longshot bias.
    q_i = p_i^α / Σ p_j^α  (α > 1 shifts mass from longshots to favourites).
    """
    if abs(alpha - 1.0) < 1e-9:
        return home_win, draw, away_win
    qh, qd, qa = home_win ** alpha, draw ** alpha, away_win ** alpha
    total = qh + qd + qa
    if total < 1e-12:
        return home_win, draw, away_win
    return qh / total, qd / total, qa / total


def _fallback_probs() -> dict:
    return {
        "home_win": 0.40,
        "draw": 0.25,
        "away_win": 0.35,
        "total_goals": {0: 0.08, 1: 0.20, 2: 0.27, 3: 0.24, 4: 0.13, 5: 0.05, 6: 0.03},
        "source": "historical_fallback",
    }
