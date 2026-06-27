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
                phase: str = "group", p_draw_ko: float = 0.25) -> tuple[dict | None, bool]:
    """
    Extract normalised P(home_win), P(draw), P(away_win) from KXWCGAME markets.

    Returns (probs_dict, has_tie_market).

    When all three markets (HOME, AWAY, TIE) are present the probs represent
    regulation-time (90 min) outcomes — including for knockout matches.
    has_tie_market=True signals that the caller should apply the reg→quiniela
    ET conversion for knockout.

    Fallback for knockout without TIE market: synthesises quiniela probs using
    the historical penalty-draw rate p_draw_ko (old behaviour, has_tie=False).
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

    if len(probs) == 3:
        total = sum(probs.values())
        if total < 0.5:
            return None, False
        return {k: v / total for k, v in probs.items()}, True

    # Knockout fallback: TIE market absent — synthesise from historical rate
    if phase == "knockout" and "home_win" in probs and "away_win" in probs:
        total = probs["home_win"] + probs["away_win"]
        if total < 0.5:
            return None, False
        r_h = probs["home_win"] / total
        r_a = probs["away_win"] / total
        p_d = p_draw_ko
        p_h = max(0.0, r_h - 0.5 * p_d)
        p_a = max(0.0, r_a - 0.5 * p_d)
        t2 = p_h + p_d + p_a
        return {"home_win": p_h / t2, "draw": p_d / t2, "away_win": p_a / t2}, False

    return None, False


def _parse_advance(markets: list[dict], home_code: str, away_code: str) -> dict | None:
    """Parse KXWCADVANCE market: who advances (includes ET + penalties)."""
    probs = {}
    for m in markets:
        t = m["ticker"]
        p = _midprice(m)
        if p is None:
            continue
        if t.endswith(f"-{home_code}"):
            probs["home"] = p
        elif t.endswith(f"-{away_code}"):
            probs["away"] = p
    if len(probs) != 2:
        return None
    total = probs["home"] + probs["away"]
    if total < 0.5:
        return None
    return {"home": probs["home"] / total, "away": probs["away"] / total}


def _parse_teamtotals(markets: list[dict], home_code: str, away_code: str) -> dict | None:
    """
    Parse KXWCTEAMTOTAL markets: P(team scores >= N goals) in regulation time.
    Returns {"home": {1: p, 2: p, ...}, "away": {1: p, 2: p, ...}}.
    """
    home_over: dict[int, float] = {}
    away_over: dict[int, float] = {}
    for m in markets:
        t = m["ticker"]
        p = _midprice(m, use_last=True)
        if p is None:
            continue
        for code, bucket in [(home_code, home_over), (away_code, away_over)]:
            match = re.search(rf"-{re.escape(code)}(\d+)$", t)
            if match:
                bucket[int(match.group(1))] = p
    if not home_over and not away_over:
        return None
    return {
        "home": home_over if home_over else None,
        "away": away_over if away_over else None,
    }


def _reg_to_quiniela(
    p_reg_h: float, p_reg_d: float, p_reg_a: float,
    p_adv_h: float | None, p_adv_a: float | None,
    p_et_draw: float,
) -> tuple[float, float, float]:
    """
    Convert regulation-time (90 min) outcome probs to quiniela (90+30 min) probs.

    p_q_draw  = P(reg draw) × P(ET draw → penalties | ET played)
    p_q_home  = P(home advances) − p_q_draw/2   [using KXWCADVANCE]
    p_q_away  = P(away advances) − p_q_draw/2

    If advance probs unavailable: ET winner ratio = reg-time winner ratio.
    """
    p_q_draw = p_reg_d * p_et_draw

    if p_adv_h is not None and p_adv_a is not None:
        p_q_home = p_adv_h - p_q_draw / 2.0
        p_q_away = p_adv_a - p_q_draw / 2.0
    else:
        denom = p_reg_h + p_reg_a
        r_h = p_reg_h / denom if denom > 1e-9 else 0.5
        r_a = 1.0 - r_h
        et_win = p_reg_d * (1.0 - p_et_draw)
        p_q_home = p_reg_h + et_win * r_h
        p_q_away = p_reg_a + et_win * r_a

    p_q_home = max(0.0, p_q_home)
    p_q_away = max(0.0, p_q_away)
    p_q_draw = max(0.0, p_q_draw)
    total = p_q_home + p_q_draw + p_q_away
    if total < 0.5:
        return 1 / 3, 1 / 3, 1 / 3
    return p_q_home / total, p_q_draw / total, p_q_away / total


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
                "early_spread_home", "early_spread_away", "early_total_goals",
                "team_totals_home", "team_totals_away"):
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


def fetch_match_probs(match: dict,
                      p_draw_knockout: float = 0.25,
                      p_et_draw_knockout: float = 0.72) -> dict:
    """
    Fetch Kalshi probabilities for one match from schedule.json.

    For group stage returns:
      {home_win, draw, away_win, total_goals, spread_home, spread_away, source}
    where all probs are regulation-time = quiniela probs.

    For knockout, additionally returns:
      {reg_home_win, reg_draw, reg_away_win,   ← regulation-time probs (KXWCGAME-TIE)
       advance_home_win, advance_away_win,       ← who advances (KXWCADVANCE)
       team_totals_home, team_totals_away}       ← per-team goals (KXWCTEAMTOTAL)
    home_win/draw/away_win are the quiniela probs (90+30 min, penalties→draw).

    p_draw_knockout: P(penalties|KO) used only in fallback when KXWCGAME has
      no TIE market. Pass knockout_draw_rate(gamma) from historical.py.
    p_et_draw_knockout: P(penalties|ET played), from knockout_et_draw_rate(gamma).
      Used to convert reg-time probs to quiniela probs.
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
    game_probs, has_tie = _parse_game(_fetch_event(game_ticker), home_code, away_code,
                                       phase=phase, p_draw_ko=p_draw_knockout)
    total_goals = _parse_totals(_fetch_event(total_ticker))

    if game_probs is None:
        result = _fallback_probs()
    elif phase == "knockout" and has_tie:
        # KXWCGAME has TIE market → reg-time probs; convert to quiniela probs
        advance_ticker  = f"KXWCADVANCE-{date_code}{home_code}{away_code}"
        teamtotal_ticker = f"KXWCTEAMTOTAL-{date_code}{home_code}{away_code}"
        advance_probs   = _parse_advance(_fetch_event(advance_ticker), home_code, away_code)
        team_totals     = _parse_teamtotals(_fetch_event(teamtotal_ticker), home_code, away_code)

        p_adv_h = advance_probs["home"] if advance_probs else None
        p_adv_a = advance_probs["away"] if advance_probs else None
        q_home, q_draw, q_away = _reg_to_quiniela(
            game_probs["home_win"], game_probs["draw"], game_probs["away_win"],
            p_adv_h, p_adv_a, p_et_draw_knockout,
        )
        spread_home, spread_away = _parse_spread(
            _fetch_event(spread_ticker), home_code, away_code,
            game_probs["home_win"], game_probs["away_win"],
        )
        result = {
            "home_win":          q_home,
            "draw":              q_draw,
            "away_win":          q_away,
            "reg_home_win":      game_probs["home_win"],
            "reg_draw":          game_probs["draw"],
            "reg_away_win":      game_probs["away_win"],
            "advance_home_win":  p_adv_h,
            "advance_away_win":  p_adv_a,
            "total_goals":       total_goals,
            "spread_home":       spread_home,
            "spread_away":       spread_away,
            "team_totals_home":  team_totals["home"] if team_totals else None,
            "team_totals_away":  team_totals["away"] if team_totals else None,
            "source": "kalshi",
        }
    else:
        # Group stage or knockout fallback (no TIE market)
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
