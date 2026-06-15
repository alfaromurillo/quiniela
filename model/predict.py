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

from model.kalshi import fetch_match_probs, _fallback_probs, _correct_bias
from model.historical import build_distributions, scoreline_probs
from model.optimizer import best_prediction, modal_prediction
from model.learn import (load_canonical, count_2026, estimate_gamma,
                          estimate_delta, estimate_alpha,
                          SIGMA_GAMMA, SIGMA_DELTA, ALPHA_0, SIGMA_ALPHA)

SCHEDULE_PATH  = ROOT / "data" / "schedule.json"
OUT_PATH       = ROOT / "site" / "data" / "predictions.json"
LOCKED_PATH    = ROOT / "site" / "data" / "locked_predictions.json"
RESULTS_PATH   = ROOT / "site" / "data" / "results.json"
LEARNING_PATH  = ROOT / "site" / "data" / "learning.json"


def _is_tbd(name: str) -> bool:
    return not name or name[0].isdigit() or name.startswith("W")


def run():
    schedule_data = json.loads(SCHEDULE_PATH.read_text())
    schedule = schedule_data["matches"]
    now = datetime.now(timezone.utc)

    # ── Adaptive model update from WC 2026 results ──
    if RESULTS_PATH.exists():
        results_json = json.loads(RESULTS_PATH.read_text())
    else:
        results_json = {"matches": {}}

    canonical        = load_canonical(results_json, schedule_data)
    gamma            = estimate_gamma()
    delta            = estimate_delta(canonical, gamma=gamma)
    alpha            = estimate_alpha()
    extra_wins, extra_draws = count_2026(canonical)
    dist             = build_distributions(gamma=gamma, extra_wins=extra_wins,
                                           extra_draws=extra_draws, delta=delta)
    n_2026           = len(canonical)
    print(f"Historical γ = {gamma:.3f}  |  WC 2026: {n_2026} results, "
          f"δ = {delta:.3f}  |  α = {alpha:.3f}")

    # Neutral historical baseline (1/3-1/3-1/3, no WC 2026 data, no Kalshi).
    # Same prediction for every match of a given phase — precompute once.
    dist_neutral = build_distributions(gamma=gamma, delta=0.0)
    from model.historical import scoreline_probs as _sp
    _sp_neutral_group    = _sp(1/3, 1/3, 1/3, "group",    dist=dist_neutral)
    _sp_neutral_knockout = _sp(1/3, 1/3, 1/3, "knockout", dist=dist_neutral)
    baseline_by_phase = {
        "group":    best_prediction(_sp_neutral_group,    "group"),
        "knockout": best_prediction(_sp_neutral_knockout, "knockout"),
    }

    # ── Load existing locked predictions ──
    if LOCKED_PATH.exists():
        locked_data = json.loads(LOCKED_PATH.read_text())
        locked: dict = locked_data.get("matches", {})
    else:
        locked = {}

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

        raw_home = kalshi["home_win"]
        raw_draw = kalshi["draw"]
        raw_away = kalshi["away_win"]
        total_goals  = kalshi.get("total_goals")
        spread_home  = kalshi.get("spread_home")
        spread_away  = kalshi.get("spread_away")
        source = kalshi.get("source", "unknown")

        # Apply favourite-longshot bias correction before building distribution
        p_home, p_draw, p_away = _correct_bias(raw_home, raw_draw, raw_away, alpha)

        # Build joint scoreline distribution using adaptive model
        score_dist = scoreline_probs(
            p_home, p_draw, p_away, phase,
            total_goals, spread_home, spread_away,
            dist=dist,
        )

        # Find best prediction
        best   = best_prediction(score_dist, phase)
        modal  = modal_prediction(score_dist)
        bh     = baseline_by_phase[phase]

        # ── Strategy-comparison baselines ────────────────────────
        # Row 2: Kalshi outcome only, raw probs (no α correction)
        _sp_out_raw = scoreline_probs(
            raw_home, raw_draw, raw_away, phase,
            None, None, None, dist=dist,
        )
        bsln_out_raw = best_prediction(_sp_out_raw, phase)

        # Row 3: Kalshi outcome only, α-corrected probs
        _sp_out_alpha = scoreline_probs(
            p_home, p_draw, p_away, phase,
            None, None, None, dist=dist,
        )
        bsln_out_alpha = best_prediction(_sp_out_alpha, phase)

        # Row 4: Full Kalshi (goles + spread), raw probs (no α)
        _sp_full_raw = scoreline_probs(
            raw_home, raw_draw, raw_away, phase,
            total_goals, spread_home, spread_away, dist=dist,
        )
        bsln_full_raw = best_prediction(_sp_full_raw, phase)

        def _hk(b):
            return {"home": b["home"], "away": b["away"]}

        # Lock prediction before kickoff; keep existing entry after kickoff
        kickoff = datetime.strptime(
            match["time_utc"], "%Y-%m-%dT%H:%M:%SZ"
        ).replace(tzinfo=timezone.utc)
        if now < kickoff:
            locked[str(mid)] = {
                "home":                  best["home"],
                "away":                  best["away"],
                "expected_pts":          best["expected_pts"],
                "modal_home":            modal["home"],
                "modal_away":            modal["away"],
                "baseline_hist_home":    bh["home"],
                "baseline_hist_away":    bh["away"],
                "baseline_out_raw_home": bsln_out_raw["home"],
                "baseline_out_raw_away": bsln_out_raw["away"],
                "baseline_out_alpha_home": bsln_out_alpha["home"],
                "baseline_out_alpha_away": bsln_out_alpha["away"],
                "baseline_full_raw_home":  bsln_full_raw["home"],
                "baseline_full_raw_away":  bsln_full_raw["away"],
                "locked_at":             now.isoformat(),
            }

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
                "modal":             _hk(modal),
                "baseline_hist":     _hk(bh),
                "baseline_out_raw":  _hk(bsln_out_raw),
                "baseline_out_alpha": _hk(bsln_out_alpha),
                "baseline_full_raw": _hk(bsln_full_raw),
            },
            "probabilities": {
                "home_win": round(raw_home, 4),
                "draw": round(raw_draw, 4),
                "away_win": round(raw_away, 4),
                "total_goals": {str(k): round(v, 4) for k, v in total_goals.items()} if total_goals else None,
            },
            "source": source,
            "tbd": False,
        }
        predictions.append(entry)
        print(f"{best['home']}-{best['away']} (E[pts]={best['expected_pts']:.2f}, src={source})")

    ts = now.isoformat()
    out = {"generated_at": ts, "matches": predictions}
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\nSaved {len(predictions)} matches → {OUT_PATH}")

    LOCKED_PATH.write_text(json.dumps(
        {"generated_at": ts, "matches": locked},
        indent=2, ensure_ascii=False,
    ))
    print(f"Saved locked predictions → {LOCKED_PATH}")

    # Preserve delta_history; append new entry only when n_2026 changes.
    existing = {}
    if LEARNING_PATH.exists():
        try:
            existing = json.loads(LEARNING_PATH.read_text())
        except Exception:
            pass
    history = existing.get("delta_history", [])
    if not history or history[-1].get("n_games") != n_2026:
        history.append({
            "n_games":      n_2026,
            "delta":        round(delta, 4),
            "generated_at": ts,
        })

    LEARNING_PATH.write_text(json.dumps({
        "generated_at":   ts,
        "gamma":          round(gamma, 4),
        "delta":          round(delta, 4),
        "alpha":          round(alpha, 4),
        "n_games_2026":   n_2026,
        "sigma_gamma":    SIGMA_GAMMA,
        "sigma_delta":    SIGMA_DELTA,
        "alpha_0":        ALPHA_0,
        "sigma_alpha":    SIGMA_ALPHA,
        "delta_history":  history,
    }, indent=2))
    print(f"Saved learning state → {LEARNING_PATH}")


if __name__ == "__main__":
    run()
