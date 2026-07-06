"""
Analyze whether Kalshi snapshots taken earlier (hours/days before kickoff)
would have scored better or worse than snapshots taken close to kickoff.

Uses the hourly snapshots[] time series collected per match in
data/kalshi_cache.json (see model/kalshi.py) together with final results
in site/data/results.json. Only group-stage matches are usable: knockout
scoreline_probs_knockout() needs reg_home_win/team_totals, which are not
stored in the lightweight snapshot entries.

Model parameters (gamma, delta, alpha) are held fixed at their current
estimates for every snapshot of every match, so the only thing that varies
across offsets is which Kalshi snapshot was fed in. This isolates the
effect of fetch timing from the model's own in-tournament learning.
"""
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from model.kalshi import TEAM_CODES, _date_code, _correct_bias
from model.historical import build_distributions, scoreline_probs
from model.optimizer import best_prediction, _points_group
from model.learn import load_canonical, count_2026, estimate_gamma, estimate_delta, estimate_alpha

SCHEDULE_PATH = ROOT / "data" / "schedule.json"
CACHE_PATH    = ROOT / "data" / "kalshi_cache.json"
RESULTS_PATH  = ROOT / "site" / "data" / "results.json"

# Target offsets (hours before kickoff) to compare, plus tolerance window.
OFFSETS_H = [72, 48, 24, 12, 6, 3, 1]
TOLERANCE = 0.5  # snapshot must be within target*TOLERANCE hours of target


def _int_key_dict(d):
    return {int(k): v for k, v in d.items()} if d else None


def _closest_snapshot(snapshots, target_h, kickoff_ts):
    best, best_diff = None, None
    for snap in snapshots:
        hrs_before = (kickoff_ts - snap["ts"]) / 3600
        if hrs_before < 0:
            continue
        diff = abs(hrs_before - target_h)
        if best_diff is None or diff < best_diff:
            best, best_diff = snap, diff
    tol = max(target_h * TOLERANCE, 1.0)
    if best is not None and best_diff <= tol:
        return best, (kickoff_ts - best["ts"]) / 3600
    return None, None


def _predict_from_snapshot(snap, phase, dist, alpha):
    p_home, p_draw, p_away = _correct_bias(
        snap["home_win"], snap["draw"], snap["away_win"], alpha
    )
    total_goals  = _int_key_dict(snap.get("total_goals"))
    spread_home  = _int_key_dict(snap.get("spread_home"))
    spread_away  = _int_key_dict(snap.get("spread_away"))
    score_dist = scoreline_probs(
        p_home, p_draw, p_away, phase, total_goals, spread_home, spread_away, dist=dist
    )
    return best_prediction(score_dist, "group")


def run():
    schedule = json.loads(SCHEDULE_PATH.read_text())["matches"]
    cache = json.loads(CACHE_PATH.read_text())
    results_json = json.loads(RESULTS_PATH.read_text())
    results = results_json.get("matches", {})
    schedule_data = json.loads(SCHEDULE_PATH.read_text())

    canonical = load_canonical(results_json, schedule_data)
    gamma = estimate_gamma()
    delta = estimate_delta(canonical, gamma=gamma)
    alpha = estimate_alpha()
    extra_wins, extra_draws = count_2026(canonical)
    dist = build_distributions(gamma=gamma, extra_wins=extra_wins,
                                extra_draws=extra_draws, delta=delta)
    print(f"Using fixed model params: gamma={gamma:.3f} delta={delta:.3f} alpha={alpha:.3f}\n")

    # bucket -> list of (match_id, points); "near_kickoff" is the
    # closest-available snapshot to kickoff (proxy for the real locked prediction)
    buckets = defaultdict(list)
    near_kickoff_pts = {}
    n_matches = 0

    for m in schedule:
        mid = str(m["id"])
        if m.get("phase") != "group":
            continue
        if mid not in results:
            continue
        res = results[mid]
        if res.get("status") != "final":
            continue
        act_a, act_b = res["home_score"], res["away_score"]

        hc, ac = TEAM_CODES.get(m["home"]), TEAM_CODES.get(m["away"])
        if not hc or not ac:
            continue
        ticker = f"KXWCGAME-{_date_code(m['time_utc'])}{hc}{ac}"
        entry = cache.get(ticker)
        if not entry:
            continue
        snapshots = entry.get("snapshots", [])
        if len(snapshots) < 2:
            continue

        kickoff_ts = datetime.strptime(
            m["time_utc"], "%Y-%m-%dT%H:%M:%SZ"
        ).replace(tzinfo=timezone.utc).timestamp()

        n_matches += 1

        # near-kickoff proxy: snapshot with smallest hours-before-kickoff
        near = min(snapshots, key=lambda s: kickoff_ts - s["ts"] if kickoff_ts - s["ts"] >= 0 else float("inf"))
        pred = _predict_from_snapshot(near, "group", dist, alpha)
        near_pts = _points_group(pred["home"], pred["away"], act_a, act_b)
        near_kickoff_pts[mid] = near_pts

        for target_h in OFFSETS_H:
            snap, actual_h = _closest_snapshot(snapshots, target_h, kickoff_ts)
            if snap is None:
                continue
            pred = _predict_from_snapshot(snap, "group", dist, alpha)
            pts = _points_group(pred["home"], pred["away"], act_a, act_b)
            buckets[target_h].append((mid, pts))

    all_near = list(near_kickoff_pts.values())
    print(f"Matches analyzed: {n_matches}")
    print(f"near kickoff (all {len(all_near)} matches): mean pts = {sum(all_near)/len(all_near):.3f}\n")

    print(f"{'Offset':>14}  {'n':>4}  {'mean pts':>9}  {'vs near-KO (paired)':>20}  {'win/tie/lose':>14}")
    for target_h in OFFSETS_H:
        rows = buckets[target_h]
        if not rows:
            continue
        pts = [p for _, p in rows]
        paired_near = [near_kickoff_pts[mid] for mid, _ in rows]
        diffs = [p - n for p, n in zip(pts, paired_near)]
        wins  = sum(1 for d in diffs if d > 0)
        ties  = sum(1 for d in diffs if d == 0)
        loses = sum(1 for d in diffs if d < 0)
        mean_diff = sum(diffs) / len(diffs)
        print(f"{str(target_h)+'h before':>14}  {len(pts):>4}  {sum(pts)/len(pts):>9.3f}  "
              f"{mean_diff:>+20.3f}  {f'{wins}/{ties}/{loses}':>14}")


if __name__ == "__main__":
    run()
