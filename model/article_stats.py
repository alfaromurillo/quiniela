"""
Numbers reported in the article's Results section.

Every quiniela point here uses the base 5/3/2/1/0 rule with no
late-round multiplier: some pools multiply points in later rounds,
which does not change the argmax, so the article evaluates on the
base scale.

Run from repo root:
    python model/article_stats.py            # prints a summary
    python model/article_stats.py --json F   # also writes all numbers to F
"""
import json
import random
import statistics as st
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from model.optimizer import _points_group as points, best_prediction, \
    modal_prediction, expected_points
from model.kalshi import (_load_cache, _clean_entry, _date_code,
                          _correct_bias, TEAM_CODES)
from model.historical import (build_distributions, build_et_kernel,
                              scoreline_probs_ipf, scoreline_probs_knockout)

GAMMA, DELTA, ALPHA = 0.84, 0.10, 1.14   # final estimates (learning.json)
N_BOOT = 10_000

# Strategy keys in locked_predictions.json → article strategy number
STRATEGIES = {
    1: "baseline_hist",
    2: "baseline_out_raw",
    3: "baseline_out_alpha",
    4: "baseline_full_raw",
    5: "modal",
}


def load():
    schedule = json.loads((ROOT / "data/schedule.json").read_text())["matches"]
    results  = json.loads((ROOT / "site/data/results.json").read_text())["matches"]
    locked   = json.loads((ROOT / "site/data/locked_predictions.json").read_text())["matches"]
    preds    = {m["id"]: m for m in json.loads(
        (ROOT / "site/data/predictions.json").read_text())["matches"]}
    return schedule, results, locked, preds


def favourite_is_home(probs: dict) -> bool:
    return probs["home_win"] >= probs["away_win"]


def match_rows():
    """One row per match, in kickoff order, with every strategy's points."""
    schedule, results, locked, preds = load()
    rows = []
    for m in sorted(schedule, key=lambda m: m["time_utc"]):
        k = str(m["id"])
        a, b = results[k]["home_score"], results[k]["away_score"]
        lk = locked[k]
        pr = preds[m["id"]]["probabilities"]
        fav_home = favourite_is_home(pr)
        row = {
            "id": m["id"], "phase": m["phase"], "round": m["round"],
            "home": m["home"], "away": m["away"], "result": (a, b),
            "pred": (lk["home"], lk["away"]), "exp": lk.get("expected_pts"),
            "source": preds[m["id"]].get("source"),
            "probs": pr,
            "p_fav": max(pr["home_win"], pr["away_win"]),
            "pts": {6: points(lk["home"], lk["away"], a, b),
                    "fav10": points(*((1, 0) if fav_home else (0, 1)), a, b),
                    "fav21": points(*((2, 1) if fav_home else (1, 2)), a, b),
                    "always00": points(0, 0, a, b)},
            "picks": {6: (lk["home"], lk["away"])},
        }
        for s, key in STRATEGIES.items():
            if key + "_home" in lk:
                pick = (lk[key + "_home"], lk[key + "_away"])
                row["picks"][s] = pick
                row["pts"][s] = points(*pick, a, b)
        rows.append(row)
    return rows


def bayes_boot(diff: list[float], seed: int = 1) -> tuple[float, float, float]:
    """Total difference and its 95% Bayesian-bootstrap interval."""
    rng = random.Random(seed)
    n, draws = len(diff), []
    for _ in range(N_BOOT):
        w = [rng.expovariate(1.0) for _ in range(n)]
        s = sum(w)
        draws.append(n * sum(wi * di for wi, di in zip(w, diff)) / s)
    draws.sort()
    return sum(diff), draws[int(0.025 * N_BOOT)], draws[int(0.975 * N_BOOT)]


def summary(rows):
    out = {}
    for name, rs in [("group", [r for r in rows if r["phase"] == "group"]),
                     ("knockout", [r for r in rows if r["phase"] == "knockout"]),
                     ("all", rows)]:
        n = len(rs)
        tot = sum(r["pts"][6] for r in rs)
        out[name] = {"n": n, "total": tot, "mean": tot / n,
                     "exact": sum(r["pts"][6] == 5 for r in rs),
                     "winner_or_better": sum(r["pts"][6] >= 2 for r in rs)}
    out["rounds"] = {}
    for rd in ["Round of 32", "Round of 16", "Quarter-final",
               "Semi-final", "Match for third place", "Final"]:
        rs = [r for r in rows if r["round"] == rd]
        out["rounds"][rd] = {"n": len(rs), "total": sum(r["pts"][6] for r in rs)}
    ex = [r for r in rows if r["exp"] is not None]
    # expected_pts of the double-points rounds was logged ×2 only after the
    # tournament; all logged values predate that change, so they are base.
    out["ex_ante"] = {"n": len(ex), "expected": st.mean(r["exp"] for r in ex),
                      "realized": st.mean(r["pts"][6] for r in ex)}
    return out


def strategies(rows):
    cmp_ = [r for r in rows if 5 in r["pts"]]
    out = {"n": len(cmp_),
           "n_group": sum(r["phase"] == "group" for r in cmp_)}
    keys = [1, 2, 3, 4, 5, 6, "fav10", "fav21", "always00"]
    for s in keys:
        out[str(s)] = {ph: sum(r["pts"][s] for r in cmp_ if r["phase"] == ph)
                       for ph in ("group", "knockout")}
        out[str(s)]["all"] = sum(r["pts"][s] for r in cmp_)
    # whole tournament for strategies available on all 104 matches
    out["all104"] = {str(s): sum(r["pts"][s] for r in rows)
                     for s in (6, "fav10", "fav21", "always00")}
    out["diffs"] = {}
    for x, y in [(2, 1), (3, 2), (4, 2), (6, 4), (6, 5), (6, "fav10")]:
        d = [r["pts"][x] - r["pts"][y] for r in cmp_]
        tot, lo, hi = bayes_boot(d)
        out["diffs"][f"{x}-{y}"] = {"total": tot, "lo": lo, "hi": hi,
                                   "better": sum(v > 0 for v in d),
                                   "worse": sum(v < 0 for v in d)}
    d = [r["pts"][6] - r["pts"]["fav10"] for r in rows]
    tot, lo, hi = bayes_boot(d)
    out["diffs"]["6-fav10_all104"] = {"total": tot, "lo": lo, "hi": hi,
                                      "better": sum(v > 0 for v in d),
                                      "worse": sum(v < 0 for v in d)}
    kinds = Counter()
    sign = lambda p: (p[0] > p[1]) - (p[0] < p[1])
    for r in cmp_:
        o, m = r["picks"][6], r["picks"][5]
        if o == m:
            continue
        if sign(m) == 0 and sign(o) != 0:
            kinds["draw_to_win"] += 1
        elif sign(m) == sign(o):
            kinds["same_winner"] += 1
        else:
            kinds["other"] += 1
    out["opt_vs_modal_kinds"] = dict(kinds)
    out["draw_picks"] = {"optimal": sum(sign(r["picks"][6]) == 0 for r in cmp_),
                         "modal": sum(sign(r["picks"][5]) == 0 for r in cmp_)}
    return out


def predicted_vs_observed(rows):
    canon = lambda p: (max(p), min(p))
    return {"predicted": Counter(canon(r["pred"]) for r in rows),
            "observed": Counter(canon(r["result"]) for r in rows)}


def by_favourite(rows, edges=(0.0, 0.45, 0.55, 0.65, 0.75, 1.0)):
    out = []
    ks = [r for r in rows if r["source"] == "kalshi"]
    for lo, hi in zip(edges[:-1], edges[1:]):
        rs = [r for r in ks if lo <= r["p_fav"] < hi]
        if not rs:
            continue
        out.append({"lo": lo, "hi": hi, "n": len(rs),
                    "p_fav": st.mean(r["p_fav"] for r in rs),
                    "model": st.mean(r["pts"][6] for r in rs),
                    "fav10": st.mean(r["pts"]["fav10"] for r in rs)})
    return out


def surprise_rows(rows):
    """
    Kalshi-only surprise measures for the matches with markets:
    p_out = market probability of the observed quiniela outcome,
    d_goals = observed total goals minus the market's expected total
    (6+ counted as 6).
    """
    out = []
    for r in rows:
        if r["source"] != "kalshi":
            continue
        a, b = r["result"]
        p = r["probs"]
        p_out = p["home_win"] if a > b else p["draw"] if a == b else p["away_win"]
        e_t = sum(int(k) * v for k, v in p["total_goals"].items())
        out.append({**r, "p_out": p_out, "d_goals": a + b - e_t})
    return out


SURPRISE_BINS = [
    ("Resultado: menos de 30 %", lambda r: r["p_out"] < 0.3),
    ("Resultado: 30 a 50 %",     lambda r: 0.3 <= r["p_out"] < 0.5),
    ("Resultado: 50 % o más",    lambda r: r["p_out"] >= 0.5),
    ("Goles: más de 1 por debajo", lambda r: r["d_goals"] < -1),
    ("Goles: a 1 o menos",         lambda r: abs(r["d_goals"]) <= 1),
    ("Goles: más de 1 por encima", lambda r: r["d_goals"] > 1),
]


def surprises(rows):
    """Model minus 'favourite wins 1-0' points per match, by surprise bin."""
    sr = surprise_rows(rows)
    out = {"n": len(sr),
           "upsets": sum(r["p_out"] < 0.3 for r in sr),
           "goals_off_2": sum(abs(r["d_goals"]) > 2 for r in sr),
           "bins": []}
    for label, keep in SURPRISE_BINS:
        g = [r for r in sr if keep(r)]
        d = [r["pts"][6] - r["pts"]["fav10"] for r in g]
        tot, lo, hi = bayes_boot(d)
        out["bins"].append({"label": label, "n": len(g),
                            "model": st.mean(r["pts"][6] for r in g),
                            "fav10": st.mean(r["pts"]["fav10"] for r in g),
                            "diff": tot / len(g), "lo": lo / len(g),
                            "hi": hi / len(g)})
    return out


ROUND_ORDER = ["G1", "G2", "G3", "R32", "R16", "QF", "SF-F"]


def _round_key(m: dict) -> str | None:
    """Round label of a non-group match (both data sources)."""
    r = m["round"].lower()
    if "round of 32" in r:
        return "R32"
    if "round of 16" in r:
        return "R16"
    if "quarter" in r:
        return "QF"
    if "semi" in r or "third" in r or "final" in r:
        return "SF-F"
    return None


def _by_round(matches: list[dict], score) -> dict[str, list]:
    """Final scores grouped by round; group rounds by order within group."""
    out: dict[str, list] = {}
    groups: dict[str, list] = {}
    for m in matches:
        if m.get("group") and "Matchday" in m["round"]:
            groups.setdefault(m["group"], []).append(m)
        else:
            k = _round_key(m)
            if k:
                out.setdefault(k, []).append(score(m))
    for lst in groups.values():
        lst.sort(key=lambda m: m.get("time_utc") or (m["date"], m.get("time", "")))
        for i, m in enumerate(lst):
            out.setdefault(f"G{i // 2 + 1}", []).append(score(m))
    return out


def rounds_compare():
    """
    Goals per match and share of draws by round, 2026 vs 2014-2022.
    Scores include extra time, so a knockout draw is a penalty shootout.
    """
    hist: dict[str, list] = {}
    for y in (2014, 2018, 2022):
        ms = json.loads((ROOT / f"data/wc{y}.json").read_text())["matches"]
        for k, v in _by_round(
                ms, lambda m: tuple(m["score"].get("et") or m["score"]["ft"])).items():
            hist.setdefault(k, []).extend(v)
    schedule, results, _, _ = load()
    new = _by_round(schedule, lambda m: (results[str(m["id"])]["home_score"],
                                         results[str(m["id"])]["away_score"]))
    out = {}
    for name, d in (("hist", hist), ("2026", new)):
        out[name] = {k: {"n": len(v),
                         "goals": st.mean(a + b for a, b in v),
                         "draws": sum(a == b for a, b in v) / len(v)}
                     for k, v in d.items()}
    return out


def model_distributions():
    """Final-parameter scoreline distributions for the 98 Kalshi matches."""
    schedule, _, _, _ = load()
    cache = _load_cache()
    dist = build_distributions(gamma=GAMMA, delta=DELTA)
    kern = build_et_kernel(GAMMA, delta=DELTA)
    out = {}
    for m in schedule:
        e = cache.get(f"KXWCGAME-{_date_code(m['time_utc'])}"
                      f"{TEAM_CODES[m['home']]}{TEAM_CODES[m['away']]}")
        if not e:
            continue
        e = _clean_entry(e)
        if e.get("source") != "kalshi":
            continue
        ph, pd, pa = _correct_bias(e["home_win"], e["draw"], e["away_win"], ALPHA)
        if "reg_home_win" in e:
            rh, rd, ra = _correct_bias(e["reg_home_win"], e["reg_draw"],
                                       e["reg_away_win"], ALPHA)
            sp = scoreline_probs_knockout(
                ph, pd, pa, rh, rd, ra, e["total_goals"], e["spread_home"],
                e["spread_away"], dist=dist, et_kernel=kern,
                team_totals={"home": e.get("team_totals_home"),
                             "away": e.get("team_totals_away")})
        else:
            sp = scoreline_probs_ipf(ph, pd, pa, "group", e["total_goals"],
                                     e["spread_home"], e["spread_away"], dist=dist)
        out[m["id"]] = sp
    return out


def rule_points(rule):
    """Point function for a rule (exact, winner+goals, winner, goals)."""
    ex, wg, w, g = rule
    def f(pa, pb, a, b):
        base = points(pa, pb, a, b)
        return {5: ex, 3: wg, 2: w, 1: g, 0: 0}[base]
    return f


def rule_sensitivity(rows, dists,
                     rules=((5, 3, 2, 1), (10, 3, 2, 1), (5, 4, 2, 1),
                            (3, 2, 2, 1))):
    """Optimal vs modal under alternative scoring rules (final model)."""
    res = {r["id"]: r["result"] for r in rows}
    cells = [(a, b) for a in range(6) for b in range(6)]
    out = []
    for rule in rules:
        f = rule_points(rule)
        differ, gain_exp, gain_obs = 0, 0.0, 0
        for mid, sp in dists.items():
            ev = {c: sum(f(*c, h, k) * p for (h, k), p in sp.items()) for c in cells}
            opt = max(cells, key=ev.get)
            mod = max(cells, key=sp.get)
            if opt != mod:
                differ += 1
            gain_exp += ev[opt] - ev[mod]
            gain_obs += f(*opt, *res[mid]) - f(*mod, *res[mid])
        out.append({"rule": rule, "n": len(dists), "differ": differ,
                    "gain_exp_per_match": gain_exp / len(dists),
                    "gain_obs": gain_obs})
    return out


def main():
    rows = match_rows()
    dists = model_distributions()
    stats = {
        "summary": summary(rows),
        "strategies": strategies(rows),
        "pred_vs_obs": {k: {f"{a}-{b}": n for (a, b), n in v.items()}
                        for k, v in predicted_vs_observed(rows).items()},
        "by_favourite": by_favourite(rows),
        "rules": rule_sensitivity(rows, dists),
        "surprises": surprises(rows),
        "rounds": rounds_compare(),
    }
    print(json.dumps(stats, indent=1, default=str))
    if "--json" in sys.argv:
        Path(sys.argv[sys.argv.index("--json") + 1]).write_text(
            json.dumps(stats, indent=1, default=str))


if __name__ == "__main__":
    main()
