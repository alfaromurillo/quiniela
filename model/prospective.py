"""
Pseudo-prospective evaluation of the final method, and comparison of
scoreline models on the frozen Kalshi prices.

The operational record (article_stats.py) scores the predictions the
system actually published, which came from successive versions of the
code.  Here every match is predicted again, in kickoff order, by one
fixed method using only what was known before kickoff:

  - Kalshi prices: the cache entry frozen 10 min before kickoff;
  - γ: estimated from WC 2014-2022 only (fixed before the tournament);
  - δ, the WC 2026 counts in the reference distribution, and α: from
    the matches that had ended (kickoff + 3 h) before this kickoff.

The ET kernel stays historical, as in operation (no 90-min scores were
recorded for WC 2026 knockout matches).  Points use the base 5/3/2/1/0
rule.

Run from repo root:
    python model/prospective.py              # prints a summary
    python model/prospective.py --json F     # also writes it to F
"""
import json
import math
import statistics as st
import sys
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from model.optimizer import _points_group as points, best_prediction, \
    modal_prediction
from model.kalshi import _load_cache, _clean_entry, _date_code, \
    _correct_bias, TEAM_CODES
from model.historical import build_distributions, build_et_kernel, \
    scoreline_probs, scoreline_probs_ipf, extra_time
from model.learn import estimate_gamma, estimate_delta, load_canonical, \
    count_2026, alpha_map
from model import marginals as mg
from model import elo
from model.article_stats import bayes_boot, rule_points

ENDED_AFTER = timedelta(hours=3)        # a match's result is known 3 h after kickoff

# name → (label for the article, needs Kalshi totals/spreads)
MODELS = {
    "hist":     "Solo historial",
    "out_raw":  "Resultado (sin α)",
    "out":      "Resultado (con α)",
    "full_raw": "Completo (sin α)",
    "final":    "Completo: proyección + IPF",
    "ipf_op":   "IPF operativo",
    "ratio":    "Reajuste por razones",
    "poisson":  "Poisson independiente",
    "dc":       "Dixon-Coles",
    "elo":      "Elo (sin mercado)",
}


def _kickoff(m: dict) -> datetime:
    return datetime.strptime(m["time_utc"], "%Y-%m-%dT%H:%M:%SZ")


def _outcome(a: int, b: int) -> str:
    return "home_win" if a > b else "away_win" if a < b else "draw"


def load():
    schedule_json = json.loads((ROOT / "data/schedule.json").read_text())
    results = json.loads((ROOT / "site/data/results.json").read_text())["matches"]
    cache = _load_cache()
    matches = sorted(schedule_json["matches"], key=_kickoff)
    entries = {}
    for m in matches:
        e = cache[f"KXWCGAME-{_date_code(m['time_utc'])}"
                  f"{TEAM_CODES[m['home']]}{TEAM_CODES[m['away']]}"]
        entries[m["id"]] = _clean_entry(e)
    return schedule_json, matches, results, entries


def _normal(P: np.ndarray) -> np.ndarray:
    return P / P.sum()


def _prior(dist: dict, phase: str, p: tuple) -> np.ndarray:
    """Reference distribution with outcome probabilities p (no totals)."""
    return mg.to_array(scoreline_probs(*p, phase, dist=dist))


def match_models(m: dict, e: dict, dist: dict, kern: dict,
                 alpha: float) -> dict[str, np.ndarray]:
    """Final scoreline distribution of every model for one match."""
    raw = (e["home_win"], e["draw"], e["away_win"])
    corr = _correct_bias(*raw, alpha)
    out = {"hist": _prior(dist, m["phase"], (1/3, 1/3, 1/3))}
    kalshi = e.get("source") == "kalshi"
    ko = m["phase"] == "knockout" and "reg_home_win" in e

    if not ko:
        phase = m["phase"]
        out["out_raw"] = _prior(dist, phase, raw)
        out["out"] = _prior(dist, phase, corr)
        if not kalshi:
            for k in ("full_raw", "final", "ipf_op", "ratio", "poisson", "dc"):
                out[k] = out["out"] if k != "full_raw" else out["out_raw"]
            return out
        tg, sa, sb = e["total_goals"], e["spread_home"], e["spread_away"]
        tabs_raw = mg.market_tables(*raw, tg, sa, sb)
        tabs = mg.market_tables(*corr, tg, sa, sb)
        out["full_raw"] = mg.fit_compatible(out["out_raw"], tabs_raw)
        out["final"] = mg.fit_compatible(out["out"], tabs)
        out["ipf_op"] = _normal(mg.to_array(scoreline_probs_ipf(
            *corr, phase, tg, sa, sb, dist=dist)))
        out["ratio"] = _normal(mg.to_array(scoreline_probs(
            *corr, phase, tg, sa, sb, dist=dist)))
        out["poisson"] = mg.fit_poisson(tabs)
        out["dc"] = mg.fit_poisson(tabs, dixon_coles=True)
        return out

    # Knockout, two stages: 90-min fit, then extra time and the quiniela
    # outcome probabilities.
    reg_raw = (e["reg_home_win"], e["reg_draw"], e["reg_away_win"])
    reg = _correct_bias(*reg_raw, alpha)
    tg, sa, sb = e["total_goals"], e["spread_home"], e["spread_away"]
    # Team-goal markets only enter the operational IPF: added to the
    # final method, they over-constrain the sparse 90-min knockout
    # reference (48 matches) and worsened the knockout log score from
    # -2.83 to -2.99.
    tt = {"home": e.get("team_totals_home"), "away": e.get("team_totals_away")}

    def finish(P_reg: np.ndarray, q: tuple) -> np.ndarray:
        return _normal(mg.to_array(extra_time(mg.to_dict(P_reg), q, kern)))

    pr_raw = _prior(dist, "knockout_reg", reg_raw)
    pr = _prior(dist, "knockout_reg", reg)
    tabs_raw = mg.market_tables(*reg_raw, tg, sa, sb)
    tabs = mg.market_tables(*reg, tg, sa, sb)
    out["out_raw"] = finish(pr_raw, raw)
    out["out"] = finish(pr, corr)
    out["full_raw"] = finish(mg.fit_compatible(pr_raw, tabs_raw), raw)
    out["final"] = finish(mg.fit_compatible(pr, tabs), corr)
    out["ipf_op"] = finish(mg.to_array(scoreline_probs_ipf(
        *reg, "knockout_reg", tg, sa, sb, dist=dist, team_totals=tt)), corr)
    out["ratio"] = finish(mg.to_array(scoreline_probs(
        *reg, "knockout_reg", tg, sa, sb, dist=dist)), corr)
    out["poisson"] = finish(mg.fit_poisson(tabs), corr)
    out["dc"] = finish(mg.fit_poisson(tabs, dixon_coles=True), corr)
    return out


_ROWS = None


def run():
    """One row per match with each model's log score, pick and points."""
    global _ROWS
    if _ROWS is None:
        _ROWS = _run()
    return _ROWS


def _run():
    schedule_json, matches, results, entries = load()
    gamma = estimate_gamma()
    kern = build_et_kernel(gamma)
    elo_hist = [x for x in elo.load_matches() if x["year"] < 2026]
    elo_26 = elo.schedule_index(matches)
    rows, cache = [], {}
    for m in matches:
        ended = tuple(j["id"] for j in matches
                      if _kickoff(j) + ENDED_AFTER <= _kickoff(m))
        if ended not in cache:
            sub = {"matches": {str(i): results[str(i)] for i in ended}}
            canon = load_canonical(sub, schedule_json)
            delta = estimate_delta(canon, gamma=gamma)
            w26, d26 = count_2026(canon)
            dist = build_distributions(gamma=gamma, extra_wins=w26,
                                       extra_draws=d26, delta=delta)
            data = []
            for i in ended:
                ei = entries[i]
                if ei.get("source") == "kalshi":
                    r = results[str(i)]
                    data.append((ei["home_win"], ei["draw"], ei["away_win"],
                                 _outcome(r["home_score"], r["away_score"])))
            cache[ended] = (delta, alpha_map(data), dist,
                            elo.fit(elo_hist + [elo_26[i] for i in ended]))
        delta, alpha, dist, elo_par = cache[ended]
        e = entries[m["id"]]
        a, b = results[str(m["id"])]["home_score"], results[str(m["id"])]["away_score"]
        cell = (min(a, 5), min(b, 5))
        models = match_models(m, e, dist, kern, alpha)
        p_elo = elo.match_probs(elo_26[m["id"]], elo_par)
        models["elo"] = _prior(dist, m["phase"], p_elo)
        row = {"id": m["id"], "phase": m["phase"], "round": m["round"],
               "home": m["home"], "away": m["away"], "result": (a, b),
               "kalshi": e.get("source") == "kalshi",
               "n_ended": len(ended), "delta": delta, "alpha": alpha,
               "elo_par": elo_par,
               "log": {}, "pick": {}, "pts": {}}
        for k, P in models.items():
            d = mg.to_dict(P)
            bp = best_prediction(d, "group")
            row["log"][k] = math.log(max(P[cell], 1e-300))
            row["pick"][k] = (bp["home"], bp["away"])
            row["pts"][k] = points(bp["home"], bp["away"], a, b)
        row["P_final"] = models["final"].tolist()
        md = modal_prediction(mg.to_dict(models["final"]))
        row["pick"]["modal"] = (md["home"], md["away"])
        row["pts"]["modal"] = points(md["home"], md["away"], a, b)
        fav_home = e["home_win"] >= e["away_win"]
        row["pts"]["fav10"] = points(*((1, 0) if fav_home else (0, 1)), a, b)
        elo_home = p_elo[0] >= p_elo[2]
        row["pts"]["elo_fav10"] = points(*((1, 0) if elo_home else (0, 1)), a, b)
        # outcome probabilities (quiniela outcome) for the 3-way scores
        row["p_out"] = {"kalshi": _correct_bias(
                            e["home_win"], e["draw"], e["away_win"], alpha),
                        "elo": p_elo}
        row["outcome"] = ("home_win", "draw", "away_win").index(_outcome(a, b))
        rows.append(row)
    return rows


def match_detail(mid: int) -> dict:
    """
    One match as the pseudo-prospective evaluation saw it: its entry,
    α and δ from the matches ended before kickoff, and every model's
    distribution (for the article's worked example).
    """
    schedule_json, matches, results, entries = load()
    m = next(x for x in matches if x["id"] == mid)
    gamma = estimate_gamma()
    ended = [j["id"] for j in matches
             if _kickoff(j) + ENDED_AFTER <= _kickoff(m)]
    canon = load_canonical({"matches": {str(i): results[str(i)]
                                        for i in ended}}, schedule_json)
    delta = estimate_delta(canon, gamma=gamma)
    w26, d26 = count_2026(canon)
    dist = build_distributions(gamma=gamma, extra_wins=w26,
                               extra_draws=d26, delta=delta)
    data = [(entries[i]["home_win"], entries[i]["draw"], entries[i]["away_win"],
             _outcome(results[str(i)]["home_score"], results[str(i)]["away_score"]))
            for i in ended if entries[i].get("source") == "kalshi"]
    alpha = alpha_map(data)
    e = entries[mid]
    return {"match": m, "entry": e, "alpha": alpha, "delta": delta,
            "gamma": gamma,
            "p": _correct_bias(e["home_win"], e["draw"], e["away_win"], alpha),
            "result": (results[str(mid)]["home_score"],
                       results[str(mid)]["away_score"]),
            "models": match_models(m, e, dist, build_et_kernel(gamma), alpha)}


def compatibility():
    """
    How far the market tables are from compatible, read literally (legacy)
    and with unquoted spread bins censored; tail mass of the operational
    IPF.  Shifts are the largest change, over all bins, that the
    compatibility projection makes to a market table.
    """
    schedule_json, matches, results, entries = load()
    gamma = estimate_gamma()
    dist = build_distributions(gamma=gamma)
    out = {"shift_legacy": [], "shift": [], "censored_sides": 0, "sides": 0,
           "sum_ipf_op": [], "tail52_ipf_op": [], "tail52_final": []}
    for m in matches:
        e = entries[m["id"]]
        if e.get("source") != "kalshi":
            continue
        ko = m["phase"] == "knockout" and "reg_home_win" in e
        p = _correct_bias(*((e["reg_home_win"], e["reg_draw"], e["reg_away_win"])
                            if ko else (e["home_win"], e["draw"], e["away_win"])),
                          1.10)
        phase = "knockout_reg" if ko else "group"
        args = (*p, e["total_goals"], e["spread_home"], e["spread_away"])
        for s in (e["spread_home"], e["spread_away"]):
            if s:
                out["sides"] += 1
                out["censored_sides"] += int(mg._last_quoted(s) < 4)
        P0 = _prior(dist, phase, p)
        for key, censor in (("shift_legacy", False), ("shift", True)):
            tabs = mg.market_tables(*args, censor=censor)
            q = mg.project(P0, tabs)
            out[key].append(max(abs(x[1] - y[1]) for ta, tq in zip(tabs, q)
                                for x, y in zip(ta, tq)))
        op = mg.to_array(scoreline_probs_ipf(*args[:3], phase, *args[3:],
                                             dist=dist))
        fin = mg.fit_compatible(P0, mg.market_tables(*args))
        out["sum_ipf_op"].append(op.sum())
        out["tail52_ipf_op"].append((op[5, 2] + op[2, 5]) / op.sum())
        out["tail52_final"].append(fin[5, 2] + fin[2, 5])
    summ = lambda v: {"median": float(st.median(v)), "max": float(max(v))}
    return {"n": len(out["shift"]),
            "censored_sides": out["censored_sides"], "sides": out["sides"],
            "shift_legacy": summ(out["shift_legacy"]),
            "incompatible_legacy": int(sum(x > 1e-3 for x in out["shift_legacy"])),
            "shift": summ(out["shift"]),
            "incompatible": int(sum(x > 1e-3 for x in out["shift"])),
            "sum_ipf_op": {"min": float(min(out["sum_ipf_op"])),
                           "max": float(max(out["sum_ipf_op"]))},
            "tail52_ipf_op": summ(out["tail52_ipf_op"]),
            "tail52_final": summ(out["tail52_final"]),
            "tail52_ipf_op_over5pct": int(sum(x > 0.05 for x in out["tail52_ipf_op"]))}


def summarize(rows):
    ks = [r for r in rows if r["kalshi"]]
    out = {"n": len(rows), "n_kalshi": len(ks),
           "alpha_final": rows[-1]["alpha"], "delta_final": rows[-1]["delta"],
           "alpha_range": (min(r["alpha"] for r in rows),
                           max(r["alpha"] for r in rows)),
           "points": {}, "logscore": {}, "diffs": {}}
    keys = list(MODELS) + ["modal", "fav10", "elo_fav10"]
    for k in keys:
        out["points"][k] = {
            "all104": sum(r["pts"][k] for r in rows),
            "kalshi": sum(r["pts"][k] for r in ks),
            "group": sum(r["pts"][k] for r in rows if r["phase"] == "group"),
            "knockout": sum(r["pts"][k] for r in rows if r["phase"] == "knockout"),
        }
    for k in MODELS:
        out["logscore"][k] = {
            "kalshi": st.mean(r["log"][k] for r in ks),
            "group": st.mean(r["log"][k] for r in ks if r["phase"] == "group"),
            "knockout": st.mean(r["log"][k] for r in ks if r["phase"] == "knockout"),
        }
        d = [r["log"]["final"] - r["log"][k] for r in ks]
        tot, lo, hi = bayes_boot(d)
        out["logscore"][k]["diff_vs_final"] = {
            "mean": tot / len(d), "lo": lo / len(d), "hi": hi / len(d)}
    for x, y, rs in [("out_raw", "hist", rows), ("out", "out_raw", rows),
                     ("final", "out", rows), ("final", "full_raw", rows),
                     ("final", "modal", rows), ("final", "fav10", rows),
                     ("final", "ipf_op", ks), ("final", "ratio", ks),
                     ("final", "poisson", ks), ("final", "dc", ks),
                     ("elo", "hist", rows), ("out", "elo", rows),
                     ("final", "elo", rows), ("elo", "elo_fav10", rows),
                     ("fav10", "elo_fav10", rows)]:
        d = [r["pts"][x] - r["pts"][y] for r in rs]
        tot, lo, hi = bayes_boot(d)
        out["diffs"][f"{x}-{y}"] = {"n": len(d), "total": tot, "lo": lo,
                                    "hi": hi,
                                    "better": sum(v > 0 for v in d),
                                    "worse": sum(v < 0 for v in d)}
    out["differ_final_modal"] = sum(r["pick"]["final"] != r["pick"]["modal"]
                                    for r in rows)
    out["draw_picks_final"] = sum(p[0] == p[1] for p in
                                  (r["pick"]["final"] for r in rows))
    out["exact_final"] = sum(r["pts"]["final"] == 5 for r in rows)
    out["outcome_scores"] = outcome_scores(rows)
    out["elo_par_final"] = rows[-1]["elo_par"]
    return out


def outcome_scores(rows):
    """Log and Brier scores of the 3-way outcome: Kalshi (with α) vs Elo."""
    ks = [r for r in rows if r["kalshi"]]
    out = {}
    for name, f in (("log", lambda p, o: math.log(p[o])),
                    ("brier", lambda p, o: sum((p[i] - (i == o)) ** 2
                                               for i in range(3)))):
        v = {src: [f(r["p_out"][src], r["outcome"]) for r in ks]
             for src in ("kalshi", "elo")}
        d = [x - y for x, y in zip(v["kalshi"], v["elo"])]
        tot, lo, hi = bayes_boot(d)
        out[name] = {"kalshi": st.mean(v["kalshi"]), "elo": st.mean(v["elo"]),
                     "diff": tot / len(d), "lo": lo / len(d),
                     "hi": hi / len(d)}
    out["same_favourite"] = sum(
        (r["p_out"]["kalshi"][0] >= r["p_out"]["kalshi"][2])
        == (r["p_out"]["elo"][0] >= r["p_out"]["elo"][2]) for r in ks)
    out["n"] = len(ks)
    return out


def _outcome_probs(P: np.ndarray) -> tuple[float, float, float]:
    return float(np.tril(P, -1).sum()), float(np.trace(P)), \
        float(np.triu(P, 1).sum())


def decomposition_g(P: np.ndarray) -> np.ndarray:
    """
    Outcome-free part of the expected points: V(a,b) = 2 P(outcome of
    (a,b)) + g(a,b), with g(a,b) = P(H=a) + P(K=b) + P(a,b).
    """
    return P.sum(1)[:, None] + P.sum(0)[None, :] + P


def poisson_pick(lam: float, mu: float, n: int = 30) -> tuple[int, int]:
    """Optimal prediction (grid 0..5) for independent Poisson(λ), Poisson(μ)."""
    g = np.arange(n)
    pl = np.exp(-lam + g * math.log(max(lam, 1e-12)) - np.cumsum(np.log(np.maximum(g, 1))))
    pm = np.exp(-mu + g * math.log(max(mu, 1e-12)) - np.cumsum(np.log(np.maximum(g, 1))))
    P = np.outer(pl, pm)
    pa, pe, pb = _outcome_probs(P)
    V = {(a, b): 2 * (pa if a > b else pb if a < b else pe)
         + pl[a] + pm[b] + pl[a] * pm[b] for a in range(6) for b in range(6)}
    return max(V, key=V.get)


def decomposition(rows):
    """
    How the final picks follow from V = 2 P(outcome) + g (Proposition 1)
    and from the Poisson map at each match's expected goals (Prop. 2).
    """
    out = {"n": len(rows), "gmax_outcome_most_likely": 0,
           "pick_is_gmax": 0, "switch_from_gmax": Counter(),
           "poisson_agree": 0, "mode_cells": Counter(), "draw_picks": []}
    for r in rows:
        P = np.array(r["P_final"])
        g = decomposition_g(P)
        a, b = np.unravel_index(g.argmax(), g.shape)
        pa, pe, pb = _outcome_probs(P)
        p_r = pa if a > b else pb if a < b else pe
        pick = tuple(r["pick"]["final"])
        if p_r >= max(pa, pe, pb) - 1e-12:
            out["gmax_outcome_most_likely"] += 1
        else:
            out["switch_from_gmax"][f"{a}-{b}->{pick[0]}-{pick[1]}"] += 1
        out["pick_is_gmax"] += pick == (a, b)
        goals = np.arange(6)
        lh, lk = float(P.sum(1) @ goals), float(P.sum(0) @ goals)
        out["poisson_agree"] += pick == poisson_pick(lh, lk)
        out["mode_cells"][f"{int(max(lh, lk))}-{int(min(lh, lk))}"] += 1
        if pick[0] == pick[1]:
            out["draw_picks"].append({"match": f"{r['home']}-{r['away']}",
                                      "pick": pick, "p": (pa, pe, pb)})
    return out


def rule_sensitivity(rows,
                     rules=((5, 3, 2, 1), (10, 3, 2, 1), (5, 4, 2, 1),
                            (3, 2, 2, 1))):
    """Optimal vs modal of the final method under other scoring rules."""
    ks = [r for r in rows if r["kalshi"]]
    cells = [(a, b) for a in range(6) for b in range(6)]
    out = []
    for rule in rules:
        f = rule_points(rule)
        differ, gain_exp, gain_obs = 0, 0.0, 0
        for r in ks:
            P = r["P_final"]
            ev = {c: sum(f(*c, h, k) * P[h][k] for h in range(6)
                         for k in range(6)) for c in cells}
            opt = max(cells, key=ev.get)
            mod = max(cells, key=lambda c: P[c[0]][c[1]])
            differ += opt != mod
            gain_exp += ev[opt] - ev[mod]
            gain_obs += f(*opt, *r["result"]) - f(*mod, *r["result"])
        out.append({"rule": rule, "n": len(ks), "differ": differ,
                    "gain_exp_per_match": gain_exp / len(ks),
                    "gain_obs": gain_obs})
    return out


def main():
    rows = run()
    stats = {"summary": summarize(rows), "rules": rule_sensitivity(rows),
             "decomposition": decomposition(rows),
             "compatibility": compatibility()}
    print(json.dumps(stats, indent=1, default=str))
    if "--json" in sys.argv:
        Path(sys.argv[sys.argv.index("--json") + 1]).write_text(
            json.dumps({"stats": stats, "rows": rows}, indent=1, default=str))


if __name__ == "__main__":
    main()
