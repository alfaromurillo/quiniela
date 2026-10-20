"""
Informed baseline that does not use the market: World Football Elo
ratings (eloratings.net) turned into outcome probabilities.

Data: data/elo/{2014,2018,2022,2026}_World_Cup_results.tsv, downloaded
from eloratings.net on 2026-10-20.  Columns: year, month, day, team 1,
team 2, goals 1, goals 2 (after extra time; a shoot-out is a draw),
tournament, venue country (blank: team 1 at home), rating change of
team 1, ratings of both teams after the match, rank changes, ranks.
The rating before the match is the rating after it minus the change
(Elo exchanges are zero-sum).

Outcome model (ordered logit, symmetric in the two teams):
    x = (R_A − R_B + 100·home_A − 100·home_B) / 400
    P(A wins) = σ(β x − θ_phase),  P(B wins) = σ(−β x − θ_phase),
    P(draw) = 1 − P(A wins) − P(B wins),
with one threshold for the group stage and one for knockout matches
(where a draw means a penalty shoot-out).  The parameters are fitted
by maximum likelihood to the World Cups 2014–2022 and to the WC 2026
matches already ended, so the baseline learns during the tournament
as α and δ do.
"""
import csv
import math
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).parent.parent
ELO_DIR = ROOT / "data" / "elo"
YEARS = (2014, 2018, 2022, 2026)
N_GROUP = {2014: 48, 2018: 48, 2022: 48, 2026: 72}
HOME = 100.0                              # Elo home advantage


def _codes() -> dict[str, str]:
    """Team name (any alias in en.teams.tsv) → Elo code."""
    out = {}
    for row in csv.reader(open(ELO_DIR / "en.teams.tsv"), delimiter="\t"):
        for name in row[1:]:
            out.setdefault(name, row[0])
    out["Czech Republic"] = "CZ"
    return out


def _int(s: str) -> int:
    return int(s.replace("−", "-"))


def load_matches() -> list[dict]:
    """Every World Cup match with pre-match ratings, in file order."""
    out = []
    for year in YEARS:
        rows = [r for r in csv.reader(
            open(ELO_DIR / f"{year}_World_Cup_results.tsv"), delimiter="\t")
            if len(r) > 11 and r[7] == "WC"]
        for i, r in enumerate(rows):
            change = _int(r[9])
            out.append({
                "year": year, "date": f"{r[0]}-{r[1]}-{r[2]}",
                "phase": "group" if i < N_GROUP[year] else "knockout",
                "a": r[3], "b": r[4], "ga": int(r[5]), "gb": int(r[6]),
                "ra": _int(r[10]) - change, "rb": _int(r[11]) + change,
                "home_a": r[8] == ""})
    return out


def x_diff(m: dict) -> float:
    return (m["ra"] - m["rb"] + HOME * m["home_a"]) / 400


def probs(x: float, beta: float, theta: float) -> tuple[float, float, float]:
    pa = 1 / (1 + math.exp(-(beta * x - theta)))
    pb = 1 / (1 + math.exp(-(-beta * x - theta)))
    return pa, 1 - pa - pb, pb


def fit(matches: list[dict]) -> dict:
    """MLE of (β, θ_group, θ_knockout) on matches with known results."""
    data = [(x_diff(m), m["phase"],
             0 if m["ga"] > m["gb"] else 1 if m["ga"] == m["gb"] else 2)
            for m in matches]

    def nll(v):
        beta, tg, tk = v
        out = 0.0
        for x, ph, o in data:
            p = probs(x, beta, tg if ph == "group" else tk)
            out -= math.log(max(p[o], 1e-12))
        return out

    r = minimize(nll, [1.5, 0.6, 0.9], method="Nelder-Mead",
                 options={"xatol": 1e-6, "fatol": 1e-9, "maxiter": 4000})
    beta, tg, tk = map(float, r.x)
    return {"beta": beta, "theta_group": tg, "theta_knockout": tk,
            "n": len(data)}


def schedule_index(schedule: list[dict]) -> dict[int, dict]:
    """Schedule match id → its Elo row, oriented as home (A) vs away (B)."""
    codes = _codes()
    elo26 = [m for m in load_matches() if m["year"] == 2026]
    out = {}
    for s in schedule:
        ca, cb = codes[s["home"]], codes[s["away"]]
        hit = [m for m in elo26 if {m["a"], m["b"]} == {ca, cb}
               and m["phase"] == s["phase"]]
        assert len(hit) == 1, (s["home"], s["away"], s["phase"])
        m = hit[0]
        if m["a"] != ca:                    # swap to the schedule's order
            m = {**m, "a": m["b"], "b": m["a"], "ga": m["gb"], "gb": m["ga"],
                 "ra": m["rb"], "rb": m["ra"], "home_a": False,
                 "home_b": m["home_a"]}
        out[s["id"]] = m
    return out


def match_probs(m: dict, par: dict) -> tuple[float, float, float]:
    x = (m["ra"] - m["rb"]
         + HOME * m.get("home_a", False) - HOME * m.get("home_b", False)) / 400
    th = par["theta_group" if m["phase"] == "group" else "theta_knockout"]
    return probs(x, par["beta"], th)
