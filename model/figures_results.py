"""
Results figures for the article (fig5–fig10).  Fig 5 shows the frozen
predictions next to the final method; the others use the final method's
pseudo-prospective picks (prospective.py) or the data alone.
Output: articulo/figures/

Run from repo root:
    python model/figures_results.py
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.stats import beta

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from model.figures import OUT_DIR, _num   # style, locale, decimal comma
from model.article_stats import (match_rows, final_rows, final_dist,
                                  by_favourite, surprises, surprise_rows,
                                  rounds_compare, ROUND_ORDER)
from model.historical import scoreline_probs
from model.optimizer import best_prediction
from model import marginals as mg

INK   = "#222222"
MUTED = "#9a9a9a"
BLUE  = "#1f4e79"
ORANGE = "#c8691c"

ES = {"Germany": "Alemania", "Curaçao": "Curazao", "Spain": "España",
      "Cape Verde": "Cabo Verde", "Ecuador": "Ecuador",
      "New Zealand": "Nueva Zelanda", "Belgium": "Bélgica",
      "Qatar": "Catar", "Switzerland": "Suiza", "Brazil": "Brasil",
      "Haiti": "Haití", "Morocco": "Marruecos", "Netherlands": "Países Bajos",
      "Tunisia": "Túnez", "Iraq": "Irak", "Norway": "Noruega",
      "France": "Francia", "Saudi Arabia": "Arabia Saudita"}


def _clean(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=INK)


def _save(fig, name):
    out = OUT_DIR / f"{name}.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Fig 5: cumulative points ───────────────────────────────────
def fig5_cumulative():
    """Pseudo-prospective strategies (prospective.py) plus the frozen record."""
    from model.prospective import run
    rows = run()
    frozen = {r["id"]: r["pts"][6] for r in match_rows()}
    series = [("final", "Óptimo (método final)", BLUE, 2.2, "-"),
              ("modal", "Marcador modal", "#5b8db8", 1.3, "-"),
              ("frozen", "Pronósticos congelados", INK, 1.0, "--"),
              ("fav10", "Favorito gana 1-0", ORANGE, 1.3, "-"),
              ("hist", "Solo historial (casi siempre 1-1)", MUTED, 1.3, "-")]
    dys = {"final": 9, "modal": 2, "frozen": -5, "fav10": -12, "hist": 0}
    x = np.arange(1, len(rows) + 1)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for key, label, color, lw, ls in series:
        pts = [frozen[r["id"]] if key == "frozen" else r["pts"][key]
               for r in rows]
        y = np.cumsum(pts)
        ax.plot(x, y, color=color, lw=lw, ls=ls)
        ax.text(x[-1] + 1.5, y[-1] + dys[key], f"{label}: {y[-1]}",
                color=color, va="center", fontsize=9)
    n_group = sum(r["phase"] == "group" for r in rows)
    ax.axvline(n_group + 0.5, color="#cccccc", lw=0.8, ls="--")
    ax.text(n_group / 2, 5, "Fase de grupos", ha="center", color=MUTED, fontsize=9)
    ax.text(n_group + (len(rows) - n_group) / 2, 5, "Eliminatoria",
            ha="center", color=MUTED, fontsize=9)
    ax.set_xlim(0, len(rows) + 1)
    ax.set_ylim(0, None)
    ax.set_xlabel("Partidos, en orden cronológico")
    ax.set_ylabel("Puntos acumulados")
    _clean(ax)
    _save(fig, "fig5_puntos_acumulados")


# ── Fig 6: predicted vs observed scorelines ────────────────────
def fig6_predicted_observed():
    rows = final_rows()
    canon = lambda p: (max(p), min(p))
    pred = Counter(canon(r["pred"]) for r in rows)
    obs  = Counter(canon(r["result"]) for r in rows)
    scores = sorted(set(pred) | set(obs),
                    key=lambda s: (obs[s], pred[s]), reverse=False)
    y = np.arange(len(scores))
    fig, ax = plt.subplots(figsize=(6.0, 5.2))
    for yi, s in zip(y, scores):
        ax.plot([obs[s], pred[s]], [yi, yi], color="#dddddd", lw=1.2, zorder=1)
    ax.scatter([obs[s] for s in scores], y, s=34, facecolor="white",
               edgecolor=INK, lw=1.1, zorder=3)
    ax.scatter([pred[s] for s in scores], y, s=34, color=BLUE, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{a}-{b}" for a, b in scores])
    top = scores.index((1, 0))
    ax.text(pred[(1, 0)], top + 0.55, "pronosticado", color=BLUE,
            ha="center", fontsize=9)
    ax.text(obs[(1, 0)], top + 0.55, "ocurrido", color=INK,
            ha="center", fontsize=9)
    ax.set_xlabel("Número de partidos (de 104)")
    ax.set_ylabel("Marcador (goles del ganador, real o\npronosticado, primero)")
    ax.set_xlim(-1, max(pred.values()) + 3)
    ax.set_ylim(-0.7, len(scores) - 0.2)
    _clean(ax)
    _save(fig, "fig6_pronosticado_observado")


# ── Fig 7: decision map ────────────────────────────────────────
def _poisson_totals(mu):
    from math import exp, factorial
    p = {t: exp(-mu) * mu**t / factorial(t) for t in range(6)}
    p[6] = max(0.0, 1 - sum(p.values()))
    return p


def fig7_decision_map():
    dist = final_dist()
    pf = np.arange(0.25, 0.96, 0.005)
    pe = np.arange(0.03, 0.52, 0.005)
    # Expected totals near the 15th, 50th and 85th percentiles of the
    # 98 matches' Kalshi totals; each match is drawn in the panel whose
    # range contains its own expected total.
    panels = [(1.9, "Pocos goles (1,9 esperados)", (0, 2.3)),
              (2.6, "Goles intermedios (2,6)", (2.3, 2.9)),
              (3.4, "Muchos goles (3,4)", (2.9, 99))]
    grids, labels = [], {}
    for mu, _, _ in panels:
        tg = _poisson_totals(mu)
        g = np.full((len(pe), len(pf)), -1)
        for j, x in enumerate(pf):
            for i, e in enumerate(pe):
                u = 1 - x - e
                if u < 0.02 or u > x - 0.01:   # favourite must lead
                    continue
                P0 = mg.to_array(scoreline_probs(x, e, u, "group", dist=dist))
                sp = mg.to_dict(mg.fit_compatible(
                    P0, mg.market_tables(x, e, u, tg, None, None)))
                b = best_prediction(sp, "group")
                lab = f"{b['home']}-{b['away']}"
                g[i, j] = labels.setdefault(lab, len(labels))
        grids.append(g)
    names = {v: k for k, v in labels.items()}
    palette = ["#c6dbef", "#9ecae1", "#fdd0a2", "#fdae6b", "#c7e9c0",
               "#a1d99b", "#dadaeb", "#fcbba1", "#d9d9d9"]
    cmap = ListedColormap(palette[:len(labels)])
    cmap.set_under("white")

    # matches: corrected probabilities, placed by their expected total
    rows = [r for r in final_rows() if r["source"] == "kalshi"]
    from model.kalshi import _correct_bias
    pts = []
    for r in rows:
        p = r["probs"]
        h, d, a = _correct_bias(p["home_win"], p["draw"], p["away_win"], 1.12)
        tg = p.get("total_goals") or {}
        et = sum(int(k) * v for k, v in tg.items()) if tg else 2.6
        pts.append((max(h, a), d, et, r["pred"][0] == r["pred"][1]))

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), sharey=True)
    for ax, g, (mu, title, cut) in zip(axes, grids, panels):
        ax.imshow(np.ma.masked_less(g, 0), origin="lower", aspect="auto",
                  cmap=cmap, vmin=0, vmax=len(labels) - 1,
                  extent=[pf[0], pf[-1], pe[0], pe[-1]], interpolation="nearest")
        for k, name in names.items():
            ii, jj = np.where(g == k)
            if len(ii) == 0:
                continue
            cx, cy = pf[int(np.median(jj))], pe[int(np.median(ii))]
            if len(ii) < 25:          # sliver: label it from outside
                ax.annotate(name, (cx, cy), xytext=(-28, -22),
                            textcoords="offset points", fontsize=10,
                            color=INK, fontweight="bold",
                            arrowprops=dict(arrowstyle="-", color=INK, lw=0.6))
                continue
            ax.text(cx, cy, name, ha="center", va="center", fontsize=10,
                    color=INK, fontweight="bold")
        sel = [p for p in pts if cut[0] <= p[2] < cut[1]]
        ax.scatter([p[0] for p in sel if not p[3]], [p[1] for p in sel if not p[3]],
                   s=10, color=INK, alpha=0.7, lw=0)
        ax.scatter([p[0] for p in sel if p[3]], [p[1] for p in sel if p[3]],
                   s=30, facecolor="none", edgecolor=INK, lw=1.2)
        ax.set_xlabel("Probabilidad de que gane el favorito")
        ax.text(0.98, 0.97, title, transform=ax.transAxes, ha="right",
                va="top", fontsize=9, color=INK)
        _clean(ax)
    axes[0].set_ylabel("Probabilidad de empate")
    fig.tight_layout()
    _save(fig, "fig7_mapa_decision")


# ── Fig 8: calibration and points by favourite strength ────────
def fig11_poisson_map():
    """Optimal prediction under independent Poisson goals (Proposition 2)."""
    from model.prospective import run, poisson_pick
    lam = np.arange(0.01, 3.6, 0.01)
    mu = np.arange(0.01, 2.4, 0.01)
    labels, g = {}, np.full((len(mu), len(lam)), -1)
    for j, x in enumerate(lam):
        for i, y in enumerate(mu):
            if y < x:
                a, b = poisson_pick(x, y)
                g[i, j] = labels.setdefault(f"{a}-{b}", len(labels))
    names = {v: k for k, v in labels.items()}
    palette = {"0-0": "#d9d9d9", "1-0": "#c6dbef", "2-0": "#fdd0a2",
               "2-1": "#c7e9c0", "3-0": "#fdae6b", "3-1": "#a1d99b",
               "3-2": "#dadaeb"}
    cmap = ListedColormap([palette[names[k]] for k in range(len(names))])
    cmap.set_under("white")

    # matches: expected goals of each team under the final method
    pts = []
    goals = np.arange(6)
    for r in run():
        P = np.array(r["P_final"])
        lh, lk = float(P.sum(1) @ goals), float(P.sum(0) @ goals)
        pick = tuple(r["pick"]["final"])
        same = pick == poisson_pick(lh, lk)
        pts.append((max(lh, lk), min(lh, lk), same))

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    ax.imshow(np.ma.masked_less(g, 0), origin="lower", aspect="auto",
              cmap=cmap, vmin=0, vmax=len(labels) - 1,
              extent=[lam[0], lam[-1], mu[0], mu[-1]],
              interpolation="nearest")
    for v in (1, 2, 3):
        ax.axvline(v, color="white", lw=0.8)
        if v < 2.4:
            ax.axhline(v, color="white", lw=0.8)
    # labels below the cloud of matches, which sits near μ ≈ 0.9
    where = {"0-0": (0.5, 0.2), "1-0": (1.45, 0.3), "2-0": (2.5, 0.3),
             "3-0": (3.3, 0.3), "2-1": (2.5, 1.6), "3-1": (3.3, 1.5),
             "3-2": (3.3, 2.2)}
    for name in names.values():
        ax.text(*where[name], name, ha="center", va="center", fontsize=11,
                color=INK, fontweight="bold")
    ax.scatter([p[0] for p in pts if p[2]], [p[1] for p in pts if p[2]],
               s=12, color=INK, alpha=0.75, lw=0)
    ax.scatter([p[0] for p in pts if not p[2]], [p[1] for p in pts if not p[2]],
               s=30, facecolor="none", edgecolor=INK, lw=1.1)
    ax.set_xlim(0, 3.6)
    ax.set_ylim(0, 2.4)
    ax.set_xlabel("Goles esperados del favorito, $\\lambda$")
    ax.set_ylabel("Goles esperados del rival, $\\mu$")
    _clean(ax)
    fig.tight_layout()
    _save(fig, "fig11_mapa_poisson")


def fig8_calibration():
    rows = [r for r in final_rows() if r["source"] == "kalshi"]
    pairs = []
    for r in rows:
        a, b = r["result"]
        o = {"home_win": a > b, "draw": a == b, "away_win": a < b}
        for k in ("home_win", "draw", "away_win"):
            pairs.append((r["probs"][k], o[k]))
    edges = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))
    ax1.plot([0, 1], [0, 1], color="#cccccc", lw=1, zorder=1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = [(p, o) for p, o in pairs if lo <= p < hi]
        if not sel:
            continue
        n, k = len(sel), sum(o for _, o in sel)
        mp = np.mean([p for p, _ in sel])
        l, u = beta.ppf([0.025, 0.975], 1 + k, 1 + n - k)
        ax1.plot([mp, mp], [l, u], color=BLUE, lw=1.2)
        ax1.scatter([mp], [k / n], color=BLUE, s=22, zorder=3)
        ax1.text(mp + 0.012, k / n, f"{n}", fontsize=7, color=MUTED, va="center")
    ax1.set_xlim(0, 1); ax1.set_ylim(0, 1)
    ax1.set_xlabel("Probabilidad según Kalshi")
    ax1.set_ylabel("Frecuencia observada")
    ax1.set_aspect("equal")
    _clean(ax1)

    bf = by_favourite(rows)
    xs = np.arange(len(bf))
    ax2.plot(xs, [b["model"] for b in bf], "o-", color=BLUE, lw=1.6)
    ax2.plot(xs, [b["fav10"] for b in bf], "o--", color=ORANGE, lw=1.2, ms=4)
    ax2.text(xs[-1] + 0.1, bf[-1]["model"], "Óptimo", color=BLUE,
             va="center", fontsize=9)
    ax2.text(xs[-1] + 0.1, bf[-1]["fav10"] + 0.08, "Favorito 1-0",
             color=ORANGE, va="center", fontsize=9)
    ax2.set_xticks(xs)
    ax2.set_xticklabels([f"{_num(b['lo'], 2)}–{_num(b['hi'], 2)}\n(n = {b['n']})"
                         for b in bf], fontsize=8)
    ax2.set_xlim(-0.4, len(bf) - 0.2)
    ax2.set_ylim(0, 3.5)
    ax2.set_xlabel("Probabilidad de que gane el favorito")
    ax2.set_ylabel("Puntos por partido")
    _clean(ax2)
    fig.tight_layout()
    _save(fig, "fig8_calibracion")


# ── Fig 9: how this World Cup differed ─────────────────────────
def _hist_group_totals():
    out = Counter()
    n = 0
    for y in (2014, 2018, 2022):
        for m in json.loads((ROOT / f"data/wc{y}.json").read_text())["matches"]:
            if "Matchday" in m["round"]:
                out[min(sum(m["score"]["ft"]), 7)] += 1
                n += 1
    return out, n


def fig9_world_cup():
    rows = [r for r in match_rows() if r["phase"] == "group"]
    h, nh = _hist_group_totals()
    c = Counter(min(sum(r["result"]), 7) for r in rows)
    ts = np.arange(8)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8),
                             gridspec_kw={"width_ratios": [1, 1.3]})
    (ax1, ax2), (ax3, ax4) = axes
    ax1.plot(ts, [100 * h[t] / nh for t in ts], "o-", color=MUTED, lw=1.4)
    ax1.plot(ts, [100 * c[t] / len(rows) for t in ts], "o-", color=BLUE, lw=1.8)
    ax1.text(1.2, 100 * h[1] / nh + 3.5, "2014–2022", color=MUTED, fontsize=9)
    ax1.text(4.3, 100 * c[4] / len(rows) + 3, "2026", color=BLUE, fontsize=9)
    ax1.set_xticks(ts)
    ax1.set_xticklabels([str(t) for t in ts[:-1]] + ["7+"])
    ax1.set_xlabel("Goles en el partido (fase de grupos)")
    ax1.set_ylabel("Partidos (%)")
    ax1.set_ylim(0, None)
    _clean(ax1)

    ks = [r for r in rows if r["source"] == "kalshi"]
    pf = [r["p_fav"] for r in ks]
    gd = []
    for r in ks:
        a, b = r["result"]
        home_fav = r["probs"]["home_win"] >= r["probs"]["away_win"]
        gd.append(a - b if home_fav else b - a)
    ax2.axhline(0, color="#cccccc", lw=0.8)
    ax2.scatter(pf, gd, s=18, color=BLUE, alpha=0.8, lw=0)
    # label → text position (data coordinates)
    notable = {("Germany", "Curaçao"): (0.62, 6.7),
               ("New Zealand", "Belgium"): (0.70, 4.6),
               ("Spain", "Cape Verde"): (0.95, -1.0),
               ("Ecuador", "Curaçao"): (0.88, -1.6),
               ("Qatar", "Switzerland"): (0.80, -2.2)}
    for r, x, y in zip(ks, pf, gd):
        pos = notable.get((r["home"], r["away"]))
        if pos:
            a, b = r["result"]
            ax2.annotate(f"{ES.get(r['home'], r['home'])} {a}-{b} "
                         f"{ES.get(r['away'], r['away'])}",
                         (x, y), xytext=pos, ha="right", va="center",
                         fontsize=8, color=INK,
                         arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
    ax2.set_ylim(-2.6, 7.2)
    ax2.set_xlabel("Probabilidad de que gane el favorito (Kalshi)")
    ax2.set_ylabel("Diferencia de goles a favor del favorito")
    _clean(ax2)

    # (c, d) goals per match and draws by round, 2026 vs 2014–2022
    rc = rounds_compare()
    labels = {"G1": "Grupos 1", "G2": "Grupos 2", "G3": "Grupos 3",
              "R32": "Dieciseisavos", "R16": "Octavos", "QF": "Cuartos",
              "SF-F": "Semis a final"}
    keys = ROUND_ORDER
    x = np.arange(len(keys))
    for ax, field, scale, ylab in [
            (ax3, "goals", 1, "Goles por partido"),
            (ax4, "draws", 100, "Empates (%)")]:
        for name, color, dx in (("hist", MUTED, -0.12), ("2026", BLUE, 0.12)):
            xs = [i + dx for i, k in enumerate(keys) if k in rc[name]]
            ys = [scale * rc[name][k][field] for k in keys if k in rc[name]]
            ax.scatter(xs, ys, s=30, color=color, zorder=3)
            ax.vlines(xs, 0, ys, color=color, lw=1.2, alpha=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels([labels[k] for k in keys], rotation=30, ha="right",
                           fontsize=9)
        ax.set_ylabel(ylab)
        ax.set_ylim(0, None)
        _clean(ax)
    ax3.text(0.02, 0.97, "2014–2022", color=MUTED, transform=ax3.transAxes,
             va="top", fontsize=9)
    ax3.text(0.02, 0.90, "2026", color=BLUE, transform=ax3.transAxes,
             va="top", fontsize=9)
    for ax, tag in zip(axes.flat, "abcd"):
        ax.text(0.0, 1.03, tag, transform=ax.transAxes, fontsize=12,
                fontweight="bold", va="bottom", ha="left")
    fig.tight_layout()
    _save(fig, "fig9_mundial_distinto")


# ── Fig 10: surprises ──────────────────────────────────────────
ES_SURPRISE = {"England": "Inglaterra", "Ghana": "Ghana",
               "Paraguay": "Paraguay", "Uruguay": "Uruguay",
               "Turkey": "Turquía", "USA": "EE. UU.", "Algeria": "Argelia",
               "Austria": "Austria", "Ecuador": "Ecuador"}

# match → label offset in points (dx, dy) and horizontal alignment
SURPRISE_LABELS = {
    ("Spain", "Cape Verde"):    ((6, -9), "left"),
    ("Ecuador", "Curaçao"):     ((4, 11), "left"),
    ("England", "Ghana"):       ((8, 3), "left"),
    ("Qatar", "Switzerland"):   ((-5, 7), "right"),
    ("Germany", "Paraguay"):    ((6, -9), "left"),
    ("Brazil", "Norway"):       ((-5, 8), "right"),
    ("Ecuador", "Germany"):     ((6, -9), "left"),
    ("Uruguay", "Cape Verde"):  ((-10, 14), "center"),
    ("Turkey", "USA"):          ((4, 12), "left"),
    ("France", "England"):      ((8, -2), "left"),
    ("Germany", "Curaçao"):     ((-6, 8), "right"),
    ("Algeria", "Austria"):     ((8, 4), "left"),
}


def fig10_surprises():
    rows = surprise_rows(final_rows())
    # Fixed axes positions: panel b's long tick labels must not narrow a.
    fig = plt.figure(figsize=(10, 9.2))
    ax1 = fig.add_axes([0.08, 0.43, 0.76, 0.53])
    ax2 = fig.add_axes([0.42, 0.06, 0.42, 0.26])
    ax1.axvline(0.3, color="#bbbbbb", lw=0.8, ls="--")
    for y in (-1, 1):
        ax1.axhline(y, color="#bbbbbb", lw=0.8, ls="--")
    colors = {1: BLUE, -1: ORANGE, 0: "#c8c8c8"}
    for r in rows:
        d = r["pts"][6] - r["pts"]["fav10"]
        ax1.scatter(r["p_out"], r["d_goals"], s=24, lw=0, zorder=3,
                    color=colors[(d > 0) - (d < 0)])
    es = dict(ES, **ES_SURPRISE)
    for r in rows:
        lab = SURPRISE_LABELS.get((r["home"], r["away"]))
        if not lab:
            continue
        (dx, dy), ha = lab
        a, b = r["result"]
        ax1.annotate(f"{es.get(r['home'], r['home'])} {a}-{b} "
                     f"{es.get(r['away'], r['away'])}",
                     (r["p_out"], r["d_goals"]), xytext=(dx, dy),
                     textcoords="offset points", ha=ha, va="center",
                     fontsize=7.5, color=INK,
                     arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.5,
                                     shrinkA=0, shrinkB=2))
    ax1.set_xlim(0, 1)
    ax1.set_ylim(-4.6, 7.4)
    # zone labels: outcome along the top, goals along the right margin
    kw = dict(color="#777777", fontsize=8.5, style="italic")
    ax1.text(0.29, 7.6, "← resultado sorpresivo", ha="right", va="bottom", **kw)
    ax1.text(0.31, 7.6, "resultado esperado →", ha="left", va="bottom", **kw)
    for y, txt in ((4.2, "más goles\nde lo esperado"),
                   (0.0, "goles como\nse esperaba"),
                   (-2.8, "menos goles\nde lo esperado")):
        ax1.text(1.015, y, txt, transform=ax1.get_yaxis_transform(),
                 ha="left", va="center", **kw)
    ax1.set_xlabel("Probabilidad que Kalshi daba al resultado que ocurrió\n"
                   "(a la victoria del ganador real, o al empate)")
    ax1.set_ylabel("Goles del partido menos goles esperados")
    _clean(ax1)

    bins = surprises(final_rows())["bins"]
    ypos = [5.4, 4.4, 3.4, 2.0, 1.0, 0.0]
    ax2.axvline(0, color="#cccccc", lw=0.8)
    for b, y in zip(bins, ypos):
        color = BLUE if b["lo"] > 0 else ORANGE if b["hi"] < 0 else INK
        ax2.plot([b["lo"], b["hi"]], [y, y], color=color, lw=1.4)
        ax2.scatter([b["diff"]], [y], color=color, s=28, zorder=3)
    names = ["resultado sorpresivo (menos de 30 %)",
             "resultado de 30 a 50 %", "resultado esperado (50 % o más)",
             "menos goles de lo esperado (más de 1)",
             "goles como se esperaba (a 1 o menos)",
             "más goles de lo esperado (más de 1)"]
    ax2.set_yticks(ypos)
    ax2.set_yticklabels([f"{n} (n = {b['n']})" for n, b in zip(names, bins)],
                        fontsize=8.5)
    ax2.set_xlim(min(b["lo"] for b in bins) - 0.15,
                 max(b["hi"] for b in bins) + 0.15)
    ax2.set_ylim(-0.6, 5.9)
    ax2.set_xlabel("Puntos por partido: método final menos “favorito 1-0”")
    _clean(ax2)
    ax2.spines["left"].set_visible(False)
    ax2.tick_params(axis="y", length=0)
    for ax, tag in ((ax1, "a"), (ax2, "b")):
        ax.text(-0.06 if tag == "a" else -0.72, 1.04, tag,
                transform=ax.transAxes, fontsize=12, fontweight="bold",
                va="bottom", ha="right")
    _save(fig, "fig10_sorpresas")


if __name__ == "__main__":
    print("Generating results figures...")
    fig5_cumulative()
    fig6_predicted_observed()
    fig11_poisson_map()
    fig7_decision_map()
    fig8_calibration()
    fig9_world_cup()
    fig10_surprises()
    print("Done.")
