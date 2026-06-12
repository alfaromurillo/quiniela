"""
Generate EPS figures for the scientific article.
Output: articulo/figures/fig{1..4}.eps

Run from repo root:
    python model/figures.py
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

import sys
ROOT      = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
OUT_DIR   = ROOT / "articulo" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Shared style ───────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size":   11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "figure.dpi":  150,
})

GOALS = list(range(6))   # 0..5


def _scoreline_matrix(phase: str, gamma: float) -> np.ndarray:
    """Return 6×6 matrix P[home, away] from historical + gamma weighting."""
    from model.historical import build_distributions, scoreline_probs
    dist = build_distributions(gamma=gamma, delta=0.0)
    # Use equal probabilities (1/3, 1/3, 1/3) to get the pure historical
    # scoreline distribution without any Kalshi reweighting.
    sp = scoreline_probs(1/3, 1/3, 1/3, phase, dist=dist)
    mat = np.zeros((6, 6))
    for (h, a), p in sp.items():
        if h < 6 and a < 6:
            mat[h][a] = p
    total = mat.sum()
    if total > 0:
        mat /= total
    return mat


def _load_gamma() -> float:
    path = ROOT / "site" / "data" / "learning.json"
    if path.exists():
        return json.loads(path.read_text()).get("gamma", 0.84)
    return 0.84


# ── Figure 1: Historical scoreline heatmaps ────────────────────
def fig1_scoreline_dist():
    gamma = _load_gamma()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    for ax, phase, title in zip(
        axes,
        ["group", "knockout"],
        ["Fase de grupos", "Fase eliminatoria"],
    ):
        mat = _scoreline_matrix(phase, gamma)
        im = ax.imshow(mat * 100, origin="lower", aspect="equal",
                       cmap="Blues", vmin=0)
        ax.set_xticks(GOALS)
        ax.set_yticks(GOALS)
        ax.set_xlabel("Goles visitante")
        ax.set_ylabel("Goles local")
        ax.set_title(title)
        # Annotate cells with percentage
        for h in GOALS:
            for a in GOALS:
                val = mat[h][a] * 100
                if val >= 0.5:
                    ax.text(a, h, f"{val:.1f}", ha="center", va="center",
                            fontsize=7,
                            color="white" if val > 8 else "black")
        plt.colorbar(im, ax=ax, label="Probabilidad (%)")

    fig.suptitle(
        f"Distribución histórica de marcadores — WC 2014+2018+2022 "
        f"($\\gamma \\approx {gamma:.2f}$)",
        fontsize=12,
    )
    fig.tight_layout()
    out = OUT_DIR / "fig1_scoreline_dist.eps"
    fig.savefig(out, format="eps", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Figure 2: Effect of gamma ──────────────────────────────────
def fig2_gamma_effect():
    gammas = [1.0, 0.84, 0.6]
    labels = ["γ = 1 (sin decaimiento)", "γ = 0.84 (estimado)", "γ = 0.6"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    # Marginal distribution of total goals for group stage
    fig, ax = plt.subplots(figsize=(7, 4))
    totals = range(0, 9)
    for gamma, label, color in zip(gammas, labels, colors):
        mat = _scoreline_matrix("group", gamma)
        marginal = [
            sum(mat[h][a] for h in GOALS for a in GOALS if h + a == t)
            for t in totals
        ]
        ax.plot(totals, [m * 100 for m in marginal],
                marker="o", label=label, color=color)

    ax.set_xlabel("Total de goles en el partido")
    ax.set_ylabel("Probabilidad (%)")
    ax.set_title("Efecto de $\\gamma$ sobre la distribución de goles totales\n"
                 "(fase de grupos, WC 2014+2018+2022)")
    ax.legend()
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = OUT_DIR / "fig2_gamma_effect.eps"
    fig.savefig(out, format="eps", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Figure 3: Kalshi reweighting for a sample match ────────────
def fig3_kalshi_reweight():
    """
    Show before/after Kalshi reweighting for the first cached match.
    Requires at least one entry in data/kalshi_cache.json.
    """
    from model.kalshi import _load_cache, _clean_entry
    from model.historical import scoreline_probs

    cache = _load_cache()
    if not cache:
        print("  fig3: no Kalshi cache found — skipping")
        return

    # Use the first cached entry
    ticker_key = next(iter(cache))
    entry = _clean_entry(cache[ticker_key])
    ph = entry.get("home_win", 0.40)
    pd = entry.get("draw", 0.25)
    pa = entry.get("away_win", 0.35)
    tg = entry.get("total_goals")
    sh = entry.get("spread_home")
    sa = entry.get("spread_away")

    gamma = _load_gamma()

    from model.historical import build_distributions, scoreline_probs
    dist = build_distributions(gamma=gamma, delta=0.0)
    sp_before = scoreline_probs(ph, pd, pa, "group",
                                total_goals_probs=None,
                                spread_home=None, spread_away=None,
                                dist=dist)
    sp_after  = scoreline_probs(ph, pd, pa, "group",
                                total_goals_probs=tg,
                                spread_home=sh, spread_away=sa,
                                dist=dist)

    def to_matrix(sp):
        mat = np.zeros((6, 6))
        for (h, a), p in sp.items():
            if h < 6 and a < 6:
                mat[h][a] = p
        return mat

    mat_b = to_matrix(sp_before)
    mat_a = to_matrix(sp_after)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, mat, title in zip(
        axes,
        [mat_b, mat_a],
        ["Sin Kalshi (solo histórico)", "Con Kalshi (reajustado)"],
    ):
        im = ax.imshow(mat * 100, origin="lower", aspect="equal",
                       cmap="Blues", vmin=0)
        ax.set_xticks(GOALS)
        ax.set_yticks(GOALS)
        ax.set_xlabel("Goles visitante")
        ax.set_ylabel("Goles local")
        ax.set_title(title)
        for h in GOALS:
            for a in GOALS:
                val = mat[h][a] * 100
                if val >= 0.5:
                    ax.text(a, h, f"{val:.1f}", ha="center", va="center",
                            fontsize=7,
                            color="white" if val > 8 else "black")
        plt.colorbar(im, ax=ax, label="Probabilidad (%)")

    fig.suptitle(
        f"Reajuste por mercados de predicción — {ticker_key[:20]}\n"
        f"P(local)={ph:.2f}, P(empate)={pd:.2f}, P(visita)={pa:.2f}",
        fontsize=11,
    )
    fig.tight_layout()
    out = OUT_DIR / "fig3_kalshi_reweight.eps"
    fig.savefig(out, format="eps", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Figure 4: Delta evolution (stub until more results) ────────
def fig4_delta_evolution():
    """
    Plot estimated delta vs number of WC 2026 matches played.
    Currently just the initial point (n=0, delta=0) plus
    any points already available from learning.json.
    """
    path = ROOT / "site" / "data" / "learning.json"
    if not path.exists():
        print("  fig4: learning.json not found — skipping")
        return

    data = json.loads(path.read_text())
    n_now   = data.get("n_games_2026", 0)
    delta_now = data.get("delta", 0.0)

    # Include the known points (will grow as results come in)
    ns     = [0, n_now] if n_now > 0 else [0]
    deltas = [0.0, delta_now] if n_now > 0 else [0.0]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ns, deltas, marker="o", color="#1f77b4", linewidth=2)
    ax.set_xlabel("Partidos del WC 2026 disputados")
    ax.set_ylabel("Estimación de $\\delta$")
    ax.set_title("Adaptación bayesiana al torneo en curso\n"
                 "($\\delta = 0$: solo historial; "
                 "$\\delta > 0$: WC 2026 pesa más)")
    ax.set_xlim(left=-1)
    ax.set_ylim(bottom=-0.05)
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.grid(axis="y", alpha=0.3)
    ax.annotate(
        f"Actual: $\\delta = {delta_now:.2f}$ ({n_now} partidos)",
        xy=(n_now, delta_now), xytext=(n_now + 2, delta_now + 0.05),
        arrowprops=dict(arrowstyle="->", color="gray"),
        fontsize=10,
    ) if n_now > 0 else None
    fig.tight_layout()
    out = OUT_DIR / "fig4_delta_evolution.eps"
    fig.savefig(out, format="eps", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Main ───────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating figures for articulo/figures/...")
    fig1_scoreline_dist()
    fig2_gamma_effect()
    fig3_kalshi_reweight()
    fig4_delta_evolution()
    print("Done.")
