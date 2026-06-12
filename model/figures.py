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
MG = 5                   # MAX_GOALS


def _load_wc_matrices(path: Path) -> tuple[dict, dict]:
    """
    Parse a WC JSON file into count matrices and totals.
    Returns (counts, totals) where counts[phase] is (MG+1)×(MG+1) int array
    and totals[phase] is the number of matches in that phase.
    Knockout scores use et (120 min) when available, else ft (90 min).
    """
    data = json.loads(path.read_text())
    counts = {
        "group":    np.zeros((MG+1, MG+1), dtype=int),
        "knockout": np.zeros((MG+1, MG+1), dtype=int),
    }
    totals = {"group": 0, "knockout": 0}
    KO_KEYS = ["round of 16", "round of 32", "quarter", "semi", "final"]
    for m in data.get("matches", []):
        sc_raw = m.get("score", {})
        ft = sc_raw.get("ft")
        if not ft or len(ft) != 2:
            continue
        rnd = m.get("round", "").lower()
        phase = "knockout" if any(k in rnd for k in KO_KEYS) else "group"
        if phase == "knockout":
            et = sc_raw.get("et")
            sc = et if (et and len(et) == 2) else ft
        else:
            sc = ft
        h, a = int(sc[0]), int(sc[1])
        if h <= MG and a <= MG:
            counts[phase][h, a] += 1
        totals[phase] += 1
    return counts, totals


def _model_matrix(phase: str, gamma: float) -> np.ndarray:
    """Return (MG+1)×(MG+1) probability matrix from γ-weighted model (δ=0)."""
    from model.historical import build_distributions, scoreline_probs
    dist = build_distributions(gamma=gamma, delta=0.0)
    sp = scoreline_probs(1/3, 1/3, 1/3, phase, dist=dist)
    mat = np.zeros((MG+1, MG+1))
    for (h, a), p in sp.items():
        if h <= MG and a <= MG:
            mat[h, a] = p
    s = mat.sum()
    return mat / s if s > 0 else mat


def _load_gamma() -> float:
    path = ROOT / "site" / "data" / "learning.json"
    if path.exists():
        return json.loads(path.read_text()).get("gamma", 0.84)
    return 0.84


# ── Figure 1: Historical + model scoreline overview ────────────
def fig1_historical_overview():
    """
    3 rows × 4 columns:
      Left block  (cols 0–1): WC 2014/2018/2022 actual scoreline matrices,
                               as proportions; annotated with n=Y (x.xx%).
      Right block (cols 2–3): Row 0 = model estimate before WC 2026 (δ=0).
                               Rows 1–2 = placeholders (fill after tournament).
    Shared colour scale (proportions) across all non-placeholder panels.
    """
    gamma = _load_gamma()

    WC_FILES = [
        (2014, ROOT / "data" / "wc2014.json"),
        (2018, ROOT / "data" / "wc2018.json"),
        (2022, ROOT / "data" / "wc2022.json"),
    ]
    wc_data = [(yr, *_load_wc_matrices(p)) for yr, p in WC_FILES]

    phases      = ["group", "knockout"]
    phase_names = ["Fase de grupos", "Eliminación directa"]

    # Model matrices for row 0 of right block
    model_mats = {ph: _model_matrix(ph, gamma) for ph in phases}

    # Shared vmax across historical proportions and model probabilities
    vmax = 0.0
    for _yr, counts, totals in wc_data:
        for ph in phases:
            tot = totals[ph]
            if tot > 0:
                vmax = max(vmax, counts[ph].max() / tot)
    for ph in phases:
        vmax = max(vmax, model_mats[ph].max())
    vmax = round(vmax * 1.10, 2)  # 10 % headroom

    # ── Layout ────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 4, figsize=(15, 9.5))
    plt.subplots_adjust(wspace=0.38, hspace=0.55,
                        left=0.07, right=0.88, top=0.88, bottom=0.07)

    cmap_hist  = "Blues"
    cmap_model = "Greens"

    def _draw_heatmap(ax, mat, cmap, title, ylabel):
        im = ax.imshow(mat, origin="lower", aspect="equal",
                       cmap=cmap, vmin=0, vmax=vmax)
        ax.set_xticks(GOALS); ax.set_yticks(GOALS)
        ax.set_xticklabels(GOALS, fontsize=6)
        ax.set_yticklabels(GOALS, fontsize=6)
        ax.set_xlabel("Goles visitante", fontsize=7)
        ax.set_ylabel(ylabel, fontsize=7)
        ax.set_title(title, fontsize=8, pad=3)
        return im

    # ── Left block: historical WC data ────────────────────────────
    im_hist = None
    for row, (yr, counts, totals) in enumerate(wc_data):
        for col, (ph, ph_name) in enumerate(zip(phases, phase_names)):
            ax = axes[row, col]
            tot = totals[ph]
            mat = counts[ph] / tot if tot > 0 else counts[ph].astype(float)
            ylabel = f"WC {yr}\nGoles local" if col == 0 else "Goles local"
            im_hist = _draw_heatmap(ax, mat, cmap_hist, ph_name, ylabel)

            for h in GOALS:
                for a in GOALS:
                    n   = counts[ph][h, a]
                    frc = n / tot if tot > 0 else 0.0
                    if n == 0:
                        continue
                    txt   = f"n={n}\n({frc*100:.1f}%)"
                    color = "white" if frc > vmax * 0.55 else "black"
                    ax.text(a, h, txt, ha="center", va="center",
                            fontsize=4.5, color=color, linespacing=1.3)

    # ── Right block row 0: model estimate before WC 2026 ──────────
    im_model = None
    for col, (ph, ph_name) in enumerate(zip(phases, phase_names)):
        ax = axes[0, 2 + col]
        mat = model_mats[ph]
        ylabel = f"Estimado inicio\nGoles local" if col == 0 else "Goles local"
        im_model = _draw_heatmap(ax, mat, cmap_model, ph_name, ylabel)

        for h in GOALS:
            for a in GOALS:
                p = mat[h, a]
                if p < 0.004:
                    continue
                txt   = f"({p*100:.1f}%)"
                color = "white" if p > vmax * 0.55 else "black"
                ax.text(a, h, txt, ha="center", va="center",
                        fontsize=5.5, color=color)

    # ── Right block rows 1–2: placeholders ────────────────────────
    placeholder_rows = [
        ("Estimado final", "Completar al final\nde la fase de grupos"),
        ("WC 2026 — real", "Completar al final\ndel torneo"),
    ]
    for ri, (row_label, note) in enumerate(placeholder_rows):
        for col, ph_name in enumerate(phase_names):
            ax = axes[1 + ri, 2 + col]
            ax.set_facecolor("#efefef")
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_edgecolor("#bbbbbb")
            ax.set_title(ph_name, fontsize=8, pad=3, color="#777777")
            ax.set_ylabel(row_label if col == 0 else "",
                          fontsize=7, color="#777777")
            ax.text(0.5, 0.5, note,
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=8, color="#999999", style="italic",
                    multialignment="center")

    # ── Block headers ──────────────────────────────────────────────
    fig.text(0.27, 0.915,
             "Datos históricos  (proporciones, n real por celda)",
             ha="center", va="bottom", fontsize=10, fontweight="bold")
    fig.text(0.73, 0.915,
             f"Modelo estimado  ($\\gamma \\approx {gamma:.2f}$, $\\delta = 0$)",
             ha="center", va="bottom", fontsize=10, fontweight="bold")

    # ── Colorbars ─────────────────────────────────────────────────
    # One Blues bar for historical block, one Greens bar for model block
    # Place them stacked on the far right
    cb_ax1 = fig.add_axes([0.895, 0.52, 0.012, 0.33])  # Blues
    cb_ax2 = fig.add_axes([0.895, 0.15, 0.012, 0.33])  # Greens
    if im_hist is not None:
        cb1 = fig.colorbar(im_hist, cax=cb_ax1)
        cb1.set_label("Proporción", fontsize=7)
        cb1.ax.tick_params(labelsize=6)
    if im_model is not None:
        cb2 = fig.colorbar(im_model, cax=cb_ax2)
        cb2.set_label("Probabilidad", fontsize=7)
        cb2.ax.tick_params(labelsize=6)

    out = OUT_DIR / "fig1_historical_overview.eps"
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
        mat = _model_matrix("group", gamma)
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
    fig1_historical_overview()
    fig2_gamma_effect()
    fig3_kalshi_reweight()
    fig4_delta_evolution()
    print("Done.")
