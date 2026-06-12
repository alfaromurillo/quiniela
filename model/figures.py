"""
Generate PNG figures for the scientific article.
Output: articulo/figures/fig{1..4}.png

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
ROOT    = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
OUT_DIR = ROOT / "articulo" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Shared style ───────────────────────────────────────────────
plt.rcParams.update({
    "font.family":    "serif",
    "font.size":      11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "figure.dpi":     150,
})

GOALS = list(range(6))   # 0..5
MG    = 5                # MAX_GOALS


def _to_canonical(raw_mat: np.ndarray) -> np.ndarray:
    """
    Aggregate home×away matrix into canonical (winner, loser) form.
    Returns canon[loser, winner] where winner = max(h,a), loser = min(h,a).
    Cells where loser > winner are left as NaN (physically impossible).

    Plotted with imshow(origin='lower'):
      x-axis = winner (col index), y-axis = loser (row index).
    """
    n = raw_mat.shape[0]
    canon = np.full((n, n), np.nan)
    for h in range(n):
        for a in range(n):
            w = max(h, a)
            l = min(h, a)
            if np.isnan(canon[l, w]):
                canon[l, w] = 0.0
            canon[l, w] += raw_mat[h, a]
    return canon


def _load_wc_matrices(path: Path) -> tuple[dict, dict]:
    """
    Parse a WC JSON file into canonical count matrices and totals.
    Returns (counts, totals) where counts[phase] is a (MG+1)×(MG+1) array
    in canonical form (NaN for impossible cells) and totals[phase] is the
    number of matches in that phase.
    Knockout scores use et (120 min) when available, else ft (90 min).
    """
    data = json.loads(path.read_text())
    raw = {
        "group":    np.zeros((MG+1, MG+1)),
        "knockout": np.zeros((MG+1, MG+1)),
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
            raw[phase][h, a] += 1
        totals[phase] += 1
    canon = {ph: _to_canonical(raw[ph]) for ph in raw}
    return canon, totals


def _model_matrix(phase: str, gamma: float) -> np.ndarray:
    """
    Return (MG+1)×(MG+1) canonical probability matrix from γ-weighted
    model (δ=0, neutral 1/3-1/3-1/3 probs, no Kalshi reweighting).
    """
    from model.historical import build_distributions, scoreline_probs
    dist = build_distributions(gamma=gamma, delta=0.0)
    sp = scoreline_probs(1/3, 1/3, 1/3, phase, dist=dist)
    raw = np.zeros((MG+1, MG+1))
    for (h, a), p in sp.items():
        if h <= MG and a <= MG:
            raw[h, a] = p
    s = raw.sum()
    raw = raw / s if s > 0 else raw
    return _to_canonical(raw)


def _load_gamma() -> float:
    path = ROOT / "site" / "data" / "learning.json"
    if path.exists():
        return json.loads(path.read_text()).get("gamma", 0.84)
    return 0.84


# ── Figure 1: Historical + model scoreline overview ────────────
def fig1_historical_overview():
    """
    3 rows × 4 columns — all panels show canonical scorelines:
      x-axis = Goles ganador (winner, 0–5)
      y-axis = Goles perdedor (loser, 0–5)
      NaN / white = upper-left triangle (loser > winner, impossible)

    Column layout (after col 2↔3 swap):
      Col 0: Fase de grupos  — historical WC 2014/2018/2022
      Col 1: Eliminación dir.— historical WC 2014/2018/2022
      Col 2: Eliminación dir.— model estimate (row 0) + placeholders
      Col 3: Fase de grupos  — model estimate (row 0) + placeholders

    Placeholders (4 lower-right panels):
      [row 1, col 2] = "WC 2026 KO real"        (fill after ~jul 19)
      [row 1, col 3] = "Estimado fin mundial"    (fill after ~jul 19)
      [row 2, col 2] = "Estimado fin de grupos"  (fill after ~jul 3)
      [row 2, col 3] = "WC 2026 Grp real"        (fill after ~jul 3)

    Single Blues colormap; one shared colorbar on the right.
    """
    gamma = _load_gamma()

    WC_FILES = [
        (2014, ROOT / "data" / "wc2014.json"),
        (2018, ROOT / "data" / "wc2018.json"),
        (2022, ROOT / "data" / "wc2022.json"),
    ]
    wc_data = [(yr, *_load_wc_matrices(p)) for yr, p in WC_FILES]

    phases      = ["group", "knockout"]
    phase_names = {"group": "Fase de grupos", "knockout": "Eliminación directa"}

    # Right block after column swap: col 2 = KO model, col 3 = Grp model
    model_mats = {ph: _model_matrix(ph, gamma) for ph in phases}

    # Shared vmax across historical proportions and model probabilities
    vmax = 0.0
    for _yr, counts, totals in wc_data:
        for ph in phases:
            tot = totals[ph]
            if tot > 0:
                vmax = max(vmax, float(np.nanmax(counts[ph] / tot)))
    for ph in phases:
        vmax = max(vmax, float(np.nanmax(model_mats[ph])))
    vmax = round(vmax * 1.10, 2)   # 10 % headroom

    # ── Layout ────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 4, figsize=(15, 9.5))
    plt.subplots_adjust(wspace=0.45, hspace=0.55,
                        left=0.07, right=0.89, top=0.88, bottom=0.07)
    cmap = "Blues"

    def _draw_heatmap(ax, mat, title, ylabel):
        im = ax.imshow(mat, origin="lower", aspect="equal",
                       cmap=cmap, vmin=0, vmax=vmax)
        ax.set_xticks(GOALS); ax.set_yticks(GOALS)
        ax.set_xticklabels(GOALS, fontsize=6)
        ax.set_yticklabels(GOALS, fontsize=6)
        ax.set_xlabel("Goles ganador", fontsize=7)
        ax.set_ylabel(ylabel, fontsize=7)
        ax.set_title(title, fontsize=8, pad=3)
        return im

    # ── Left block: historical WC data ────────────────────────
    im_last = None
    left_phases = ["group", "knockout"]   # col 0 = Grp, col 1 = KO

    for row, (yr, counts, totals) in enumerate(wc_data):
        for col_idx, ph in enumerate(left_phases):
            ax  = axes[row, col_idx]
            tot = totals[ph]
            arr = counts[ph]   # canonical, NaN for impossible cells
            prop = np.where(np.isnan(arr), np.nan, arr / tot) if tot > 0 else arr.copy()

            ylabel = f"WC {yr}\nGoles perdedor" if col_idx == 0 else "Goles perdedor"
            im_last = _draw_heatmap(ax, prop, phase_names[ph], ylabel)

            for loser in GOALS:
                for winner in GOALS:
                    n = arr[loser, winner]
                    if np.isnan(n) or n == 0:
                        continue
                    frc   = n / tot if tot > 0 else 0.0
                    txt   = f"n={int(n)}\n({frc*100:.1f}%)"
                    color = "white" if frc > vmax * 0.55 else "black"
                    ax.text(winner, loser, txt, ha="center", va="center",
                            fontsize=4.5, color=color, linespacing=1.3)

    # ── Right block row 0: model estimate (δ=0, neutral probs) ─
    # Column order after swap: col 2 = KO model, col 3 = Grp model
    right_phases = ["knockout", "group"]   # col 2 = KO, col 3 = Grp

    for rcol, ph in enumerate(right_phases):
        ax  = axes[0, 2 + rcol]
        mat = model_mats[ph]
        ylabel = "Estimado inicio\nGoles perdedor" if rcol == 0 else "Goles perdedor"
        im_last = _draw_heatmap(ax, mat, phase_names[ph], ylabel)

        for loser in GOALS:
            for winner in GOALS:
                p = mat[loser, winner]
                if np.isnan(p) or p < 0.004:
                    continue
                txt   = f"({p*100:.1f}%)"
                color = "white" if p > vmax * 0.55 else "black"
                ax.text(winner, loser, txt, ha="center", va="center",
                        fontsize=5.5, color=color)

    # ── Right block rows 1–2: placeholders ────────────────────
    # (r, c, phase_header, bold_label, italic_note)
    placeholders = [
        (1, 2, "Eliminación directa",
         "WC 2026 KO real",
         "Completar al final\ndel torneo (~jul 19)"),
        (1, 3, "Fase de grupos",
         "Estimado fin\ndel mundial",
         "Completar al final\ndel torneo (~jul 19)"),
        (2, 2, "Eliminación directa",
         "Estimado fin\nde grupos",
         "Completar al fin\nde la fase de grupos (~jul 3)"),
        (2, 3, "Fase de grupos",
         "WC 2026 Grp real",
         "Completar al fin\nde la fase de grupos (~jul 3)"),
    ]
    for r, c, ph_hdr, bold_lbl, italic_note in placeholders:
        ax = axes[r, c]
        ax.set_facecolor("#efefef")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor("#bbbbbb")
        ax.set_title(ph_hdr, fontsize=8, pad=3, color="#777777")
        if c == 2:
            ax.set_ylabel("", fontsize=7, color="#777777")
        ax.text(0.5, 0.57, bold_lbl, ha="center", va="center",
                transform=ax.transAxes, fontsize=9, color="#444444",
                fontweight="bold", multialignment="center")
        ax.text(0.5, 0.28, italic_note, ha="center", va="center",
                transform=ax.transAxes, fontsize=7.5, color="#999999",
                style="italic", multialignment="center")

    # ── Block headers ──────────────────────────────────────────
    fig.text(0.255, 0.915,
             "Datos históricos  (proporciones, n por celda)",
             ha="center", va="bottom", fontsize=10, fontweight="bold")
    fig.text(0.715, 0.915,
             f"Modelo estimado  ($\\gamma \\approx {gamma:.2f}$, $\\delta = 0$)",
             ha="center", va="bottom", fontsize=10, fontweight="bold")

    # ── Single shared colorbar ─────────────────────────────────
    cb_ax = fig.add_axes([0.905, 0.10, 0.013, 0.70])
    if im_last is not None:
        cb = fig.colorbar(im_last, cax=cb_ax)
        cb.set_label("Proporción / Probabilidad", fontsize=8)
        cb.ax.tick_params(labelsize=7)

    out = OUT_DIR / "fig1_historical_overview.png"
    fig.savefig(out, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Figure 2: Effect of gamma ──────────────────────────────────
def fig2_gamma_effect():
    gammas = [1.0, 0.84, 0.6]
    labels = ["γ = 1 (sin decaimiento)", "γ = 0.84 (estimado)", "γ = 0.6"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    fig, ax = plt.subplots(figsize=(7, 4))
    totals_range = range(0, 9)
    for gamma, label, color in zip(gammas, labels, colors):
        mat = _model_matrix("group", gamma)
        # Marginal P(total=t): sum canonical[loser, winner] over loser+winner=t
        marginal = []
        for t in totals_range:
            prob = 0.0
            for l in GOALS:
                w = t - l
                if 0 <= w <= MG and w >= l:
                    v = mat[l, w]
                    if not np.isnan(v):
                        prob += v
            marginal.append(prob)
        ax.plot(list(totals_range), [m * 100 for m in marginal],
                marker="o", label=label, color=color)

    ax.set_xlabel("Total de goles en el partido")
    ax.set_ylabel("Probabilidad (%)")
    ax.set_title("Efecto de $\\gamma$ sobre la distribución de goles totales\n"
                 "(fase de grupos, WC 2014+2018+2022)")
    ax.legend()
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = OUT_DIR / "fig2_gamma_effect.png"
    fig.savefig(out, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Figure 3: Kalshi reweighting for a sample match ────────────
def fig3_kalshi_reweight():
    """
    Show before/after Kalshi reweighting for the first cached match.
    Requires at least one entry in data/kalshi_cache.json.
    Uses home×away (not canonical) to show the asymmetric reweighting.
    """
    from model.kalshi import _load_cache, _clean_entry
    from model.historical import scoreline_probs

    cache = _load_cache()
    if not cache:
        print("  fig3: no Kalshi cache found — skipping")
        return

    ticker_key = next(iter(cache))
    entry  = _clean_entry(cache[ticker_key])
    ph_val = entry.get("home_win", 0.40)
    pd_val = entry.get("draw", 0.25)
    pa_val = entry.get("away_win", 0.35)
    tg     = entry.get("total_goals")
    sh     = entry.get("spread_home")
    sa     = entry.get("spread_away")

    gamma = _load_gamma()

    from model.historical import build_distributions
    dist = build_distributions(gamma=gamma, delta=0.0)
    sp_before = scoreline_probs(ph_val, pd_val, pa_val, "group",
                                total_goals_probs=None,
                                spread_home=None, spread_away=None,
                                dist=dist)
    sp_after  = scoreline_probs(ph_val, pd_val, pa_val, "group",
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
        f"P(local)={ph_val:.2f}, P(empate)={pd_val:.2f}, "
        f"P(visita)={pa_val:.2f}",
        fontsize=11,
    )
    fig.tight_layout()
    out = OUT_DIR / "fig3_kalshi_reweight.png"
    fig.savefig(out, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Figure 4: Delta evolution ──────────────────────────────────
def fig4_delta_evolution():
    """
    Plot estimated delta vs number of WC 2026 matches played.
    """
    path = ROOT / "site" / "data" / "learning.json"
    if not path.exists():
        print("  fig4: learning.json not found — skipping")
        return

    data      = json.loads(path.read_text())
    n_now     = data.get("n_games_2026", 0)
    delta_now = data.get("delta", 0.0)

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
    if n_now > 0:
        ax.annotate(
            f"Actual: $\\delta = {delta_now:.2f}$ ({n_now} partidos)",
            xy=(n_now, delta_now),
            xytext=(n_now + 2, delta_now + 0.05),
            arrowprops=dict(arrowstyle="->", color="gray"),
            fontsize=10,
        )
    fig.tight_layout()
    out = OUT_DIR / "fig4_delta_evolution.png"
    fig.savefig(out, format="png", dpi=150, bbox_inches="tight")
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
