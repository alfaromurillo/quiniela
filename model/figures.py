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


def _load_wc2026_group_real() -> tuple[np.ndarray, int]:
    """
    Canonical count matrix + total for completed WC 2026 group-stage matches,
    read from site/data/results.json + data/schedule.json.
    """
    schedule_data = json.loads((ROOT / "data" / "schedule.json").read_text())
    results_path  = ROOT / "site" / "data" / "results.json"
    if not results_path.exists():
        return np.zeros((MG+1, MG+1)), 0
    results_json = json.loads(results_path.read_text())
    by_id = {m["id"]: m for m in schedule_data["matches"]}
    raw = np.zeros((MG+1, MG+1))
    total = 0
    for mid_str, r in results_json.get("matches", {}).items():
        if r.get("status") != "final":
            continue
        m = by_id.get(int(mid_str))
        if not m or m["phase"] != "group":
            continue
        h, a = r.get("home_score"), r.get("away_score")
        if h is None or a is None:
            continue
        if h <= MG and a <= MG:
            raw[h, a] += 1
        total += 1
    return _to_canonical(raw), total


def _end_of_group_estimate_matrix(gamma: float, delta: float) -> np.ndarray:
    """
    Canonical probability matrix for the group phase using the model as it
    stood at the end of the group stage: γ-weighted historical data plus the
    72 completed WC 2026 group matches, weighted (1+δ), neutral 1/3-1/3-1/3
    outcome probabilities (no Kalshi reweighting).
    """
    from model.learn import load_canonical, count_2026
    from model.historical import build_distributions, scoreline_probs

    schedule_data = json.loads((ROOT / "data" / "schedule.json").read_text())
    results_path  = ROOT / "site" / "data" / "results.json"
    results_json  = json.loads(results_path.read_text()) if results_path.exists() else {"matches": {}}
    canonical     = load_canonical(results_json, schedule_data)
    extra_wins, extra_draws = count_2026(canonical)

    dist = build_distributions(gamma=gamma, delta=delta,
                                extra_wins=extra_wins, extra_draws=extra_draws)
    sp = scoreline_probs(1/3, 1/3, 1/3, "group", dist=dist)
    raw = np.zeros((MG+1, MG+1))
    for (h, a), p in sp.items():
        if h <= MG and a <= MG:
            raw[h, a] = p
    s = raw.sum()
    raw = raw / s if s > 0 else raw
    return _to_canonical(raw)


# ── Figure 1: Historical + model scoreline overview ────────────
def fig1_historical_overview():
    """
    3 rows × 4 columns — all panels show canonical scorelines:
      x-axis = Goles ganador (winner, 0–5)
      y-axis = Goles perdedor (loser, 0–5)
      NaN / white = upper-left triangle (loser > winner, impossible)

    Column layout (phases grouped together):
      Col 0: Fase de grupos  — historical WC 2014/2018/2022
      Col 1: Fase de grupos  — model estimate (row 0) + placeholders
      Col 2: Eliminación dir.— historical WC 2014/2018/2022
      Col 3: Eliminación dir.— model estimate (row 0) + placeholders

    Placeholders in model columns (rows 1–2):
      [row 1, col 1] = "WC 2026 Grp real"       (fill after ~jul 3)
      [row 2, col 1] = "Estimado fin de grupos"  (fill after ~jul 3)
      [row 1, col 3] = "Estimado fin del mundial"(fill after ~jul 19)
      [row 2, col 3] = "WC 2026 KO real"         (fill after ~jul 19)

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
    # 5-column GridSpec: cols 0-1 = Fase de grupos, col 2 = spacer
    # (near-zero width), cols 3-4 = Eliminación directa.
    # The spacer has negligible width but receives wspace padding on
    # both sides, so the inter-block gap = 2 × the intra-block gap.
    import matplotlib.gridspec as gridspec
    fig = plt.figure(figsize=(15, 9.5))
    gs = gridspec.GridSpec(
        3, 5, figure=fig,
        width_ratios=[1, 1, 0.001, 1, 1],
        wspace=0.30, hspace=0.42,
        left=0.07, right=0.89, top=0.88, bottom=0.07,
    )
    _GC = [0, 1, 3, 4]   # logical col i → GridSpec col _GC[i]
    axes = np.array([[fig.add_subplot(gs[r, c]) for c in _GC]
                     for r in range(3)])
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

    # ── Historical data: col 0 = Grp hist, col 2 = KO hist ────
    im_last = None
    # (col_idx, phase): odd model cols (1, 3) are handled separately
    hist_cols = [(0, "group"), (2, "knockout")]

    for row, (yr, counts, totals) in enumerate(wc_data):
        for col, ph in hist_cols:
            ax  = axes[row, col]
            tot = totals[ph]
            arr = counts[ph]   # canonical, NaN for impossible cells
            prop = np.where(np.isnan(arr), np.nan, arr / tot) if tot > 0 else arr.copy()

            ylabel = f"WC {yr}\nGoles perdedor" if col == 0 else "Goles perdedor"
            im_last = _draw_heatmap(ax, prop, "Datos históricos", ylabel)

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

    # ── Model estimate row 0: col 1 = Grp model, col 3 = KO model
    model_cols = [(1, "group"), (3, "knockout")]

    for col, ph in model_cols:
        ax  = axes[0, col]
        mat = model_mats[ph]
        im_last = _draw_heatmap(ax, mat, "Estimado inicio", "Goles perdedor")

        for loser in GOALS:
            for winner in GOALS:
                p = mat[loser, winner]
                if np.isnan(p) or p < 0.004:
                    continue
                txt   = f"({p*100:.1f}%)"
                color = "white" if p > vmax * 0.55 else "black"
                ax.text(winner, loser, txt, ha="center", va="center",
                        fontsize=5.5, color=color)

        # γ/δ annotation in the upper-left white region (loser > winner)
        ax.text(0.04, 0.97, f"$\\gamma \\approx {gamma:.2f}$\n$\\delta = 0$",
                transform=ax.transAxes, fontsize=6.5, color="#555555",
                va="top", ha="left", linespacing=1.5)

    # ── Row 1, col 1: end-of-group-stage model estimate (δ > 0) ────
    delta_eog = 0.1   # δ estimated when the 72nd (last) group match locked
    ax = axes[1, 1]
    mat_eog = _end_of_group_estimate_matrix(gamma, delta_eog)
    _draw_heatmap(ax, mat_eog, "Estimado fin de grupos", "Goles perdedor")
    for loser in GOALS:
        for winner in GOALS:
            p = mat_eog[loser, winner]
            if np.isnan(p) or p < 0.004:
                continue
            txt   = f"({p*100:.1f}%)"
            color = "white" if p > vmax * 0.55 else "black"
            ax.text(winner, loser, txt, ha="center", va="center",
                    fontsize=5.5, color=color)
    ax.text(0.04, 0.97, f"$\\gamma \\approx {gamma:.2f}$\n$\\delta \\approx {delta_eog:.2f}$",
            transform=ax.transAxes, fontsize=6.5, color="#555555",
            va="top", ha="left", linespacing=1.5)

    # ── Row 2, col 1: real WC 2026 group-stage results (72 matches) ─
    ax = axes[2, 1]
    real_counts, real_total = _load_wc2026_group_real()
    prop = (np.where(np.isnan(real_counts), np.nan, real_counts / real_total)
            if real_total > 0 else real_counts.copy())
    _draw_heatmap(ax, prop, "WC 2026 — datos reales", "Goles perdedor")
    for loser in GOALS:
        for winner in GOALS:
            n = real_counts[loser, winner]
            if np.isnan(n) or n == 0:
                continue
            frc = n / real_total if real_total > 0 else 0.0
            txt = f"n={int(n)}\n({frc*100:.1f}%)"
            color = "white" if frc > vmax * 0.55 else "black"
            ax.text(winner, loser, txt, ha="center", va="center",
                    fontsize=4.5, color=color, linespacing=1.3)

    # ── Placeholders: knockout model column (tournament in progress) ─
    # (r, c, bold_label, italic_note)
    placeholders = [
        (1, 3,
         "Estimado fin\ndel mundial",
         "Completar al final\ndel torneo (~jul 19)"),
        (2, 3,
         "WC 2026 KO real",
         "Completar al final\ndel torneo (~jul 19)"),
    ]
    for r, c, bold_lbl, italic_note in placeholders:
        ax = axes[r, c]
        ax.set_facecolor("#efefef")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor("#bbbbbb")
        ax.text(0.5, 0.57, bold_lbl, ha="center", va="center",
                transform=ax.transAxes, fontsize=9, color="#444444",
                fontweight="bold", multialignment="center")
        ax.text(0.5, 0.28, italic_note, ha="center", va="center",
                transform=ax.transAxes, fontsize=7.5, color="#999999",
                style="italic", multialignment="center")

    # ── Phase block headers (span two columns each) ────────────
    left_cx  = (axes[0, 0].get_position().x0
                + axes[0, 1].get_position().x1) / 2
    right_cx = (axes[0, 2].get_position().x0
                + axes[0, 3].get_position().x1) / 2
    fig.text(left_cx,  0.915, "Fase de grupos",
             ha="center", va="bottom", fontsize=10, fontweight="bold")
    fig.text(right_cx, 0.915, "Eliminación directa",
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
    gammas = [1.0, 0.84]
    labels = ["$\\gamma = 1$ (sin decaimiento)",
              "$\\gamma \\approx 0.84$ (estimado)"]
    colors = ["#1f77b4", "#ff7f0e"]

    totals_range = list(range(0, 9))
    phases = [("group",    "Fase de grupos"),
              ("knockout", "Eliminación directa")]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.0), sharey=False)
    plt.subplots_adjust(wspace=0.35)

    # Pre-compute matrices for both gammas × both phases (needed for insets)
    mats = {(ph, g): _model_matrix(ph, g) for ph, _ in phases for g in gammas}

    for ax, (phase, phase_label) in zip(axes, phases):
        # ── Line plots ──────────────────────────────────────────
        for gamma, label, color in zip(gammas, labels, colors):
            mat = mats[(phase, gamma)]
            marginal = []
            for t in totals_range:
                prob = 0.0
                for lo in GOALS:
                    w = t - lo
                    if 0 <= w <= MG and w >= lo:
                        v = mat[lo, w]
                        if not np.isnan(v):
                            prob += v
                marginal.append(prob)
            ax.plot(totals_range, [m * 100 for m in marginal],
                    marker="o", label=label, color=color, linewidth=1.8)

        ax.set_xlabel("Total de goles en el partido", fontsize=11)
        ax.set_ylabel("Probabilidad (%)", fontsize=11)
        ax.set_title(phase_label, fontsize=12)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.grid(axis="y", alpha=0.3)
        ax.legend(loc="lower left", fontsize=9)

        # ── Inset bar chart: scorelines with biggest γ change ───
        mat_1   = mats[(phase, 1.0)]
        mat_084 = mats[(phase, 0.84)]

        diffs = []
        for lo in GOALS:
            for wi in range(lo, MG + 1):
                p1  = mat_1[lo, wi]
                p84 = mat_084[lo, wi]
                if np.isnan(p1) or np.isnan(p84):
                    continue
                diffs.append((abs(p84 - p1), lo, wi, p1, p84))
        diffs.sort(reverse=True)
        top5 = diffs[:5]

        slabels  = [f"{wi}-{lo}" for _, lo, wi, _, _   in top5]
        p1_vals  = [p1  * 100   for _, _,  _,  p1, _  in top5]
        p84_vals = [p84 * 100   for _, _,  _,  _,  p84 in top5]

        inset = ax.inset_axes([0.515, 0.52, 0.465, 0.43])
        xi = np.arange(len(slabels))
        bw = 0.38
        inset.bar(xi - bw/2, p1_vals,  bw, color="#1f77b4",
                  alpha=0.85, label="$\\gamma=1$")
        inset.bar(xi + bw/2, p84_vals, bw, color="#ff7f0e",
                  alpha=0.85, label="$\\hat{\\gamma}$")
        inset.set_xticks(xi)
        inset.set_xticklabels(slabels, fontsize=6.5)
        inset.set_ylabel("P (%)", fontsize=6.5, labelpad=2)
        inset.tick_params(labelsize=6, pad=1)
        inset.grid(axis="y", alpha=0.3, linewidth=0.5)
        inset.legend(fontsize=6, loc="upper right",
                     framealpha=0.75, borderpad=0.4)
        inset.set_title("Marcadores con mayor cambio",
                        fontsize=6.5, pad=2)
        for sp in inset.spines.values():
            sp.set_linewidth(0.5)

    fig.suptitle(
        "Efecto de $\\gamma$ sobre la distribución de goles totales\n"
        "(WC 2014+2018+2022, ponderación geométrica por torneo)",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out = OUT_DIR / "fig2_gamma_effect.png"
    fig.savefig(out, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Figure 3: Kalshi reweighting for a sample match ────────────
def fig3_kalshi_reweight():
    """
    Show before/after Kalshi reweighting for South Korea vs Czech Republic
    (WC 2026 group stage, jornada 1) — the same match used in Section 5.
    Falls back to the first available cached match if not found.
    Uses home×away (not canonical) to show asymmetric reweighting clearly.
    Marks the optimal prediction on the 'Con Kalshi' panel.
    """
    from model.kalshi import _load_cache, _clean_entry
    from model.historical import scoreline_probs, scoreline_probs_ipf, build_distributions
    from model.optimizer import expected_points, best_prediction

    cache = _load_cache()
    if not cache:
        print("  fig3: no Kalshi cache found — skipping")
        return

    # Prefer Korea vs Czech (cross-references Section 5 example)
    preferred = "KXWCGAME-26JUN11KORCZE"
    ticker_key = preferred if preferred in cache else next(iter(cache))
    entry  = _clean_entry(cache[ticker_key])
    ph_val = entry.get("home_win", 0.40)
    pd_val = entry.get("draw", 0.25)
    pa_val = entry.get("away_win", 0.35)
    tg     = entry.get("total_goals")
    sh     = entry.get("spread_home")
    sa     = entry.get("spread_away")

    gamma = _load_gamma()
    dist  = build_distributions(gamma=gamma, delta=0.0)

    sp_before = scoreline_probs(ph_val, pd_val, pa_val, "group",
                                total_goals_probs=None,
                                spread_home=None, spread_away=None,
                                dist=dist)
    sp_after  = scoreline_probs_ipf(ph_val, pd_val, pa_val, "group",
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
    best  = best_prediction(sp_after, "group")
    opt_h, opt_a = best["home"], best["away"]

    vmax = max(mat_b.max(), mat_a.max()) * 100

    if ticker_key == preferred:
        xlabel_str = "Goles Rep. Checa"
        ylabel_str = "Goles Corea del Sur"
    else:
        xlabel_str = "Goles Equipo B"
        ylabel_str = "Goles Equipo A"

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, mat, title, mark_opt in zip(
        axes,
        [mat_b, mat_a],
        ["Sin Kalshi (solo histórico)", "Con Kalshi (IPF)"],
        [False, True],
    ):
        im = ax.imshow(mat * 100, origin="lower", aspect="equal",
                       cmap="Blues", vmin=0, vmax=vmax)
        ax.set_xticks(GOALS); ax.set_yticks(GOALS)
        ax.set_xticklabels(GOALS, fontsize=9)
        ax.set_yticklabels(GOALS, fontsize=9)
        ax.set_xlabel(xlabel_str, fontsize=11)
        ax.set_ylabel(ylabel_str, fontsize=11)
        ax.set_title(title, fontsize=12)
        for h in GOALS:
            for a in GOALS:
                val = mat[h][a] * 100
                if val >= 0.5:
                    color = "white" if val > vmax * 0.60 else "black"
                    ax.text(a, h, f"{val:.1f}", ha="center", va="center",
                            fontsize=7.5, color=color)
        if mark_opt:
            rect = plt.Rectangle(
                (opt_a - 0.5, opt_h - 0.5), 1, 1,
                linewidth=2.5, edgecolor="#e63946", facecolor="none",
            )
            ax.add_patch(rect)
            ax.text(opt_a, opt_h - 0.42, "óptimo",
                    ha="center", va="top", fontsize=7,
                    color="#e63946", fontweight="bold")
        plt.colorbar(im, ax=ax, label="Probabilidad (%)")

    # Readable match label
    match_label = ("Corea del Sur vs. Rep. Checa"
                   if ticker_key == preferred
                   else ticker_key[:24])
    fig.suptitle(
        f"Reajuste IPF por mercados de predicción de Kalshi\n"
        f"{match_label} — "
        f"$p_A={ph_val:.2f}$, $p_E={pd_val:.2f}$, $p_B={pa_val:.2f}$",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    out = OUT_DIR / "fig3_kalshi_reweight.png"
    fig.savefig(out, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Figure 4: Delta evolution ──────────────────────────────────
# Jornada boundaries (cumulative games after each jornada ends)
_JORNADA_LABELS = [
    (16,  "MD1"),   # Matchday 1: 16 games
    (32,  "MD2"),   # Matchday 2: 16 games
    (48,  "MD3"),   # Matchday 3: 16 games
    (64,  "R32"),   # Round of 32: 16 games
    (72,  "R16"),   # Round of 16: 8 games
    (76,  "QF"),    # Quarter-finals: 4 games
    (78,  "SF"),    # Semi-finals: 2 games
    (80,  "F"),     # Final + 3rd place: 2 games
]


def fig4_delta_evolution():
    """
    Plot estimated δ vs cumulative WC 2026 matches played.
    Reads delta_history from learning.json for full trajectory.
    Jornada boundaries are marked with vertical dashed lines.
    """
    path = ROOT / "site" / "data" / "learning.json"
    if not path.exists():
        print("  fig4: learning.json not found — skipping")
        return

    data    = json.loads(path.read_text())
    history = data.get("delta_history", [])

    # Build (n_games, delta) series; always start at (0, 0)
    if history:
        ns     = [0] + [h["n_games"] for h in history]
        deltas = [0.0] + [h["delta"]  for h in history]
    else:
        n_now     = data.get("n_games_2026", 0)
        delta_now = data.get("delta", 0.0)
        ns     = [0, n_now] if n_now > 0 else [0]
        deltas = [0.0, delta_now] if n_now > 0 else [0.0]

    n_last     = ns[-1]
    delta_last = deltas[-1]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(ns, deltas, marker="o", color="#1f77b4", linewidth=2,
            markersize=5, zorder=3)

    # Jornada boundary lines — always show all to give tournament context
    y_top = max(max(deltas) * 1.15, 0.08)
    ax.set_ylim(bottom=-0.05, top=y_top)
    for n_boundary, label in _JORNADA_LABELS:
        ax.axvline(n_boundary, color="#cccccc", linewidth=0.9,
                   linestyle="--", zorder=1)
        ax.text(n_boundary, y_top * 0.97, label,
                ha="center", va="top", fontsize=7.5, color="#888888")

    ax.set_xlabel("Partidos del WC 2026 disputados", fontsize=11)
    ax.set_ylabel("Estimación de $\\delta$", fontsize=11)
    ax.set_title(
        "Adaptación bayesiana al torneo en curso\n"
        "($\\delta = 0$: solo historial; "
        "$\\delta > 0$: WC 2026 pesa más)",
        fontsize=11,
    )
    ax.set_xlim(left=-1, right=82)
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--", zorder=2)
    ax.grid(axis="y", alpha=0.3)

    if n_last > 0:
        ax.annotate(
            f"Actual: $\\delta = {delta_last:.2f}$ ({n_last} partidos)",
            xy=(n_last, delta_last),
            xytext=(min(n_last + 4, 70), delta_last + 0.04),
            arrowprops=dict(arrowstyle="->", color="gray"),
            fontsize=9,
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
