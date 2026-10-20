"""
Fit a scoreline distribution to Kalshi's separately quoted marginals.

The markets give marginal tables of the score grid {0..5}²: signed goal
margin M = h − k (outcome + spread markets), total goals T = h + k, and,
in the knockout stage, each team's goals.  Quoted separately, these
tables are rarely compatible: no distribution on the grid reproduces
all of them (e.g. P(T even) must equal P(M even)), so plain IPF cycles
and its output depends on the order of the projections.

The final method has two steps:

1. Compatibility projection.  Find the marginal tables q* that some
   distribution on the grid has and that are closest to the market
   tables t in the sense of  min Σ_k KL(t_k ‖ q_k).  The objective is
   concave in the joint distribution P (each q_k is linear in P), and
   EM finds its maximiser: P ← P · mean_k t_k(b_k(c)) / q_k(b_k(c)).
   The optimal marginals are unique.  When the market tables are
   compatible, q* = t.

2. IPF from the historical reference distribution to q*.  Since q* is
   compatible, IPF converges to the I-projection: the distribution
   closest to the reference in Kullback–Leibler divergence that has
   exactly the marginals q* (Csiszár 1975).  It sums to 1.

Censored markets.  Kalshi rarely lists "wins by over 3.5" (185 of 196
spread quotes lack it) and team-total markets stop at varying
thresholds.  A missing threshold is not a zero probability: the last
quoted bin is open ("wins by 3 or more").  The legacy parser stored the
missing bins as 0, which forced IPF to put the mass of high totals on
5-2 and 2-5.

Parametric competitors (independent Poisson, Dixon–Coles) are fitted to
the same market tables by maximising Σ_k Σ_b t_kb log q_kb.

Grid convention: the cell with 5 goals stands for 5 or more.
"""
import math

import numpy as np
from scipy.optimize import minimize

G = 6                                    # goals 0..5 per team
H, K = np.meshgrid(range(G), range(G), indexing="ij")
TOT, MAR = H + K, H - K


# ── Market tables ────────────────────────────────────────────────────────────

def _last_quoted(cond: dict) -> int:
    """Highest margin bin with positive probability; later bins are unquoted."""
    pos = [m for m in range(1, 5) if cond.get(m, 0.0) > 0.0]
    return max(pos) if pos else 1


def margin_table(p_a: float, p_e: float, p_b: float,
                 spread_a: dict | None, spread_b: dict | None,
                 censor: bool = True) -> list:
    """
    Signed-margin table: draw, Team A by m, Team B by m.  spread_*[m] is
    P(win by m | win), m = 4 meaning 4+; trailing zeros are unquoted
    markets and merge into an open bin "by m or more" (censor=False
    reads them literally, as the legacy fits did).
    """
    tab = [(MAR == 0, p_e)]
    for p_side, cond, sign in ((p_a, spread_a, 1), (p_b, spread_b, -1)):
        m_side = sign * MAR
        if not cond or sum(cond.values()) <= 0:
            tab.append((m_side >= 1, p_side))
            continue
        s = sum(cond.values())
        top = _last_quoted(cond) if censor else 4
        for m in range(1, top):
            tab.append((m_side == m, p_side * cond.get(m, 0.0) / s))
        tab.append((m_side >= top,
                    p_side * sum(cond.get(m, 0.0) for m in range(top, 5)) / s))
    return tab


def total_table(total_goals: dict) -> list:
    """Total-goals table: T = 0..5 and T ≥ 6."""
    s = sum(total_goals.values())
    tab = [(TOT == t, total_goals.get(t, 0.0) / s) for t in range(6)]
    tab.append((TOT >= 6, total_goals.get(6, 0.0) / s))
    return tab


def team_table(over: dict | None, axis: int) -> list | None:
    """
    One team's goals from P(goals ≥ n) at the quoted thresholds n; bins run
    between consecutive quoted thresholds, the last one open.
    """
    if not over:
        return None
    ns = sorted(n for n in over if 1 <= n <= 5)
    if not ns:
        return None
    p, prev = {}, 1.0
    for n in ns:                         # enforce a decreasing survival curve
        prev = min(prev, max(over[n], 0.0))
        p[n] = prev
    g = H if axis == 0 else K
    edges = [0] + ns
    tab = []
    for lo, hi in zip(edges, edges[1:] + [None]):
        mass = (1.0 if lo == 0 else p[lo]) - (p[hi] if hi else 0.0)
        cells = (g >= lo) if hi is None else (g >= lo) & (g < hi)
        tab.append((cells, max(mass, 0.0)))
    return tab


def market_tables(p_a, p_e, p_b, total_goals, spread_a, spread_b,
                  team_totals: dict | None = None,
                  censor: bool = True) -> list:
    """All market tables available for one match (see module docstring)."""
    tabs = [margin_table(p_a, p_e, p_b, spread_a, spread_b, censor)]
    if total_goals:
        tabs.append(total_table(total_goals))
    for key, axis in (("home", 0), ("away", 1)):
        t = team_table((team_totals or {}).get(key), axis)
        if t:
            tabs.append(t)
    return tabs


def table_fit(P: np.ndarray, tabs: list) -> float:
    """Σ_k Σ_b t_kb log q_kb (the projection's objective)."""
    out = 0.0
    for tab in tabs:
        for cells, t in tab:
            if t > 0:
                out += t * math.log(max(P[cells].sum(), 1e-300))
    return out


def _compile(tabs: list) -> list:
    """Each table as (bin label of every cell, targets); bins partition the grid."""
    out = []
    for tab in tabs:
        lab = np.full(G * G, -1)
        for i, (cells, _) in enumerate(tab):
            lab[cells.ravel()] = i
        assert (lab >= 0).all(), "a table's bins must cover the grid"
        out.append((lab, np.array([t for _, t in tab], dtype=float)))
    return out


def _ratio(P: np.ndarray, lab: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Per-cell target / current mass of the cell's bin (0 for empty bins)."""
    q = np.bincount(lab, P, minlength=len(t))
    r = np.divide(t, q, out=np.zeros_like(t), where=q > 0)
    return r[lab]


# ── Step 1: compatibility projection (EM) ────────────────────────────────────

def project(P0: np.ndarray, tabs: list,
            tol: float = 1e-11, max_iter: int = 20000) -> list:
    """
    Closest compatible tables: argmax over P of Σ_k Σ_b t_kb log q_kb(P),
    by EM from P0.  Returns tables with the same bins and targets q*.
    """
    comp = _compile(tabs)
    P = (P0 / P0.sum()).ravel()
    for _ in range(max_iter):
        R = sum(_ratio(P, lab, t) for lab, t in comp)
        P_new = P * R / len(comp)
        P_new /= P_new.sum()
        done = np.abs(P_new - P).max() < tol
        P = P_new
        if done:
            break
    P = P.reshape(G, G)
    return [[(cells, float(P[cells].sum())) for cells, _ in tab] for tab in tabs]


# ── Step 2: IPF to compatible tables ─────────────────────────────────────────

def ipf(P0: np.ndarray, tabs: list,
        tol: float = 1e-12, max_iter: int = 5000) -> np.ndarray:
    """
    Cyclic proportional fitting of P0 to the tables' targets.  A table's
    bins are disjoint, so all of them are rescaled at once.
    """
    comp = _compile(tabs)
    P = (P0 / P0.sum()).ravel()
    for _ in range(max_iter):
        old = P.copy()
        for lab, t in comp:
            q = np.bincount(lab, P, minlength=len(t))
            f = np.divide(t, q, out=np.ones_like(t), where=q > 0)
            P = P * f[lab]
        if np.abs(P - old).max() < tol:
            break
    P = P.reshape(G, G)
    return P / P.sum()


def residual(P: np.ndarray, tabs: list) -> float:
    """Largest gap between a table's targets and P's marginals."""
    return max(abs(P[cells].sum() - t) for tab in tabs for cells, t in tab)


def fit_compatible(P0: np.ndarray, tabs: list) -> np.ndarray:
    """
    Final method: compatibility projection, then IPF from P0.  When the
    tables are already compatible the projection returns them unchanged,
    so IPF is tried first and the (slow) projection runs only if IPF
    cannot reproduce the tables.
    """
    P = ipf(P0, tabs)
    if residual(P, tabs) < 1e-7:
        return P
    return ipf(P0, project(P0, tabs))


# ── Parametric competitors ───────────────────────────────────────────────────

def _pois_col(lam: float) -> np.ndarray:
    p = np.array([math.exp(-lam) * lam**g / math.factorial(g) for g in range(G)])
    p[-1] = max(1.0 - p[:-1].sum(), 0.0)          # 5 stands for 5+
    return p


def poisson_grid(la: float, lb: float, rho: float = 0.0) -> np.ndarray | None:
    """Independent Poisson; rho ≠ 0 adds Dixon–Coles low-score dependence."""
    P = np.outer(_pois_col(la), _pois_col(lb))
    if rho:
        tau = np.ones((2, 2))
        tau[0, 0] = 1 - la * lb * rho
        tau[0, 1] = 1 + la * rho
        tau[1, 0] = 1 + lb * rho
        tau[1, 1] = 1 - rho
        if (tau <= 0).any():
            return None
        P[:2, :2] *= tau
    return P / P.sum()


def fit_poisson(tabs: list, dixon_coles: bool = False) -> np.ndarray:
    """Poisson (or Dixon–Coles) maximising Σ t log q over the market tables."""
    def loss(x):
        P = poisson_grid(math.exp(x[0]), math.exp(x[1]),
                         x[2] if dixon_coles else 0.0)
        return 1e6 if P is None else -table_fit(P, tabs)

    x0 = [math.log(1.3), math.log(1.0)] + ([0.0] if dixon_coles else [])
    best = None
    for start in (x0, [math.log(2.0), math.log(0.6)] + x0[2:],
                  [math.log(0.6), math.log(2.0)] + x0[2:]):
        r = minimize(loss, start, method="Nelder-Mead",
                     options={"xatol": 1e-7, "fatol": 1e-10, "maxiter": 4000})
        if best is None or r.fun < best.fun:
            best = r
    x = best.x
    return poisson_grid(math.exp(x[0]), math.exp(x[1]),
                        x[2] if dixon_coles else 0.0)


# ── dict ↔ array ─────────────────────────────────────────────────────────────

def to_array(probs: dict) -> np.ndarray:
    P = np.zeros((G, G))
    for (h, k), p in probs.items():
        P[min(h, G - 1), min(k, G - 1)] += p
    return P


def to_dict(P: np.ndarray) -> dict:
    return {(h, k): float(P[h, k]) for h in range(G) for k in range(G)}
