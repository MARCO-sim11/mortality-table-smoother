"""joiner.py - Module 4: the three zone curves become one continuous mu_x on [0, omega3], then a life table.

Spec 4.1: "assembler les courbes des trois zones en une table mu_x continue sur [0, omega3] en appliquant une interpolation
douce au voisinage de chaque frontiere (omega1, omega2)". The steps, in the order they run in `join_zones`::

    1. CHECK        the three results are zone I, II, III of the same config; 0 <= omega1 < omega2 < omega3; sigma > 0;
    2. AGE GRID     0, 1, ..., omega3  (spec 4.3: "tableau de longueur omega3 + 1, indices = ages");
    3. WEIGHTS      w_i(x) = (1 + tanh((omega_i - x) / sigma)) / 2 on the window [omega_i - 2 sigma, omega_i + 2 sigma]
                    (spec 4.2), 1 below it, 0 above it: `blend_weight`;
    4. CURVES       each zone's mu at the ages where its weight is not 0: its own `mu_hat` where it has one, else its
                    fitted formula evaluated from `params`: `evaluate_curve`;
    5. BLEND        mu = w1 mu_I + (1 - w1) [ w2 mu_II + (1 - w2) mu_III ], the spec's
                    mu_joint = w mu_left + (1 - w) mu_right applied at omega1 then at omega2;
    6. LEADING GAP  ages below the first age of zone I (a table that starts at 20, say) are NaN, not invented;

then `mu_to_q` (spec 4.4: q = 1 - exp(-mu)) and `build_life_table` (spec 4.4: columns age, mu, qx, px, lx, dx, ex).

Why `evaluate_curve` exists. The window around omega1 reaches omega1 + 2 sigma, past the ages zone I was fitted on, and
down to omega1 - 2 sigma, before zone II starts: a `FitResult` only stores `mu_hat` on its own ages, so the joiner has to
evaluate the fitted formulas elsewhere, from `params` (all nine methods; each is checked against its own `mu_hat`).
Where a zone is used "as is" (outside the windows, spec 4.2) the value returned is its stored `mu_hat`, bit for bit.

Decisions the specs leave open or contradict, made here and flagged to the user:

  * CONTINUOUS EDGES. With the literal weight, w(omega - 2 sigma) = 0.982, not 1: cutting the window there leaves a step of
    1.8% of the gap between the two zones at each window edge. By default the tanh is rescaled inside the window,
    w = (1 + tanh(u) / tanh(2)) / 2 with u = (omega - x) / sigma, so that it reaches exactly 1 and 0 at the edges: same
    window, same midpoint (w = 1/2 at omega), continuous (with a slope break of 1.2% of the gap per age at sigma = 3).
    Measured on integer ages, the largest second difference of the weight is, in % of the gap (literal / continuous):
    4.1 / 4.3 at the default sigma = 3 (the steep middle dominates: no difference), but 1.8 / 0.7 at sigma = 7.5 and
    1.8 / 0.4 at sigma = 10, where the literal edge step is the largest irregularity of the table. So the choice only
    matters for a wide window; `CONTINUOUS_EDGES = False` restores the spec's formula literally.
  * THREE ZONES, TWO WINDOWS. The spec blends two curves at a time; for three zones the weights are the products
    I: w1, II: (1 - w1) w2, III: (1 - w1)(1 - w2), which sum to 1 at every age, so the windows may overlap
    (sigma large, zone II short: `validate_config` warns) without anything breaking.
  * LIFE TABLE. See `build_life_table`: radix 100 000, q = 1 - exp(-mu) (the spec says UDD, the formula is the constant-force
    bridge), L_x with a_x = 1/2 (UDD, as in the lab's table_builder), and the survivors beyond omega3 live on at the constant
    force mu(omega3) instead of being dropped.

Not here (module 5, evaluator + tab 3): choosing which method feeds each zone, and the "Valider la selection" button that
calls `join_zones` and `build_life_table` and fills `mu_joint` and `life_table` of the session state.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import expit

from src.config import ZONE_ALL, FitResult, SmootherConfig
from src.zone1_poly import _natural_spline_eval

WINDOW = 2.0                 # half-width of the transition window, in units of sigma (spec 4.2)
CONTINUOUS_EDGES = True      # rescale the tanh so that the weight is exactly 1 / 0 at the window edges (see above)
RADIX = 100_000.0            # l_0 of the life table
LIFE_TABLE_COLUMNS = ["age", "mu", "qx", "px", "lx", "dx", "ex"]


# --------------------------------------------------------------------------
# Step 4: a zone's curve at any age, from its parameters
# --------------------------------------------------------------------------


def evaluate_curve(result: FitResult, ages) -> np.ndarray:
    """mu at `ages` of the curve a `FitResult` describes, computed from `result.method` and `result.params` alone.

    This is what lets the joiner read a zone outside the ages it was fitted on. Plain-age basis everywhere (the one the
    params are reported in), x the age, t = x - x0 for Kannisto::

        poly3         exp(a0 + a1 x + a2 x^2 + a3 x^3)
        spline        natural cubic spline through (knots, ln_mu_knots) in ln mu, straight line beyond the end knots
        spline_all    the same formula (the global spline: knots over all the ages)
        gompertz      exp(alpha + beta x)
        makeham       A + B c^x
        makeham2      A + B c^x + D d^x
        perks         (A + B c^x) / (1 + D c^x)
        kannisto      sigmoid(ln a + b (x - x0))
        beard         B c^x / (1 + K B c^x)
        coale_kisker  -ln(1 - q),  q = min(q_omega2 e^(k (x - x0)), 1 - 1e-15)

    The formulas are those of the zone modules written once more from the reported parameters; the tests check that each
    reproduces the method's own `mu_hat` on its ages, so the two cannot drift apart unnoticed. Overflow gives inf, never
    a warning (the joiner only evaluates where the blend weight is not 0).

    Raises ValueError for a method this module does not know.
    """
    x = np.asarray(ages, dtype=float)
    p, method = result.params, result.method
    with np.errstate(over="ignore", invalid="ignore"):
        if method == "poly3":
            return np.exp(p["a0"] + p["a1"] * x + p["a2"] * x ** 2 + p["a3"] * x ** 3)
        if method in ("spline", "spline_all"):
            return np.exp(_natural_spline_eval(np.asarray(p["knots"], float), np.asarray(p["ln_mu_knots"], float), x))
        if method == "gompertz":
            return np.exp(p["alpha"] + p["beta"] * x)
        if method == "makeham":
            return p["A"] + p["B"] * p["c"] ** x
        if method == "makeham2":
            return p["A"] + p["B"] * p["c"] ** x + p["D"] * p["d"] ** x
        if method == "perks":
            cx = p["c"] ** x
            return (p["A"] + p["B"] * cx) / (1.0 + p["D"] * cx)
        if method == "kannisto":
            return expit(np.log(p["a"]) + p["b"] * (x - p["x0"]))
        if method == "beard":
            z = p["B"] * p["c"] ** x
            return z / (1.0 + p["K"] * z)
        if method == "coale_kisker":
            q = np.minimum(p["q_omega2"] * np.exp(p["k"] * (x - p["x0"])), 1.0 - 1e-15)
            return -np.log1p(-q)
    raise ValueError(f"Raccord : méthode inconnue « {method} » (aucune formule pour évaluer sa courbe).")


def zone_values(result: FitResult, ages: np.ndarray) -> np.ndarray:
    """A zone's mu at `ages`: its stored `mu_hat` where it has one (used as is), `evaluate_curve` elsewhere.

    Used by `join_zones` and by the evaluator, which reads zone III on the data ages of its own rows."""
    stored = {float(a): m for a, m in zip(result.ages, result.mu_hat) if np.isfinite(m)}
    out = np.array([stored.get(float(a), np.nan) for a in ages], dtype=float)
    missing = ~np.isfinite(out)
    if missing.any():                                  # the formula only where nothing is stored
        out[missing] = evaluate_curve(result, np.asarray(ages, dtype=float)[missing])
    return out


# --------------------------------------------------------------------------
# Steps 1-3: weights
# --------------------------------------------------------------------------


def blend_weight(x, omega: float, sigma: float) -> np.ndarray:
    """Weight of the zone on the LEFT of `omega` at ages `x` (spec 4.2); the right zone has 1 - w.

    With u = (omega - x) / sigma::

        u >  2    -> 1       (below the window: the left zone alone)
        u < -2    -> 0       (above the window: the right zone alone)
        else         (1 + tanh(u) / tanh(2)) / 2      if CONTINUOUS_EDGES   (default: exactly 1 at u = 2, 0 at u = -2)
                     (1 + tanh(u)) / 2                otherwise             (the spec's formula: 0.982 / 0.018 at the edges)

    Either way w = 1/2 at omega and w(omega + d) = 1 - w(omega - d). Raises ValueError unless sigma > 0.
    """
    if not sigma > 0:
        raise ValueError("Raccord : σ doit être strictement positif.")
    u = (omega - np.asarray(x, dtype=float)) / sigma
    t = np.tanh(u)
    inside = 0.5 * (1.0 + (t / np.tanh(WINDOW) if CONTINUOUS_EDGES else t))
    return np.where(u > WINDOW, 1.0, np.where(u < -WINDOW, 0.0, inside))


# --------------------------------------------------------------------------
# The join (steps 1-6)
# --------------------------------------------------------------------------


def join_zones(z1: FitResult, z2: FitResult, z3: FitResult, cfg: SmootherConfig) -> np.ndarray:
    """Assemble the three zone curves into mu_x for x = 0, 1, ..., omega3 (spec 4.3): an array of length omega3 + 1.

    Steps (see the module docstring for the reasons)::

        w1 = blend_weight(x, omega1, sigma),  w2 = blend_weight(x, omega2, sigma),   sigma = cfg.sigma_join
        mu(x) = w1 mu_I(x) + (1 - w1) [ w2 mu_II(x) + (1 - w2) mu_III(x) ]

    Outside the windows [omega_i - 2 sigma, omega_i + 2 sigma] exactly one zone has weight 1, and its stored `mu_hat` is
    returned unchanged. Inside, the zone curves are evaluated from their parameters wherever they were not fitted
    (`evaluate_curve`): zone I a few ages beyond omega1, zone II on both sides of its range, zone III below omega2.
    A zone's curve is only evaluated where its weight is positive, so a polynomial that explodes far outside its zone
    cannot poison the result with 0 * inf.

    Ages below the first age of zone I (a table that does not start at 0) are NaN: no zone describes them.

    Parameters
    ----------
    z1, z2, z3 : FitResult
        The chosen result of zone I, II and III (`zone` = 'zone1', 'zone2', 'zone3'), for ONE sex. A global method's result
        (`zone` = 'all', e.g. the spline over all the ages) is accepted in any of the three places: it is the same curve.
    cfg : SmootherConfig
        Reads `omega1`, `omega2`, `omega3` and `sigma_join` (the half-width parameter sigma of the tanh; the window is
        2 sigma each side).

    Returns
    -------
    numpy.ndarray, shape (omega3 + 1,)
        mu at age = index. Pass it to `build_life_table`.

    Raises
    ------
    ValueError
        A result is not of the zone it is passed as, bounds not 0 <= omega1 < omega2 < omega3, sigma <= 0, a method
        `evaluate_curve` does not know, or a zone I result with no age.
    """
    for result, zone in ((z1, "zone1"), (z2, "zone2"), (z3, "zone3")):
        if result.zone not in (zone, ZONE_ALL):
            raise ValueError(f"Raccord : le résultat passé pour la {zone} est de la {result.zone} (méthode « {result.method} »).")
    if not 0 <= cfg.omega1 < cfg.omega2 < cfg.omega3:
        raise ValueError(f"Raccord : il faut 0 ≤ ω₁ < ω₂ < ω₃ (reçu : {cfg.omega1}, {cfg.omega2}, {cfg.omega3}).")
    if len(z1.ages) == 0:
        raise ValueError("Raccord : le résultat de la zone I ne couvre aucun âge.")

    ages = np.arange(0, cfg.omega3 + 1, dtype=float)
    w1 = blend_weight(ages, cfg.omega1, cfg.sigma_join)
    w2 = blend_weight(ages, cfg.omega2, cfg.sigma_join)
    mu = np.zeros_like(ages)
    for result, weight in ((z1, w1), (z2, (1.0 - w1) * w2), (z3, (1.0 - w1) * (1.0 - w2))):
        used = weight > 0.0
        if used.any():
            mu[used] += weight[used] * zone_values(result, ages[used])
    mu[ages < float(np.min(z1.ages))] = np.nan
    return mu


# --------------------------------------------------------------------------
# Spec 4.4: from mu to a life table
# --------------------------------------------------------------------------


def mu_to_q(mu) -> np.ndarray:
    """q = 1 - exp(-mu) (spec 4.4), through `expm1` so that a tiny mu keeps its precision (q = 1e-18, not 0).

    The spec says "sous hypothese UDD" but writes the constant-force bridge; that formula is what is used, here and in
    `join_zones`' consumers, so that Coale-Kisker's q round-trips exactly (zone3 gives mu = -ln(1 - q))."""
    return -np.expm1(-np.asarray(mu, dtype=float))


def build_life_table(mu_hat, radix: float = RADIX, open_tail: bool = True) -> pd.DataFrame:
    """The life table of a mu curve (spec 4.4): a DataFrame with columns [age, mu, qx, px, lx, dx, ex].

    `mu_hat[i]` is the force of mortality at age `age0 + i`, where age0 is the first age with a value (the output of
    `join_zones` starts at index 0, which is age 0; leading NaN are dropped, so a table starting at 20 starts at 20)::

        qx = 1 - exp(-mu)          px = exp(-mu) = 1 - qx
        l_first = radix,   l_{x+1} = l_x p_x,   d_x = l_x q_x
        L_x = l_{x+1} + d_x / 2    (person-years lived in the year of age x, deaths uniform: a_x = 1/2, as the lab's table)
        T_x = sum over y >= x of L_y,   e_x = T_x / l_x      (complete expectation of life at exact age x, as HMD's ex)

    The table stops at the last age of `mu_hat` WITHOUT forcing q = 1 there (omega3's q is 0.4-0.6 on a real table, and
    Coale-Kisker's target q(omega3) = 0.6 would be overwritten), so l_{omega3} p_{omega3} people are still alive at the end.
    They are not dropped from e_x: with `open_tail` (default) they live on at the constant force mu(omega3), which adds
    l_{last+1} / mu_last person-years to the last L, so that e_last = (1 + p) / 2 + p / mu_last (p = exp(-mu_last));
    with `open_tail=False` only the year they have just completed is counted (e_last = (1 + p) / 2), the finite truncation
    of the lab's table. Consequently sum(dx) = radix - l_{last+1}, not radix.

    Parameters
    ----------
    mu_hat : array-like
        Force of mortality by age, e.g. `join_zones`' result. Leading NaN are dropped; every remaining value must be
        finite and >= 0.
    radix : float
        l at the first age. Default 100 000.
    open_tail : bool
        See above.

    Raises
    ------
    ValueError
        Nothing left after dropping the leading NaN, a NaN or infinite value after the first age, a negative rate, or
        a non-positive radix.
    """
    if not radix > 0:
        raise ValueError("Table de vie : la racine (radix) doit être strictement positive.")
    mu_all = np.asarray(mu_hat, dtype=float).ravel()
    finite = np.isfinite(mu_all)
    if not finite.any():
        raise ValueError("Table de vie : aucun taux fini dans mu_hat.")
    first = int(np.argmax(finite))
    mu = mu_all[first:]
    if not np.all(np.isfinite(mu)):
        raise ValueError("Table de vie : taux manquant ou infini après le premier âge (trou dans la courbe).")
    if np.any(mu < 0):
        raise ValueError("Table de vie : taux de mortalité négatif.")

    qx = mu_to_q(mu)
    px = np.exp(-mu)
    lx = radix * np.concatenate(([1.0], np.cumprod(px[:-1])))
    dx = lx * qx
    l_next = lx * px                                                    # survivors one year later (beyond the table at the end)
    big_l = l_next + 0.5 * dx
    if open_tail and mu[-1] > 0:
        big_l[-1] += l_next[-1] / mu[-1]
    big_t = np.cumsum(big_l[::-1])[::-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        ex = np.where(lx > 0, big_t / lx, np.nan)
    return pd.DataFrame({"age": np.arange(first, first + mu.size, dtype="int64"), "mu": mu, "qx": qx, "px": px,
                         "lx": lx, "dx": dx, "ex": ex})[LIFE_TABLE_COLUMNS]
