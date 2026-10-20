"""zone2_makeham2.py - Module 3d (Makeham 2nd order): Zone II (ages omega1 .. omega2), Poisson MLE by Nelder-Mead.

Model (spec 3d; methodology doc, sections 3.3 and 3.8)::

    mu_x = A + B c^x + D d^x,        A >= 0, B > 0, c > 1, D >= 0, d > 1, d != c      (5 parameters)
    D_x ~ Poisson(E_x mu_x)          ell(A, B, c, D, d) = sum_x [ D_x ln(mu_x) - E_x mu_x ]

Makeham with a second exponential term, to follow the transition towards the oldest ages at the price of two more
parameters (the specs call it "Makeham 2e ordre"; the methodology doc writes the second base `e`, the spec `d`; `d` is
used here). D = 0 gives Makeham back, so its maximum log-likelihood can never be below Makeham's on the same ages.

The two exponentials play symmetric roles (swap (B, c) with (D, d) and the curve is the same), so the fit is reported in
a canonical order, c <= d: (B, c) is the slower-growing term, (D, d) the steeper one.

Estimation, `fit_makeham2` (spec 3d: "Nelder-Mead via scipy.optimize.minimize, objective -ell(theta)", "initialisation:
partir des parametres Makeham estimes + perturbation +-10% pour les parametres additionnels")::

    seed    = the Makeham fit (A, B, c), plus a second exponential with base d = 1.1 c whose amplitude is 10% of the
              Makeham term at the oldest age
    simplex = the seed and one vertex per parameter, each moved by about 10%
    Nelder-Mead on the half deviance (= -ell up to a constant), restarted from the best point while it improves

Choices made here, flagged to the user (they mirror zone2_perks, whose docstring gives the reasons):

  * COORDINATES. Unconstrained, O(1)-scaled, with the age centred on the mean age of the deaths (xbar), b = B c^xbar and
    g = D d^xbar:   A = sA th_A^2,  b = exp(th_b),  c = exp(exp(th_c)),  g = th_g^2 * (b w_max / v_max),
    d = exp(exp(th_d)), with w = c^(x - xbar), v = d^(x - xbar), the subscript max meaning the oldest age of the seed.
    A and D are SQUARES (reachable exactly, no drift to 1e-17); one unit of th_g is a second term as large as the first at
    the oldest age. B = b c^-xbar and D = g d^-xbar are reported. Nelder-Mead stops within its tolerance of a wall, never
    on it (on Makeham data it ends with a tiny D and a base d almost equal to c, the two terms nearly merged), so A (or
    D) is reported as exactly 0 when the model constrained to 0 - the other parameters re-fitted
    (`fit_common.refit_with_zero`) - is within `cfg.nr_tol` of the free fit.
  * SEED. The second term must start small (10% at the oldest age) and steeper than the first: a second term that
    starts large sends the simplex to the label-swapped copy of the same optimum or to a worse local one.
  * ITERATION CAP. `fit_common.NM_MAXITER` per run, not `cfg.nr_maxiter` (a Newton cap, 200); `cfg.nr_tol` is the
    tolerance of the objective.

Conventions of the zone modules (see zone1_poly, zone2_gompertz, fit_common): input = the rows of ONE sex and ONE zone,
output arrays have the input length and order, `loglik` is `fit_common.poisson_loglik`, `params['n_params']` is the
number of free parameters, impossible input raises ValueError (French message).
"""
from __future__ import annotations

import numpy as np

from src.config import FitResult, SmootherConfig
from src.fit_common import check_columns, half_deviance, nelder_mead, poisson_loglik, refit_with_zero
from src.zone2_makeham import fit_makeham

N_PARAMS = 5
SEED_BASE_RATIO = 1.1        # d = 1.1 c at the seed: a second exponential clearly steeper than the first
SEED_AMPLITUDE = 0.1         # D d^x = 10% of B c^x at the oldest age of the seed
SIMPLEX_STEP = 0.1           # about 10% on each parameter of the initial simplex


def _order(b: float, c: float, g: float, f: float) -> tuple[float, float, float, float]:
    """The two exponentials are interchangeable: return (b, c, g, f) with the slower-growing one first (c <= f).

    Nelder-Mead may end with the terms either way round (it did in the lab, whose seed put the steep term first), and
    the reported (B, c) and (D, d) must not depend on that accident of the search.
    """
    return (g, f, b, c) if c > f else (b, c, g, f)


def fit_makeham2(ages, Dx, Ex, cfg: SmootherConfig | None = None) -> FitResult:
    """Zone II, method `makeham2`: mu_x = A + B c^x + D d^x, Poisson maximum likelihood by Nelder-Mead.

    Algorithm (spec 3d; see the module docstring for the coordinates and the departures from the letter of the spec)::

        1. seed = Makeham fit (A, B, c) of the same ages (`fit_makeham`, with the same cfg), and a second exponential
           d = 1.1 c with D d^x = 10% of B c^x at the oldest age
        2. initial simplex: the seed and, for each of A, ln B, ln ln c, D, ln ln d, the seed moved by about 10%
        3. Nelder-Mead on the half deviance, restarted from the best point while a restart still improves it by
           more than cfg.nr_tol (at most `fit_common.NM_RESTARTS` restarts)
        4. back to (A, B, c, D, d), ordered so that c <= d; A, then D, are set to exactly 0 when the constrained
           re-fit is within cfg.nr_tol of the free one

    Every age with E > 0 enters the likelihood, ages with D = 0 included. Ages with E <= 0 or a non-finite D or E
    are ignored and still get a rate in the output: `mu_hat` is the fitted curve at EVERY input age.

    Parameters
    ----------
    ages, Dx, Ex : array-like, shape (n,)
        The rows of ONE sex and ONE zone (both bounds included), one row per age; Dx >= 0 (non-integer allowed).
    cfg : SmootherConfig, optional
        Reads `nr_tol` (tolerance of the objective, default 1e-8) and is handed to the Makeham seed fit.

    Returns
    -------
    FitResult
        method='makeham2', zone='zone2'; `mu_hat` at every input age; `params` = {'A', 'B', 'c', 'D', 'd',
        'n_params': 5} (floats, plain-age basis, c <= d; A = 0.0 or D = 0.0 exactly when the constraint binds);
        `loglik` the full Poisson log-likelihood; `converged` True when Nelder-Mead met its tolerances and a final
        restart gained nothing; `n_iter` the Nelder-Mead iterations summed over its runs; `message` a French
        diagnostic. When D = 0 the base d is meaningless (the term vanishes): the fit has reduced to Makeham.

    Raises
    ------
    ValueError
        Columns of different lengths, a duplicated or non-finite age, negative deaths, fewer than 5 ages with E > 0
        (five parameters), or deaths at fewer than 2 distinct ages.
    """
    cfg = cfg if cfg is not None else SmootherConfig()
    x, d, e = check_columns(ages, Dx, Ex)
    use = np.isfinite(d) & np.isfinite(e) & (e > 0)
    if int(use.sum()) < N_PARAMS or np.unique(x[use & (d > 0)]).size < 2:
        raise ValueError(f"Makeham 2e ordre : {int(use.sum())} âge(s) avec exposition, il en faut au moins {N_PARAMS}, "
                         f"avec des décès à au moins 2 âges différents (zone II trop courte, ou presque aucun décès).")
    xu, du, eu = x[use], d[use], e[use]

    seed = fit_makeham(xu, du, eu, cfg)
    A_s, B_s, c_s = (seed.params[k] for k in ("A", "B", "c"))
    d_s = SEED_BASE_RATIO * c_s
    xbar = float(np.average(xu, weights=du))
    s_a = du.sum() / eu.sum()                                          # scale of A: the crude overall rate
    x_top = xu.max() - xbar
    b_s = B_s * c_s ** xbar
    s_g = b_s * c_s ** x_top / d_s ** x_top                            # g v_max = b w_max for th_g = 1

    def unpack(th):                                                    # -> A, b, c, g, f (g = D d^xbar, f = d)
        return s_a * th[0] ** 2, np.exp(th[1]), np.exp(np.exp(th[2])), s_g * th[3] ** 2, np.exp(np.exp(th[4]))

    def curve(th, ages_):
        A, b, c, g, f = unpack(th)
        return A + b * c ** (ages_ - xbar) + g * f ** (ages_ - xbar)

    def objective(th):
        with np.errstate(over="ignore", invalid="ignore"):
            return half_deviance(du, eu, curve(th, xu))

    th0 = np.array([np.sqrt(A_s / s_a), np.log(b_s), np.log(np.log(c_s)), np.sqrt(SEED_AMPLITUDE), np.log(np.log(d_s))])
    steps = SIMPLEX_STEP * np.array([max(th0[0], 0.1), 1.0, 1.0, max(th0[3], 0.1), 1.0])
    simplex = np.vstack([th0] + [th0 + (-1.0) ** i * steps[i] * np.eye(N_PARAMS)[i] for i in range(N_PARAMS)])
    nm = nelder_mead(objective, simplex, fatol=cfg.nr_tol)

    th = nm.x.copy()
    for zero, fixed in ((0, ()), (3, (4,))):                           # A and D can sit exactly on their wall
        on_wall = refit_with_zero(objective, th, zero, fixed, tol=cfg.nr_tol) if th[zero] != 0.0 else None
        th = th if on_wall is None else on_wall
    A, b, c, g, f = unpack(th)
    b, c, g, f = _order(b, c, g, f)                                    # canonical order: c <= d
    B, D = b / c ** xbar, g / f ** xbar
    with np.errstate(over="ignore", invalid="ignore"):
        mu_hat = A + b * c ** (x - xbar) + g * f ** (x - xbar)

    message = (f"Nelder-Mead sur {N_PARAMS} paramètres : {nm.n_iter} itérations en {nm.runs} lancer(s), départ Makeham "
               f"(A = {A_s:.3g}, c = {c_s:.5f}) ; c = {c:.5f}, d = {f:.5f}.")
    if D == 0.0:
        message += " D = 0 (borne) : le second terme disparaît, Makeham 2e ordre se réduit à Makeham."
    if A == 0.0:
        message += " A = 0 (borne)."
    if not nm.converged:
        message += " ATTENTION : Nelder-Mead n'a pas convergé (tolérances non atteintes ou redémarrages épuisés)."
    n_out = x.size - int(use.sum())
    if n_out:
        message += f" {n_out} âge{'s' if n_out > 1 else ''} ignoré{'s' if n_out > 1 else ''} (Ex = 0 ou valeur manquante)."

    return FitResult(
        method="makeham2", zone="zone2", ages=x, mu_hat=mu_hat,
        params={"A": float(A), "B": float(B), "c": float(c), "D": float(D), "d": float(f), "n_params": N_PARAMS},
        loglik=poisson_loglik(d, e, mu_hat),
        converged=nm.converged, n_iter=nm.n_iter, message=message)
