"""zone2_perks.py - Module 3d (Perks): Zone II (ages omega1 .. omega2), Perks law, Poisson MLE by Nelder-Mead.

Model (spec 3d; methodology doc, sections 3.4 and 3.8)::

    mu_x = (A + B c^x) / (1 + D c^x),        A >= 0, B > 0, D >= 0, c > 1
    D_x ~ Poisson(E_x mu_x)                  ell(A, B, c, D) = sum_x [ D_x ln(mu_x) - E_x mu_x ]

A logistic form of the force of mortality: where D c^x << 1 it is Makeham, and as x grows the denominator catches up
and mu is pulled towards the finite plateau B / D instead of exploding, so Perks is the model that can DECELERATE.
D = 0 gives Makeham back, hence its maximum log-likelihood can never be below Makeham's on the same ages. D >= 0
means Perks can only slow the growth down: on data that accelerate beyond Makeham the optimum sits ON the wall D = 0
and the fit equals Makeham's (France 2020, ages 20-85, is such a case: an exhaustive global search agrees).

Estimation, `fit_perks` (spec 3d: "Nelder-Mead via scipy.optimize.minimize, objective -ell(theta)", "initialisation:
partir des parametres Makeham estimes + perturbation +-10% pour les parametres additionnels")::

    seed   = the Makeham fit (A, B, c), plus the extra parameter D set so that D c^x is 10% at the oldest age
    simplex = the seed and one vertex per parameter, each moved by about 10%
    Nelder-Mead on the half deviance (= -ell up to a constant), restarted from the best point while it improves

The derivative-free route is the specs' choice: with a fractional form and four parameters the Newton Hessian is badly
conditioned (doc 3.8). Choices made here, flagged to the user:

  * COORDINATES. The optimiser works on unconstrained, O(1)-scaled coordinates, with the age centred on the mean age of
    the deaths (xbar), b = B c^xbar and d = D c^xbar: that removes the extreme correlation between ln B and ln c, which
    otherwise makes the simplex crawl along a curved valley. The model is the same; B = b c^-xbar and D = d c^-xbar are
    reported.
        A = sA th_A^2 (sA = the crude overall rate),  b = exp(th_b),  c = exp(exp(th_c)),  D c^xbar = d = th_D^2 / w_max
    A and D are SQUARES so that they can reach exactly 0 and the optimiser converges there; with exp() they drift towards
    1e-17 for thousands of iterations (the lab's version needed 2200, here ~500). Nelder-Mead stops within its tolerance
    of a wall, never on it, so A (or D) is reported as exactly 0 when the model constrained to 0 - the other parameters
    re-fitted (`fit_common.refit_with_zero`) - is within `cfg.nr_tol` of the free fit: the curve is the same to that
    precision and the report says what the data support.
  * OBJECTIVE. The half deviance rather than -ell: same minimiser, but -ell is of the order of 1e6 and its rounding noise
    would exceed the tolerance.
  * ITERATION CAP. Nelder-Mead needs ~500 to 2000 iterations here, whereas `cfg.nr_maxiter` (200) caps NEWTON
    iterations, so it is not used; `fit_common.NM_MAXITER` is. `cfg.nr_tol` is the tolerance on the objective.

Conventions of the zone modules (see zone1_poly, zone2_gompertz, fit_common): input = the rows of ONE sex and ONE zone,
output arrays have the input length and order, `loglik` is `fit_common.poisson_loglik`, `params['n_params']` is the
number of free parameters, impossible input raises ValueError (French message).
"""
from __future__ import annotations

import numpy as np

from src.config import FitResult, SmootherConfig
from src.fit_common import check_columns, half_deviance, nelder_mead, poisson_loglik, refit_with_zero
from src.zone2_makeham import fit_makeham

N_PARAMS = 4
SEED_DECELERATION = 0.1      # D c^x at the oldest age of the seed: Perks starts 10% away from Makeham there
SIMPLEX_STEP = 0.1           # about 10% on each parameter of the initial simplex


def fit_perks(ages, Dx, Ex, cfg: SmootherConfig | None = None) -> FitResult:
    """Zone II, method `perks`: mu_x = (A + B c^x) / (1 + D c^x), Poisson maximum likelihood by Nelder-Mead.

    Algorithm (spec 3d; see the module docstring for the coordinates and the departures from the letter of the spec)::

        1. seed = Makeham fit (A, B, c) of the same ages (`fit_makeham`, with the same cfg), and D with D c^x = 10% at
           the oldest age
        2. initial simplex: the seed and, for each of A, ln B, ln ln c, D, the seed moved by about 10%
        3. Nelder-Mead on the half deviance, restarted from the best point while a restart still improves it by
           more than cfg.nr_tol (at most `fit_common.NM_RESTARTS` restarts)
        4. back to (A, B, c, D); A, then D, are set to exactly 0 when the constrained re-fit is within cfg.nr_tol of
           the free one

    Every age with E > 0 enters the likelihood, ages with D = 0 included. Ages with E <= 0 or a non-finite D or E
    are ignored and still get a rate in the output: `mu_hat` is the Perks curve at EVERY input age.

    Parameters
    ----------
    ages, Dx, Ex : array-like, shape (n,)
        The rows of ONE sex and ONE zone (both bounds included), one row per age; Dx >= 0 (non-integer allowed).
    cfg : SmootherConfig, optional
        Reads `nr_tol` (tolerance of the objective, default 1e-8) and is handed to the Makeham seed fit
        (`nr_maxiter`, `nr_tol`). The Nelder-Mead caps are those of `fit_common`, not settings of the app.

    Returns
    -------
    FitResult
        method='perks', zone='zone2'; `mu_hat` at every input age; `params` = {'A', 'B', 'c', 'D', 'n_params': 4}
        (floats, plain-age basis of the spec; A = 0.0 or D = 0.0 exactly when the constraint binds); `loglik` the full
        Poisson log-likelihood; `converged` True when Nelder-Mead met its tolerances and a final restart gained nothing;
        `n_iter` the Nelder-Mead iterations summed over its runs; `message` a French diagnostic (iterations, runs, the
        Makeham seed, and a warning when A or D sits on 0 - D = 0 means Perks has reduced to Makeham).

    Raises
    ------
    ValueError
        Columns of different lengths, a duplicated or non-finite age, negative deaths, fewer than 4 ages with E > 0
        (four parameters), or deaths at fewer than 2 distinct ages.
    """
    cfg = cfg if cfg is not None else SmootherConfig()
    x, d, e = check_columns(ages, Dx, Ex)
    use = np.isfinite(d) & np.isfinite(e) & (e > 0)
    if int(use.sum()) < N_PARAMS or np.unique(x[use & (d > 0)]).size < 2:
        raise ValueError(f"Perks : {int(use.sum())} âge(s) avec exposition, il en faut au moins {N_PARAMS}, "
                         f"avec des décès à au moins 2 âges différents (zone II trop courte, ou presque aucun décès).")
    xu, du, eu = x[use], d[use], e[use]

    seed = fit_makeham(xu, du, eu, cfg)
    A_s, B_s, c_s = (seed.params[k] for k in ("A", "B", "c"))
    xbar = float(np.average(xu, weights=du))
    s_a = du.sum() / eu.sum()                                          # scale of A: the crude overall rate
    w_max = c_s ** (xu.max() - xbar)                                   # w at the oldest age, for the seed's c

    def unpack(th):                                                    # -> A, b, c, d (b = B c^xbar, d = D c^xbar)
        return s_a * th[0] ** 2, np.exp(th[1]), np.exp(np.exp(th[2])), th[3] ** 2 / w_max

    def curve(th, ages_):
        A, b, c, dd = unpack(th)
        w = c ** (ages_ - xbar)
        return (A + b * w) / (1.0 + dd * w)

    def objective(th):
        with np.errstate(over="ignore", invalid="ignore"):
            return half_deviance(du, eu, curve(th, xu))

    th0 = np.array([np.sqrt(A_s / s_a), np.log(B_s * c_s ** xbar), np.log(np.log(c_s)), np.sqrt(SEED_DECELERATION)])
    steps = SIMPLEX_STEP * np.array([max(th0[0], 0.1), 1.0, 1.0, max(th0[3], 0.1)])
    simplex = np.vstack([th0] + [th0 + (-1.0) ** i * steps[i] * np.eye(N_PARAMS)[i] for i in range(N_PARAMS)])
    nm = nelder_mead(objective, simplex, fatol=cfg.nr_tol)

    th = nm.x.copy()
    for zero in (0, 3):                                                # A and D can sit exactly on their wall
        on_wall = refit_with_zero(objective, th, zero, tol=cfg.nr_tol) if th[zero] != 0.0 else None
        th = th if on_wall is None else on_wall
    A, b, c, dd = unpack(th)
    D, B = dd / c ** xbar, b / c ** xbar
    with np.errstate(over="ignore", invalid="ignore"):
        w_all = c ** (x - xbar)
        mu_hat = (A + b * w_all) / (1.0 + dd * w_all)

    message = (f"Nelder-Mead sur {N_PARAMS} paramètres : {nm.n_iter} itérations en {nm.runs} lancer(s), départ Makeham "
               f"(A = {A_s:.3g}, c = {c_s:.5f}) ; c = {c:.5f}.")
    if D == 0.0:
        message += " D = 0 (borne) : Perks se réduit à Makeham, les données ne montrent pas de décélération."
    if A == 0.0:
        message += " A = 0 (borne)."
    if not nm.converged:
        message += " ATTENTION : Nelder-Mead n'a pas convergé (tolérances non atteintes ou redémarrages épuisés)."
    n_out = x.size - int(use.sum())
    if n_out:
        message += f" {n_out} âge{'s' if n_out > 1 else ''} ignoré{'s' if n_out > 1 else ''} (Ex = 0 ou valeur manquante)."

    return FitResult(
        method="perks", zone="zone2", ages=x, mu_hat=mu_hat,
        params={"A": float(A), "B": float(B), "c": float(c), "D": float(D), "n_params": N_PARAMS},
        loglik=poisson_loglik(d, e, mu_hat),
        converged=nm.converged, n_iter=nm.n_iter, message=message)
