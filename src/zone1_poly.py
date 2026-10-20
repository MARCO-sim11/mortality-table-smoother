"""zone1_poly.py - Module 3a: Zone I (ages 0 .. omega1), cubic polynomial and natural cubic spline in ln(mu).

Childhood mortality is not monotone (infant spike, minimum around 9-12, then a rise), which rules out
Gompertz/Makeham there. Both methods fit a flexible curve to the LOG of the crude rate by weighted
least squares (methodology doc, section 2):

    y_x = ln(m_x_raw) = ln(D_x / E_x),      w_x = D_x,      a_hat = (A' W A)^-1 A' W y

The weight D_x is the inverse of Var[ln m_x] ~ 1/D_x (delta method), so this is WLS with the
heteroscedasticity corrected, and the solution is closed form: one small linear solve, no iteration.
`A` is the design matrix: [1, x, x^2, x^3] for `poly3`, the natural-spline basis for `spline`.

Public API (spec 3, "fit_*(ages, Dx, Ex, cfg) -> FitResult", one function per method key of config.METHODS):

    fit_poly3(ages, Dx, Ex, cfg)    -> FitResult(method='poly3',  zone='zone1', ...)
    fit_spline(ages, Dx, Ex, cfg)   -> FitResult(method='spline', zone='zone1', ...)

Conventions shared with the other zone modules (to be kept when they are written):

  * Input = the rows of ONE sex and ONE zone, bounds included (`config.zone_slice`). Output arrays have
    exactly the input length and order, so the joiner can stack the zones.
  * `loglik` is the full Poisson log-likelihood of `fit_common.poisson_loglik` (ages with D_x = 0 included: they
    contribute -E_x mu_x). The WLS estimate does not maximise it, but evaluating it there puts zone I on the
    same scale as the Poisson-MLE zone II models, which is what AIC/BIC (evaluator) need.
  * Closed-form methods report `n_iter = 0` and `converged = True` when the solution is finite.
  * `params['n_params']` is the number of free parameters of THAT fit (the evaluator's k for AIC/BIC).
    For `spline` it depends on the knots actually placed, so config.METHODS cannot give it.
  * A fit that cannot be done (too few usable ages) raises ValueError with a French message; the app turns
    that into an "erreur" row for the method without stopping the others.
"""
from __future__ import annotations

from math import comb

import numpy as np
from scipy.interpolate import CubicSpline

from src.config import FitResult, SmootherConfig
from src.fit_common import check_columns, poisson_loglik

DEGREE = 3          # polynomial degree of `poly3` (spec 3a)
KNOT_STEP = 5       # spline rule of spec 3a: one knot every 5 ages
MIN_KNOTS = 3       # fewer knots and a natural cubic spline is just a straight line


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------


def _prepare(ages, Dx, Ex):
    """Check the three columns (`fit_common.check_columns`) and flag the ages that can enter a log-rate fit.

    Returns (x, d, e, ok): float arrays in the caller's order, and `ok`, a boolean mask that is True where
    ln(D/E) is defined and weighted: D > 0, E > 0 and both finite. Ages with D = 0 are therefore left out of
    the WLS (their weight D would be 0 anyway, spec 3a) but stay in the arrays, so the curve is still
    evaluated there.
    """
    x, d, e = check_columns(ages, Dx, Ex)
    return x, d, e, np.isfinite(d) & np.isfinite(e) & (d > 0) & (e > 0)


def _weighted_solve(A: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Closed-form WLS, a_hat = (A' W A)^-1 A' W y with W = diag(w), through `np.linalg.solve` (spec 3a:
    no `np.polyfit`, no explicit inverse). Builds A'W by scaling the columns of A', never the n x n W."""
    AtW = A.T * w
    return np.linalg.solve(AtW @ A, AtW @ y)


def _count(n: int, singular: str, plural: str) -> str:
    return f"{n} {singular if n == 1 else plural}"


# --------------------------------------------------------------------------
# 3a - poly3: cubic polynomial in ln(mu)
# --------------------------------------------------------------------------


def _to_age_basis(b: np.ndarray, centre: float, half: float) -> np.ndarray:
    """Re-express  sum_k b_k ((x - centre)/half)^k  as  sum_j a_j x^j  (binomial expansion).

        a_j = sum_{k >= j} b_k * C(k, j) * (-centre)^(k - j) / half^k
    """
    a = np.zeros_like(b)
    for j in range(b.size):
        for k in range(j, b.size):
            a[j] += b[k] * comb(k, j) * (-centre) ** (k - j) / half ** k
    return a


def fit_poly3(ages, Dx, Ex, cfg: SmootherConfig | None = None) -> FitResult:
    """Zone I, method `poly3`: ln(mu_x) = a0 + a1 x + a2 x^2 + a3 x^3, estimated by weighted least squares.

    Model and estimator (spec 3a; methodology doc, section 2.1)::

        mu_x   = exp(a0 + a1 x + a2 x^2 + a3 x^3)
        a_hat  = (A' W A)^-1 A' W y,   A = [1, x, x^2, x^3],   W = diag(D_x),   y_x = ln(D_x / E_x)

    The solve is `np.linalg.solve` on the 4 x 4 normal equations; there is no `np.polyfit` and no
    iteration. For conditioning only, the ages are centred and scaled to [-1, 1] before the solve; this
    changes nothing mathematically, and the coefficients returned are converted back to the plain-age basis
    of the spec. (Measured on exact cubic data: the raw solve keeps 12 digits on ages 0-59 but only 8 on a
    window such as 60-109, where cond(A'WA) reaches 1e18; the scaled solve keeps 14 everywhere.)

    Ages with D_x = 0 (or E_x = 0) are excluded from the fit, since ln(0) is undefined (w_x = D_x = 0
    cancels them anyway). They are still in the output: `mu_hat` is the polynomial evaluated at EVERY
    input age, so a zero-death age gets a smoothed rate.

    Parameters
    ----------
    ages : array-like, shape (n,)
        Ages of the zone (rows of ONE sex, both bounds included, one row per age). Order is kept.
    Dx : array-like, shape (n,)
        Observed deaths, >= 0 (non-integer values from rebuilt deaths are accepted).
    Ex : array-like, shape (n,)
        Central exposure to risk, same length. Ages with Ex <= 0 are excluded from the fit.
    cfg : SmootherConfig, optional
        Accepted for the common `fit_*` protocol (spec 3) and not used: zone I needs no setting beyond the
        age range, which `config.zone_slice` already applied (omega1 is the last row).

    Returns
    -------
    FitResult
        method='poly3', zone='zone1'; `ages` the input ages as floats; `mu_hat` exp(polynomial) at each age;
        `params` = {'a0', 'a1', 'a2', 'a3'} (plain-age basis, floats) and 'n_params' = 4; `loglik` the
        Poisson log-likelihood at the WLS estimate (see the module docstring); `converged` True when the
        solution is finite; `n_iter` 0 (closed form); `message` a French diagnostic with the number of ages
        used and excluded.

    Raises
    ------
    ValueError
        Columns of different lengths, a duplicated or non-finite age, negative deaths, or fewer than 4
        distinct ages with D > 0 and E > 0 (a degree-3 polynomial has 4 coefficients).
    """
    x, d, e, ok = _prepare(ages, Dx, Ex)
    n_used = int(ok.sum())
    if np.unique(x[ok]).size < DEGREE + 1:
        raise ValueError(f"Polynôme de degré {DEGREE} : {n_used} âge(s) avec décès et exposition, "
                         f"il en faut au moins {DEGREE + 1} (zone I trop courte, ou trop d'âges sans décès).")

    xs = x[ok]
    centre, half = 0.5 * (xs.max() + xs.min()), 0.5 * (xs.max() - xs.min())

    def design(ages_):                                    # [1, t, t^2, t^3] with t = (x - centre) / half
        t = (ages_ - centre) / half
        return np.column_stack([t ** k for k in range(DEGREE + 1)])

    b = _weighted_solve(design(xs), np.log(d[ok] / e[ok]), d[ok])
    mu_hat = np.exp(design(x) @ b)
    a = _to_age_basis(b, centre, half)

    n_out = x.size - n_used
    message = (f"WLS en forme fermée (poids = Dx) sur {_count(n_used, 'âge', 'âges')} sur {x.size}"
               + (f" ; {_count(n_out, 'âge exclu', 'âges exclus')} du fit (Dx = 0 ou Ex = 0), "
                  f"lissé(s) en sortie." if n_out else "."))
    return FitResult(
        method="poly3", zone="zone1", ages=x, mu_hat=mu_hat,
        params={**{f"a{j}": float(a[j]) for j in range(DEGREE + 1)}, "n_params": DEGREE + 1},
        loglik=poisson_loglik(d, e, mu_hat),
        converged=bool(np.all(np.isfinite(b)) and np.all(np.isfinite(mu_hat))),
        n_iter=0, message=message)


# --------------------------------------------------------------------------
# 3a - spline: natural cubic spline in ln(mu)
# --------------------------------------------------------------------------


def _knot_ages(x: np.ndarray, ok: np.ndarray, step: int) -> np.ndarray:
    """Knot ages of the spline: one every `step` ages, both ends of the zone included, usable ages only.

    The grid starts at the first age of the zone and moves by `step`. The last age of the zone is always a
    candidate, because the joiner needs mu at omega1: it replaces the last grid point when it is closer than
    half a step (zone 0..11 gives 0, 5, 11) and is appended otherwise (zone 0..13 gives 0, 5, 10, 13).
    A candidate age with D = 0 (or E = 0) is dropped, not moved (spec 3a: "exclus des noeuds").
    """
    if x.size == 0:
        return x
    lo, hi = x.min(), x.max()
    grid = np.arange(lo, hi + 1e-9, step)
    if grid[-1] < hi:
        if hi - grid[-1] < step / 2:
            grid[-1] = hi
        else:
            grid = np.append(grid, hi)
    return np.intersect1d(grid, x[ok])


def _natural_spline_eval(knots: np.ndarray, values: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Evaluate the natural cubic spline through (knots, values) at the ages `x`.

    `values` has shape (K,) or (K, m) (m splines at once; the identity gives the basis functions). Inside
    [knots[0], knots[-1]] this is `scipy.interpolate.CubicSpline(..., bc_type='natural')`. Outside it is
    continued by the straight line tangent at the end knot: that is what "natural" means (zero curvature
    beyond the boundary knots), whereas scipy's default would carry on the last cubic piece.
    """
    cs = CubicSpline(knots, values, axis=0, bc_type="natural")
    out = np.array(cs(x), dtype=float)
    for edge, i, outside in ((knots[0], 0, x < knots[0]), (knots[-1], -1, x > knots[-1])):
        if outside.any():
            dx = (x[outside] - edge).reshape((-1,) + (1,) * (out.ndim - 1))
            out[outside] = values[i] + cs(edge, 1) * dx
    return out


def fit_spline(ages, Dx, Ex, cfg: SmootherConfig | None = None, knot_step: int = KNOT_STEP) -> FitResult:
    """Zone I, method `spline`: natural cubic spline in ln(mu_x), values at the knots estimated by WLS.

    Model (spec 3a; methodology doc, section 2.1, "Variante")::

        ln mu_x = S(x),   S = natural cubic spline with knots tau_1 < ... < tau_K, S(tau_k) = beta_k
        beta_hat = (A' W A)^-1 A' W y,   W = diag(D_x),   y_x = ln(D_x / E_x)
        A[x, k]  = value at x of the natural spline equal to 1 at tau_k and 0 at the other knots

    The spec asks for `scipy.interpolate.CubicSpline(bc_type='natural')` with one knot every 5 ages, and the
    methodology doc says that estimation stays WLS "with the same structure, A being the spline basis". The
    two are joined here: the columns of A are the cardinal natural splines, built with `CubicSpline`, and
    beta is solved with `np.linalg.solve`. Interpolating the raw ln(m_x) at the knots instead would copy the
    sampling noise of single ages into the curve; with WLS the curve is pulled by the ages with many deaths
    (infants) and averages the noisy ones. When every knot is also an observed age and there are no other
    ages, the two coincide. A natural spline with K knots has K degrees of freedom, hence
    `params['n_params'] = K`.

    Knots (spec 3a: "1 noeud tous les 5 ages", D = 0 ages "exclus des noeuds mais interpoles en sortie"):
    from the first age of the zone every `knot_step` ages, plus the last age (see `_knot_ages`), keeping only
    ages with D > 0 and E > 0. Ages with D = 0 are out of the fit and out of the knots, but `mu_hat` is the
    spline evaluated at EVERY input age; beyond the first/last knot (a zone whose end age has no death) the
    curve continues as a straight line in ln(mu), as a natural spline does.

    Parameters
    ----------
    ages, Dx, Ex : array-like, shape (n,)
        Same contract as `fit_poly3`: the rows of ONE sex and ONE zone, one row per age.
    cfg : SmootherConfig, optional
        Accepted for the common `fit_*` protocol and not used (the knot rule is fixed by the spec).
    knot_step : int, default 5
        Spacing of the knot grid, in ages (>= 1). Not a setting of the app; for experiments and tests.

    Returns
    -------
    FitResult
        method='spline', zone='zone1'; `mu_hat` exp(S(x)) at each input age; `params` = {'knots': [ages],
        'ln_mu_knots': [fitted ln(mu) at the knots], 'n_params': K}; `loglik`, `converged`, `n_iter`
        and `message` as in `fit_poly3` (`message` lists the knots).

    Raises
    ------
    ValueError
        Same input checks as `fit_poly3`, `knot_step` < 1, or fewer than 3 knots (zone shorter than about
        10 ages, or too many knot ages without deaths): the spline would degenerate into a line.
    """
    if knot_step < 1:
        raise ValueError("knot_step doit valoir au moins 1.")
    x, d, e, ok = _prepare(ages, Dx, Ex)
    knots = _knot_ages(x, ok, knot_step)
    if knots.size < MIN_KNOTS:
        raise ValueError(f"Spline naturelle : {_count(knots.size, 'noeud', 'noeuds')} (un tous les {knot_step} âges, "
                         f"âges sans décès exclus), il en faut au moins {MIN_KNOTS} (zone I trop courte, "
                         f"ou trop d'âges sans décès).")

    xs = x[ok]
    A = _natural_spline_eval(knots, np.eye(knots.size), xs)          # (n_used, K) cardinal basis
    beta = _weighted_solve(A, np.log(d[ok] / e[ok]), d[ok])
    mu_hat = np.exp(_natural_spline_eval(knots, beta, x))

    n_out = x.size - int(ok.sum())
    message = (f"WLS en forme fermée (poids = Dx), spline cubique naturelle à {knots.size} nœuds "
               f"({', '.join(f'{k:g}' for k in knots)})"
               + (f" ; {_count(n_out, 'âge exclu', 'âges exclus')} du fit et des nœuds (Dx = 0 ou Ex = 0), "
                  f"interpolé(s) en sortie." if n_out else "."))
    return FitResult(
        method="spline", zone="zone1", ages=x, mu_hat=mu_hat,
        params={"knots": knots.tolist(), "ln_mu_knots": beta.tolist(), "n_params": int(knots.size)},
        loglik=poisson_loglik(d, e, mu_hat),
        converged=bool(np.all(np.isfinite(beta)) and np.all(np.isfinite(mu_hat))),
        n_iter=0, message=message)
