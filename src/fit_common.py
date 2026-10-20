"""fit_common.py - what the zone modules (3a-3e) do the same way: check their three input columns, compute the
Poisson log-likelihood they report, and drive the Nelder-Mead optimiser the derivative-free methods share.

Not in the file list of the specs: it exists so that the zone modules share ONE definition of `loglik`
(methodology doc, section 3.5, "Log-vraisemblance commune (Poisson)") instead of each carrying a copy that
could drift, since the evaluator compares the log-likelihoods of different methods through AIC/BIC.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln

NM_MAXITER = 2000        # iterations of ONE Nelder-Mead run (the methodology table expects ~100, real data needs ~1000)
NM_XATOL = 1e-8          # simplex size, in the optimiser's own (unconstrained, O(1)-scaled) coordinates
NM_RESTARTS = 5          # extra runs from the best point while they still improve the objective


def check_columns(ages, Dx, Ex) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate the `fit_*` inputs and return them as float arrays `(x, d, e)`, in the caller's order.

    The three columns are the rows of ONE sex and ONE zone (`config.zone_slice`), one row per age.

    Raises ValueError (French message, shown in the app's status table) for vectors that are not 1-D and of
    equal length, a non-finite or duplicated age, or a negative death count. Non-finite deaths or exposures
    are NOT refused: such an age carries no information and the fits ignore it, like an age with E = 0.
    """
    x = np.asarray(ages, dtype=float)
    d = np.asarray(Dx, dtype=float)
    e = np.asarray(Ex, dtype=float)
    if not (x.ndim == d.ndim == e.ndim == 1 and x.size == d.size == e.size):
        raise ValueError("ages, Dx et Ex doivent être trois vecteurs de même longueur.")
    if not np.all(np.isfinite(x)):
        raise ValueError("Âge non fini dans la zone.")
    if np.unique(x).size != x.size:
        raise ValueError("Âges dupliqués dans la zone : une seule ligne par âge est attendue.")
    if np.any(d[np.isfinite(d)] < 0):
        raise ValueError("Nombre de décès négatif dans la zone.")
    return x, d, e


def poisson_loglik(Dx, Ex, mu) -> float:
    """Full Poisson log-likelihood of the deaths given the rates:  sum_x [ D ln(E mu) - E mu - ln(D!) ].

    This is the `loglik` of every `FitResult`. It is the doc's ell(theta) = sum[D ln(mu) - E mu] plus the
    terms that do not depend on theta (D ln E - ln D!), so it is a true log-likelihood: comparable between
    methods fitted to the same ages, whatever the estimator (WLS in zone I, MLE in zones II-III).

    `ln(D!)` is `gammaln(D + 1)`, so non-integer D (deaths rebuilt from rates) are accepted. An age with
    D = 0 contributes -E mu. Ages with E <= 0 or a non-finite D or E contribute nothing (nothing at risk,
    nothing observed, or no information), which is also what the fits do with them. `mu` must be > 0
    wherever E > 0.
    """
    d = np.asarray(Dx, dtype=float)
    e = np.asarray(Ex, dtype=float)
    mu = np.asarray(mu, dtype=float)
    k = np.isfinite(d) & np.isfinite(e) & (e > 0)
    m = e[k] * mu[k]
    return float(np.sum(d[k] * np.log(m) - m - gammaln(d[k] + 1.0)))


def half_deviance(Dx, Ex, mu) -> float:
    """Half the Poisson deviance, sum_x [ D ln(D / (E mu)) - D + E mu ]  (the D ln term is 0 where D = 0).

    It is the log-likelihood turned upside down and anchored at the saturated model:
    half_deviance = ell(saturated) - ell(mu), so it is >= 0, is 0 only for mu = D / E, and the mu that
    minimises it is exactly the Poisson MLE. Optimisers are given THIS rather than -ell because -ell is of
    the order of D ln E (1e6 and more for a national table) and rounding noise there would be larger than the
    tolerance; half_deviance is of the order of the number of ages when the model fits.

    Ages with E <= 0 or a non-finite D or E are skipped, as in `poisson_loglik`. Returns +inf when any used
    rate is not strictly positive (or not finite), so a Nelder-Mead simplex simply steps away from it.
    """
    d = np.asarray(Dx, dtype=float)
    e = np.asarray(Ex, dtype=float)
    mu = np.asarray(mu, dtype=float)
    k = np.isfinite(d) & np.isfinite(e) & (e > 0)
    m = e[k] * mu[k]
    if not np.all(np.isfinite(m)) or np.any(m <= 0):
        return float("inf")
    dk = d[k]
    pos = dk > 0
    return float(np.sum(dk[pos] * np.log(dk[pos] / m[pos])) - dk.sum() + m.sum())


class NelderMeadResult(NamedTuple):
    x: np.ndarray        # best point found
    fun: float           # objective there
    n_iter: int          # iterations, summed over all runs
    n_fev: int           # objective evaluations, summed over all runs
    converged: bool      # the last run met both tolerances AND improved by no more than `fatol`
    runs: int            # Nelder-Mead runs made (1 + restarts used)


def nelder_mead(fun, simplex, *, fatol: float = 1e-8, xatol: float = NM_XATOL, maxiter: int = NM_MAXITER,
                restarts: int = NM_RESTARTS, restart_step: float = 0.05) -> NelderMeadResult:
    """Minimise `fun` by Nelder-Mead (scipy, adaptive coefficients) from `simplex`, restarting while it improves.

    Nelder-Mead is the method the specs prescribe for the derivative-free zone models (3d, 3e). A single run can
    stop on a collapsed simplex short of the minimum, so the run is repeated from the best point, with a fresh
    simplex of side `restart_step`, until a run improves the objective by no more than `fatol`.

    Parameters
    ----------
    fun : callable
        Objective of one 1-D array; may return +inf for points outside its domain.
    simplex : array-like, shape (n + 1, n)
        Initial simplex; its first vertex is the starting point. The caller chooses the vertices (the
        optimiser's coordinates should be O(1)-scaled so that one `restart_step` is a modest move).
    fatol : float
        Convergence tolerance on the objective (scipy's `fatol`) and threshold of a restart's improvement.
    xatol : float
        Convergence tolerance on the simplex size (scipy's `xatol`).
    maxiter : int
        Iteration cap of each run.
    restarts : int
        Maximum number of extra runs.

    `converged` is True when the final run reports success and the restart before declaring victory
    improved the objective by <= `fatol`; it is False when a run hit `maxiter` or when the restarts were used up
    while the objective was still falling. With `restarts = 0` it is the single run's own success flag.
    """
    simplex = np.asarray(simplex, dtype=float)
    opts = dict(maxiter=maxiter, xatol=xatol, fatol=fatol, adaptive=True)
    best = minimize(fun, simplex[0], method="Nelder-Mead", options=dict(opts, initial_simplex=simplex))
    n_iter, n_fev, runs = int(best.nit), int(best.nfev), 1
    converged = bool(best.success)
    signs = np.where(np.arange(simplex.shape[1]) % 2 == 0, 1.0, -1.0)
    for _ in range(restarts):
        fresh = best.x + restart_step * np.vstack([np.zeros(simplex.shape[1]), np.diag(signs)])
        nxt = minimize(fun, best.x, method="Nelder-Mead", options=dict(opts, initial_simplex=fresh))
        n_iter += int(nxt.nit)
        n_fev += int(nxt.nfev)
        runs += 1
        gain = float(best.fun - nxt.fun)
        if nxt.fun < best.fun:
            best = nxt
        converged = bool(nxt.success) and gain <= fatol
        if gain <= fatol:
            break
    return NelderMeadResult(np.asarray(best.x, dtype=float), float(best.fun), n_iter, n_fev, converged, runs)


def refit_with_zero(fun, theta, zero: int, fixed=(), *, tol: float, step: float = 0.05):
    """Is coordinate `zero` of `theta` really on its wall? Re-optimise the others with it set to 0 and compare.

    Nelder-Mead stops within its tolerance of a wall, never exactly on it, and a parameter that is truly 0 may
    leave a small positive value that the other parameters have partly compensated (a second exponential with a base
    almost equal to the first one is the typical case). Setting it to 0 and leaving the rest alone would then look
    costly although the constrained model is as good as the free one, so the remaining coordinates are re-fitted.

    Parameters
    ----------
    fun : callable
        The objective being minimised (a half deviance), of the full coordinate vector.
    theta : array-like
        The best point found, in the optimiser's coordinates (a coordinate equal to 0 means the parameter is 0).
    zero : int
        The coordinate to set to 0.
    fixed : sequence of int
        Coordinates that become meaningless once `zero` is 0 (the base of an exponential whose amplitude is 0): they
        are held at their current values instead of being re-fitted.
    tol : float
        The wall is accepted when the constrained optimum is at most `tol` above the free one.
    step : float
        Side of the Nelder-Mead simplex of the re-fit, in the optimiser's coordinates.

    Returns the new coordinate vector (coordinate `zero` exactly 0, the others re-optimised), or None when the wall
    costs more than `tol`, i.e. when the parameter is genuinely positive.
    """
    theta = np.asarray(theta, dtype=float)
    free = [i for i in range(theta.size) if i != zero and i not in tuple(fixed)]
    base = theta.copy()
    base[zero] = 0.0

    def reduced(z):
        t = base.copy()
        t[free] = z
        return fun(t)

    x0 = theta[free]
    signs = np.where(np.arange(len(free)) % 2 == 0, 1.0, -1.0)
    simplex = np.vstack([x0] + [x0 + step * signs[j] * np.eye(len(free))[j] for j in range(len(free))])
    nm = nelder_mead(reduced, simplex, fatol=tol)
    if nm.fun > fun(theta) + tol:
        return None
    out = base.copy()
    out[free] = nm.x
    return out
