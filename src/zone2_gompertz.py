"""zone2_gompertz.py - Module 3b: Zone II (ages omega1 .. omega2), Gompertz law, Poisson MLE by Newton-Raphson 2x2.

Model (spec 3b; methodology doc, sections 3.1, 3.5 and 3.6)::

    mu_x = B c^x = exp(alpha + beta x),      alpha = ln B,  beta = ln c,      theta = (alpha, beta) in R^2

    D_x ~ Poisson(E_x mu_x)          ell(alpha, beta) = sum_x [ D_x (alpha + beta x) - E_x exp(alpha + beta x) ]

Why Poisson maximum likelihood and not least squares on ln(D/E) (doc 3.5): the variance of ln(D/E) is ~1/D
(unweighted OLS trusts the thinly observed ages as much as the others), an age with D = 0 has no logarithm but
a perfectly good likelihood term (-E mu), and the counts really are Poisson. OLS is kept for one thing only,
the starting point of the iteration.

Estimation, `fit_gompertz`:

  * start (alpha0, beta0): OLS of ln(D/E) on x over the ages with D > 0 (spec 3b: "regression OLS sur log(mx_raw)");
  * Newton-Raphson, theta <- theta - H(theta)^-1 grad ell(theta), with the analytic score and Hessian of the spec::

        d ell/d alpha = sum [ D - E mu ]            d ell/d beta = sum [ x D - x E mu ]
        H = - sum E mu * [[1, x], [x, x^2]]          (negative definite as soon as two distinct ages have E mu > 0)

    ell is therefore strictly concave: there is one maximum, it exists whenever deaths are observed at two or
    more distinct ages, and a Newton step always points uphill.

Two decisions the specs leave open, made here:

  * STOPPING RULE. `cfg.nr_tol` is compared with the remaining log-likelihood gap estimated by the Newton
    decrement, 1/2 grad' (-H)^-1 grad, which is how far ell still is from its maximum to second order. It is
    free of the scale of the ages, unlike the norm of the gradient (whose beta-component is multiplied by x).
  * STEP HALVING. A full Newton step that would lower ell (or overflow exp) is halved until ell does not
    decrease. From the OLS start this almost never triggers; it is a guard, not a change of method, and it
    keeps the iteration from diverging on a poor start.

Conventions of the zone modules (see zone1_poly and fit_common): input = the rows of ONE sex and ONE zone, output
arrays have the input length and order, `loglik` is `fit_common.poisson_loglik`, `params['n_params']` is the
number of free parameters, an impossible fit raises ValueError (French message).
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np

from src.config import FitResult, SmootherConfig
from src.fit_common import check_columns, poisson_loglik

N_PARAMS = 2
MAX_HALVINGS = 60        # a Newton step is never cut by more than 2^-60


class _Newton(NamedTuple):
    alpha: float
    beta: float
    n_iter: int          # Newton steps applied
    converged: bool
    gap: float           # estimated remaining log-likelihood gap at the returned point (nan if singular)
    note: str            # '' unless something went wrong


def _ols_start(x: np.ndarray, d: np.ndarray, e: np.ndarray) -> tuple[float, float]:
    """OLS of y = ln(D/E) on x (ages with D > 0 and E > 0 only), unweighted as in spec 3b:
    beta0 = sum (x - xbar)(y - ybar) / sum (x - xbar)^2,  alpha0 = ybar - beta0 xbar. Needs 2 distinct ages."""
    y = np.log(d / e)
    xc = x - x.mean()
    beta0 = float(np.sum(xc * (y - y.mean())) / np.sum(xc ** 2))
    return float(y.mean() - beta0 * x.mean()), beta0


def _loglik(alpha: float, beta: float, x: np.ndarray, d: np.ndarray, e: np.ndarray) -> float:
    """The doc's ell(alpha, beta) = sum [ D (alpha + beta x) - E exp(alpha + beta x) ] (constants dropped, which
    changes neither the maximiser nor any Newton step). -inf when exp overflows."""
    with np.errstate(over="ignore", invalid="ignore"):
        ll = float(np.sum(d * (alpha + beta * x) - e * np.exp(alpha + beta * x)))
    return ll if np.isfinite(ll) else -np.inf


def _newton(x: np.ndarray, d: np.ndarray, e: np.ndarray, alpha: float, beta: float,
            maxiter: int, tol: float) -> _Newton:
    """Newton-Raphson 2x2 on ell(alpha, beta) from (alpha, beta), on the ages x (all with E > 0).

    At each point: score g and Hessian H as in the module docstring, step s = (-H)^-1 g, remaining gap
    g's / 2. Stops as converged when the gap is <= tol (so the returned point is one where the test held, and
    `n_iter` counts the steps taken to reach it: 0 when the start already satisfies it), and as not converged
    after `maxiter` steps, on a singular Hessian, or when no step length in 1, 1/2, ..., 2^-60 makes ell
    increase. A step is accepted when ell stays finite and does not fall by more than rounding noise.
    """
    ll = _loglik(alpha, beta, x, d, e)
    if not np.isfinite(ll):
        return _Newton(alpha, beta, 0, False, np.nan, "point de départ MCO hors domaine (exp débordée)")
    gap = np.nan
    for n_iter in range(maxiter + 1):
        em = e * np.exp(alpha + beta * x)
        r = d - em
        score = np.array([r.sum(), (x * r).sum()])
        neg_h = np.array([[em.sum(), (x * em).sum()], [(x * em).sum(), (x * x * em).sum()]])
        try:
            step = np.linalg.solve(neg_h, score)
        except np.linalg.LinAlgError:
            return _Newton(alpha, beta, n_iter, False, np.nan, "hessien singulier")
        gap = 0.5 * float(score @ step)
        if gap <= tol:
            return _Newton(alpha, beta, n_iter, True, gap, "")
        if n_iter == maxiter:
            break
        t = 1.0
        for _ in range(MAX_HALVINGS + 1):
            new_ll = _loglik(alpha + t * step[0], beta + t * step[1], x, d, e)
            if new_ll >= ll - 1e-12 * (1.0 + abs(ll)):          # finite (else -inf) and not lower than before
                break
            t *= 0.5
        else:
            return _Newton(alpha, beta, n_iter, False, gap, "aucun pas de Newton ne fait monter la log-vraisemblance")
        alpha, beta, ll = alpha + t * step[0], beta + t * step[1], new_ll
    return _Newton(alpha, beta, maxiter, False, gap, "")


def fit_gompertz(ages, Dx, Ex, cfg: SmootherConfig | None = None) -> FitResult:
    """Zone II, method `gompertz`: mu_x = exp(alpha + beta x), Poisson maximum likelihood by Newton-Raphson 2x2.

    Algorithm (spec 3b; methodology doc, section 3.6)::

        1. (alpha0, beta0) = OLS of ln(D_x / E_x) on x, ages with D > 0
        2. repeat  theta <- theta + (-H)^-1 grad ell   until the remaining log-likelihood gap is <= cfg.nr_tol
           or cfg.nr_maxiter steps have been taken, with the analytic grad ell and H of the module docstring
        3. B = exp(alpha), c = exp(beta)

    Every age with D >= 0 and E > 0 enters the likelihood, ages with D = 0 included (they contribute -E mu,
    which is why this is not a WLS on logarithms). Ages with E <= 0 or a non-finite D or E are ignored, and
    still get a rate in the output: `mu_hat` is exp(alpha + beta x) at EVERY input age.

    Parameters
    ----------
    ages : array-like, shape (n,)
        Ages of the zone (rows of ONE sex, both bounds included, one row per age). Order is kept.
    Dx : array-like, shape (n,)
        Observed deaths, >= 0 (non-integer values from rebuilt deaths are accepted).
    Ex : array-like, shape (n,)
        Central exposure to risk, same length.
    cfg : SmootherConfig, optional
        Reads `nr_maxiter` (cap on Newton steps, default 200) and `nr_tol` (default 1e-8, see the stopping
        rule in the module docstring). Defaults of the spec are used when it is omitted.

    Returns
    -------
    FitResult
        method='gompertz', zone='zone2'; `ages` the input ages as floats; `mu_hat` the fitted rates at every
        input age; `params` = {'alpha', 'beta', 'B', 'c', 'n_params': 2} (floats, B = e^alpha, c = e^beta);
        `loglik` the full Poisson log-likelihood at the estimate (`fit_common.poisson_loglik`); `converged`
        True when the stopping rule was met; `n_iter` the number of Newton steps applied; `message` a French
        diagnostic (iterations, c, and a warning when beta <= 0, i.e. mortality not increasing with age).

        When the iteration stops without converging (cap reached, singular Hessian, no ascent step) a
        FitResult is still returned, with `converged=False`, the last iterate and the reason in `message`.

    Raises
    ------
    ValueError
        Columns of different lengths, a duplicated or non-finite age, negative deaths, or fewer than 2 distinct
        ages with D > 0 and E > 0 (the OLS start, and the likelihood itself, need two ages with deaths).
    """
    cfg = cfg if cfg is not None else SmootherConfig()
    x, d, e = check_columns(ages, Dx, Ex)
    use = np.isfinite(d) & np.isfinite(e) & (e > 0)                   # ages in the likelihood
    start = use & (d > 0)                                             # ages with a logarithm, for the OLS start
    if np.unique(x[start]).size < 2:
        raise ValueError(f"Gompertz : {int(start.sum())} âge(s) avec décès et exposition, il en faut au moins 2 "
                         f"(zone II trop courte, ou presque aucun décès).")

    alpha0, beta0 = _ols_start(x[start], d[start], e[start])
    nr = _newton(x[use], d[use], e[use], alpha0, beta0, cfg.nr_maxiter, cfg.nr_tol)
    mu_hat = np.exp(nr.alpha + nr.beta * x)
    c = float(np.exp(nr.beta))

    n_out = x.size - int(use.sum())
    steps = f"{nr.n_iter} itération{'s' if nr.n_iter > 1 else ''}"
    if nr.converged:
        message = f"Newton-Raphson 2×2 convergé en {steps} (départ MCO) ; c = e^β = {c:.4f}."
    else:
        why = nr.note or f"limite nr_maxiter = {cfg.nr_maxiter} atteinte (écart de log-vraisemblance estimé : {nr.gap:.3g} > nr_tol = {cfg.nr_tol:g})"
        message = f"Newton-Raphson 2×2 NON convergé après {steps} : {why}."
    if nr.beta <= 0:
        message += " Attention : β ≤ 0, la mortalité ne croît pas avec l'âge sur cette zone (loi de Gompertz peu pertinente)."
    if n_out:
        message += f" {n_out} âge{'s' if n_out > 1 else ''} ignoré{'s' if n_out > 1 else ''} (Ex = 0 ou valeur manquante)."

    return FitResult(
        method="gompertz", zone="zone2", ages=x, mu_hat=mu_hat,
        params={"alpha": float(nr.alpha), "beta": float(nr.beta), "B": float(np.exp(nr.alpha)), "c": c,
                "n_params": N_PARAMS},
        loglik=poisson_loglik(d, e, mu_hat),
        converged=nr.converged, n_iter=nr.n_iter, message=message)
