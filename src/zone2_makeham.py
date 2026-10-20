"""zone2_makeham.py - Module 3c: Zone II (ages omega1 .. omega2), Makeham law, Poisson MLE by profile on c.

Model (spec 3c; methodology doc, sections 3.2 and 3.7)::

    mu_x = A + B c^x,        A >= 0 (the age-independent "accident" term), B > 0, c > 1
    D_x ~ Poisson(E_x mu_x)  ell(A, B, c) = sum_x [ D_x ln(A + B c^x) - E_x (A + B c^x) ]

Makeham is Gompertz plus a constant: A = 0 gives Gompertz back, so its maximum log-likelihood can never be
below Gompertz's on the same ages.

Estimation, `fit_makeham` (spec 3c: "profil sur c + Newton-Raphson 2x2 sur (A, B) a c fixe")::

    for each c of a grid on [1.05, 1.15] (100 points):   solve the (A, B) sub-problem        <- _solve_ab
    keep the c whose profile log-likelihood ell*(c) = max_{A,B} ell(A, B, c) is the largest
    refine c around the winning grid cell, then take (A, B) at that c

For a fixed c, mu = A + B c^x is LINEAR in (A, B), so the sub-problem has the analytic score and Hessian of the spec::

    d ell/dA = sum [ D/mu - E ]            d ell/dB = sum [ (D/mu - E) c^x ]
    H = - sum (D / mu^2) * [[1, c^x], [c^x, c^(2x)]]          (negative definite: ell is concave in (A, B))

and is handled in three steps, because of the constraint A >= 0 that a bare Newton iteration ignores
(the lab's first Newton version settled near the A = 0 wall, 2.7 log-likelihood units short of the optimum):

  1. the A = 0 point, B = sum D / sum (E c^x), is the exact optimum on that boundary; if leaving it cannot raise the
     log-likelihood by more than `nr_tol` (Kuhn-Tucker: the score of A there is <= 0, or so small that the gain
     g^2 / (2 S) estimated with the Schur complement S of the curvature is below the tolerance) it is THE
     constrained optimum and no iteration is needed (Makeham collapsed to Gompertz);
  2. likewise the B = 0 point (constant force);
  3. otherwise the optimum is interior (A > 0, B > 0), ell is concave, and Newton-Raphson 2x2 with step halving
     converges to it from a feasible start.

Where this departs from the letter of the specs (all flagged to the user):

  * C REFINEMENT. The spec stops at the best grid value c*. A 100-point grid leaves c up to 0.0005 off, which
    costs 0.3 to 1.5 log-likelihood units on France 2020 (enough to blur an AIC comparison), so c is then refined by a
    bounded scalar search inside the winning cell. `REFINE_C = False` gives the spec's literal behaviour.
  * INITIALISATION. The spec starts at A0 = 0.001, B0 = exp(alpha_Gompertz), c0 = exp(beta_Gompertz). With a profile
    the starting point is needed at each fixed c: A0 = 10% of the crude overall rate and the B0 that reproduces the
    observed number of deaths. Both are feasible for every c of the grid, which a single (B0, c0) is not.
  * COORDINATES. B c^x is computed as b w_x with w_x = c^(x - x_ref) <= 1 and b = B c^(x_ref), x_ref the oldest age:
    c^x alone spans ten orders of magnitude over a zone and makes the Hessian ill-conditioned. Newton is invariant
    to this rescaling; B = b / c^(x_ref) is reported.

Conventions of the zone modules (see zone1_poly, zone2_gompertz, fit_common): input = the rows of ONE sex and ONE zone,
output arrays have the input length and order, `loglik` is `fit_common.poisson_loglik`, `params['n_params']` is the
number of free parameters, impossible input raises ValueError (French message).
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np
from scipy.optimize import minimize_scalar

from src.config import FitResult, SmootherConfig
from src.fit_common import check_columns, half_deviance, poisson_loglik

C_BOUNDS = (1.05, 1.15)      # spec 3c: grid search on c in [1.05, 1.15]
GRID_N = 100                 # ... with 100 points
REFINE_C = True              # polish c inside the winning grid cell (see the module docstring)
N_PARAMS = 3
MAX_HALVINGS = 60


class _AB(NamedTuple):
    A: float
    b: float            # B c^(x_ref): the exponential term's rate at the oldest age of the fit
    dev: float          # half deviance at the solution (the profile objective, to be minimised)
    n_iter: int         # Newton steps
    converged: bool
    kind: str           # 'A=0' | 'B=0' | 'interior' | 'maxiter' | 'stuck'
    gap: float          # estimated log-likelihood still to gain at the returned point (0 on a wall, nan if stuck)


def _stays_on_wall(g: float, h_pp: float, h_pq: float, h_qq: float, tol: float) -> bool:
    """Kuhn-Tucker test at a boundary point whose other coordinate q is already optimal.

    p is the coordinate that would leave its wall, `g` its score there and h_** the (positive) entries of -H.
    Moving p (with q re-optimised) can raise the log-likelihood by about g^2 / (2 S), S = h_pp - h_pq^2 / h_qq
    being the curvature of p once q has adjusted. The point is accepted as the constrained optimum when that gain is
    at most `tol` (always when g <= 0): the same remaining-gap rule as the interior iteration, and immune to the
    +-1e-11 of rounding noise that a bare `g <= 0` mistakes for a reason to iterate on data lying exactly on the wall.
    """
    if g <= 0.0:
        return True
    schur = h_pp - h_pq * h_pq / h_qq if h_qq > 0.0 else 0.0
    return schur <= 0.0 or 0.5 * g * g / schur <= tol


def _solve_ab(w: np.ndarray, d: np.ndarray, e: np.ndarray, maxiter: int, tol: float) -> _AB:
    """Maximise sum [ D ln(A + b w) - E (A + b w) ] over A >= 0, b >= 0, for the fixed positive vector `w`.

    `w`, `d`, `e` hold only the ages of the likelihood (E > 0) and `d` has deaths at two or more ages. The three
    steps of the module docstring: the A = 0 boundary (exact, closed form), the b = 0 boundary, else Newton-Raphson
    2x2 on the interior with the stopping rule of zone2_gompertz (remaining log-likelihood gap
    1/2 g' (-H)^-1 g <= tol) and step halving, up to `maxiter` steps. Returns the solution and how it was reached.
    """
    d_tot, e_tot, ew = d.sum(), e.sum(), float(np.sum(e * w))

    b0 = d_tot / ew                                                    # optimum on the boundary A = 0
    q = d / (b0 * w) ** 2
    if _stays_on_wall(np.sum(d / (b0 * w)) - e_tot, q.sum(), (q * w).sum(), (q * w * w).sum(), tol):
        return _AB(0.0, b0, half_deviance(d, e, b0 * w), 0, True, "A=0", 0.0)  # not worth leaving A = 0
    a0 = d_tot / e_tot                                                 # optimum on the boundary b = 0 (constant force)
    q = d / a0 ** 2
    if _stays_on_wall(np.sum(w * d) / a0 - ew, (q * w * w).sum(), (q * w).sum(), q.sum(), tol):
        return _AB(a0, 0.0, half_deviance(d, e, np.full_like(w, a0)), 0, True, "B=0", 0.0)

    A, b = 0.1 * d_tot / e_tot, 0.9 * d_tot / ew                       # feasible start: reproduces the observed deaths
    dev = half_deviance(d, e, A + b * w)
    gap = float("nan")
    for n_iter in range(maxiter + 1):
        mu = A + b * w
        r = d / mu - e
        score = np.array([r.sum(), (w * r).sum()])
        q = d / mu ** 2
        neg_h = np.array([[q.sum(), (q * w).sum()], [(q * w).sum(), (q * w * w).sum()]])
        try:
            step = np.linalg.solve(neg_h, score)
        except np.linalg.LinAlgError:
            return _AB(A, b, dev, n_iter, False, "stuck", float("nan"))
        gap = 0.5 * float(score @ step)
        if gap <= tol:
            return _AB(A, b, dev, n_iter, True, "interior", gap)
        if n_iter == maxiter:
            return _AB(A, b, dev, maxiter, False, "maxiter", gap)
        t = 1.0
        for _ in range(MAX_HALVINGS + 1):
            new_dev = half_deviance(d, e, (A + t * step[0]) + (b + t * step[1]) * w)
            if new_dev <= dev + 1e-12 * (1.0 + abs(dev)):               # finite (else inf) and not worse
                break
            t *= 0.5
        else:
            return _AB(A, b, dev, n_iter, False, "stuck", gap)
        A, b, dev = A + t * step[0], b + t * step[1], new_dev
    return _AB(A, b, dev, maxiter, False, "maxiter", gap)               # unreachable: the loop returns


def fit_makeham(ages, Dx, Ex, cfg: SmootherConfig | None = None) -> FitResult:
    """Zone II, method `makeham`: mu_x = A + B c^x, Poisson maximum likelihood by profile on c (spec 3c).

    Algorithm (see the module docstring for the reasons and the departures from the letter of the spec)::

        1. grid c_j = linspace(1.05, 1.15, 100); for each, (A_j, B_j) = argmax over A >= 0, B >= 0 of ell(A, B, c_j)
           (Kuhn-Tucker test on the A = 0 and B = 0 boundaries, else Newton-Raphson 2x2, cfg.nr_maxiter / cfg.nr_tol)
        2. c* = the c_j with the largest profile log-likelihood, then refined inside [c_(j-1), c_(j+1)]
        3. (A, B) at c*

    Every age with E > 0 enters the likelihood, ages with D = 0 included. Ages with E <= 0 or a non-finite D or E
    are ignored and still get a rate in the output: `mu_hat` is A + B c^x at EVERY input age.

    Parameters
    ----------
    ages, Dx, Ex : array-like, shape (n,)
        The rows of ONE sex and ONE zone (both bounds included), one row per age; Dx >= 0 (non-integer allowed).
    cfg : SmootherConfig, optional
        Reads `nr_maxiter` and `nr_tol` for the Newton-Raphson of every (A, B) sub-problem (defaults of the spec if
        omitted). The grid itself (c in [1.05, 1.15], 100 points) is fixed by the spec, not a setting.

    Returns
    -------
    FitResult
        method='makeham', zone='zone2'; `mu_hat` at every input age; `params` = {'A', 'B', 'c', 'n_params': 3}
        (floats; A = 0.0 exactly when the constraint binds); `loglik` the full Poisson log-likelihood
        (`fit_common.poisson_loglik`); `converged` True when every (A, B) sub-problem converged; `n_iter` the Newton
        steps summed over all sub-problems (grid and refinement; the closed-form boundary cases cost none);
        `message` a French diagnostic: the number of steps, c, and a warning when A or B sits on its bound
        (Makeham then reduces to Gompertz or to a constant) or when c sits on an end of [1.05, 1.15]
        (the constrained optimum may lie beyond it).

    Raises
    ------
    ValueError
        Columns of different lengths, a duplicated or non-finite age, negative deaths, fewer than 3 ages with
        E > 0 (three parameters), or deaths at fewer than 2 distinct ages.
    """
    cfg = cfg if cfg is not None else SmootherConfig()
    x, d, e = check_columns(ages, Dx, Ex)
    use = np.isfinite(d) & np.isfinite(e) & (e > 0)
    if int(use.sum()) < N_PARAMS or np.unique(x[use & (d > 0)]).size < 2:
        raise ValueError(f"Makeham : {int(use.sum())} âge(s) avec exposition, il en faut au moins {N_PARAMS}, "
                         f"avec des décès à au moins 2 âges différents (zone II trop courte, ou presque aucun décès).")
    xu, du, eu = x[use], d[use], e[use]
    x_ref = float(xu.max())
    steps = [0]

    def solve(c: float) -> _AB:
        sol = _solve_ab(c ** (xu - x_ref), du, eu, cfg.nr_maxiter, cfg.nr_tol)
        steps[0] += sol.n_iter
        return sol

    grid = np.linspace(*C_BOUNDS, GRID_N)
    sols = [solve(float(c)) for c in grid]
    k = int(np.argmin([s.dev for s in sols]))
    c = float(grid[k])
    if REFINE_C:
        lo, hi = float(grid[max(k - 1, 0)]), float(grid[min(k + 1, GRID_N - 1)])
        res = minimize_scalar(lambda cc: solve(cc).dev, bounds=(lo, hi), method="bounded", options={"xatol": 1e-12})
        if res.fun <= sols[k].dev:
            c = float(res.x)
    final = solve(c)
    A, B = final.A, final.b / c ** x_ref
    mu_hat = A + final.b * c ** (x - x_ref)
    converged = final.converged and all(s.converged for s in sols)

    message = (f"Profil de vraisemblance sur c ∈ [{C_BOUNDS[0]}, {C_BOUNDS[1]}] ({GRID_N} valeurs"
               f"{', affiné' if REFINE_C else ''}), (A, B) par Newton-Raphson 2×2 : {steps[0]} pas au total ; c = {c:.5f}.")
    if final.kind == "A=0":
        message += " A = 0 (borne) : Makeham se réduit à Gompertz."
    elif final.kind == "B=0":
        message += " B = 0 (borne) : force de mortalité constante, c n'est pas identifié."
    if abs(c - C_BOUNDS[0]) < 1e-9 or abs(c - C_BOUNDS[1]) < 1e-9:
        message += f" c est à une borne de [{C_BOUNDS[0]}, {C_BOUNDS[1]}] : l'optimum se trouve peut-être au-delà."
    if not converged:
        bad = sum(not s.converged for s in sols) + (not final.converged)
        message += f" ATTENTION : {bad} sous-problème(s) (A, B) non convergé(s)."
    n_out = x.size - int(use.sum())
    if n_out:
        message += f" {n_out} âge{'s' if n_out > 1 else ''} ignoré{'s' if n_out > 1 else ''} (Ex = 0 ou valeur manquante)."

    return FitResult(
        method="makeham", zone="zone2", ages=x, mu_hat=mu_hat,
        params={"A": float(A), "B": float(B), "c": c, "n_params": N_PARAMS},
        loglik=poisson_loglik(d, e, mu_hat),
        converged=converged, n_iter=steps[0], message=message)
