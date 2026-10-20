"""zone3.py - Module 3e: Zone III (ages omega2 .. omega3), the oldest ages: Kannisto, Beard, Coale-Kisker.

Where the data thin out (few exposed, deaths in tens) and mortality decelerates instead of exploding like Gompertz
(methodology doc, section 4). Three methods, one per key of `config.METHODS`, all in this module (spec: zone3.py)::

    fit_kannisto(ages, Dx, Ex, cfg)       mu_x = a e^(b t) / (1 + a e^(b t)),  t = x - omega2     2 parameters, plateau 1
    fit_beard(ages, Dx, Ex, cfg)          mu_x = B c^x / (1 + K B c^x)                            3 parameters, plateau 1/K
    fit_coale_kisker(ages, Dx, Ex, cfg)   q_x = q_omega2 e^(k (x - omega2)), q(omega3) = 0.6      1 parameter, deterministic

WHAT THE RESULT COVERS (the one place where zone III differs from zones I and II). The joiner needs mu on every age up to
omega3 (spec 4.3: "tableau de longueur omega3 + 1"), yet a table such as HMD stops at 109 while omega3 = 110 by default, and
that is exactly the extrapolation this zone exists for. So `FitResult.ages` is ALWAYS `omega2, omega2 + 1, ..., omega3` and
`mu_hat` the curve on all of them, whatever rows were supplied; the input rows are only what the curve is fitted to and
what `loglik` is computed on.

Decisions the specs leave open or contradict, made here and flagged to the user:

  * LIKELIHOOD: POISSON, not the spec's Bernoulli. Spec 3e writes "log-vraisemblance Bernoulli (D ~ Binomial(E, mu)
    approximee)". But D > E occurs at the oldest ages (France 2020: age 105 for both sexes, 107-109 for men, rates up to
    2.0), where a binomial has a negative number of survivors and no likelihood. Poisson, D ~ Poisson(E mu), has no such
    limit, is the likelihood of every other zone module (`fit_common.poisson_loglik`, so AIC/BIC share one scale), and is
    the one whose gradient the methodology doc actually prints for Kannisto (its formula is binomial, its gradients
    (D - E mu)(1 - mu) are Poisson's). The user's lab made the same choice.
  * ORIGIN OF THE AGE in Kannisto: t = x - omega2. The spec's grid a in [0.001, 1], b in [0.05, 0.2] has no point near the
    optimum with absolute ages (a would be 7e-8) but does with t, where a is the odds at omega2 (0.06-0.10 on France
    2020). The absolute-age a of the literature is a e^(-b omega2); `params['x0']` gives the origin.
  * COALE-KISKER ANCHOR: the crude rate observed at age omega2, turned into q by q = 1 - exp(-m), exactly as the joiner
    does (spec 4.4); the target is the spec's 0.6 at omega3. The anchor is one raw age, so for a small population it is
    noisy and so is the whole extrapolation.

Conventions of the zone modules (see zone1_poly, zone2_gompertz, fit_common): `ages, Dx, Ex` are the rows of ONE sex (zone
rows omega2..omega3, both bounds included), `loglik` is `fit_common.poisson_loglik` over the supplied rows, `n_iter` counts
Newton steps / Nelder-Mead iterations, `params['n_params']` is the number of parameters, impossible input raises ValueError
(French message).
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np
from scipy.special import expit

from src.config import FitResult, SmootherConfig
from src.fit_common import check_columns, half_deviance, nelder_mead, poisson_loglik, refit_with_zero

LN_A_RANGE = (-10.0, 0.0)    # grid on ln a: contains the spec's a in [0.001, 1], and the methodology's [-10, 0]
B_RANGE = (0.05, 0.20)       # grid on b (spec 3e)
Q_TARGET = 0.6               # Coale-Kisker: q at omega3 (spec 3e: "q_cible conventionnelle : 0.6")
MAX_HALVINGS = 60
TRUST_LOGIT = 4.0            # a Newton step never moves the logit eta = ln a + b t of any age by more than this
SIMPLEX_STEP = 0.1           # about 10% on each coordinate of Beard's initial simplex


def _zone_ages(cfg: SmootherConfig) -> np.ndarray:
    """The ages the result covers: omega2 .. omega3, both included."""
    return np.arange(cfg.omega2, cfg.omega3 + 1, dtype=float)


def _inputs(name: str, ages, Dx, Ex, cfg, n_min: int):
    """Check the columns and keep the ages of the likelihood (E > 0, D and E finite).

    Returns (cfg, x, d, e, use). Raises ValueError when fewer than `n_min` ages are usable (one per parameter) or when
    deaths are seen at fewer than 2 distinct ages (no curve can be fitted to a single point).
    """
    cfg = cfg if cfg is not None else SmootherConfig()
    x, d, e = check_columns(ages, Dx, Ex)
    use = np.isfinite(d) & np.isfinite(e) & (e > 0)
    if int(use.sum()) < n_min or np.unique(x[use & (d > 0)]).size < 2:
        raise ValueError(f"{name} : {int(use.sum())} âge(s) avec exposition, il en faut au moins {n_min}, avec des décès "
                         f"à au moins 2 âges différents (zone III trop courte, ou presque aucun décès).")
    return cfg, x, d, e, use


def _result(method: str, cfg, x, d, e, mu_of, params: dict, converged: bool, n_iter: int, message: str) -> FitResult:
    """Assemble the FitResult: the curve on omega2..omega3, the log-likelihood on the supplied rows."""
    ages_out = _zone_ages(cfg)
    n_out = int(np.sum(~(np.isfinite(d) & np.isfinite(e) & (e > 0))))
    if n_out:
        message += f" {n_out} âge{'s' if n_out > 1 else ''} ignoré{'s' if n_out > 1 else ''} (Ex = 0 ou valeur manquante)."
    with np.errstate(over="ignore", invalid="ignore"):
        mu_hat, mu_rows = mu_of(ages_out), mu_of(x)
    return FitResult(method=method, zone="zone3", ages=ages_out, mu_hat=mu_hat, params=params,
                     loglik=poisson_loglik(d, e, mu_rows), converged=converged, n_iter=n_iter, message=message)


# --------------------------------------------------------------------------
# Kannisto
# --------------------------------------------------------------------------


class _Newton(NamedTuple):
    ln_a: float
    b: float
    n_iter: int
    converged: bool
    gap: float           # estimated log-likelihood still to gain at the returned point (nan if singular)
    n_fisher: int        # iterations that used the expected information because the observed Hessian was not definite
    note: str


def _grid_half_deviance(ln_a: np.ndarray, b: np.ndarray, t: np.ndarray, d: np.ndarray, e: np.ndarray) -> np.ndarray:
    """Half Poisson deviance of Kannisto on the whole (ln a, b) grid at once: array of shape (len(ln_a), len(b)).

    +inf where the rate underflows to 0 at an age with deaths (outside the domain), as in `fit_common.half_deviance`.
    """
    m = e * expit(ln_a[:, None, None] + b[None, :, None] * t[None, None, :])
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(d > 0, d * np.log(d / m), 0.0)
    dev = np.where((m <= 0) & (d > 0), np.inf, term - d + m)
    return dev.sum(axis=2)


def _newton(t: np.ndarray, d: np.ndarray, e: np.ndarray, ln_a: float, b: float, maxiter: int, tol: float) -> _Newton:
    """Newton-Raphson 2x2 on the Poisson log-likelihood of Kannisto in (ln a, b), from (ln a, b).

    With eta = ln a + b t and mu = sigmoid(eta), the score is  sum (D - E mu)(1 - mu) (1, t)  (the doc's gradients, in
    ln a rather than a), and the observed Hessian is -sum w (1, t)'(1, t) with  w = mu (1 - mu)(D + E - 2 E mu).
    Unlike Gompertz this log-likelihood is NOT concave (w < 0 where mu > (1 + D/E)/2, near the plateau), so where
    the observed Hessian is not negative definite the expected information, w = E mu (1 - mu)^2 > 0, is used instead
    (Fisher scoring, always valid). Stopping rule and step halving as in zone2_gompertz: remaining log-likelihood gap
    1/2 g' (-H)^-1 g <= tol; a step is accepted when the half deviance does not rise.

    One addition for the sigmoid: a trust region. Near the plateau the score (D - E mu)(1 - mu) and the information
    both vanish with (1 - mu), their ratio explodes, and a full step can throw the iterate into the flat saturated region
    (mu = 1 at every age), from which it never returns, while halving only asks that the deviance not rise. The step is
    therefore scaled so that no age's logit moves by more than TRUST_LOGIT = 4. From 64 random starts over the whole
    (ln a, b) plane this takes the success rate from 61 to 64; from the grid's best point, the start used by
    `fit_kannisto`, it never matters.
    """
    dev = half_deviance(d, e, expit(ln_a + b * t))
    n_fisher, gap = 0, float("nan")
    for n_iter in range(maxiter + 1):
        mu = expit(ln_a + b * t)
        r = (d - e * mu) * (1.0 - mu)
        score = np.array([r.sum(), (t * r).sum()])
        w = mu * (1.0 - mu) * (d + e - 2.0 * e * mu)
        info = np.array([[w.sum(), (t * w).sum()], [(t * w).sum(), (t * t * w).sum()]])
        if not (info[0, 0] > 0.0 and np.linalg.det(info) > 0.0):
            w = e * mu * (1.0 - mu) ** 2
            info = np.array([[w.sum(), (t * w).sum()], [(t * w).sum(), (t * t * w).sum()]])
            n_fisher += 1
        try:
            step = np.linalg.solve(info, score)
        except np.linalg.LinAlgError:
            return _Newton(ln_a, b, n_iter, False, float("nan"), n_fisher, "hessien singulier")
        gap = 0.5 * float(score @ step)
        if gap <= tol:
            return _Newton(ln_a, b, n_iter, True, gap, n_fisher, "")
        if n_iter == maxiter:
            break
        reach = float(np.max(np.abs(step[0] + step[1] * t)))
        if reach > TRUST_LOGIT:
            step = step * (TRUST_LOGIT / reach)
        s = 1.0
        for _ in range(MAX_HALVINGS + 1):
            new_dev = half_deviance(d, e, expit(ln_a + s * step[0] + (b + s * step[1]) * t))
            if new_dev <= dev + 1e-12 * (1.0 + abs(dev)):
                break
            s *= 0.5
        else:
            return _Newton(ln_a, b, n_iter, False, gap, n_fisher, "aucun pas de Newton ne fait baisser la déviance")
        ln_a, b, dev = ln_a + s * step[0], b + s * step[1], new_dev
    return _Newton(ln_a, b, maxiter, False, gap, n_fisher, "")


def fit_kannisto(ages, Dx, Ex, cfg: SmootherConfig | None = None) -> FitResult:
    """Zone III, method `kannisto`: mu_x = a e^(b t) / (1 + a e^(b t)), t = x - omega2, Poisson maximum likelihood.

    Model and algorithm (spec 3e; methodology doc, sections 4.1 and 4.3 bis)::

        mu_x = sigmoid(ln a + b t),      D_x ~ Poisson(E_x mu_x),      a, b > 0       (plateau 1; Gompertz while mu << 1)

        1. grid search on (ln a, b) in [-10, 0] x [0.05, 0.20], cfg.grid_n x cfg.grid_n points: the best point is a
           safe start, because the Poisson log-likelihood of this sigmoid is not concave and has flat regions;
        2. Newton-Raphson 2x2 from that point (observed Hessian, expected information where it is not definite,
           step halving), cfg.nr_maxiter steps at most, stopped on the remaining log-likelihood gap <= cfg.nr_tol.

    The refinement is unconstrained: a b outside [0.05, 0.20] is found if the data want it. Every row with E > 0 enters
    the likelihood, ages with D = 0 included. See the module docstring for the Poisson-versus-Bernoulli and the age-origin
    decisions, and for what `ages` and `mu_hat` cover.

    Parameters
    ----------
    ages, Dx, Ex : array-like, shape (n,)
        The rows of ONE sex in the zone (normally omega2..omega3, a table stopping earlier is fine), one row per age.
    cfg : SmootherConfig, optional
        Reads `omega2`, `omega3` (the ages of the result and the origin of t), `grid_n`, `nr_maxiter`, `nr_tol`.
        The spec defaults are used when omitted.

    Returns
    -------
    FitResult
        method='kannisto', zone='zone3'; `ages` = omega2..omega3; `mu_hat` the curve on all of them, extrapolated beyond
        the last row; `params` = {'a': odds at omega2 (a e^(b t) at t = 0), 'b', 'x0': omega2, 'n_params': 2}; `loglik`
        the full Poisson log-likelihood at the supplied rows; `converged` True when Newton met its tolerance;
        `n_iter` the Newton steps; `message` a French diagnostic (grid size, steps, Fisher steps, b <= 0 warning).

    Raises
    ------
    ValueError
        Columns of different lengths, a duplicated or non-finite age, negative deaths, fewer than 2 ages with E > 0,
        deaths at fewer than 2 distinct ages, or a grid with no admissible point.
    """
    cfg, x, d, e, use = _inputs("Kannisto", ages, Dx, Ex, cfg, 2)
    t = x[use] - cfg.omega2
    du, eu = d[use], e[use]
    n = max(int(cfg.grid_n), 2)
    ln_a_grid, b_grid = np.linspace(*LN_A_RANGE, n), np.linspace(*B_RANGE, n)
    dev = _grid_half_deviance(ln_a_grid, b_grid, t, du, eu)
    if not np.isfinite(dev).any():
        raise ValueError("Kannisto : aucun point de la grille (ln a, b) n'est admissible pour ces données.")
    i, j = np.unravel_index(int(np.argmin(dev)), dev.shape)
    nr = _newton(t, du, eu, float(ln_a_grid[i]), float(b_grid[j]), cfg.nr_maxiter, cfg.nr_tol)

    a, b, x0 = float(np.exp(nr.ln_a)), nr.b, float(cfg.omega2)
    steps = f"{nr.n_iter} pas de Newton" + (f" dont {nr.n_fisher} en scoring de Fisher" if nr.n_fisher else "")
    if nr.converged:
        message = f"Grille {n}×{n} sur (ln a, b), puis Newton-Raphson 2×2 convergé en {steps} ; a = {a:.4f} (cote à ω₂), b = {b:.5f}."
    else:
        why = nr.note or f"limite nr_maxiter = {cfg.nr_maxiter} atteinte (écart estimé {nr.gap:.3g} > nr_tol = {cfg.nr_tol:g})"
        message = f"Grille {n}×{n} sur (ln a, b), puis Newton-Raphson 2×2 NON convergé après {steps} : {why}."
    if b <= 0:
        message += " Attention : b ≤ 0, la mortalité ne croît pas avec l'âge sur cette zone."
    return _result("kannisto", cfg, x, d, e, lambda ages_: expit(nr.ln_a + b * (ages_ - x0)),
                   {"a": a, "b": float(b), "x0": x0, "n_params": 2}, nr.converged, nr.n_iter, message)


# --------------------------------------------------------------------------
# Beard
# --------------------------------------------------------------------------


def fit_beard(ages, Dx, Ex, cfg: SmootherConfig | None = None) -> FitResult:
    """Zone III, method `beard`: mu_x = B c^x / (1 + K B c^x), Poisson maximum likelihood by Nelder-Mead.

    Model and algorithm (spec 3e: "Nelder-Mead. Initialisation depuis Kannisto"; doc 4.2)::

        mu_x = B c^x / (1 + K B c^x),     B > 0, c > 1, K >= 0                    (plateau 1/K; K = 0 is Gompertz)

        1. seed = the Kannisto fit, which IS a Beard curve with K = 1 (B = a e^(-b omega2), c = e^b): Beard can
           never be below Kannisto on the same rows;
        2. Nelder-Mead on the half deviance (= -loglik up to a constant), initial simplex about 10% around the seed,
           restarted while it improves (`fit_common.nelder_mead`);
        3. K is reported as exactly 0 when the model constrained to K = 0 (Gompertz), the other parameters re-fitted,
           is within cfg.nr_tol of the free fit (`fit_common.refit_with_zero`).

    The optimiser works on unconstrained, O(1)-scaled coordinates with the age counted from omega2: (ln beta, ln gamma,
    sqrt K) with beta = B c^omega2 (the odds-like level at omega2) and gamma = ln c. K is a SQUARE so that it can
    reach 0. B = beta e^(-gamma omega2) and c = e^gamma are reported. The doc proposes Newton-Raphson 3x3 here; the spec
    says Nelder-Mead, which is used (the lab's BFGS with numerical gradient did not report convergence on France 2020,
    this does). The caps are `fit_common`'s (not cfg.nr_maxiter, a Newton cap); cfg.nr_tol is the tolerance of the
    objective. Poisson likelihood and result ages: see the module docstring.

    Parameters
    ----------
    ages, Dx, Ex : array-like, shape (n,)
        The rows of ONE sex in the zone, one row per age.
    cfg : SmootherConfig, optional
        Reads `omega2`, `omega3`, `grid_n`, `nr_maxiter`, `nr_tol` (the Kannisto seed uses all of them).

    Returns
    -------
    FitResult
        method='beard', zone='zone3'; `ages` = omega2..omega3; `mu_hat` the curve on all of them; `params` =
        {'B', 'c', 'K', 'n_params': 3} (plain-age basis of the doc; K = 0.0 exactly when it sits on its wall);
        `loglik` the full Poisson log-likelihood at the supplied rows; `converged` True when Nelder-Mead met its
        tolerances and a final restart gained nothing; `n_iter` the Nelder-Mead iterations; `message` a French
        diagnostic (K = 0 means no plateau was detected: Beard has reduced to Gompertz).

    Raises
    ------
    ValueError
        Same input checks as `fit_kannisto`, with at least 3 ages with E > 0 (three parameters).
    """
    cfg, x, d, e, use = _inputs("Beard", ages, Dx, Ex, cfg, 3)
    kan = fit_kannisto(x, d, e, cfg)
    t, du, eu = x[use] - cfg.omega2, d[use], e[use]

    def curve(th, tt):
        z = np.exp(th[0]) * np.exp(np.exp(th[1]) * tt)
        return z / (1.0 + th[2] ** 2 * z)

    def objective(th):
        with np.errstate(over="ignore", invalid="ignore"):
            return half_deviance(du, eu, curve(th, t))

    th0 = np.array([np.log(kan.params["a"]), np.log(kan.params["b"]), 1.0])         # Kannisto = Beard with K = 1
    simplex = np.vstack([th0] + [th0 + (-1.0) ** i * SIMPLEX_STEP * np.eye(3)[i] for i in range(3)])
    nm = nelder_mead(objective, simplex, fatol=cfg.nr_tol)
    th = nm.x.copy()
    on_wall = refit_with_zero(objective, th, 2, tol=cfg.nr_tol) if th[2] != 0.0 else None
    th = th if on_wall is None else on_wall

    beta, gamma, K, x0 = float(np.exp(th[0])), float(np.exp(th[1])), float(th[2] ** 2), float(cfg.omega2)
    B, c = beta * np.exp(-gamma * x0), float(np.exp(gamma))
    message = (f"Nelder-Mead sur 3 paramètres : {nm.n_iter} itérations en {nm.runs} lancer(s), départ Kannisto "
               f"(a = {kan.params['a']:.4f}, b = {kan.params['b']:.5f}) ; c = {c:.5f}, K = {K:.4f}.")
    if K == 0.0:
        message += " K = 0 (borne) : Beard se réduit à Gompertz, aucun plateau n'est détecté."
    if not nm.converged:
        message += " ATTENTION : Nelder-Mead n'a pas convergé (tolérances non atteintes ou redémarrages épuisés)."
    return _result("beard", cfg, x, d, e, lambda ages_: curve(th, ages_ - x0),
                   {"B": float(B), "c": c, "K": K, "n_params": 3}, nm.converged, nm.n_iter, message)


# --------------------------------------------------------------------------
# Coale-Kisker
# --------------------------------------------------------------------------


def fit_coale_kisker(ages, Dx, Ex, cfg: SmootherConfig | None = None, q_target: float = Q_TARGET) -> FitResult:
    """Zone III, method `coale_kisker`: deterministic log-linear extrapolation of q from omega2 to omega3 (spec 3e).

    No estimation, no likelihood maximisation, no iteration (spec: "formule directe")::

        q_omega2 = 1 - exp(-D / E)      at the row of age omega2 (the observed anchor)
        k        = (ln q_target - ln q_omega2) / (omega3 - omega2),       q_target = 0.6 by default (q at omega3)
        q_x      = q_omega2 exp(k (x - omega2)),        mu_x = -ln(1 - q_x)

    q_x is a straight line in ln q from the anchor to the regulatory target, and `mu_hat` is the rate that gives back
    exactly that q through the constant-force bridge q = 1 - exp(-mu) the joiner applies (spec 4.4). `mu_hat` equals the
    crude rate at omega2 and is only the extrapolation beyond. The anchor is ONE raw age: a small population gives a noisy
    anchor, and the whole curve moves with it. The rows after omega2 play no role in the curve; they are only the rows
    on which `loglik` is evaluated, so that the evaluator can put this curve beside the fitted ones (it is the
    Poisson log-likelihood of a curve that was not fitted to them, and `n_params = 1` counts k).

    Parameters
    ----------
    ages, Dx, Ex : array-like, shape (n,)
        The rows of ONE sex in the zone; the row of age exactly `cfg.omega2` is required, with E > 0 and D > 0.
    cfg : SmootherConfig, optional
        Reads `omega2` (the anchor) and `omega3` (the target age).
    q_target : float, default 0.6
        q at omega3, strictly between 0 and 1. 0.6 is the spec's convention and not a setting of the app; it is far
        below what France 2020 shows (observed q already 0.63-0.86 at 109), which is why the curve fits that table badly.

    Returns
    -------
    FitResult
        method='coale_kisker', zone='zone3'; `ages` = omega2..omega3; `mu_hat` as above; `params` = {'k',
        'q_omega2', 'q_target', 'x0': omega2, 'n_params': 1} (x0 so that the curve can be evaluated from `params` alone); `loglik` the Poisson log-likelihood at the supplied rows;
        `converged` True and `n_iter` 0 (closed form); `message` a French diagnostic, with a warning when the anchor is
        already at or above the target (k <= 0: the extrapolation would not rise).

    Raises
    ------
    ValueError
        Columns of different lengths, a duplicated or non-finite age, negative deaths, `q_target` outside ]0, 1[, no row
        at age omega2, or an anchor row without exposure or without deaths (q = 0 has no logarithm).
    """
    if not 0.0 < q_target < 1.0:
        raise ValueError("q_target doit être strictement compris entre 0 et 1.")
    cfg = cfg if cfg is not None else SmootherConfig()
    x, d, e = check_columns(ages, Dx, Ex)
    anchor = np.flatnonzero(x == float(cfg.omega2))
    if anchor.size == 0:
        raise ValueError(f"Coale-Kisker : la ligne de l'âge ω₂ = {cfg.omega2} (point d'ancrage) est absente.")
    da, ea = d[anchor[0]], e[anchor[0]]
    if not (np.isfinite(da) and np.isfinite(ea) and ea > 0 and da > 0):
        raise ValueError(f"Coale-Kisker : l'âge ω₂ = {cfg.omega2} n'a pas d'exposition ou pas de décès, "
                         f"le taux d'ancrage est inutilisable.")
    q2 = float(-np.expm1(-da / ea))                                             # 1 - exp(-m)
    x0 = float(cfg.omega2)
    k = float((np.log(q_target) - np.log(q2)) / (cfg.omega3 - cfg.omega2))

    def mu_of(ages_):
        q = np.minimum(q2 * np.exp(k * (ages_ - x0)), 1.0 - 1e-15)
        return -np.log1p(-q)

    message = (f"Extrapolation déterministe de q de ω₂ = {cfg.omega2} (q = {q2:.4f}, observé) à ω₃ = {cfg.omega3} "
               f"(cible {q_target:g}) ; k = {k:.5f}.")
    if k <= 0:
        message += " Attention : l'ancrage atteint déjà la cible, k ≤ 0 (l'extrapolation ne croît pas)."
    return _result("coale_kisker", cfg, x, d, e, mu_of,
                   {"k": k, "q_omega2": q2, "q_target": float(q_target), "x0": x0, "n_params": 1}, True, 0, message)
