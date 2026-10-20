"""spline_all.py - a natural cubic spline in ln(mu) over EVERY age, from the first loaded age to omega3.

Not one of the specs' seven modules: asked for afterwards ("choisir splines sur toute la longueur des âges"). The zone methods
each describe one zone; this one is fitted once on all the ages and is then offered in every zone (`config.GLOBAL_METHODS`).
Choose it in all three zones and the final table is this single curve; choose it in one zone and it is blended with the
neighbouring zone's curve by the usual tanh windows.

The estimator is `zone1_poly.fit_spline`, unchanged, run on the whole table instead of on zone I::

    ln mu_x = S(x),   S natural cubic spline, knots every 5 ages from the first age (plus the last), WLS with weights D_x

so the knot rule, the exclusion of the ages without deaths, the straight-line continuation beyond the end knots and the
closed-form solve are the spline of zone I (see that module). What changes is the label of the result, `method='spline_all'`,
`zone='all'`, and the span: the input is every row up to omega3 (`config.zone_slice(df, cfg, 'all')`).

`params['n_params']` is the number of knots of the whole fit. A zone does not pay for all of them: `config.zone_view` hands
each zone the knots that fall inside its own bounds.

Public API::

    fit_spline_all(ages, Dx, Ex, cfg) -> FitResult(method='spline_all', zone='all', ...)
"""
from __future__ import annotations

from dataclasses import replace

from src.config import ZONE_ALL, FitResult, SmootherConfig
from src.zone1_poly import fit_spline


def fit_spline_all(ages, Dx, Ex, cfg: SmootherConfig | None = None, knot_step: int = 5) -> FitResult:
    """Natural cubic spline in ln(mu_x) over all the ages given (see the module docstring).

    Parameters
    ----------
    ages, Dx, Ex : array-like, shape (n,)
        The rows of ONE sex from the first loaded age to omega3, one row per age (`config.zone_slice(df, cfg, 'all')`).
    cfg : SmootherConfig, optional
        Accepted for the common `fit_*` protocol and not used.
    knot_step : int, default 5
        Knot spacing, as in `fit_spline` (not a setting of the app).

    Returns
    -------
    FitResult
        method='spline_all', zone='all'; `mu_hat` the spline at every input age; `params` = {'knots', 'ln_mu_knots',
        'n_params' (= the number of knots)}; `message` the spline's, mentioning that it covers all ages.

    Raises
    ------
    ValueError
        As `fit_spline`: invalid columns, `knot_step` < 1, or fewer than 3 usable knots.
    """
    try:
        res = fit_spline(ages, Dx, Ex, cfg, knot_step=knot_step)
    except ValueError as exc:                                      # the spline's messages talk about zone I
        raise ValueError(str(exc).replace("zone I trop courte", "table trop courte")) from exc
    return replace(res, method="spline_all", zone=ZONE_ALL,
                   message=res.message.replace("spline cubique naturelle", "spline cubique naturelle sur tous les âges", 1))
