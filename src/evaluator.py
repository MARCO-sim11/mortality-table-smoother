"""evaluator.py - Module 5: how well does each fitted method fit, and which one to keep in each zone.

Spec 5.1: "calculer et comparer les criteres de qualite pour chaque methode et chaque zone. Permettre la selection de
la methode finale par zone." Spec 5.2 lists the criteria, all computed here from the FITTED rates only (a `FitResult`'s
curve read on the data ages of its own zone), so every method, zone I to III, is judged the same way::

    chi2 de Pearson     sum (D - Ehat)^2 / Ehat,   Ehat = E mu_hat,   compared with chi2(n - p)
    deviance            2 sum [ D ln(D / Ehat) - (D - Ehat) ]          (the D ln term is 0 where D = 0)
    AIC, BIC            -2 loglik + 2 p,   -2 loglik + p ln(n)
    runs test           sign runs of the deviance residuals: too few = a systematic misfit
    deviance residuals  sign(D - Ehat) sqrt(d_x),  d_x the age's term of the deviance (methodology: within [-2, +2])
    Whittaker-Henderson sum w (q - z)^2 + h sum (Delta^2 z)^2,   q the crude q, z the smoothed q

and the spec's suggested rule, `suggest_selection`: the lowest AIC of each zone, among the methods whose runs test is
not significant (p > 0.05).

What is compared with what. AIC and BIC are only meaningful BETWEEN methods of the same zone (same data rows); the table
therefore carries `delta_aic`, the gap to the best AIC of the zone. `loglik` is recomputed here with
`fit_common.poisson_loglik` on the zone's data rows (for the zone-III methods too, which are fitted on rows that stop at
the table's last age but whose curve runs to omega3), so all nine methods share one scale; `p` is the method's own
`params['n_params']`, because the spline's depends on its knots; `n` is the number of data rows with E > 0 in the zone.

Decisions the specs leave open, made here and flagged to the user:

  * WHITTAKER-HENDERSON. Spec 5.2 gives the formula but neither the weights nor h. Weights: w = E / mean(E) (exposure, scaled
    so that the fidelity term does not depend on the size of the population); h = 1 by default, a setting of tab 3; order 2
    (spec 5.2 writes Delta^2; the methodology doc writes Delta^n). The table gives the two terms separately and their sum
    at h, because only their ratio is what h trades off, and across methods the roughness alone is the comparable number.
  * RUNS TEST. Normal approximation of the Wald-Wolfowitz distribution, two-sided; zero residuals are dropped. If every
    residual has the same sign the test is not defined (p = NaN) and such a method does NOT pass the "p > 0.05" rule.
  * SUGGESTION. A method that did not converge is never suggested while another of its zone did; if no method of a zone
    passes the runs test, the lowest AIC is suggested anyway (and `runs_ok` says so).

Tab 3 of the app (which calls this module, shows the table and the residual and fit charts, lets the user pick one
method per zone and runs `joiner.join_zones` on "Valider la selection") is in app.py.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd
from scipy import stats

from src import config as C
from src.config import FitResult, SmootherConfig
from src.fit_common import poisson_loglik
from src.joiner import mu_to_q, zone_values

WH_H = 1.0                    # Whittaker-Henderson smoothing parameter h (a setting of tab 3)
WH_ORDER = 2                  # spec 5.2: Delta^2
RUNS_ALPHA = 0.05             # spec 5.3: "runs test non significatif (p > 0.05)"
RESIDUAL_BOUND = 2.0          # methodology 6.1: deviance residuals expected within [-2, +2]

CRITERIA_LABELS = {
    "sex": "Sexe", "zone": "Zone", "method": "Clé", "label": "Méthode", "k": "p", "n": "n", "loglik": "Log-vraisemblance",
    "chi2": "χ² Pearson", "df": "ddl", "chi2_p": "p(χ²)", "deviance": "Déviance", "aic": "AIC", "delta_aic": "ΔAIC",
    "bic": "BIC", "runs": "Runs", "runs_p": "p(runs)", "runs_ok": "Runs OK", "outside": "|r| > 2",
    "wh_fidelity": "WH fidélité", "wh_roughness": "WH rugosité", "wh_total": "WH total",
    "converged": "Convergé", "suggested": "Suggérée",
}
TABLE_COLUMNS = ["zone", "method", "label", "k", "n", "loglik", "chi2", "df", "chi2_p", "deviance", "aic", "delta_aic",
                 "bic", "runs", "runs_p", "runs_ok", "outside", "wh_fidelity", "wh_roughness", "wh_total",
                 "converged", "suggested"]


class Chi2Result(NamedTuple):
    statistic: float
    df: int
    p_value: float


class RunsResult(NamedTuple):
    n_runs: int
    n_pos: int
    n_neg: int
    z: float
    p_value: float


# --------------------------------------------------------------------------
# The criteria of spec 5.2
# --------------------------------------------------------------------------


def _as_float(*arrays):
    return tuple(np.asarray(a, dtype=float) for a in arrays)


def deviance_terms(D, E, mu) -> np.ndarray:
    """Each age's term of the Poisson deviance, d_x = 2 [ D ln(D / Ehat) - (D - Ehat) ], Ehat = E mu (spec 5.2).

    The D ln(D / Ehat) term is 0 where D = 0 (the limit of x ln x), and d_x is clipped at 0 against rounding noise at
    ages where D = Ehat. d_x >= 0, and the sum is twice the `fit_common.half_deviance` of the fit."""
    D, E, mu = _as_float(D, E, mu)
    e_hat = E * mu
    with np.errstate(divide="ignore", invalid="ignore"):
        first = np.where(D > 0, D * np.log(D / e_hat), 0.0)
    return np.maximum(2.0 * (first - (D - e_hat)), 0.0)


def deviance_residuals(D, E, mu) -> np.ndarray:
    """sign(D - Ehat) sqrt(d_x) (spec 5.2): the age-by-age diagnostic. About N(0, 1) when the model is right, so most
    should lie within [-2, +2] (methodology 6.1). Their squares add up to the deviance."""
    D, E, mu = _as_float(D, E, mu)
    return np.sign(D - E * mu) * np.sqrt(deviance_terms(D, E, mu))


def pearson_chi2(D, E, mu, n_params: int) -> Chi2Result:
    """Pearson's chi2 = sum (D - Ehat)^2 / Ehat over the ages with Ehat > 0 (spec 5.2), its degrees of freedom
    n - n_params, and the upper-tail p-value of chi2(n - n_params). p is NaN when there is no degree of freedom left.
    Unreliable where Ehat < 5 (the oldest ages of zone III), like any chi2 on sparse counts."""
    D, E, mu = _as_float(D, E, mu)
    e_hat = E * mu
    ok = e_hat > 0
    stat = float(np.sum((D[ok] - e_hat[ok]) ** 2 / e_hat[ok]))
    df = int(ok.sum()) - int(n_params)
    return Chi2Result(stat, df, float(stats.chi2.sf(stat, df)) if df >= 1 else float("nan"))


def aic(loglik: float, n_params: int) -> float:
    """AIC = -2 loglik + 2 p (spec 5.2)."""
    return -2.0 * float(loglik) + 2.0 * int(n_params)


def bic(loglik: float, n_params: int, n_obs: int) -> float:
    """BIC = -2 loglik + p ln(n) (spec 5.2), n the number of data ages."""
    return -2.0 * float(loglik) + int(n_params) * float(np.log(n_obs))


def runs_test(residuals) -> RunsResult:
    """Wald-Wolfowitz runs test on the SIGNS of the residuals, in age order (spec 5.2).

    A run is a stretch of residuals of one sign. With n+ positive and n- negative residuals, the number of runs R has,
    under independence, mean 2 n+ n- / n + 1 and variance 2 n+ n- (2 n+ n- - n) / (n^2 (n - 1)); z = (R - mean) / sd,
    two-sided p from the normal. Too few runs: whole stretches of ages over- or under-estimated (a wrong functional
    form); too many: the residuals alternate more than chance. Zero residuals are dropped. All residuals of one sign:
    z and p are NaN (nothing to test), and `n_runs` is 1.
    """
    r = np.asarray(residuals, dtype=float)
    signs = np.sign(r[np.isfinite(r) & (r != 0)])
    n_pos, n_neg = int(np.sum(signs > 0)), int(np.sum(signs < 0))
    n = n_pos + n_neg
    if n_pos == 0 or n_neg == 0:
        return RunsResult(1 if n else 0, n_pos, n_neg, float("nan"), float("nan"))
    n_runs = 1 + int(np.sum(signs[1:] != signs[:-1]))
    mean = 2.0 * n_pos * n_neg / n + 1.0
    var = 2.0 * n_pos * n_neg * (2.0 * n_pos * n_neg - n) / (n ** 2 * (n - 1.0))
    z = (n_runs - mean) / np.sqrt(var) if var > 0 else 0.0
    return RunsResult(n_runs, n_pos, n_neg, float(z), float(2.0 * stats.norm.sf(abs(z))))


def whittaker_henderson(q_obs, z, weights, h: float = WH_H, order: int = WH_ORDER) -> dict:
    """WH(z) = sum w (q - z)^2 + h sum (Delta^order z)^2 (spec 5.2), as a diagnostic of a curve already fitted.

    Returns {'fidelity', 'roughness', 'total', 'h', 'order'}. `q_obs` and `z` are the crude and smoothed q by
    consecutive ages, `weights` the w. With no more than `order` ages the roughness is 0."""
    q_obs, z, weights = _as_float(q_obs, z, weights)
    fidelity = float(np.sum(weights * (q_obs - z) ** 2))
    roughness = float(np.sum(np.diff(z, order) ** 2)) if z.size > order else 0.0
    return {"fidelity": fidelity, "roughness": roughness, "total": fidelity + h * roughness, "h": h, "order": order}


# --------------------------------------------------------------------------
# One method, all methods
# --------------------------------------------------------------------------


def _zone_data(df_sex: pd.DataFrame, cfg: SmootherConfig, zone: str):
    """The rows of the zone with something to fit (E > 0, finite), in age order: (ages, D, E)."""
    sub = C.zone_slice(df_sex, cfg, zone).sort_values("age")
    ok = np.isfinite(sub["Dx"]) & np.isfinite(sub["Ex"]) & (sub["Ex"] > 0)
    sub = sub[ok.to_numpy()]
    return sub["age"].to_numpy(dtype=float), sub["Dx"].to_numpy(dtype=float), sub["Ex"].to_numpy(dtype=float)


def n_params_of(result: FitResult) -> int:
    """The method's own parameter count (`params['n_params']`; the registry's for a result that lacks it)."""
    return int(result.params.get("n_params", C.method_spec(result.method).n_params))


def evaluate_method(result: FitResult, ages, D, E, h: float = WH_H) -> dict:
    """All the criteria of one method on the data rows (`ages`, `D`, `E`) of its zone: a dict of the table's columns
    (`loglik, k, n, chi2, df, chi2_p, deviance, aic, bic, runs, runs_p, runs_ok, outside, wh_*, converged`).

    The curve is read at `ages` with `joiner.zone_values` (the stored `mu_hat` of the method where it has one). `n` is
    the number of rows; there must be at least one."""
    ages, D, E = _as_float(ages, D, E)
    mu = zone_values(result, ages)
    k, n = n_params_of(result), int(ages.size)
    loglik = poisson_loglik(D, E, mu)
    chi = pearson_chi2(D, E, mu, k)
    res = deviance_residuals(D, E, mu)
    runs = runs_test(res)
    wh = whittaker_henderson(mu_to_q(D / E), mu_to_q(mu), E / np.mean(E), h)
    return {"k": k, "n": n, "loglik": loglik, "chi2": chi.statistic, "df": chi.df, "chi2_p": chi.p_value,
            "deviance": float(np.sum(deviance_terms(D, E, mu))), "aic": aic(loglik, k), "bic": bic(loglik, k, n),
            "runs": runs.n_runs, "runs_p": runs.p_value, "runs_ok": bool(runs.p_value > RUNS_ALPHA),
            "outside": float(np.mean(np.abs(res) > RESIDUAL_BOUND)), "wh_fidelity": wh["fidelity"],
            "wh_roughness": wh["roughness"], "wh_total": wh["total"], "converged": bool(result.converged)}


def suggest_selection(table: pd.DataFrame, alpha: float = RUNS_ALPHA) -> dict:
    """The spec's suggested rule (5.3): per zone, the lowest AIC among the methods whose runs test is not significant.

    Returns {zone: method_key or None}. Among a zone's methods, those with a non-finite AIC are ignored; a method that
    did not converge is ignored when another converged; if none passes the runs test (p > alpha; a NaN p does not pass)
    the lowest AIC is returned anyway. None for a zone with no usable method."""
    out = {}
    for zone in C.ZONES:
        pool = table[(table["zone"] == zone) & np.isfinite(table["aic"])]
        if pool.empty:
            out[zone] = None
            continue
        if pool["converged"].any():
            pool = pool[pool["converged"]]
        passing = pool[pool["runs_p"] > alpha]
        out[zone] = str((passing if len(passing) else pool).sort_values("aic").iloc[0]["method"])
    return out


def zone_results(fit_results: dict, cfg: SmootherConfig) -> list:
    """[(method spec, zone, result as that zone sees it)] for every estimated method of one sex, grouped by zone.

    A zone method appears once, in its own zone. A global method (the spline over all the ages) appears once PER ZONE, as
    `config.zone_view` gives it: the same curve, with the number of parameters of the knots inside that zone. Within a zone
    the order is the registry's (zone methods, then the global ones)."""
    out = [(spec, zone, C.zone_view(fit_results[spec.key], zone, cfg))
           for spec in C.every_method() if spec.key in fit_results for zone in C.zones_served(spec)]
    return sorted(out, key=lambda item: C.ZONES.index(item[1]))               # stable: keeps the registry order inside a zone


def evaluate_fits(fit_results: dict, df_sex: pd.DataFrame, cfg: SmootherConfig, h: float = WH_H) -> pd.DataFrame:
    """The comparison table of spec 5.3: one row per estimated method of ONE sex, every criterion of spec 5.2 (a global
    method gets one row in each zone, read on that zone's data with that zone's share of the parameters: `zone_results`).

    Parameters
    ----------
    fit_results : {method_key: FitResult}
        What the zone modules returned for this sex (`session_state.fit_results[sex]`).
    df_sex : DataFrame
        This sex's rows of the loaded table (age, Dx, Ex, ...).
    cfg : SmootherConfig
        This sex's settings (the zone bounds).
    h : float
        Whittaker-Henderson parameter.

    Returns
    -------
    DataFrame with the columns `TABLE_COLUMNS`: zone ('zone1'..'zone3'), method (key), label, k, n, loglik, chi2, df,
    chi2_p, deviance, aic, delta_aic (to the best AIC of the zone), bic, runs, runs_p, runs_ok, outside (share of
    |residuals| > 2), wh_fidelity, wh_roughness, wh_total, converged, suggested (the rule of `suggest_selection`).
    Rows are grouped by zone and follow the registry order inside it (`zone_results`); empty when there is nothing to
    evaluate. A zone with no data row with exposure leaves its methods out.
    """
    rows = []
    for method, zone, result in zone_results(fit_results, cfg):
        ages, D, E = _zone_data(df_sex, cfg, zone)
        if ages.size == 0:
            continue
        rows.append({"zone": zone, "method": method.key, "label": method.label,
                     **evaluate_method(result, ages, D, E, h)})
    if not rows:
        return pd.DataFrame(columns=TABLE_COLUMNS)
    table = pd.DataFrame(rows)
    table["delta_aic"] = table["aic"] - table.groupby("zone")["aic"].transform("min")
    picks = suggest_selection(table)
    table["suggested"] = [picks[z] == m for z, m in zip(table["zone"], table["method"])]
    return table[TABLE_COLUMNS]


def residual_series(fit_results: dict, df_sex: pd.DataFrame, cfg: SmootherConfig) -> pd.DataFrame:
    """The deviance residuals of every estimated method, by age, for the residual chart of tab 3 (spec 5.3: "une ligne
    par methode"): columns zone, method, label, age, residual. A global method has its residuals of the three zones one
    after the other (ages increasing, the two boundary ages twice)."""
    parts = []
    for method, zone, result in zone_results(fit_results, cfg):
        ages, D, E = _zone_data(df_sex, cfg, zone)
        if ages.size:
            parts.append(pd.DataFrame({"zone": zone, "method": method.key, "label": method.label, "age": ages,
                                       "residual": deviance_residuals(D, E, zone_values(result, ages))}))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["zone", "method", "label", "age", "residual"])
