"""Module 5 - evaluator.py. Run:   python -m pytest tests/test_evaluator.py -v

Evidence that does not replay the code's own formulas:
  * the deviance against scipy's xlogy and against the Poisson log-likelihoods of scipy.stats and fit_common;
  * the runs test by ENUMERATING every arrangement of signs: with the right mean and variance, z has mean 0 and
    variance 1 exactly over all of them;
  * hand loops for chi2 and Whittaker-Henderson, scipy's chi2 and normal tails for the p-values;
  * the log-likelihood recomputed here equals the one each of the nine methods stored itself;
  * the spec's selection rule on hand-built tables, and on real France 2020 fits.
"""
from __future__ import annotations

import importlib
import itertools
import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from scipy.special import gammaln, xlogy

import synthetic as S
from src import config as C
from src import evaluator as EV
from src import loader as L
from src.evaluator import (aic, bic, deviance_residuals, deviance_terms, evaluate_fits, evaluate_method, pearson_chi2,
                           residual_series, runs_test, suggest_selection, whittaker_henderson)
from src.fit_common import half_deviance, poisson_loglik
from src.spline_all import fit_spline_all
from pathlib import Path

BOTH = ("Female", "Male")
ROOT = Path(__file__).resolve().parent.parent


def fit_all(df_sex, cfg):
    """Every method, fitted on its own zone with the module the registry names."""
    out = {}
    for m in C.METHODS.values():
        sub = C.zone_slice(df_sex, cfg, m.zone)
        fit = getattr(importlib.import_module(f"src.{m.module}"), m.fit_name)
        out[m.key] = fit(sub["age"].to_numpy(), sub["Dx"].to_numpy(), sub["Ex"].to_numpy(), cfg)
    return out


@pytest.fixture(scope="module")
def france():
    df = L.load_data(ROOT / "sample_data" / "FRA_taux_bruts_moment_2020_female_male.csv")
    cfg = C.default_config(df["age"].unique())
    data = {s: df[df["sex"] == s] for s in BOTH}
    return data, cfg, {s: fit_all(data[s], cfg) for s in BOTH}


@pytest.fixture(scope="module")
def synthetic():
    t = S.make_sex_table("Female")                                        # ages 0..100, Makeham + infant term
    df = pd.DataFrame({"sex": "Female", "age": t["age"], "Dx": t["Dx"].astype(float), "Ex": t["Ex"], "mx_raw": t["Dx"] / t["Ex"]})
    cfg = C.default_config(df["age"].unique())
    return df, cfg, fit_all(df, cfg)


# --------------------------------------------------------------------------
# Deviance and residuals
# --------------------------------------------------------------------------


D0 = np.array([0.0, 3.0, 12.0, 40.0, 101.0, 7.0])
E0 = np.array([1000.0, 900.0, 800.0, 700.0, 600.0, 500.0])
MU0 = np.array([0.001, 0.004, 0.015, 0.05, 0.17, 0.02])


def test_the_deviance_terms_are_the_spec_formula_and_twice_the_half_deviance():
    ehat = E0 * MU0
    np.testing.assert_allclose(deviance_terms(D0, E0, MU0), 2.0 * (xlogy(D0, D0 / ehat) - (D0 - ehat)), rtol=1e-13)
    assert deviance_terms(D0, E0, MU0).sum() == pytest.approx(2.0 * half_deviance(D0, E0, MU0), rel=1e-13)
    assert deviance_terms(D0, E0, MU0)[0] == pytest.approx(2.0 * ehat[0], rel=1e-13)           # D = 0: the term is 2 Ehat
    assert np.all(deviance_terms(D0, E0, MU0) >= 0)
    assert np.all(deviance_terms(D0 + 1, E0, (D0 + 1) / E0) < 1e-12)                          # zero at Ehat = D, up to rounding


def test_the_deviance_is_twice_the_gap_to_the_saturated_poisson_loglik():
    d = np.array([0, 3, 12, 40, 101, 7])
    mu = MU0
    saturated = stats.poisson.logpmf(d, d).sum()
    fitted = stats.poisson.logpmf(d, E0 * mu).sum()
    assert deviance_terms(d, E0, mu).sum() == pytest.approx(2.0 * (saturated - fitted), rel=1e-12)
    assert poisson_loglik(d, E0, mu) == pytest.approx(fitted, rel=1e-13)


def test_deviance_residuals_carry_the_sign_of_the_error_and_square_to_the_deviance():
    r = deviance_residuals(D0, E0, MU0)
    assert np.array_equal(np.sign(r), np.sign(D0 - E0 * MU0))
    assert (r ** 2).sum() == pytest.approx(deviance_terms(D0, E0, MU0).sum(), rel=1e-13)
    assert np.all(np.abs(deviance_residuals(D0 + 1, E0, (D0 + 1) / E0)) < 1e-6)               # sqrt of a ~1e-14 rounding error


def test_deviance_residuals_are_about_standard_normal_when_the_model_is_true():
    """Poisson draws from the true rates: the residuals' mean ~ 0, sd ~ 1, and about 95% inside [-2, 2]."""
    rng = np.random.default_rng(0)
    e = np.full(4000, 2000.0)
    mu = np.full(4000, 0.02)
    r = deviance_residuals(rng.poisson(e * mu), e, mu)
    assert abs(r.mean()) < 0.05 and 0.95 < r.std() < 1.05 and np.mean(np.abs(r) < 2.0) > 0.93


def test_deviance_terms_never_go_negative_and_residuals_never_turn_into_nan_on_exact_fits():
    """Rounding makes 2 [D ln(D/Ehat) - (D - Ehat)] come out as -4e-13 for about 0.3% of exact fits; unclipped, the square root
    of that is NaN. 20 000 random exact fits (fixed seed)."""
    rng = np.random.default_rng(0)
    D, E = rng.uniform(1, 5000, 20000), rng.uniform(100, 1e6, 20000)
    terms, res = deviance_terms(D, E, D / E), deviance_residuals(D, E, D / E)
    assert np.all(terms >= 0) and not np.any(np.isnan(res)) and np.max(np.abs(res)) < 1e-5


# --------------------------------------------------------------------------
# Chi2, AIC, BIC
# --------------------------------------------------------------------------


def test_pearson_chi2_matches_a_hand_loop_the_degrees_of_freedom_and_scipys_tail():
    stat = 0.0
    for d, e, mu in zip(D0, E0, MU0):
        stat += (d - e * mu) ** 2 / (e * mu)
    r = pearson_chi2(D0, E0, MU0, n_params=2)
    assert r.statistic == pytest.approx(stat, rel=1e-13) and r.df == 4
    assert r.p_value == pytest.approx(stats.chi2.sf(stat, 4), rel=1e-12)


def test_pearson_chi2_has_no_p_value_without_a_degree_of_freedom_and_skips_empty_ages():
    assert math.isnan(pearson_chi2(D0, E0, MU0, n_params=6).p_value) and pearson_chi2(D0, E0, MU0, 6).df == 0
    assert pearson_chi2(D0, E0, MU0, n_params=9).df == -3
    e = E0.copy()
    e[2] = 0.0                                                               # Ehat = 0: no contribution, one age less
    r = pearson_chi2(D0, e, MU0, n_params=1)
    assert r.df == 4 and r.statistic == pytest.approx(pearson_chi2(np.delete(D0, 2), np.delete(e, 2), np.delete(MU0, 2), 1).statistic)


def test_aic_and_bic_are_the_spec_formulas():
    assert aic(-100.0, 3) == 206.0 and aic(-100.5, 0) == 201.0
    assert bic(-100.0, 3, 50) == pytest.approx(200.0 + 3 * math.log(50), rel=1e-14)
    assert bic(-100.0, 3, 8) < bic(-100.0, 3, 50) and bic(-100.0, 5, 50) > aic(-100.0, 5)            # BIC punishes p more when ln n > 2


# --------------------------------------------------------------------------
# Runs test
# --------------------------------------------------------------------------


def test_runs_are_counted_on_signs_and_zero_residuals_are_dropped():
    assert runs_test([1, -1, 1, -1, 1, -1]).n_runs == 6
    assert runs_test([1, 1, 1, -1, -1, -1]).n_runs == 2
    r = runs_test([1.0, 0.0, 2.0, -1.0, 0.0, -3.0, 5.0])                    # zeros vanish: + + - - +
    assert (r.n_runs, r.n_pos, r.n_neg) == (3, 3, 2)
    assert runs_test([np.nan, 1.0, -1.0, 1.0]).n_runs == 3


def test_all_residuals_of_one_sign_leave_the_test_undefined():
    for res in ([1.0, 2.0, 3.0], [-1.0, -0.5]):
        r = runs_test(res)
        assert r.n_runs == 1 and math.isnan(r.z) and math.isnan(r.p_value)
    assert runs_test([]).n_runs == 0 and runs_test([0.0, 0.0]).n_runs == 0


@pytest.mark.parametrize("n_pos,n_neg", [(4, 4), (3, 5), (6, 2), (5, 5)])
def test_the_runs_statistic_has_mean_zero_and_variance_one_over_every_arrangement(n_pos, n_neg):
    """Exact moments: over all arrangements of the signs, z = (R - mean) / sd has mean 0 and variance 1 if (and only if)
    the mean and the variance of the runs distribution are the right ones."""
    n = n_pos + n_neg
    zs = []
    for positions in itertools.combinations(range(n), n_pos):
        res = np.full(n, -1.0)
        res[list(positions)] = 1.0
        zs.append(runs_test(res).z)
    zs = np.array(zs)
    assert zs.mean() == pytest.approx(0.0, abs=1e-12) and (zs ** 2).mean() == pytest.approx(1.0, abs=1e-12)


def test_the_runs_p_value_is_the_two_sided_normal_tail_and_extreme_sequences_are_rejected():
    res = np.array([1] * 10 + [-1] * 10, dtype=float)                        # 2 runs out of an expected 11
    r = runs_test(res)
    assert r.n_runs == 2 and r.z < -3.5 and r.p_value == pytest.approx(2 * stats.norm.sf(abs(r.z)), rel=1e-12)
    assert r.p_value < 0.001
    alt = runs_test(np.tile([1.0, -1.0], 10))                                # maximal alternation: too many runs
    assert alt.z > 3.5 and alt.p_value < 0.001
    shuffled = runs_test(np.random.default_rng(3).choice([-1.0, 1.0], size=200))
    assert shuffled.p_value > 0.01                                           # random signs are not rejected (fixed seed)


def test_runs_test_is_calibrated_on_random_signs():
    """Under independent signs the p-value is roughly uniform: about 5% below 0.05 over 400 sequences (fixed seed)."""
    rng = np.random.default_rng(11)
    p = np.array([runs_test(rng.choice([-1.0, 1.0], size=60)).p_value for _ in range(400)])
    assert 0.02 < np.mean(p < 0.05) < 0.09


# --------------------------------------------------------------------------
# Whittaker-Henderson
# --------------------------------------------------------------------------


def test_whittaker_henderson_is_the_spec_formula_with_its_two_terms_reported_apart():
    q = np.array([0.01, 0.012, 0.02, 0.026, 0.05, 0.07])
    z = np.array([0.011, 0.014, 0.019, 0.028, 0.045, 0.072])
    w = np.array([1.0, 2.0, 1.5, 1.0, 0.5, 1.0])
    wh = whittaker_henderson(q, z, w, h=3.0)
    fidelity = sum(w[i] * (q[i] - z[i]) ** 2 for i in range(6))
    rough = sum((z[i + 2] - 2 * z[i + 1] + z[i]) ** 2 for i in range(4))
    assert wh["fidelity"] == pytest.approx(fidelity, rel=1e-13) and wh["roughness"] == pytest.approx(rough, rel=1e-13)
    assert wh["total"] == pytest.approx(fidelity + 3.0 * rough, rel=1e-13) and (wh["h"], wh["order"]) == (3.0, 2)


def test_whittaker_henderson_roughness_is_zero_for_a_straight_line_and_the_order_is_a_parameter():
    z = np.linspace(0.01, 0.1, 10)
    assert whittaker_henderson(z, z, np.ones(10))["roughness"] == pytest.approx(0.0, abs=1e-30)
    cubic = np.arange(10.0) ** 3
    assert whittaker_henderson(cubic, cubic, np.ones(10), order=2)["roughness"] > 0
    assert whittaker_henderson(cubic, cubic, np.ones(10), order=3)["roughness"] == pytest.approx(7 * 36.0, rel=1e-12)    # 7 third differences, each 6
    assert whittaker_henderson(z[:2], z[:2], np.ones(2))["roughness"] == 0.0                                          # too short to differ


# --------------------------------------------------------------------------
# One method, all methods
# --------------------------------------------------------------------------


def test_evaluate_method_is_built_from_the_individual_criteria(synthetic):
    df, cfg, fits = synthetic
    sub = C.zone_slice(df, cfg, "zone2")
    ages, D, E = (sub[c].to_numpy(dtype=float) for c in ("age", "Dx", "Ex"))
    r = fits["makeham"]
    ev = evaluate_method(r, ages, D, E)
    mu = r.mu_hat
    assert ev["k"] == 3 and ev["n"] == len(ages) and ev["converged"] is True
    assert ev["loglik"] == pytest.approx(poisson_loglik(D, E, mu), rel=1e-13)
    assert ev["aic"] == pytest.approx(aic(ev["loglik"], 3)) and ev["bic"] == pytest.approx(bic(ev["loglik"], 3, len(ages)))
    assert ev["chi2"] == pytest.approx(pearson_chi2(D, E, mu, 3).statistic) and ev["df"] == len(ages) - 3
    assert ev["deviance"] == pytest.approx(2 * half_deviance(D, E, mu), rel=1e-12)
    assert ev["outside"] == pytest.approx(np.mean(np.abs(deviance_residuals(D, E, mu)) > 2.0))
    assert ev["runs_ok"] == (ev["runs_p"] > 0.05)


def test_the_whittaker_henderson_fidelity_does_not_depend_on_the_size_of_the_population(synthetic):
    df, cfg, fits = synthetic
    sub = C.zone_slice(df, cfg, "zone2")
    ages, D, E = (sub[c].to_numpy(dtype=float) for c in ("age", "Dx", "Ex"))
    r = fits["gompertz"]
    one, ten = evaluate_method(r, ages, D, E), evaluate_method(r, ages, 10 * D, 10 * E)
    assert one["wh_fidelity"] == pytest.approx(ten["wh_fidelity"], rel=1e-12)           # weights are E / mean(E)
    assert evaluate_method(r, ages, D, E, h=5.0)["wh_total"] - one["wh_fidelity"] == pytest.approx(5.0 * one["wh_roughness"], rel=1e-9)


def test_the_whittaker_henderson_terms_use_the_crude_q_from_the_rate_and_exposure_weights(france):
    """Zone III has rates up to 1.3: q = 1 - exp(-m) differs a lot from m there, and the weights are E / mean(E).
    Both written out by hand against the table's values."""
    data, cfg, fits = france
    sub = C.zone_slice(data["Female"], cfg, "zone3").sort_values("age")
    D, E = sub["Dx"].to_numpy(), sub["Ex"].to_numpy()
    r = fits["Female"]["kannisto"]
    z = 1.0 - np.exp(-r.mu_hat[:len(D)])
    q = 1.0 - np.exp(-D / E)
    w = E / E.mean()
    ev = evaluate_method(r, sub["age"].to_numpy(), D, E, h=7.0)
    assert ev["wh_fidelity"] == pytest.approx(np.sum(w * (q - z) ** 2), rel=1e-12)
    assert ev["wh_roughness"] == pytest.approx(np.sum((z[2:] - 2 * z[1:-1] + z[:-2]) ** 2), rel=1e-12)
    assert ev["wh_fidelity"] != pytest.approx(np.sum(w * (D / E - z) ** 2), rel=1e-3)          # the rate would give another number


def test_the_loglik_is_recomputed_from_the_data_not_read_from_the_result(france):
    import dataclasses
    data, cfg, fits = france
    result = fits["Male"]["gompertz"]
    sub = C.zone_slice(data["Male"], cfg, "zone2")
    args = (sub["age"].to_numpy(dtype=float), sub["Dx"].to_numpy(dtype=float), sub["Ex"].to_numpy(dtype=float))
    honest = evaluate_method(result, *args)
    lying = evaluate_method(dataclasses.replace(result, loglik=123.0), *args)
    assert lying["loglik"] == honest["loglik"] == pytest.approx(result.loglik, rel=1e-12) and lying["aic"] == honest["aic"]


@pytest.mark.parametrize("sex", BOTH)
def test_the_table_does_not_depend_on_the_order_of_the_rows(france, sex):
    """The runs test and Whittaker-Henderson are about consecutive ages: shuffled rows must give the same table."""
    data, cfg, fits = france
    shuffled = data[sex].sample(frac=1.0, random_state=4)
    pd.testing.assert_frame_equal(evaluate_fits(fits[sex], shuffled, cfg), evaluate_fits(fits[sex], data[sex], cfg))
    pd.testing.assert_frame_equal(residual_series(fits[sex], shuffled, cfg), residual_series(fits[sex], data[sex], cfg))


@pytest.mark.parametrize("sex", BOTH)
def test_the_loglik_recomputed_here_is_the_one_each_of_the_nine_methods_stored(france, sex):
    """One scale for AIC: the table's loglik (on the zone's data rows) must equal what each module computed itself."""
    data, cfg, fits = france
    table = evaluate_fits(fits[sex], data[sex], cfg).set_index("method")
    for key, result in fits[sex].items():
        assert table.loc[key, "loglik"] == pytest.approx(result.loglik, rel=1e-12, abs=1e-9), key
        assert table.loc[key, "k"] == result.params["n_params"] and bool(table.loc[key, "converged"]) is result.converged


@pytest.mark.parametrize("sex", BOTH)
def test_the_table_of_all_methods_is_complete_and_consistent(france, sex):
    data, cfg, fits = france
    t = evaluate_fits(fits[sex], data[sex], cfg)
    assert list(t.columns) == EV.TABLE_COLUMNS and list(t["method"]) == list(C.METHODS)        # registry order
    assert list(t["zone"]) == [C.METHODS[m].zone for m in t["method"]] and list(t["label"]) == [C.METHODS[m].label for m in t["method"]]
    n_rows = {z: len(C.zone_slice(data[sex], cfg, z)) for z in C.ZONES}
    assert list(t["n"]) == [n_rows[z] for z in t["zone"]]                                     # zone III: 25 data rows, not 26
    assert np.array_equal(t["df"], t["n"] - t["k"])
    for zone, sub in t.groupby("zone"):
        assert sub["delta_aic"].min() == 0.0 and np.all(sub["delta_aic"] >= 0)
        assert sub["suggested"].sum() == 1
        np.testing.assert_allclose(sub["aic"], -2 * sub["loglik"] + 2 * sub["k"], rtol=1e-13)
        np.testing.assert_allclose(sub["bic"], -2 * sub["loglik"] + sub["k"] * np.log(sub["n"]), rtol=1e-13)
    assert np.all(t["wh_total"] >= t["wh_fidelity"]) and np.all((t["outside"] >= 0) & (t["outside"] <= 1))


def test_france_2020_the_numbers_of_the_table_and_the_suggested_methods(france):
    """Pinned (measured): women, zone II, Makeham 2nd order wins by ~1800 AIC; zone III, Kannisto is suggested over Beard
    (Beard has the lower AIC by 6 but fails the runs test, p = 0.025); Coale-Kisker has only negative residuals."""
    data, cfg, fits = france
    t = {s: evaluate_fits(fits[s], data[s], cfg).set_index("method") for s in BOTH}
    pinned_aic = {"poly3": 444.0901, "spline": 456.8613, "gompertz": 3163.6535, "makeham": 2739.2455, "makeham2": 946.2302,
                  "perks": 2741.2455, "kannisto": 617.8678, "beard": 611.8201, "coale_kisker": 17301.9744}
    for key, value in pinned_aic.items():
        assert t["Female"].loc[key, "aic"] == pytest.approx(value, abs=2e-3 if key != "makeham2" else 5e-3), key
    assert t["Female"].loc["beard", "runs_p"] == pytest.approx(0.0249, abs=5e-4) and not t["Female"].loc["beard", "runs_ok"]
    assert t["Female"].loc["kannisto", "runs_ok"] and math.isnan(t["Female"].loc["coale_kisker", "runs_p"])
    assert [m for m in C.METHODS if t["Female"].loc[m, "suggested"]] == ["poly3", "makeham2", "kannisto"]
    assert [m for m in C.METHODS if t["Male"].loc[m, "suggested"]] == ["poly3", "makeham2", "beard"]
    assert t["Female"].loc["perks", "aic"] - t["Female"].loc["makeham", "aic"] == pytest.approx(2.0, abs=1e-3)   # Perks = Makeham + a useless p


def test_a_method_without_data_in_its_zone_or_an_empty_result_gives_an_empty_or_shorter_table(france):
    data, cfg, fits = france
    assert evaluate_fits({}, data["Female"], cfg).empty and list(evaluate_fits({}, data["Female"], cfg).columns) == EV.TABLE_COLUMNS
    only3 = evaluate_fits({"kannisto": fits["Female"]["kannisto"]}, data["Female"], cfg)
    assert list(only3["method"]) == ["kannisto"] and only3["suggested"].tolist() == [True] and only3["delta_aic"].tolist() == [0.0]
    short = data["Female"][data["Female"]["age"] < 85]                       # no row in zone III at all
    assert list(evaluate_fits(fits["Female"], short, cfg)["zone"].unique()) == ["zone1", "zone2"]


def test_rows_without_exposure_are_left_out_of_n_and_of_the_criteria(synthetic):
    df, cfg, fits = synthetic
    broken = df.copy()
    broken.loc[broken["age"] == 50, "Ex"] = 0.0
    base = evaluate_fits(fits, df, cfg).set_index("method")
    t = evaluate_fits(fits, broken, cfg).set_index("method")
    assert t.loc["gompertz", "n"] == base.loc["gompertz", "n"] - 1 and t.loc["poly3", "n"] == base.loc["poly3", "n"]
    assert np.isfinite(t["aic"]).all()


# --------------------------------------------------------------------------
# The suggested selection (spec 5.3) on hand-built tables
# --------------------------------------------------------------------------


def table(rows):
    """rows: (zone, method, aic, runs_p, converged)."""
    return pd.DataFrame([{"zone": z, "method": m, "aic": a, "runs_p": p, "converged": c} for z, m, a, p, c in rows])


def test_the_rule_is_the_lowest_aic_among_the_methods_that_pass_the_runs_test():
    t = table([("zone2", "gompertz", 300.0, 0.40, True), ("zone2", "makeham", 100.0, 0.01, True),
               ("zone2", "makeham2", 150.0, 0.30, True), ("zone2", "perks", 120.0, 0.06, True)])
    assert suggest_selection(t)["zone2"] == "perks"                          # makeham has the lowest AIC but fails the runs test


def test_when_no_method_passes_the_runs_test_the_lowest_aic_is_suggested_anyway():
    t = table([("zone1", "poly3", 100.0, 0.001, True), ("zone1", "spline", 90.0, 0.01, True)])
    assert suggest_selection(t)["zone1"] == "spline"


def test_a_nan_runs_p_value_does_not_pass_and_p_equal_to_alpha_does_not_either():
    t = table([("zone3", "kannisto", 200.0, 0.06, True), ("zone3", "coale_kisker", 10.0, np.nan, True), ("zone3", "beard", 50.0, 0.05, True)])
    assert suggest_selection(t)["zone3"] == "kannisto"                       # CK (NaN) and Beard (p = 0.05, not > 0.05) both fail


def test_a_non_converged_method_is_never_suggested_while_another_converged():
    t = table([("zone2", "makeham", 100.0, 0.5, False), ("zone2", "gompertz", 300.0, 0.5, True)])
    assert suggest_selection(t)["zone2"] == "gompertz"
    allfail = table([("zone2", "makeham", 100.0, 0.5, False), ("zone2", "gompertz", 300.0, 0.5, False)])
    assert suggest_selection(allfail)["zone2"] == "makeham"                   # nobody converged: the rule still answers


def test_infinite_aic_is_ignored_and_a_zone_without_a_usable_method_has_no_suggestion():
    t = table([("zone1", "poly3", np.inf, 0.5, True), ("zone1", "spline", 50.0, 0.5, True), ("zone2", "makeham", np.nan, 0.5, True)])
    s = suggest_selection(t)
    assert s == {"zone1": "spline", "zone2": None, "zone3": None}
    assert suggest_selection(t, alpha=0.9)["zone1"] == "spline"             # alpha is a parameter; nothing passes: lowest AIC


def test_the_alpha_of_the_rule_is_a_parameter():
    t = table([("zone2", "makeham", 100.0, 0.04, True), ("zone2", "gompertz", 200.0, 0.30, True)])
    assert suggest_selection(t)["zone2"] == "gompertz" and suggest_selection(t, alpha=0.01)["zone2"] == "makeham"


# --------------------------------------------------------------------------
# Residual series for the chart
# --------------------------------------------------------------------------


@pytest.mark.parametrize("sex", BOTH)
def test_the_residual_series_has_one_line_per_method_on_its_zones_data_ages(france, sex):
    data, cfg, fits = france
    rs = residual_series(fits[sex], data[sex], cfg)
    assert list(rs.columns) == ["zone", "method", "label", "age", "residual"] and set(rs["method"]) == set(C.METHODS)
    kan = rs[rs["method"] == "kannisto"]
    assert list(kan["age"]) == list(range(85, 110))                          # the 25 data rows; the curve to 110 has no data
    sub = C.zone_slice(data[sex], cfg, "zone3")
    np.testing.assert_allclose(kan["residual"], deviance_residuals(sub["Dx"], sub["Ex"], fits[sex]["kannisto"].mu_hat[:25]), rtol=1e-12)
    assert residual_series({}, data[sex], cfg).empty


# --------------------------------------------------------------------------
# The global spline: one row per zone, read on that zone's data
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def with_global(france):
    data, cfg, fits = france
    out = {}
    for sex in BOTH:
        sub = C.zone_slice(data[sex], cfg, "all")
        out[sex] = {**fits[sex], "spline_all": fit_spline_all(sub["age"].to_numpy(), sub["Dx"].to_numpy(), sub["Ex"].to_numpy(), cfg)}
    return data, cfg, out


def test_the_zone_results_are_grouped_by_zone_with_the_global_method_last_in_each(with_global):
    _, cfg, fits = with_global
    got = [(spec.key, zone, r.zone) for spec, zone, r in EV.zone_results(fits["Female"], cfg)]
    assert got == [("poly3", "zone1", "zone1"), ("spline", "zone1", "zone1"), ("spline_all", "zone1", "zone1"),
                   ("gompertz", "zone2", "zone2"), ("makeham", "zone2", "zone2"), ("makeham2", "zone2", "zone2"),
                   ("perks", "zone2", "zone2"), ("spline_all", "zone2", "zone2"),
                   ("kannisto", "zone3", "zone3"), ("beard", "zone3", "zone3"), ("coale_kisker", "zone3", "zone3"),
                   ("spline_all", "zone3", "zone3")]
    assert [spec.key for spec, zone, r in EV.zone_results(fits["Female"], cfg)][:2] == ["poly3", "spline"]


def test_without_the_global_method_the_table_is_the_nine_rows_of_before(with_global, france):
    data, cfg, fits = with_global
    nine = EV.evaluate_fits(france[2]["Female"], data["Female"], cfg)
    assert list(nine["method"]) == list(C.METHODS) and len(nine) == 9


def test_the_global_spline_gets_a_row_in_every_zone_charged_for_its_knots_in_that_zone(with_global):
    data, cfg, fits = with_global
    t = EV.evaluate_fits(fits["Female"], data["Female"], cfg)
    g = t[t["method"] == "spline_all"]
    assert len(t) == 12 and list(g["zone"]) == ["zone1", "zone2", "zone3"] and list(g["label"].unique()) == ["Spline globale (tous âges)"]
    knots = np.array(fits["Female"]["spline_all"].params["knots"])
    assert len(knots) == 23
    assert list(g["k"]) == [5, 14, 6]                                              # knots in [0,20], [20,85], [85,110]
    assert list(g["k"]) == [int(((knots >= lo) & (knots <= hi)).sum()) for lo, hi in ((0, 20), (20, 85), (85, 110))]
    assert list(g["n"]) == [21, 66, 25]
    np.testing.assert_allclose(g["aic"], -2 * g["loglik"] + 2 * g["k"], rtol=1e-13)


def test_each_zone_row_of_the_global_spline_is_the_poisson_loglik_of_the_one_curve_on_that_zones_ages(with_global):
    data, cfg, fits = with_global
    g, d = fits["Female"]["spline_all"], data["Female"]
    t = EV.evaluate_fits(fits["Female"], d, cfg)
    for zone in C.ZONES:
        sub = C.zone_slice(d, cfg, zone)
        mu = g.mu_hat[sub["age"].to_numpy().astype(int)]                          # the one curve, read on this zone's ages
        D, E = sub["Dx"].to_numpy(), sub["Ex"].to_numpy()                      # deaths are not whole numbers here: gammaln, not logpmf
        want = float(np.sum(xlogy(D, E * mu) - E * mu - gammaln(D + 1.0)))
        row = t[(t["method"] == "spline_all") & (t["zone"] == zone)].iloc[0]
        assert row["loglik"] == pytest.approx(want, rel=1e-12)
    assert t.groupby("zone")["delta_aic"].min().eq(0).all() and t.groupby("zone")["suggested"].sum().eq(1).all()


def test_the_global_spline_residuals_are_the_deviance_residuals_of_one_curve_zone_after_zone(with_global):
    data, cfg, fits = with_global
    g, d = fits["Female"]["spline_all"], data["Female"]
    rs = EV.residual_series(fits["Female"], d, cfg)
    r = rs[rs["method"] == "spline_all"]
    assert list(r["zone"].unique()) == ["zone1", "zone2", "zone3"] and len(r) == 21 + 66 + 25
    assert (np.diff(r["age"]) >= 0).all() and (np.diff(r["age"]) == 0).sum() == 2           # increasing, the two boundary ages twice
    D, E = d.set_index("age").loc[r["age"], "Dx"].to_numpy(), d.set_index("age").loc[r["age"], "Ex"].to_numpy()
    mu = g.mu_hat[r["age"].to_numpy().astype(int)]
    want = np.sign(D - E * mu) * np.sqrt(2.0 * (xlogy(D, D / (E * mu)) - (D - E * mu)))
    np.testing.assert_allclose(r["residual"], want, rtol=1e-10, atol=1e-12)


def test_on_france_2020_women_the_global_spline_wins_the_aic_of_zones_ii_and_iii(with_global):
    """A finding, not a rule: with exposures this large the flexible curve beats the parametric laws of zones II and III
    (Makeham 2nd order 946 against 640 in zone II), so the suggestion moves to it when it is ticked."""
    data, cfg, fits = with_global
    t = EV.evaluate_fits(fits["Female"], data["Female"], cfg)
    assert {z: m for z, m in zip(t["zone"], t["method"]) if t.loc[(t["zone"] == z) & (t["method"] == m), "suggested"].iloc[0]} \
        == {"zone1": "poly3", "zone2": "spline_all", "zone3": "spline_all"}
