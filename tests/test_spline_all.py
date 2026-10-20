"""spline_all.py - the natural cubic spline over every age. Run:   python -m pytest tests/test_spline_all.py -v

Not a spec module (asked for afterwards). Evidence:
  * it IS the zone-I spline estimator on a longer span: same numbers as `fit_spline`, only the label and the zone differ;
  * exact recovery of a known natural spline, with scipy's CubicSpline as the independent oracle;
  * the knot rule on ages 0..109 and the FitResult / registry contract.
"""
from __future__ import annotations

import importlib

import numpy as np
import pytest
from scipy.interpolate import CubicSpline

import synthetic as S
from src import config as C
from src.spline_all import fit_spline_all
from src.zone1_poly import fit_spline


def table(hi=100, sex="Female"):
    t = S.make_sex_table(sex).iloc[:hi + 1]
    return t["age"].to_numpy(), t["Dx"].to_numpy().astype(float), t["Ex"].to_numpy()


def test_it_is_the_zone_one_spline_estimator_run_on_every_age():
    a, d, e = table(100)
    g, z = fit_spline_all(a, d, e), fit_spline(a, d, e)
    assert (g.method, g.zone) == ("spline_all", "all") and (z.method, z.zone) == ("spline", "zone1")
    np.testing.assert_array_equal(g.mu_hat, z.mu_hat)
    np.testing.assert_array_equal(g.ages, a)
    assert g.params == z.params and g.loglik == z.loglik and g.converged and g.n_iter == 0
    assert "tous les âges" in g.message and "tous les âges" not in z.message


def test_knots_are_every_five_ages_plus_the_last_one_and_n_params_counts_them():
    a = np.arange(0, 110.0)
    d = 50.0 + 0 * a
    e = 1e5 * np.exp(-0.03 * a)
    g = fit_spline_all(a, d, e)
    assert g.params["knots"] == [*range(0, 106, 5), 109] and g.params["n_params"] == 23 == len(g.params["ln_mu_knots"])


def test_a_known_natural_spline_is_recovered_exactly():
    knots = np.arange(0.0, 101.0, 5.0)
    values = np.log(1e-4) + 0.075 * knots + 0.8 * np.exp(-knots / 6.0)                 # an arbitrary smooth ln mu at the knots
    ages = np.arange(0.0, 101.0)
    ln_mu = CubicSpline(knots, values, bc_type="natural")(ages)                      # the oracle: scipy's natural spline
    e = 3e5 * np.exp(-0.04 * ages)
    g = fit_spline_all(ages, e * np.exp(ln_mu), e)
    np.testing.assert_allclose(g.params["knots"], knots)
    np.testing.assert_allclose(g.params["ln_mu_knots"], values, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(np.log(g.mu_hat), ln_mu, rtol=0, atol=1e-9)


def test_an_age_without_deaths_is_smoothed_in_the_output_but_never_a_knot():
    a, d, e = table(100)
    d = d.copy()
    d[50] = 0.0                                            # a knot age with no death
    g = fit_spline_all(a, d, e)
    assert 50.0 not in g.params["knots"] and len(g.mu_hat) == 101 and g.mu_hat[50] > 0


def test_a_table_too_short_for_three_knots_is_refused_in_terms_of_the_table_not_zone_one():
    a, d, e = table(6)
    with pytest.raises(ValueError, match="table trop courte") as err:
        fit_spline_all(a, d, e)
    assert "zone I" not in str(err.value)


def test_the_registry_finds_the_function_in_the_module_it_names():
    m = C.GLOBAL_METHODS["spline_all"]
    assert getattr(importlib.import_module(f"src.{m.module}"), m.fit_name) is fit_spline_all
