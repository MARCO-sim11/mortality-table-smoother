"""src/fit_common.py - the input checks and the Poisson log-likelihood shared by the zone modules.
Run:   python -m pytest tests/test_fit_common.py -v
"""
from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.stats import poisson

from src.fit_common import (NelderMeadResult, check_columns, half_deviance, nelder_mead, poisson_loglik,
                            refit_with_zero)


def test_check_columns_returns_float_arrays_in_the_callers_order():
    x, d, e = check_columns([3, 1, 2], [5, 0, 7], [100, 200, 300])
    assert [a.dtype for a in (x, d, e)] == [np.float64] * 3
    np.testing.assert_array_equal(x, [3, 1, 2])
    np.testing.assert_array_equal(d, [5, 0, 7])


@pytest.mark.parametrize("args,msg", [
    (([1, 2, 3], [1, 2], [1, 2, 3]), "même longueur"),                      # Dx too short
    (([1, 2, 3], [1, 2, 3], [1, 2]), "même longueur"),                      # Ex too short
    (([[1, 2]], [[1, 2]], [[1, 2]]), "même longueur"),                      # not 1-D
    (([1, 2, np.nan], [1, 2, 3], [1, 2, 3]), "non fini"),
    (([1, 2, 2], [1, 2, 3], [1, 2, 3]), "dupliqués"),
    (([1, 2, 3], [1, -2, 3], [1, 2, 3]), "négatif"),
])
def test_check_columns_refuses_malformed_input_in_french(args, msg):
    with pytest.raises(ValueError, match=msg):
        check_columns(*args)


def test_check_columns_lets_missing_deaths_or_exposure_through():
    """Such an age carries no information; the fits ignore it (they do not fail on it)."""
    x, d, e = check_columns([1, 2, 3], [1, np.nan, 3], [1, 2, np.inf])
    assert np.isnan(d[1]) and np.isinf(e[2])


def test_poisson_loglik_equals_the_scipy_log_pmf_for_integer_deaths():
    d = np.array([0, 3, 12, 40, 101])
    e = np.array([1000.0, 900.0, 800.0, 700.0, 600.0])
    mu = np.array([0.001, 0.004, 0.015, 0.05, 0.17])
    assert poisson_loglik(d, e, mu) == pytest.approx(poisson.logpmf(d, e * mu).sum(), rel=1e-12)


def test_poisson_loglik_accepts_non_integer_deaths_through_the_gamma_function():
    """D = 2.5, E mu = 3:  2.5 ln 3 - 3 - ln Gamma(3.5)  (scipy's logpmf is -inf at a non-integer count)."""
    got = poisson_loglik([2.5], [1000.0], [0.003])
    assert poisson.logpmf(2.5, 3.0) == -np.inf
    assert got == pytest.approx(2.5 * math.log(3.0) - 3.0 - math.lgamma(3.5), rel=1e-12)


def test_poisson_loglik_counts_zero_deaths_and_skips_ages_without_information():
    e = np.array([1000.0, 0.0, 500.0, 400.0])
    d = np.array([0.0, 0.0, np.nan, 3.0])
    mu = np.array([0.002, 0.5, 0.01, 0.006])
    only = [0, 3]                                                            # age 1: E = 0, age 2: D missing
    assert poisson_loglik(d, e, mu) == pytest.approx(poisson.logpmf(d[only], e[only] * mu[only]).sum(), rel=1e-12)
    assert poisson_loglik([0.0], [1000.0], [0.002]) == pytest.approx(-2.0)   # D = 0: just -E mu


# --------------------------------------------------------------------------
# half_deviance, nelder_mead, refit_with_zero
# --------------------------------------------------------------------------


def test_half_deviance_is_the_saturated_loglik_minus_the_loglik():
    d = np.array([0.0, 3.0, 12.0, 40.0, 101.0])
    e = np.array([1000.0, 900.0, 800.0, 700.0, 600.0])
    mu = np.array([0.001, 0.004, 0.015, 0.05, 0.17])
    sat = np.where(d > 0, d * np.log(np.where(d > 0, d, 1.0)) - d, 0.0) - np.array([math.lgamma(x + 1) for x in d])
    assert half_deviance(d, e, mu) == pytest.approx(sat.sum() - poisson_loglik(d, e, mu), rel=1e-12)
    assert half_deviance(d + 1, e, (d + 1) / e) == pytest.approx(0.0, abs=1e-12)          # zero exactly at mu = D / E
    assert half_deviance(d, e, mu * 1.3) > 0
    assert half_deviance(d, e, d / e) == float("inf")                     # mu = 0 at the D = 0 age: outside the domain


def test_half_deviance_skips_uninformative_ages_and_is_infinite_outside_the_domain():
    e, d, mu = np.array([1000.0, 0.0, 500.0]), np.array([5.0, 4.0, np.nan]), np.array([0.005, 0.5, 0.01])
    assert half_deviance(d, e, mu) == pytest.approx(half_deviance(d[:1], e[:1], mu[:1]), rel=1e-12)
    assert half_deviance([5.0], [1000.0], [0.0]) == float("inf")
    assert half_deviance([5.0], [1000.0], [-0.1]) == float("inf")
    assert half_deviance([5.0], [1000.0], [np.inf]) == float("inf")


def rosenbrock(t):
    return 100.0 * (t[1] - t[0] ** 2) ** 2 + (1.0 - t[0]) ** 2


def test_nelder_mead_finds_the_minimum_of_a_curved_valley_and_restarts_when_needed():
    simplex = np.array([[-1.2, 1.0], [-1.1, 1.0], [-1.2, 1.1]])
    r = nelder_mead(rosenbrock, simplex, fatol=1e-12, xatol=1e-9)
    assert isinstance(r, NelderMeadResult) and r.converged
    np.testing.assert_allclose(r.x, [1.0, 1.0], atol=1e-5)
    assert r.fun < 1e-10 and r.n_iter > 0 and r.n_fev >= r.n_iter and r.runs >= 2


def test_nelder_mead_reports_a_run_that_hits_its_iteration_cap_as_not_converged():
    r = nelder_mead(rosenbrock, np.array([[-1.2, 1.0], [-1.1, 1.0], [-1.2, 1.1]]), maxiter=5, restarts=0)
    assert r.converged is False and r.runs == 1 and r.n_iter == 5


ROSEN_SIMPLEX = np.array([[-1.2, 1.0], [-1.1, 1.0], [-1.2, 1.1]])


def test_nelder_mead_restarts_keep_the_better_point_and_add_up_the_iterations():
    """With 30 iterations per run no run converges: each restart must start from the best point so far, keep the
    result if it is better, and the iteration count is the sum over the runs."""
    kw = dict(fatol=1e-12, xatol=1e-9, maxiter=30)
    one = nelder_mead(rosenbrock, ROSEN_SIMPLEX, restarts=0, **kw)
    two = nelder_mead(rosenbrock, ROSEN_SIMPLEX, restarts=1, **kw)
    four = nelder_mead(rosenbrock, ROSEN_SIMPLEX, restarts=3, **kw)
    assert (one.runs, two.runs, four.runs) == (1, 2, 4)
    assert four.fun < two.fun < one.fun                                   # each restart improves on the best so far
    assert (one.n_iter, two.n_iter, four.n_iter) == (30, 60, 120)         # every run hit its cap of 30
    assert not (one.converged or two.converged or four.converged)


def test_nelder_mead_is_not_converged_while_the_last_restart_was_still_improving():
    """100 iterations: the first run stops short (1e-10), the second CONVERGES yet still gained 1e-10 > fatol, so the
    verdict is 'not converged' (restarts used up); one more restart gains nothing and then it is."""
    kw = dict(fatol=1e-12, xatol=1e-9, maxiter=100)
    two = nelder_mead(rosenbrock, ROSEN_SIMPLEX, restarts=1, **kw)
    assert two.runs == 2 and two.converged is False
    assert two.n_iter < 200                                              # the second run stopped by itself, not on the cap
    four = nelder_mead(rosenbrock, ROSEN_SIMPLEX, restarts=3, **kw)
    assert four.converged is True and four.runs == 3 and four.fun == pytest.approx(two.fun, abs=1e-15)


def test_nelder_mead_survives_an_objective_that_is_infinite_outside_its_domain():
    def fun(t):
        return float("inf") if t[0] < 0 else (t[0] - 0.5) ** 2 + (t[1] - 1.0) ** 2
    r = nelder_mead(fun, np.array([[1.0, 0.0], [1.2, 0.0], [1.0, 0.2]]), fatol=1e-12)
    np.testing.assert_allclose(r.x, [0.5, 1.0], atol=1e-5)


def test_refit_with_zero_accepts_a_wall_that_the_other_parameters_can_make_up_for():
    """f = (t1 + t0 - 3)^2 + 1e-6 t0^2: at (0.05, 2.95) naively zeroing t0 costs 2.5e-3, yet the re-fitted t1 = 3
    reaches 0 - the compensation a nearly merged pair of exponentials produces."""
    f = lambda t: (t[1] + t[0] - 3.0) ** 2 + 1e-6 * t[0] ** 2
    start = np.array([0.05, 2.95])
    assert f([0.0, 2.95]) > 1e-3
    out = refit_with_zero(f, start, 0, tol=1e-8)
    assert out is not None and out[0] == 0.0 and out[1] == pytest.approx(3.0, abs=1e-5)


def test_refit_with_zero_refuses_a_parameter_that_is_genuinely_positive():
    f = lambda t: (t[0] - 0.3) ** 2 + (t[1] - 2.0) ** 2                  # separable: nothing can make up for t0 = 0
    assert refit_with_zero(f, np.array([0.3, 2.0]), 0, tol=1e-8) is None
    assert refit_with_zero(f, np.array([0.3, 2.0]), 0, tol=0.2) is not None            # unless the tolerance allows it


def test_refit_with_zero_holds_the_fixed_coordinates_where_they_are():
    f = lambda t: (t[1] + t[0] - 3.0) ** 2 + 1e-6 * t[0] ** 2 + 0.0 * t[2]
    out = refit_with_zero(f, np.array([0.05, 2.95, 7.7]), 0, fixed=(2,), tol=1e-8)
    assert out is not None and out[2] == 7.7 and out[0] == 0.0
