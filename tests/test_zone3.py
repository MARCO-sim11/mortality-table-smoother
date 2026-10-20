"""Module 3e - zone3.py (Kannisto, Beard, Coale-Kisker). Run:   python -m pytest tests/test_zone3.py -v

Evidence that does not replay the code's own formulas:
  * exact recovery from noise-free expected counts (the score vanishes at the true parameters);
  * the Poisson score equations at the optimum, and a global search by differential evolution;
  * nesting: Beard (K = 1 is Kannisto, K = 0 is Gompertz) can never be below Kannisto, and finds each limit;
  * Coale-Kisker checked against its formula written out by hand, and through the joiner's q = 1 - exp(-mu) bridge;
  * rates above 1 (D > E, as at the oldest ages of France 2020), where the spec's binomial likelihood does not exist;
  * the user's smoothing_lab, and what the result covers: omega2..omega3 even when the table stops earlier.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import differential_evolution
from scipy.special import expit

import synthetic as S
from src import config as C
from src import loader as L
from src import zone3 as Z
from src.fit_common import NelderMeadResult, half_deviance, poisson_loglik
from src.zone3 import _newton, fit_beard, fit_coale_kisker, fit_kannisto

BOTH = ("Female", "Male")
ROOT = Path(__file__).resolve().parent.parent
POST3 = Path(os.environ.get(
    "MORTALITY_POST3_DIR",
    r"C:\Users\simar\OneDrive\Bureau\Persos\LinkedIn\Life_Risks_Series\Post3-!Smoothing1"))

W2, W3 = 85, 110                                              # the spec's default zone III bounds
AGES = np.arange(W2, 110, dtype=float)                        # a table that stops at 109, like HMD
EXPO = 2.0e5 * np.exp(-0.22 * (AGES - W2))
KAN = dict(ln_a=-2.5, b=0.15)                                 # odds at omega2 = e^-2.5 = 0.082
BEARD = dict(beta=0.06, gamma=0.14, K=0.8)                    # plateau 1 / K = 1.25


def kan_mu(ln_a, b, ages=AGES):
    return expit(ln_a + b * (ages - W2))


def beard_mu(beta, gamma, K, ages=AGES):
    z = beta * np.exp(gamma * (ages - W2))
    return z / (1.0 + K * z)


def table(mu, seed=None, expo=EXPO):
    d = EXPO * mu if seed is None else np.random.default_rng(seed).poisson(expo * mu).astype(float)
    return AGES.copy(), d, expo.copy()


def kan_table(seed=None, **kw):
    return table(kan_mu(**{**KAN, **kw}), seed)


def beard_table(seed=None, **kw):
    return table(beard_mu(**{**BEARD, **kw}), seed)


def synthetic_zone3(sex="Female"):
    t = S.make_sex_table(sex).iloc[W2:]                       # ages 85..100 of the synthetic law
    return t["age"].to_numpy(), t["Dx"].to_numpy().astype(float), t["Ex"].to_numpy()


# --------------------------------------------------------------------------
# What the result covers
# --------------------------------------------------------------------------


@pytest.mark.parametrize("fit", [fit_kannisto, fit_beard, fit_coale_kisker])
def test_the_result_covers_omega2_to_omega3_even_when_the_table_stops_at_109(fit):
    a, d, e = kan_table(1)
    assert a.max() == 109
    r = fit(a, d, e, None)
    np.testing.assert_array_equal(r.ages, np.arange(85, 111))
    assert r.mu_hat.shape == (26,) and np.all(np.isfinite(r.mu_hat)) and np.all(r.mu_hat > 0)
    assert r.zone == "zone3"


@pytest.mark.parametrize("fit", [fit_kannisto, fit_beard, fit_coale_kisker])
def test_the_bounds_come_from_the_config(fit):
    a, d, e = kan_table(2)
    r = fit(a, d, e, C.SmootherConfig(omega2=90, omega3=105))
    np.testing.assert_array_equal(r.ages, np.arange(90, 106))
    assert r.mu_hat.shape == (16,)


def test_the_curve_extrapolates_beyond_the_last_row_exactly_on_a_noise_free_table():
    a, d, e = kan_table()
    r = fit_kannisto(a, d, e, None)
    assert r.mu_hat[-1] == pytest.approx(float(kan_mu(**KAN, ages=np.array([110.0]))[0]), rel=1e-6)
    assert np.all(np.diff(r.mu_hat) > 0) and r.mu_hat[-1] < 1.0


# --------------------------------------------------------------------------
# Kannisto
# --------------------------------------------------------------------------


def test_kannisto_recovers_the_truth_from_noise_free_expected_counts():
    r = fit_kannisto(*kan_table(), None)
    assert r.params["a"] == pytest.approx(np.exp(KAN["ln_a"]), rel=1e-7)
    assert r.params["b"] == pytest.approx(KAN["b"], rel=1e-7)
    np.testing.assert_allclose(r.mu_hat[:25], kan_mu(**KAN), rtol=1e-6)         # nr_tol = 1e-8 on the loglik gap
    assert r.converged and r.n_iter <= 8 and r.params["x0"] == 85.0


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_kannisto_poisson_score_equations_hold_at_the_optimum(seed):
    """sum (D - E mu)(1 - mu) = 0 and sum t (D - E mu)(1 - mu) = 0: the doc's gradients (in ln a and b), at zero."""
    a, d, e = kan_table(seed)
    r = fit_kannisto(a, d, e, None)
    mu, t = r.mu_hat[:25], a - 85.0
    g = (d - e * mu) * (1.0 - mu)
    scale = d.sum()
    assert abs(g.sum()) / scale < 1e-6 and abs((t * g).sum()) / scale < 1e-6


@pytest.mark.parametrize("seed", [4, 5, 6])
def test_kannisto_reaches_the_global_optimum_found_by_differential_evolution(seed):
    a, d, e = kan_table(seed)
    r = fit_kannisto(a, d, e, None)
    t = a - 85.0
    de = differential_evolution(lambda th: half_deviance(d, e, expit(th[0] + th[1] * t)), [(-10, 2), (0.01, 0.4)],
                                seed=seed, popsize=40, tol=1e-13, maxiter=3000, polish=True)
    assert half_deviance(d, e, r.mu_hat[:25]) <= de.fun + 1e-6


def test_the_refinement_improves_on_the_grid_and_a_tiny_grid_still_gets_there():
    a, d, e = kan_table(7)
    fine = fit_kannisto(a, d, e, C.SmootherConfig())
    coarse = fit_kannisto(a, d, e, C.SmootherConfig(grid_n=4))
    assert coarse.params["b"] == pytest.approx(fine.params["b"], rel=1e-6)           # Newton finds the same optimum
    grid_only = fit_kannisto(a, d, e, C.SmootherConfig(nr_tol=1e12))                 # a tolerance that accepts the grid point
    assert grid_only.n_iter == 0 and grid_only.loglik < fine.loglik - 1e-3
    assert np.min(np.abs(np.linspace(0.05, 0.20, 100) - grid_only.params["b"])) < 1e-12


def test_newton_from_the_plateau_uses_expected_information_where_the_hessian_is_not_definite():
    """Near mu = 1 the Poisson log-likelihood of the sigmoid is not concave (w < 0 where mu > (1 + D/E) / 2): a pure
    Newton step there goes the wrong way. Fisher scoring takes over, and the optimum reached is the same."""
    a, d, e = kan_table(8)
    t = a - 85.0
    ref = fit_kannisto(a, d, e, None)
    rng = np.random.default_rng(0)
    starts = [(2.0, 0.3), (1.0, 0.2), (-9.0, 0.05), (0.0, 0.4)] + [(rng.uniform(-10, 3), rng.uniform(0.01, 0.5)) for _ in range(60)]
    for start in starts:                                                 # 3 of these 64 never converge without the trust region
        nr = _newton(t, d, e, *start, 500, 1e-10)
        assert nr.converged and nr.note == "", start
        assert (nr.ln_a, nr.b) == pytest.approx((np.log(ref.params["a"]), ref.params["b"]), abs=1e-4), start
    assert _newton(t, d, e, 2.0, 0.3, 500, 1e-10).n_fisher > 0


def test_the_deviance_never_rises_along_the_newton_iteration():
    """Step halving: from any start, step k + 1 is accepted only if the half deviance does not rise."""
    a, d, e = kan_table(8)
    t = a - 85.0
    rng = np.random.default_rng(1)
    starts = [(2.0, 0.3), (1.0, 0.2), (-9.0, 0.05), (0.0, 0.4)] + [(rng.uniform(-10, 3), rng.uniform(0.01, 0.5)) for _ in range(30)]
    for start in starts:
        devs = []
        for k in range(0, 10):
            nr = _newton(t, d, e, *start, k, 0.0)                        # stop after exactly k steps
            devs.append(half_deviance(d, e, expit(nr.ln_a + nr.b * t)))
        assert all(b <= a_ + 1e-9 * (1 + abs(a_)) for a_, b in zip(devs, devs[1:])), start


def test_a_newton_step_never_moves_a_logit_by_more_than_the_trust_radius():
    """From the saturated start the unrestricted step is (-6.7, -4.2): 100 logit units at the oldest age."""
    a, d, e = kan_table(8)
    t = a - 85.0
    nr = _newton(t, d, e, 2.0, 0.3, 1, 0.0)                              # exactly one step, from (2, 0.3)
    moved = np.abs((nr.ln_a - 2.0) + (nr.b - 0.3) * t)
    assert moved.max() <= Z.TRUST_LOGIT + 1e-9 and nr.n_iter == 1


def test_the_starting_point_is_the_best_of_the_grid_and_the_grid_size_is_the_configs():
    """nr_tol = 1e12 stops at once: what comes back IS the best of the cfg.grid_n x cfg.grid_n grid, checked against a
    grid minimum computed here, point by point."""
    a, d, e = kan_table(7)
    t = a - 85.0
    for n in (100, 12):
        r = fit_kannisto(a, d, e, C.SmootherConfig(nr_tol=1e12, grid_n=n))
        best = min((half_deviance(d, e, expit(la + b * t)), la, b)
                   for la in np.linspace(-10.0, 0.0, n) for b in np.linspace(0.05, 0.20, n))
        assert r.n_iter == 0 and f"Grille {n}×{n}" in r.message
        assert half_deviance(d, e, r.mu_hat[:25]) == pytest.approx(best[0], rel=1e-12)
        assert (np.log(r.params["a"]), r.params["b"]) == pytest.approx((best[1], best[2]), abs=1e-12)


def test_the_stopping_gap_is_the_loglik_still_to_gain():
    """1/2 g' I^-1 g estimates the half deviance still to lose: checked on iterates near the optimum (where the observed
    Hessian is definite, so the estimate is tight) against the true distance to the converged deviance."""
    a, d, e = kan_table(23)
    t = a - 85.0
    ref = fit_kannisto(a, d, e, None)
    la0, b0 = np.log(ref.params["a"]), ref.params["b"]
    star = _newton(t, d, e, la0, b0, 200, 1e-13)
    dev_star = half_deviance(d, e, expit(star.ln_a + star.b * t))
    checked = 0
    for start in [(la0 + 0.4, b0 - 0.012), (la0 - 0.5, b0 + 0.02)]:
        for k in range(1, 4):
            part = _newton(t, d, e, *start, k, 0.0)
            if 1e-6 < part.gap < 300.0:
                true_gap = half_deviance(d, e, expit(part.ln_a + part.b * t)) - dev_star
                assert true_gap == pytest.approx(part.gap, rel=0.1)
                checked += 1
    assert checked >= 4


def test_a_decreasing_curve_gives_a_negative_b_and_says_so():
    ages = AGES.copy()
    mu = expit(-1.0 - 0.05 * (ages - 85.0))
    r = fit_kannisto(ages, EXPO * mu, EXPO.copy(), None)
    assert r.converged and r.params["b"] == pytest.approx(-0.05, rel=1e-4) and "b ≤ 0" in r.message
    assert "b ≤ 0" not in fit_kannisto(*kan_table(), None).message


def test_deaths_at_a_single_age_are_not_enough():
    a, d, e = kan_table(24)
    one = np.zeros_like(d)
    one[5] = 40.0
    for fit in (fit_kannisto, fit_beard):
        with pytest.raises(ValueError, match="au moins 2 âges différents"):
            fit(a, one, e, None)


def test_cfg_nr_maxiter_caps_the_newton_steps_and_the_failure_is_reported():
    a, d, e = kan_table(9)
    r = fit_kannisto(a, d, e, C.SmootherConfig(nr_maxiter=0, nr_tol=1e-14))
    assert r.converged is False and "NON convergé" in r.message and "nr_maxiter = 0" in r.message
    assert np.all(np.isfinite(r.mu_hat)) and np.isfinite(r.loglik)


def test_rates_above_one_are_fitted_where_the_binomial_likelihood_does_not_exist():
    """France 2020 has D > E at 105 and 107-109 (rates up to 2.0): E - D < 0, so no binomial. Poisson copes."""
    a, d, e = kan_table(10)
    d[-4:] = 1.5 * e[-4:]
    assert np.all(d[-4:] > e[-4:])
    r = fit_kannisto(a, d, e, None)
    assert r.converged and np.isfinite(r.loglik) and np.all(np.isfinite(r.mu_hat))


def test_kannisto_ages_with_zero_deaths_stay_in_the_likelihood():
    a, d, e = kan_table(11)
    d0 = d.copy()
    d0[-5:] = 0.0
    full, without = fit_kannisto(a, d0, e, None), fit_kannisto(a[:-5], d0[:-5], e[:-5], None)
    assert abs(full.params["b"] - without.params["b"]) > 1e-4 and np.isfinite(full.loglik)


def test_kannisto_message_and_parameters_are_readable():
    r = fit_kannisto(*kan_table(12), None)
    assert "Grille 100×100" in r.message and "convergé" in r.message and "cote à ω₂" in r.message
    assert set(r.params) == {"a", "b", "x0", "n_params"} and r.params["n_params"] == 2
    absolute_a = r.params["a"] * np.exp(-r.params["b"] * r.params["x0"])            # the literature's a, ~1e-7
    assert 1e-9 < absolute_a < 1e-5


# --------------------------------------------------------------------------
# Beard
# --------------------------------------------------------------------------


def test_beard_recovers_the_truth_from_noise_free_expected_counts():
    a, d, e = beard_table()
    r = fit_beard(a, d, e, None)
    np.testing.assert_allclose(r.mu_hat[:25], beard_mu(**BEARD), rtol=1e-6)
    assert r.params["K"] == pytest.approx(BEARD["K"], rel=2e-3)
    assert r.params["c"] == pytest.approx(np.exp(BEARD["gamma"]), rel=1e-4)
    assert r.params["B"] == pytest.approx(BEARD["beta"] * np.exp(-BEARD["gamma"] * 85.0), rel=5e-3)
    assert r.loglik >= poisson_loglik(d, e, beard_mu(**BEARD)) - 1e-6 and r.converged


def test_beard_with_k_equal_to_one_is_kannisto():
    """Kannisto written as Beard: B = a e^(-b omega2), c = e^b, K = 1. A Kannisto truth gives K = 1, and the same fit."""
    a, d, e = kan_table()
    k, b = fit_kannisto(a, d, e, None), fit_beard(a, d, e, None)
    assert b.params["K"] == pytest.approx(1.0, rel=2e-3)
    assert b.params["c"] == pytest.approx(np.exp(k.params["b"]), rel=1e-4)
    np.testing.assert_allclose(b.mu_hat, k.mu_hat, rtol=1e-6)


def test_beard_is_never_below_kannisto_and_gains_when_the_plateau_is_not_one():
    a, d, e = beard_table(1)
    assert fit_beard(a, d, e, None).loglik - fit_kannisto(a, d, e, None).loglik > 5
    for a, d, e in [kan_table(2), synthetic_zone3("Female"), synthetic_zone3("Male")]:
        assert fit_beard(a, d, e, None).loglik >= fit_kannisto(a, d, e, None).loglik - 1e-7


def test_beard_with_a_gompertz_truth_reports_k_exactly_zero():
    """No plateau in the data: the optimum is on the wall K = 0 (Beard reduces to Gompertz), reported as 0.0."""
    a = AGES.copy()
    mu = 0.01 * np.exp(0.1 * (a - 85.0))                                  # reaches 0.12 at 109: nowhere near a plateau
    r = fit_beard(a, EXPO * mu, EXPO.copy(), None)
    assert r.params["K"] == 0.0 and "K = 0 (borne)" in r.message and "Gompertz" in r.message
    assert r.params["c"] == pytest.approx(np.exp(0.1), rel=1e-5)
    np.testing.assert_allclose(r.mu_hat[:25], mu, rtol=1e-5)


@pytest.mark.parametrize("seed", [2, 3])
def test_beard_reaches_the_global_optimum_found_by_differential_evolution(seed):
    a, d, e = beard_table(seed)
    r = fit_beard(a, d, e, None)
    t = a - 85.0

    def obj(th):
        z = np.exp(th[0]) * np.exp(np.exp(th[1]) * t)
        with np.errstate(over="ignore", invalid="ignore"):
            return half_deviance(d, e, z / (1.0 + th[2] ** 2 * z))

    de = differential_evolution(obj, [(-8, 2), (np.log(0.02), np.log(0.5)), (0, 5)], seed=seed, popsize=20, tol=1e-9,
                                maxiter=1000, polish=True)
    assert half_deviance(d, e, r.mu_hat[:25]) <= de.fun + 1e-4


def test_beard_cfg_nr_tol_is_the_tolerance_of_the_objective():
    a, d, e = beard_table(4)
    tight = fit_beard(a, d, e, C.SmootherConfig(nr_tol=1e-12))
    loose = fit_beard(a, d, e, C.SmootherConfig(nr_tol=1e-3))
    assert loose.n_iter < tight.n_iter and loose.loglik <= tight.loglik + 1e-9


def test_a_failed_optimisation_is_reported_not_raised(monkeypatch):
    def fake(fun, simplex, **kw):
        return NelderMeadResult(np.asarray(simplex)[0], float(fun(np.asarray(simplex)[0])), 5, 15, False, 6)
    monkeypatch.setattr(Z, "nelder_mead", fake)
    r = fit_beard(*beard_table(5), None)
    assert r.converged is False and r.n_iter == 5 and "n'a pas convergé" in r.message
    assert np.all(np.isfinite(r.mu_hat)) and np.isfinite(r.loglik)


def test_beard_parameters_are_in_the_plain_age_basis_of_the_doc():
    r = fit_beard(*beard_table(6), None)
    assert set(r.params) == {"B", "c", "K", "n_params"} and r.params["n_params"] == 3
    z = r.params["B"] * r.params["c"] ** r.ages
    np.testing.assert_allclose(r.mu_hat, z / (1.0 + r.params["K"] * z), rtol=1e-8)


# --------------------------------------------------------------------------
# Coale-Kisker
# --------------------------------------------------------------------------


def test_coale_kisker_follows_its_formula_written_out_by_hand():
    a, d, e = kan_table(13)
    r = fit_coale_kisker(a, d, e, None)
    m2 = d[0] / e[0]                                                     # the crude rate at omega2 = 85
    q2 = 1.0 - np.exp(-m2)
    k = (np.log(0.6) - np.log(q2)) / (110 - 85)
    q = q2 * np.exp(k * (np.arange(85, 111) - 85))
    assert r.params["q_omega2"] == pytest.approx(q2, rel=1e-12) and r.params["k"] == pytest.approx(k, rel=1e-12)
    np.testing.assert_allclose(1.0 - np.exp(-r.mu_hat), q, rtol=1e-12)           # the joiner's bridge gives q back
    assert q[-1] == pytest.approx(0.6, rel=1e-12) and r.mu_hat[0] == pytest.approx(m2, rel=1e-12)


def test_coale_kisker_curve_depends_only_on_the_anchor_but_loglik_on_all_the_rows():
    a, d, e = kan_table(14)
    d2 = d.copy()
    d2[10:] *= 3.0                                                       # change the data AFTER omega2
    r1, r2 = fit_coale_kisker(a, d, e, None), fit_coale_kisker(a, d2, e, None)
    np.testing.assert_array_equal(r1.mu_hat, r2.mu_hat)
    assert r1.loglik != r2.loglik and r1.loglik == pytest.approx(poisson_loglik(d, e, r1.mu_hat[:25]), rel=1e-14)


def test_coale_kisker_is_closed_form_and_n_params_counts_k():
    r = fit_coale_kisker(*kan_table(15), None)
    assert (r.method, r.converged, r.n_iter) == ("coale_kisker", True, 0)
    assert set(r.params) == {"k", "q_omega2", "q_target", "x0", "n_params"} and r.params["n_params"] == 1
    assert r.params["x0"] == 85.0
    assert r.params["q_target"] == 0.6 and "déterministe" in r.message


def test_coale_kisker_q_target_is_an_option_and_must_lie_strictly_between_0_and_1():
    a, d, e = kan_table(16)
    r = fit_coale_kisker(a, d, e, None, q_target=0.9)
    assert 1.0 - np.exp(-r.mu_hat[-1]) == pytest.approx(0.9, rel=1e-12) and r.params["q_target"] == 0.9
    for bad in (0.0, 1.0, -0.2, 1.5):
        with pytest.raises(ValueError, match="entre 0 et 1"):
            fit_coale_kisker(a, d, e, None, q_target=bad)


def test_coale_kisker_warns_when_the_anchor_is_already_above_the_target():
    a, d, e = kan_table()
    d = d.copy()
    d[0] = 2.0 * e[0]                                                    # m = 2: q = 0.86 > 0.6
    r = fit_coale_kisker(a, d, e, None)
    assert r.params["k"] < 0 and "k ≤ 0" in r.message
    assert np.all(np.isfinite(r.mu_hat))


@pytest.mark.parametrize("what", ["absent", "no exposure", "no deaths", "missing deaths"])
def test_coale_kisker_needs_a_usable_anchor_row(what):
    a, d, e = kan_table(17)
    if what == "absent":
        a, d, e = a[1:], d[1:], e[1:]
        match = "absente"
    else:
        d, e = d.copy(), e.copy()
        match = "pas d'exposition ou pas de décès"
        if what == "no exposure":
            e[0] = 0.0
        elif what == "no deaths":
            d[0] = 0.0
        else:
            d[0] = np.nan
    with pytest.raises(ValueError, match=match):
        fit_coale_kisker(a, d, e, None)


def test_coale_kisker_never_returns_an_infinite_rate_for_ages_beyond_the_target():
    a, d, e = kan_table(18)
    r = fit_coale_kisker(a, d, e, C.SmootherConfig(omega2=85, omega3=100))        # the data run to 109 > omega3
    assert np.all(np.isfinite(r.mu_hat)) and np.isfinite(r.loglik)


# --------------------------------------------------------------------------
# The FitResult protocol and the rules shared with the other zone modules
# --------------------------------------------------------------------------


@pytest.mark.parametrize("fit,key,k", [(fit_kannisto, "kannisto", 2), (fit_beard, "beard", 3), (fit_coale_kisker, "coale_kisker", 1)])
def test_the_fitresult_protocol(fit, key, k):
    a, d, e = kan_table(19)
    before = (a.copy(), d.copy(), e.copy())
    r = fit(a, d, e, C.SmootherConfig())
    assert isinstance(r, C.FitResult) and (r.method, r.zone) == (key, "zone3")
    assert r.params["n_params"] == k == C.METHODS[key].n_params
    assert r.loglik == pytest.approx(poisson_loglik(d, e, r.mu_hat[:25]), rel=1e-12)       # the supplied rows, ages 85..109
    assert isinstance(r.n_iter, int) and r.message
    for got, was in zip((a, d, e), before):
        np.testing.assert_array_equal(got, was)


@pytest.mark.parametrize("fit", [fit_kannisto, fit_beard, fit_coale_kisker])
def test_the_config_defaults_of_the_spec_are_used_when_it_is_omitted(fit):
    a, d, e = kan_table(20)
    np.testing.assert_array_equal(fit(a, d, e, None).mu_hat, fit(a, d, e, C.SmootherConfig()).mu_hat)


@pytest.mark.parametrize("fit", [fit_kannisto, fit_beard, fit_coale_kisker])
def test_the_order_of_the_rows_does_not_matter(fit):
    a, d, e = kan_table(21)
    order = np.random.default_rng(0).permutation(a.size)
    np.testing.assert_allclose(fit(a[order], d[order], e[order], None).mu_hat, fit(a, d, e, None).mu_hat, rtol=1e-5)


@pytest.mark.parametrize("fit", [fit_kannisto, fit_beard])
@pytest.mark.parametrize("bad", ["exposure", "deaths"])
def test_a_row_without_exposure_or_with_missing_deaths_is_ignored(fit, bad):
    a, d, e = kan_table(22)
    d2, e2 = d.copy(), e.copy()
    if bad == "exposure":
        e2[10] = 0.0
    else:
        d2[10] = np.nan
    keep = np.arange(a.size) != 10
    r, clean = fit(a, d2, e2, None), fit(a[keep], d[keep], e[keep], None)
    np.testing.assert_allclose(r.mu_hat, clean.mu_hat, rtol=1e-4)
    assert np.isfinite(r.loglik) and "1 âge ignoré" in r.message


@pytest.mark.parametrize("fit", [fit_kannisto, fit_beard, fit_coale_kisker])
@pytest.mark.parametrize("args,msg", [
    (lambda a, d, e: (a, d[:-1], e), "même longueur"),
    (lambda a, d, e: (np.r_[a[:-1], a[0]], d, e), "dupliqués"),
    (lambda a, d, e: (a, np.r_[-1.0, d[1:]], e), "négatif"),
    (lambda a, d, e: (np.r_[np.nan, a[1:]], d, e), "non fini"),
])
def test_malformed_input_is_refused(fit, args, msg):
    with pytest.raises(ValueError, match=msg):
        fit(*args(*kan_table(23)), None)


def test_too_few_ages_or_no_deaths_are_refused():
    a, d, e = kan_table(24)
    with pytest.raises(ValueError, match="au moins 2"):
        fit_kannisto(a[:1], d[:1], e[:1], None)
    with pytest.raises(ValueError, match="au moins 3"):
        fit_beard(a[:2], d[:2], e[:2], None)
    for fit in (fit_kannisto, fit_beard):
        with pytest.raises(ValueError, match="au moins"):
            fit(a, np.zeros_like(d), e, None)


# --------------------------------------------------------------------------
# Real data: France 2020 (HMD), the standard file shipped in sample_data/
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def france():
    df = L.load_data(ROOT / "sample_data" / "FRA_taux_bruts_moment_2020_female_male.csv")
    cfg = C.default_config(df["age"].unique())
    return {sex: C.zone_slice(df[df["sex"] == sex], cfg, "zone3") for sex in BOTH}, cfg


PINNED = {                       # measured on the sample file, ages 85..109; agree with smoothing_lab (Kannisto to 1e-9)
    "Female": dict(a=0.06152037383966885, b=0.160578608136796, ll_k=-306.93391924408866,
                   B=1.0892452327014293e-07, c=1.168618646501523, K=0.8585834418626123, ll_b=-302.91004605127057,
                   k=0.09520321281798189, q2=0.055525886462929795),
    "Male": dict(a=0.10037808421753801, b=0.15556234258491924, ll_k=-133.93548548321277,
                 B=4.279685984900475e-07, c=1.1564171609596805, K=0.7425110236576202, ll_b=-127.26774022051332,
                 k=0.07634130604740373, q2=0.08897869900206913),
}


@pytest.mark.parametrize("sex", BOTH)
def test_france_2020_zone_three_estimates(france, sex):
    tables, cfg = france
    a, d, e = (tables[sex][c].to_numpy() for c in ("age", "Dx", "Ex"))
    assert (a.min(), a.max()) == (85, 109) and np.any(d > e)             # rates above 1 are really in the data
    p = PINNED[sex]
    k, b, ck = fit_kannisto(a, d, e, cfg), fit_beard(a, d, e, cfg), fit_coale_kisker(a, d, e, cfg)
    assert (k.params["a"], k.params["b"], k.loglik) == pytest.approx((p["a"], p["b"], p["ll_k"]), rel=1e-7)
    assert (b.params["B"], b.params["c"], b.params["K"]) == pytest.approx((p["B"], p["c"], p["K"]), rel=1e-4)
    assert b.loglik == pytest.approx(p["ll_b"], abs=1e-4)
    assert (ck.params["k"], ck.params["q_omega2"]) == pytest.approx((p["k"], p["q2"]), rel=1e-9)
    assert k.converged and b.converged and b.loglik > k.loglik and b.params["K"] < 1.0
    assert k.mu_hat[-1] < 1.0 and np.all(np.diff(k.mu_hat) > 0)         # extrapolated to 110, increasing, bounded by 1


@pytest.mark.parametrize("sex", BOTH)
def test_france_2020_coale_kisker_with_the_spec_target_fits_far_worse_than_the_laws(france, sex):
    """The conventional q(110) = 0.6 is below what the table shows (q already 0.63-0.86 at 109): measured, not hidden."""
    tables, cfg = france
    a, d, e = (tables[sex][c].to_numpy() for c in ("age", "Dx", "Ex"))
    ck, k = fit_coale_kisker(a, d, e, cfg), fit_kannisto(a, d, e, cfg)
    assert ck.loglik < k.loglik - 1000
    assert fit_coale_kisker(a, d, e, cfg, q_target=0.95).loglik > ck.loglik          # a higher target repairs part of it


@pytest.mark.parametrize("sex", BOTH)
def test_france_2020_beard_reaches_the_global_optimum(france, sex):
    tables, cfg = france
    a, d, e = (tables[sex][c].to_numpy() for c in ("age", "Dx", "Ex"))
    r, t = fit_beard(a, d, e, cfg), a - 85.0

    def obj(th):
        z = np.exp(th[0]) * np.exp(np.exp(th[1]) * t)
        with np.errstate(over="ignore", invalid="ignore"):
            return half_deviance(d, e, z / (1.0 + th[2] ** 2 * z))

    de = differential_evolution(obj, [(-8, 2), (np.log(0.02), np.log(0.5)), (0, 5)], seed=3, popsize=20, tol=1e-9,
                                maxiter=1000, polish=True)
    assert half_deviance(d, e, r.mu_hat[:25]) <= de.fun + 1e-4


@pytest.fixture(scope="module")
def smoothing_lab():
    sys.path.insert(0, str(POST3))
    pytest.importorskip("smoothing_lab.zone3_kannisto")
    from smoothing_lab import zone3_beard, zone3_kannisto
    return zone3_kannisto, zone3_beard


@pytest.mark.parametrize("sex", BOTH)
def test_france_2020_agrees_with_the_smoothing_lab_implementations(smoothing_lab, france, sex):
    """Kannisto: the lab works in absolute ages (a ~ 1e-7), here a is the odds at omega2: a_abs = a e^(-b omega2)."""
    lab_k, lab_b = smoothing_lab
    tables, cfg = france
    a, d, e = (tables[sex][c].to_numpy() for c in ("age", "Dx", "Ex"))
    k, b = fit_kannisto(a, d, e, cfg), fit_beard(a, d, e, cfg)
    lk, lb = lab_k.fit_kannisto_profile(a, d, e), lab_b.fit_beard(a, d, e)
    assert k.params["a"] * np.exp(-k.params["b"] * 85.0) == pytest.approx(lk.a, rel=1e-4)
    assert k.params["b"] == pytest.approx(lk.b, rel=1e-5)
    np.testing.assert_allclose(k.mu_hat[:25], lk.mu(a), rtol=1e-4)
    assert b.loglik == pytest.approx(poisson_loglik(d, e, lb.mu(a)), abs=1e-3)
