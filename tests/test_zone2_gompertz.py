"""Module 3b - zone2_gompertz.py. Run:   python -m pytest tests/test_zone2_gompertz.py -v

Evidence that does not replay the code's own formulas:
  * exact recovery: noise-free expected counts D = E mu(theta) put the score at zero at the TRUE theta;
  * the Poisson score equations at the optimum: sum(D) = sum(E mu_hat) and sum(x D) = sum(x E mu_hat);
  * independent optimisers: scipy's Nelder-Mead on the log-likelihood, and the user's smoothing_lab;
  * the validation tests of spec section 8 (NR in < 50 iterations, log-lik above the OLS one);
  * calibration: over many simulated tables, (beta_hat - beta) / se is centred, unit-variance.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import minimize

import synthetic as S
from src import config as C
from src import loader as L
from src.fit_common import poisson_loglik
from src.zone2_gompertz import _newton, _ols_start, fit_gompertz

BOTH = ("Female", "Male")
ROOT = Path(__file__).resolve().parent.parent
POST3 = Path(os.environ.get(
    "MORTALITY_POST3_DIR",
    r"C:\Users\simar\OneDrive\Bureau\Persos\LinkedIn\Life_Risks_Series\Post3-!Smoothing1"))

ALPHA, BETA = -10.8, 0.095                                   # B = 2.0e-5, c = 1.0997
AGES = np.arange(20, 86, dtype=float)
EXPO = 4.0e5 * np.exp(-0.06 * (AGES - 20))                   # 4e5 at 20, 8e3 at 85


def expected_counts(alpha=ALPHA, beta=BETA, ages=AGES, expo=EXPO):
    return expo * np.exp(alpha + beta * ages)


def poisson_table(seed, alpha=ALPHA, beta=BETA):
    d = np.random.default_rng(seed).poisson(expected_counts(alpha, beta)).astype(float)
    return AGES.copy(), d, EXPO.copy()


def neg_loglik(theta, a, d, e):
    return -np.sum(d * (theta[0] + theta[1] * a) - e * np.exp(theta[0] + theta[1] * a))


def nelder_mead(a, d, e, start):
    return minimize(neg_loglik, start, args=(a, d, e), method="Nelder-Mead",
                    options=dict(xatol=1e-11, fatol=1e-11, maxiter=40000)).x


def synthetic_zone2(sex="Female"):
    t = S.make_sex_table(sex).iloc[20:86]                    # Makeham + infant term: Gompertz is misspecified
    return t["age"].to_numpy(), t["Dx"].to_numpy().astype(float), t["Ex"].to_numpy()


# --------------------------------------------------------------------------
# The estimator
# --------------------------------------------------------------------------


def test_exact_recovery_from_noise_free_expected_counts():
    """The score vanishes at the true (alpha, beta), so the MLE IS the truth, whatever the exposure."""
    r = fit_gompertz(AGES, expected_counts(), EXPO, None)
    assert r.params["alpha"] == pytest.approx(ALPHA, abs=1e-9)
    assert r.params["beta"] == pytest.approx(BETA, abs=1e-11)
    assert r.params["B"] == pytest.approx(np.exp(ALPHA), rel=1e-9)
    assert r.params["c"] == pytest.approx(np.exp(BETA), rel=1e-11)
    np.testing.assert_allclose(r.mu_hat, np.exp(ALPHA + BETA * AGES), rtol=1e-9)
    assert r.converged and r.n_iter < 10


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_the_poisson_score_equations_hold_at_the_optimum(seed):
    """For the log link, the MLE reproduces the observed deaths and their age-weighted total exactly."""
    a, d, e = poisson_table(seed)
    r = fit_gompertz(a, d, e, None)
    fitted = e * r.mu_hat
    assert fitted.sum() == pytest.approx(d.sum(), rel=1e-6)
    assert (a * fitted).sum() == pytest.approx((a * d).sum(), rel=1e-6)


@pytest.mark.parametrize("seed", [4, 5, 6])
def test_it_agrees_with_an_independent_nelder_mead_maximisation(seed):
    a, d, e = poisson_table(seed)
    r = fit_gompertz(a, d, e, None)
    oracle = nelder_mead(a, d, e, [r.params["alpha"] + 0.05, r.params["beta"] - 0.002])
    np.testing.assert_allclose([r.params["alpha"], r.params["beta"]], oracle, atol=1e-6)
    assert -neg_loglik([r.params["alpha"], r.params["beta"]], a, d, e) >= -neg_loglik(oracle, a, d, e) - 1e-7


def test_ages_without_deaths_stay_in_the_likelihood():
    """The old, thin ages with D = 0 say the rate is low there: dropping them changes the fit, and the
    estimate with them is the one an independent optimiser finds on the FULL table."""
    a, d, e = poisson_table(7)
    d[-6:] = 0.0                                                         # ages 80..85 observe nothing
    full = fit_gompertz(a, d, e, None)
    without = fit_gompertz(a[:-6], d[:-6], e[:-6], None)
    oracle = nelder_mead(a, d, e, [full.params["alpha"] + 0.05, full.params["beta"] - 0.002])
    np.testing.assert_allclose([full.params["alpha"], full.params["beta"]], oracle, atol=1e-6)
    assert abs(full.params["beta"] - without.params["beta"]) > 1e-3
    assert np.all(np.isfinite(full.mu_hat))


def test_estimates_are_calibrated_against_the_fisher_information():
    """z = (beta_hat - beta) / se over 150 simulated tables: mean ~ 0 and sd ~ 1 (se^2 = [I^-1]_22 with
    I = sum E mu [[1, x], [x, x^2]]). A biased score or a wrong Hessian would shift or stretch it."""
    mu = np.exp(ALPHA + BETA * AGES)
    info = np.array([[np.sum(EXPO * mu), np.sum(AGES * EXPO * mu)],
                     [np.sum(AGES * EXPO * mu), np.sum(AGES ** 2 * EXPO * mu)]])
    se_beta = np.sqrt(np.linalg.inv(info)[1, 1])
    z = np.array([(fit_gompertz(*poisson_table(1000 + s), None).params["beta"] - BETA) / se_beta
                  for s in range(150)])
    assert abs(z.mean()) < 0.3 and 0.8 < z.std() < 1.2


def test_the_ols_start_is_the_unweighted_regression_on_the_log_rates():
    a, d, e = poisson_table(8)
    alpha0, beta0 = _ols_start(a, d, e)
    oracle = np.polyfit(a, np.log(d / e), 1)                             # [slope, intercept]
    assert (alpha0, beta0) == pytest.approx((oracle[1], oracle[0]), rel=1e-9)


# --------------------------------------------------------------------------
# Newton-Raphson behaviour
# --------------------------------------------------------------------------


@pytest.mark.parametrize("start", [(-30.0, 0.0), (-5.0, 0.3), (0.0, 0.0), (-12.0, 0.5), (-8.0, -0.2), (5.0, 1.0)])
def test_step_halving_brings_newton_home_from_far_starts(start):
    """A pure Newton step from a start with mu far too small is enormous (the Hessian is ~0 there) and lands
    in overflow; with halving every start reaches the same, unique maximum of the concave log-likelihood."""
    a, d, e = poisson_table(9)
    ref = fit_gompertz(a, d, e, None)
    nr = _newton(a, d, e, *start, 500, 1e-10)
    assert nr.converged and nr.note == ""
    assert (nr.alpha, nr.beta) == pytest.approx((ref.params["alpha"], ref.params["beta"]), abs=1e-5)


def test_one_newton_step_is_theta_minus_the_inverse_hessian_times_the_score():
    """The step of the spec, written out by hand with the spec's score and Hessian, from the OLS start."""
    a, d, e = poisson_table(22)
    alpha0, beta0 = _ols_start(a, d, e)
    mu = np.exp(alpha0 + beta0 * a)
    score = np.array([np.sum(d - e * mu), np.sum(d * a - e * a * mu)])
    hess = -np.array([[np.sum(e * mu), np.sum(a * e * mu)], [np.sum(a * e * mu), np.sum(a ** 2 * e * mu)]])
    expected = np.array([alpha0, beta0]) - np.linalg.inv(hess) @ score
    r = fit_gompertz(a, d, e, C.SmootherConfig(nr_maxiter=1))
    assert r.n_iter == 1
    assert (r.params["alpha"], r.params["beta"]) == pytest.approx(tuple(expected), rel=1e-10)


def test_the_stopping_gap_is_the_loglik_still_to_gain():
    """`gap` = 1/2 grad' (-H)^-1 grad estimates ell(theta_hat) - ell(theta) to second order: checked on the
    iterates that are close enough to the maximum for the estimate to be tight but far enough to be measurable."""
    a, d, e = poisson_table(23)
    star = _newton(a, d, e, *_ols_start(a, d, e), 200, 1e-13)
    ll_star = poisson_loglik(d, e, np.exp(star.alpha + star.beta * a))
    checked = 0
    for k in range(0, 8):
        nr = _newton(a, d, e, -12.0, 0.1, k, 0.0)                        # gaps 6e3, 2e3, 1e2, 1, 1e-4, ~0
        if 1e-6 < nr.gap < 10:
            true_gap = ll_star - poisson_loglik(d, e, np.exp(nr.alpha + nr.beta * a))
            assert true_gap == pytest.approx(nr.gap, rel=0.1)
            checked += 1
    assert checked == 2


def test_the_loglik_never_decreases_along_the_iteration():
    a, d, e = poisson_table(10)
    values = []
    for k in range(0, 14):
        nr = _newton(a, d, e, -5.0, 0.3, k, 0.0)                         # stop after exactly k steps
        values.append(poisson_loglik(d, e, np.exp(nr.alpha + nr.beta * a)))
    assert all(b >= a_ - 1e-9 for a_, b in zip(values, values[1:]))
    assert values[-1] > values[0] + 1e3                                  # and it did travel a long way


def test_spec_validation_newton_converges_in_under_50_iterations_and_beats_ols():
    """Spec section 8, J2 afternoon: 'Convergence NR en < 50 iterations. log-lik > log-lik OLS.'"""
    for a, d, e in [synthetic_zone2("Female"), synthetic_zone2("Male"), poisson_table(11)]:
        r = fit_gompertz(a, d, e, None)
        keep = d > 0
        alpha0, beta0 = _ols_start(a[keep], d[keep], e[keep])
        assert r.converged and r.n_iter < 50
        assert r.loglik > poisson_loglik(d, e, np.exp(alpha0 + beta0 * a))


def test_cfg_nr_maxiter_caps_the_steps_and_the_failure_is_reported_not_raised():
    a, d, e = poisson_table(12)
    r = fit_gompertz(a, d, e, C.SmootherConfig(nr_maxiter=1))
    assert r.converged is False and r.n_iter == 1
    assert "NON convergé" in r.message and "nr_maxiter = 1" in r.message
    assert np.all(np.isfinite(r.mu_hat)) and np.isfinite(r.loglik)


def test_cfg_nr_tol_is_the_remaining_loglik_gap():
    """A tolerance larger than the gap at the OLS start accepts it: no step, the OLS line comes back."""
    a, d, e = poisson_table(13)
    r = fit_gompertz(a, d, e, C.SmootherConfig(nr_tol=1e12))
    alpha0, beta0 = _ols_start(a, d, e)
    assert r.converged and r.n_iter == 0
    assert (r.params["alpha"], r.params["beta"]) == pytest.approx((alpha0, beta0), rel=1e-12)
    tight = fit_gompertz(a, d, e, C.SmootherConfig(nr_tol=1e-12))
    loose = fit_gompertz(a, d, e, C.SmootherConfig(nr_tol=1e-2))
    assert loose.n_iter <= tight.n_iter and loose.loglik <= tight.loglik + 1e-9


def test_a_singular_hessian_stops_the_iteration_and_says_so():
    """One single distinct age makes -H rank 1 (x = 1 keeps the arithmetic exact): no step can be computed."""
    nr = _newton(np.array([1.0, 1.0]), np.array([50.0, 60.0]), np.array([1e4, 1e4]), -5.0, 0.0, 10, 1e-8)
    assert nr.converged is False and nr.note == "hessien singulier" and np.isnan(nr.gap)


def test_a_failure_note_of_the_iteration_reaches_the_message_instead_of_the_cap_text(monkeypatch):
    import src.zone2_gompertz as G
    monkeypatch.setattr(G, "_newton", lambda *args: G._Newton(-9.0, 0.08, 4, False, np.nan, "hessien singulier"))
    r = fit_gompertz(*poisson_table(24), None)
    assert r.converged is False and r.n_iter == 4
    assert "NON convergé après 4 itérations : hessien singulier." in r.message and "nr_maxiter" not in r.message


def test_a_non_increasing_mortality_still_converges_and_is_flagged():
    a, d, e = AGES.copy(), expected_counts(alpha=-6.0, beta=-0.02), EXPO.copy()
    r = fit_gompertz(a, d, e, None)
    assert r.converged and r.params["beta"] == pytest.approx(-0.02, abs=1e-8)
    assert "β ≤ 0" in r.message
    assert "β ≤ 0" not in fit_gompertz(*poisson_table(14), None).message


# --------------------------------------------------------------------------
# The FitResult protocol and the rules shared with the other zone modules
# --------------------------------------------------------------------------


def test_the_fitresult_protocol():
    a, d, e = poisson_table(15)
    before = (a.copy(), d.copy(), e.copy())
    r = fit_gompertz(a, d, e, C.SmootherConfig())
    assert isinstance(r, C.FitResult) and (r.method, r.zone) == ("gompertz", "zone2")
    np.testing.assert_array_equal(r.ages, a)
    assert r.mu_hat.shape == a.shape and np.all(r.mu_hat > 0)
    assert set(r.params) == {"alpha", "beta", "B", "c", "n_params"} and r.params["n_params"] == 2
    assert r.params["B"] == pytest.approx(np.exp(r.params["alpha"])) and r.params["c"] == pytest.approx(np.exp(r.params["beta"]))
    np.testing.assert_allclose(r.mu_hat, r.params["B"] * r.params["c"] ** a, rtol=1e-12)
    assert r.loglik == pytest.approx(poisson_loglik(d, e, r.mu_hat), rel=1e-14)
    assert r.converged is True and isinstance(r.n_iter, int) and "convergé" in r.message
    for got, was in zip((a, d, e), before):
        np.testing.assert_array_equal(got, was)


def test_the_config_defaults_of_the_spec_are_used_when_it_is_omitted():
    a, d, e = poisson_table(16)
    np.testing.assert_array_equal(fit_gompertz(a, d, e, None).mu_hat, fit_gompertz(a, d, e, C.SmootherConfig()).mu_hat)


def test_ages_keep_the_callers_order():
    a, d, e = poisson_table(17)
    order = np.random.default_rng(0).permutation(a.size)
    r1, r2 = fit_gompertz(a, d, e, None), fit_gompertz(a[order], d[order], e[order], None)
    np.testing.assert_allclose(r2.mu_hat, r1.mu_hat[order], rtol=1e-9)
    np.testing.assert_array_equal(r2.ages, a[order])


def test_non_integer_deaths_from_rebuilt_rates_are_accepted():
    a, d, e = poisson_table(18)
    r = fit_gompertz(a, d + 0.37, e, None)
    assert r.converged and np.isfinite(r.loglik)


@pytest.mark.parametrize("bad", ["exposure", "deaths"])
def test_an_age_without_exposure_or_with_missing_deaths_is_ignored_but_still_smoothed(bad):
    a, d, e = poisson_table(19)
    d2, e2 = d.copy(), e.copy()
    if bad == "exposure":
        e2[10] = 0.0
    else:
        d2[10] = np.nan
    keep = np.arange(a.size) != 10
    r = fit_gompertz(a, d2, e2, None)
    clean = fit_gompertz(a[keep], d[keep], e[keep], None)
    assert (r.params["alpha"], r.params["beta"]) == pytest.approx((clean.params["alpha"], clean.params["beta"]), abs=1e-7)
    assert np.all(np.isfinite(r.mu_hat)) and r.mu_hat[10] > 0 and np.isfinite(r.loglik)
    assert "1 âge ignoré" in r.message


# --------------------------------------------------------------------------
# Refusals (French messages)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("args,msg", [
    (lambda a, d, e: (a, d[:-1], e), "même longueur"),
    (lambda a, d, e: (np.r_[a[:-1], a[0]], d, e), "dupliqués"),
    (lambda a, d, e: (a, np.r_[-1.0, d[1:]], e), "négatif"),
    (lambda a, d, e: (np.r_[np.nan, a[1:]], d, e), "non fini"),
])
def test_malformed_input_is_refused(args, msg):
    with pytest.raises(ValueError, match=msg):
        fit_gompertz(*args(*poisson_table(20)), None)


def test_two_ages_with_deaths_are_needed():
    a, d, e = poisson_table(21)
    only_one = np.zeros_like(d)
    only_one[30] = 50.0
    with pytest.raises(ValueError, match="au moins 2"):
        fit_gompertz(a, only_one, e, None)
    with pytest.raises(ValueError, match="au moins 2"):
        fit_gompertz(a, np.zeros_like(d), e, None)
    two = only_one.copy()
    two[50] = 60.0
    assert fit_gompertz(a, two, e, None).converged                       # two ages are enough: the MLE exists


def test_a_two_age_zone_is_fitted_exactly():
    """Two ages, two parameters: the curve passes through both crude rates."""
    a, d, e = np.array([60.0, 70.0]), np.array([100.0, 400.0]), np.array([1e5, 1e5])
    r = fit_gompertz(a, d, e, None)
    np.testing.assert_allclose(r.mu_hat, d / e, rtol=1e-6)


# --------------------------------------------------------------------------
# Real data: France 2020 (HMD), the standard file shipped in sample_data/
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def france():
    df = L.load_data(ROOT / "sample_data" / "FRA_taux_bruts_moment_2020_female_male.csv")
    cfg = C.default_config(df["age"].unique())
    return {sex: C.zone_slice(df[df["sex"] == sex], cfg, "zone2") for sex in BOTH}, cfg


PINNED = {                       # measured on the sample file, ages 20..85; Nelder-Mead and smoothing_lab agree
    "Female": (-11.166673414392907, 0.09502310616418592),
    "Male": (-9.95608878294974, 0.08707959247386884),
}


@pytest.mark.parametrize("sex", BOTH)
def test_france_2020_zone_two_estimates_and_the_spec_validation(france, sex):
    tables, cfg = france
    t = tables[sex]
    a, d, e = (t[c].to_numpy() for c in ("age", "Dx", "Ex"))
    assert (a.min(), a.max()) == (20, 85)
    r = fit_gompertz(a, d, e, cfg)
    assert (r.params["alpha"], r.params["beta"]) == pytest.approx(PINNED[sex], rel=1e-7)
    keep = d > 0
    alpha0, beta0 = _ols_start(a[keep], d[keep], e[keep])
    assert r.converged and r.n_iter < 50 and 1.08 < r.params["c"] < 1.11
    assert r.loglik > poisson_loglik(d, e, np.exp(alpha0 + beta0 * a))
    oracle = nelder_mead(a, d, e, [r.params["alpha"] + 0.05, r.params["beta"] - 0.002])
    np.testing.assert_allclose([r.params["alpha"], r.params["beta"]], oracle, atol=1e-6)


@pytest.fixture(scope="module")
def smoothing_lab():
    sys.path.insert(0, str(POST3))
    pytest.importorskip("smoothing_lab.zone2_gompertz")
    from smoothing_lab import zone2_gompertz as mod
    return mod


@pytest.mark.parametrize("sex", BOTH)
def test_france_2020_agrees_with_the_smoothing_lab_implementation(smoothing_lab, france, sex):
    tables, cfg = france
    a, d, e = (tables[sex][c].to_numpy() for c in ("age", "Dx", "Ex"))
    r, lab = fit_gompertz(a, d, e, cfg), smoothing_lab.fit_gompertz(a, d, e)
    np.testing.assert_allclose([r.params["alpha"], r.params["beta"]], [lab.alpha, lab.beta], rtol=1e-8)
    np.testing.assert_allclose(r.mu_hat, lab.mu(a), rtol=1e-8)
