"""Module 3d (Makeham 2nd order) - zone2_makeham2.py. Run:   python -m pytest tests/test_zone2_makeham2.py -v

Evidence that does not replay the code's own formulas:
  * exact recovery from noise-free expected counts, in the canonical order c <= d (also from a truth written the
    other way round: the two exponentials are interchangeable);
  * the global optimum found independently by differential evolution;
  * the score of every free parameter vanishes at an interior optimum (central finite differences);
  * nesting: it can never be below Makeham (D = 0), and collapses onto it with D = 0 exactly when the data are Makeham;
  * the user's smoothing_lab.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import differential_evolution

import synthetic as S
from src import config as C
from src import loader as L
from src import zone2_makeham2 as M2
from src.fit_common import NelderMeadResult, half_deviance, poisson_loglik
from src.zone2_makeham import fit_makeham
from src.zone2_makeham2 import _order, fit_makeham2

BOTH = ("Female", "Male")
ROOT = Path(__file__).resolve().parent.parent
POST3 = Path(os.environ.get(
    "MORTALITY_POST3_DIR",
    r"C:\Users\simar\OneDrive\Bureau\Persos\LinkedIn\Life_Risks_Series\Post3-!Smoothing1"))

TRUTH = dict(A=2.0e-4, B=1.5e-5, c=1.105, D=1.0e-11, d=1.3)   # the steep term is as large as the main one at 85
AGES = np.arange(20, 86, dtype=float)
EXPO = 4.0e5 * np.exp(-0.06 * (AGES - 20))


def mk2_mu(A, B, c, D, d, ages=AGES):
    return A + B * c ** ages + D * d ** ages


def expected_table(**kw):
    return AGES.copy(), EXPO * mk2_mu(**{**TRUTH, **kw}), EXPO.copy()


def poisson_table(seed, **kw):
    d = np.random.default_rng(seed).poisson(EXPO * mk2_mu(**{**TRUTH, **kw})).astype(float)
    return AGES.copy(), d, EXPO.copy()


def makeham_table(seed=None):
    mu = 2.0e-4 + 1.5e-5 * 1.105 ** AGES
    d = EXPO * mu if seed is None else np.random.default_rng(seed).poisson(EXPO * mu).astype(float)
    return AGES.copy(), d, EXPO.copy()


def synthetic_zone2(sex="Female"):
    t = S.make_sex_table(sex).iloc[20:86]
    return t["age"].to_numpy(), t["Dx"].to_numpy().astype(float), t["Ex"].to_numpy()


def global_optimum(a, d, e):
    """Half deviance at the optimum of differential evolution on the same model (independent global search)."""
    s_a, xb = d.sum() / e.sum(), 60.0

    def obj(t):
        mu = (s_a * t[0] ** 2 + np.exp(t[1]) * np.exp(np.exp(t[2]) * (a - xb))
              + np.exp(t[3]) * np.exp(np.exp(t[4]) * (a - xb)))
        return half_deviance(d, e, mu)

    bounds = [(0, 3), (-8, -1), (np.log(np.log(1.02)), np.log(np.log(1.3))), (-40, 0),
              (np.log(np.log(1.02)), np.log(np.log(2.0)))]
    return differential_evolution(obj, bounds, seed=2, popsize=40, tol=1e-13, maxiter=4000, polish=True).fun


def log_scores(r, a, d, e):
    """d ell / d ln(theta_i) / sum(D) by central differences, for the five positive parameters."""
    base = np.array([r.params[k] for k in ("A", "B", "c", "D", "d")])
    out = []
    for i in range(5):
        h = 1e-5
        up, dn = base.copy(), base.copy()
        up[i] *= np.exp(h)
        dn[i] *= np.exp(-h)
        out.append((poisson_loglik(d, e, mk2_mu(*up, ages=a)) - poisson_loglik(d, e, mk2_mu(*dn, ages=a))) / (2 * h))
    return np.array(out) / d.sum()


# --------------------------------------------------------------------------
# The estimator
# --------------------------------------------------------------------------


def test_exact_recovery_from_noise_free_expected_counts():
    a, d, e = expected_table()
    r = fit_makeham2(a, d, e, None)
    np.testing.assert_allclose(r.mu_hat, mk2_mu(**TRUTH), rtol=1e-5)
    for k in ("A", "B", "c", "d"):
        assert r.params[k] == pytest.approx(TRUTH[k], rel=2e-3), k
    assert r.params["D"] == pytest.approx(TRUTH["D"], rel=5e-2)
    assert r.loglik >= poisson_loglik(d, e, mk2_mu(**TRUTH)) - 1e-6 and r.converged


def test_the_two_exponentials_are_reported_in_the_canonical_order_c_below_d():
    """The same curve written with the steep term first must come back with the slow term as (B, c)."""
    swapped = dict(A=2.0e-4, B=1.0e-11, c=1.3, D=1.5e-5, d=1.105)
    a, d, e = expected_table(**swapped)
    r = fit_makeham2(a, d, e, None)
    assert r.params["c"] < r.params["d"]
    assert r.params["c"] == pytest.approx(1.105, rel=2e-3) and r.params["d"] == pytest.approx(1.3, rel=2e-3)
    assert r.params["B"] == pytest.approx(1.5e-5, rel=2e-3)
    np.testing.assert_allclose(r.mu_hat, mk2_mu(**swapped), rtol=1e-5)
    for sex_seed in (1, 2, 3):
        p = fit_makeham2(*poisson_table(sex_seed), None).params
        assert p["c"] <= p["d"]


def test_order_puts_the_slower_growing_exponential_first_and_leaves_the_rest_alone():
    """Pure function: swap the (amplitude, base) pairs when c > d, never touch them otherwise. Nelder-Mead from the
    Makeham seed happens to return c < d on every realistic table tried, so the swap is tested here directly."""
    assert _order(1.0, 1.3, 2.0, 1.1) == (2.0, 1.1, 1.0, 1.3)
    assert _order(1.0, 1.1, 2.0, 1.3) == (1.0, 1.1, 2.0, 1.3)
    assert _order(1.0, 1.2, 2.0, 1.2) == (1.0, 1.2, 2.0, 1.2)                 # equal bases: nothing to swap
    b, c, g, f = _order(3.0, 1.4, 5.0, 1.2)
    assert (c, f) == (1.2, 1.4) and (b, g) == (5.0, 3.0)                      # each amplitude travels with its base


def test_both_bases_stay_above_one_even_when_the_data_ask_for_a_decaying_second_term():
    """d > 1 and c > 1 are part of the model: a term that decays with age (an infant-like excess) cannot be fitted by
    d < 1, so the optimiser must stop at the wall d -> 1 instead of crossing it."""
    a = AGES.copy()
    mu = 1.0e-4 + 1.5e-5 * 1.105 ** a + 3.0e-4 * 0.93 ** (a - 20)
    r = fit_makeham2(a, EXPO * mu, EXPO.copy(), None)
    assert r.params["c"] > 1.0 and r.params["d"] > 1.0 and r.converged


@pytest.mark.parametrize("seed", [1, 2])
def test_it_reaches_the_global_optimum_found_by_differential_evolution(seed):
    a, d, e = poisson_table(seed)
    r = fit_makeham2(a, d, e, None)
    assert half_deviance(d, e, r.mu_hat) <= global_optimum(a, d, e) + 1e-4
    assert r.params["D"] > 0


@pytest.mark.parametrize("seed", [3, 4])
def test_every_free_parameter_has_a_vanishing_score_at_an_interior_optimum(seed):
    a, d, e = poisson_table(seed)
    r = fit_makeham2(a, d, e, None)
    assert min(r.params[k] for k in ("A", "B", "D")) > 0
    np.testing.assert_allclose(log_scores(r, a, d, e), 0.0, atol=5e-5)


def test_a_cannot_go_negative_even_when_the_data_would_like_it_to():
    a, d, e = expected_table(A=-5e-5)
    assert mk2_mu(**{**TRUTH, "A": -5e-5}).min() > 0
    r = fit_makeham2(a, d, e, None)
    assert r.params["A"] == 0.0 and "A = 0 (borne)" in r.message
    assert r.loglik < poisson_loglik(d, e, mk2_mu(**{**TRUTH, "A": -5e-5})) - 1e-3


def test_makeham2_is_never_below_makeham_and_gains_clearly_when_a_second_term_exists():
    a, d, e = poisson_table(5)
    assert fit_makeham2(a, d, e, None).loglik - fit_makeham(a, d, e, None).loglik > 10
    for a, d, e in [makeham_table(6), synthetic_zone2("Female"), synthetic_zone2("Male")]:
        assert fit_makeham2(a, d, e, None).loglik >= fit_makeham(a, d, e, None).loglik - 1e-7


def test_on_makeham_data_makeham2_collapses_onto_makeham_with_d_exactly_zero():
    a, d, e = makeham_table()
    r, m = fit_makeham2(a, d, e, None), fit_makeham(a, d, e, None)
    assert r.params["D"] == 0.0 and "D = 0 (borne)" in r.message and "se réduit à Makeham" in r.message
    assert r.loglik == pytest.approx(m.loglik, abs=1e-6)
    np.testing.assert_allclose(r.mu_hat, m.mu_hat, rtol=1e-5)


def test_spec_validation_makeham2_converges_and_the_fitresult_is_complete():
    """Spec section 8, J3 afternoon: 'Tous les modeles convergent. FitResult complet pour chaque methode.'"""
    for a, d, e in [synthetic_zone2("Female"), synthetic_zone2("Male"), poisson_table(7), makeham_table(8)]:
        r = fit_makeham2(a, d, e, None)
        assert r.converged and np.isfinite(r.loglik) and np.all(np.isfinite(r.mu_hat)) and r.message


# --------------------------------------------------------------------------
# Configuration, convergence reporting, the FitResult protocol
# --------------------------------------------------------------------------


def test_cfg_nr_tol_is_the_tolerance_of_the_objective_and_nr_maxiter_is_not_used_by_nelder_mead():
    a, d, e = poisson_table(9)
    tight = fit_makeham2(a, d, e, C.SmootherConfig(nr_tol=1e-12))
    loose = fit_makeham2(a, d, e, C.SmootherConfig(nr_tol=1e-3))
    assert loose.n_iter < tight.n_iter and loose.loglik <= tight.loglik + 1e-9
    capped = fit_makeham2(a, d, e, C.SmootherConfig(nr_maxiter=1))             # caps the Makeham seed only
    assert capped.converged and capped.loglik == pytest.approx(fit_makeham2(a, d, e, None).loglik, abs=1e-4)


def test_a_failed_optimisation_is_reported_not_raised(monkeypatch):
    def fake(fun, simplex, **kw):
        return NelderMeadResult(np.asarray(simplex)[0], float(fun(np.asarray(simplex)[0])), 9, 30, False, 6)
    monkeypatch.setattr(M2, "nelder_mead", fake)
    r = fit_makeham2(*poisson_table(10), None)
    assert r.converged is False and r.n_iter == 9 and "n'a pas convergé" in r.message
    assert np.all(np.isfinite(r.mu_hat)) and np.isfinite(r.loglik)


def test_the_fitresult_protocol():
    a, d, e = poisson_table(11)
    before = (a.copy(), d.copy(), e.copy())
    r = fit_makeham2(a, d, e, C.SmootherConfig())
    assert isinstance(r, C.FitResult) and (r.method, r.zone) == ("makeham2", "zone2")
    np.testing.assert_array_equal(r.ages, a)
    assert r.mu_hat.shape == a.shape and np.all(r.mu_hat > 0)
    assert set(r.params) == {"A", "B", "c", "D", "d", "n_params"} and r.params["n_params"] == 5
    np.testing.assert_allclose(r.mu_hat, mk2_mu(*(r.params[k] for k in ("A", "B", "c", "D", "d")), ages=a), rtol=1e-8)
    assert r.loglik == pytest.approx(poisson_loglik(d, e, r.mu_hat), rel=1e-14)
    assert isinstance(r.n_iter, int) and r.n_iter > 0 and "itérations" in r.message
    for got, was in zip((a, d, e), before):
        np.testing.assert_array_equal(got, was)


def test_the_config_defaults_of_the_spec_are_used_when_it_is_omitted():
    a, d, e = poisson_table(12)
    np.testing.assert_array_equal(fit_makeham2(a, d, e, None).mu_hat, fit_makeham2(a, d, e, C.SmootherConfig()).mu_hat)


def test_ages_keep_the_callers_order():
    a, d, e = poisson_table(13)
    order = np.random.default_rng(0).permutation(a.size)
    r1, r2 = fit_makeham2(a, d, e, None), fit_makeham2(a[order], d[order], e[order], None)
    np.testing.assert_allclose(r2.mu_hat, r1.mu_hat[order], rtol=1e-5)


@pytest.mark.parametrize("bad", ["exposure", "deaths"])
def test_an_age_without_exposure_or_with_missing_deaths_is_ignored_but_still_smoothed(bad):
    a, d, e = poisson_table(14)
    d2, e2 = d.copy(), e.copy()
    if bad == "exposure":
        e2[10] = 0.0
    else:
        d2[10] = np.nan
    keep = np.arange(a.size) != 10
    r, clean = fit_makeham2(a, d2, e2, None), fit_makeham2(a[keep], d[keep], e[keep], None)
    np.testing.assert_allclose(r.mu_hat[keep], clean.mu_hat, rtol=1e-3)
    assert np.all(np.isfinite(r.mu_hat)) and r.mu_hat[10] > 0 and "1 âge ignoré" in r.message


@pytest.mark.parametrize("args,msg", [
    (lambda a, d, e: (a, d[:-1], e), "même longueur"),
    (lambda a, d, e: (np.r_[a[:-1], a[0]], d, e), "dupliqués"),
    (lambda a, d, e: (a, np.r_[-1.0, d[1:]], e), "négatif"),
    (lambda a, d, e: (np.r_[np.nan, a[1:]], d, e), "non fini"),
])
def test_malformed_input_is_refused(args, msg):
    with pytest.raises(ValueError, match=msg):
        fit_makeham2(*args(*poisson_table(15)), None)


def test_five_ages_and_two_ages_with_deaths_are_needed():
    a, d, e = poisson_table(16)
    with pytest.raises(ValueError, match="au moins 5"):
        fit_makeham2(a[:4], d[:4], e[:4], None)
    one = np.zeros_like(d)
    one[30] = 50.0
    with pytest.raises(ValueError, match="au moins 5"):
        fit_makeham2(a, one, e, None)


# --------------------------------------------------------------------------
# Real data: France 2020 (HMD), the standard file shipped in sample_data/
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def france():
    df = L.load_data(ROOT / "sample_data" / "FRA_taux_bruts_moment_2020_female_male.csv")
    cfg = C.default_config(df["age"].unique())
    return {sex: C.zone_slice(df[df["sex"] == sex], cfg, "zone2") for sex in BOTH}, cfg


PINNED = {                      # measured on the sample file, ages 20..85; differential evolution finds the same optimum
    "Female": dict(loglik=-468.1151, A=0.0, c=1.0862, d=1.3545),
    "Male": dict(loglik=-556.5543, A=1.0493e-4, c=1.0889, d=1.5860),
}


@pytest.mark.parametrize("sex", BOTH)
def test_france_2020_optimum_is_the_global_one_and_the_gain_over_makeham_is_large(france, sex):
    tables, cfg = france
    a, d, e = (tables[sex][c].to_numpy() for c in ("age", "Dx", "Ex"))
    r = fit_makeham2(a, d, e, cfg)
    pin = PINNED[sex]
    assert r.converged and r.loglik == pytest.approx(pin["loglik"], abs=2e-3)
    assert r.params["A"] == pytest.approx(pin["A"], abs=2e-6)
    assert (r.params["A"] == 0.0) == (sex == "Female") and ("A = 0 (borne)" in r.message) == (sex == "Female")
    assert r.params["c"] == pytest.approx(pin["c"], rel=1e-3) and r.params["d"] == pytest.approx(pin["d"], rel=1e-3)
    assert r.loglik - fit_makeham(a, d, e, cfg).loglik > 300
    assert half_deviance(d, e, r.mu_hat) <= global_optimum(a, d, e) + 1e-4


@pytest.fixture(scope="module")
def smoothing_lab():
    sys.path.insert(0, str(POST3))
    pytest.importorskip("smoothing_lab.zone2_makeham2")
    from smoothing_lab import zone2_makeham2 as mod
    return mod


@pytest.mark.parametrize("sex", BOTH)
def test_france_2020_agrees_with_the_smoothing_lab_implementation(smoothing_lab, france, sex):
    """The lab's labels may be swapped (its seed makes the steep term first) but the curve is the same."""
    tables, cfg = france
    a, d, e = (tables[sex][c].to_numpy() for c in ("age", "Dx", "Ex"))
    r, lab = fit_makeham2(a, d, e, cfg), smoothing_lab.fit_makeham2(a, d, e)
    assert r.loglik == pytest.approx(poisson_loglik(d, e, lab.mu(a)), abs=2e-3)
    np.testing.assert_allclose(r.mu_hat, lab.mu(a), rtol=2e-3)
