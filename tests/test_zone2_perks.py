"""Module 3d (Perks) - zone2_perks.py. Run:   python -m pytest tests/test_zone2_perks.py -v

Evidence that does not replay the code's own formulas:
  * exact recovery from noise-free expected counts;
  * the global optimum found independently by differential evolution (Nelder-Mead alone can stop short);
  * the score of every free parameter vanishes at an interior optimum (central finite differences);
  * nesting: Perks can never be below Makeham (D = 0), and collapses onto it, with D = 0 exactly, when the data are
    Makeham or accelerate beyond it (which is what France 2020 does);
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
from src import zone2_perks as P
from src.fit_common import NelderMeadResult, half_deviance, poisson_loglik
from src.zone2_makeham import fit_makeham
from src.zone2_perks import fit_perks

BOTH = ("Female", "Male")
ROOT = Path(__file__).resolve().parent.parent
POST3 = Path(os.environ.get(
    "MORTALITY_POST3_DIR",
    r"C:\Users\simar\OneDrive\Bureau\Persos\LinkedIn\Life_Risks_Series\Post3-!Smoothing1"))

TRUTH = dict(A=2.0e-4, B=1.5e-5, c=1.105, D=2.0e-4)           # D c^85 ~ 0.9: the curve visibly decelerates by 85
AGES = np.arange(20, 86, dtype=float)
EXPO = 4.0e5 * np.exp(-0.06 * (AGES - 20))


def perks_mu(A, B, c, D, ages=AGES):
    w = c ** ages
    return (A + B * w) / (1.0 + D * w)


def expected_table(**kw):
    return AGES.copy(), EXPO * perks_mu(**{**TRUTH, **kw}), EXPO.copy()


def poisson_table(seed, **kw):
    d = np.random.default_rng(seed).poisson(EXPO * perks_mu(**{**TRUTH, **kw})).astype(float)
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
        w = np.exp(np.exp(t[2]) * (a - xb))
        return half_deviance(d, e, (s_a * t[0] ** 2 + np.exp(t[1]) * w) / (1.0 + np.exp(t[3]) * w))

    return differential_evolution(obj, [(0, 3), (-8, -1), (np.log(np.log(1.03)), np.log(np.log(1.3))), (-30, 4)],
                                  seed=1, popsize=40, tol=1e-13, maxiter=3000, polish=True).fun


def log_scores(r, a, d, e):
    """d ell / d ln(theta_i), relative to sum(D), by central differences (A, B, c, D all positive here)."""
    base = np.array([r.params[k] for k in ("A", "B", "c", "D")])
    out = []
    for i in range(4):
        h = 1e-5
        up, dn = base.copy(), base.copy()
        up[i] *= np.exp(h)
        dn[i] *= np.exp(-h)
        out.append((poisson_loglik(d, e, perks_mu(*up, ages=a)) - poisson_loglik(d, e, perks_mu(*dn, ages=a))) / (2 * h))
    return np.array(out) / d.sum()


# --------------------------------------------------------------------------
# The estimator
# --------------------------------------------------------------------------


def test_exact_recovery_from_noise_free_expected_counts():
    a, d, e = expected_table()
    r = fit_perks(a, d, e, None)
    for k in ("A", "B", "c", "D"):
        assert r.params[k] == pytest.approx(TRUTH[k], rel=1e-3), k
    np.testing.assert_allclose(r.mu_hat, perks_mu(**TRUTH), rtol=1e-6)
    assert r.loglik >= poisson_loglik(d, e, perks_mu(**TRUTH)) - 1e-6
    assert r.converged and r.params["D"] > 0


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_it_reaches_the_global_optimum_found_by_differential_evolution(seed):
    a, d, e = poisson_table(seed)
    r = fit_perks(a, d, e, None)
    assert half_deviance(d, e, r.mu_hat) <= global_optimum(a, d, e) + 1e-4
    assert r.params["D"] > 0


@pytest.mark.parametrize("seed", [4, 5])
def test_every_free_parameter_has_a_vanishing_score_at_an_interior_optimum(seed):
    a, d, e = poisson_table(seed)
    r = fit_perks(a, d, e, None)
    assert r.params["A"] > 0 and r.params["D"] > 0
    np.testing.assert_allclose(log_scores(r, a, d, e), 0.0, atol=2e-5)


def test_perks_is_never_below_makeham_and_the_data_decide_if_it_gains():
    a, d, e = poisson_table(6)
    gain = fit_perks(a, d, e, None).loglik - fit_makeham(a, d, e, None).loglik
    assert gain > 10                                                     # a decelerating truth: Perks must win clearly
    for a, d, e in [makeham_table(7), synthetic_zone2("Female"), synthetic_zone2("Male")]:
        assert fit_perks(a, d, e, None).loglik >= fit_makeham(a, d, e, None).loglik - 1e-7


def test_on_makeham_data_perks_collapses_onto_makeham_with_d_exactly_zero():
    a, d, e = makeham_table()
    r, m = fit_perks(a, d, e, None), fit_makeham(a, d, e, None)
    assert r.params["D"] == 0.0 and "D = 0 (borne)" in r.message and "Makeham" in r.message
    assert r.params["A"] == pytest.approx(m.params["A"], rel=1e-4) and r.params["c"] == pytest.approx(m.params["c"], rel=1e-5)
    assert r.loglik == pytest.approx(m.loglik, abs=1e-6)
    np.testing.assert_allclose(r.mu_hat, m.mu_hat, rtol=1e-5)


def test_a_equal_to_zero_is_reported_exactly_when_the_data_have_no_constant_term():
    a, d, e = expected_table(A=0.0)
    r = fit_perks(a, d, e, None)
    assert r.params["A"] == 0.0 and "A = 0 (borne)" in r.message and r.params["D"] > 0
    np.testing.assert_allclose(r.mu_hat, perks_mu(**{**TRUTH, "A": 0.0}), rtol=1e-6)


def test_a_cannot_go_negative_even_when_the_data_would_like_it_to():
    """A truth with A < 0 (rates still positive): the model has A >= 0, so the optimum is ON the wall, A = 0 exactly,
    and the constrained log-likelihood stays below the truth's, which is outside the model."""
    a, d, e = expected_table(A=-5e-5)
    assert perks_mu(**{**TRUTH, "A": -5e-5}).min() > 0
    r = fit_perks(a, d, e, None)
    assert r.params["A"] == 0.0 and "A = 0 (borne)" in r.message
    assert r.loglik < poisson_loglik(d, e, perks_mu(**{**TRUTH, "A": -5e-5})) - 1e-3


def test_the_curve_flattens_towards_the_plateau_b_over_d():
    """The defining feature of Perks: with D > 0 the rate tends to B / D, far below the Makeham extrapolation."""
    a, d, e = expected_table(D=5e-4)
    r = fit_perks(a, d, e, None)
    far = np.array([150.0])
    plateau = r.params["B"] / r.params["D"]
    mu_far = perks_mu(r.params["A"], r.params["B"], r.params["c"], r.params["D"], ages=far)[0]
    assert plateau == pytest.approx(TRUTH["B"] / 5e-4, rel=2e-3)
    assert 0.9 * plateau < mu_far < 1.02 * plateau


def test_spec_validation_perks_converges_and_the_fitresult_is_complete():
    """Spec section 8, J3 afternoon: 'Tous les modeles convergent. FitResult complet pour chaque methode.'"""
    for a, d, e in [synthetic_zone2("Female"), synthetic_zone2("Male"), poisson_table(8), makeham_table(9)]:
        r = fit_perks(a, d, e, None)
        assert r.converged and np.isfinite(r.loglik) and np.all(np.isfinite(r.mu_hat)) and r.message


# --------------------------------------------------------------------------
# Configuration, convergence reporting, the FitResult protocol
# --------------------------------------------------------------------------


def test_cfg_nr_tol_is_the_tolerance_of_the_objective_and_nr_maxiter_is_not_used_by_nelder_mead():
    a, d, e = poisson_table(10)
    tight = fit_perks(a, d, e, C.SmootherConfig(nr_tol=1e-12))
    loose = fit_perks(a, d, e, C.SmootherConfig(nr_tol=1e-3))
    assert loose.n_iter < tight.n_iter and loose.loglik <= tight.loglik + 1e-9
    capped = fit_perks(a, d, e, C.SmootherConfig(nr_maxiter=1))                # caps the Makeham seed only
    assert capped.converged and capped.loglik == pytest.approx(fit_perks(a, d, e, None).loglik, abs=1e-5)


def test_a_failed_optimisation_is_reported_not_raised(monkeypatch):
    def fake(fun, simplex, **kw):
        return NelderMeadResult(np.asarray(simplex)[0], float(fun(np.asarray(simplex)[0])), 7, 20, False, 6)
    monkeypatch.setattr(P, "nelder_mead", fake)
    r = fit_perks(*poisson_table(11), None)
    assert r.converged is False and r.n_iter == 7 and "n'a pas convergé" in r.message
    assert np.all(np.isfinite(r.mu_hat)) and np.isfinite(r.loglik)


def test_the_fitresult_protocol():
    a, d, e = poisson_table(12)
    before = (a.copy(), d.copy(), e.copy())
    r = fit_perks(a, d, e, C.SmootherConfig())
    assert isinstance(r, C.FitResult) and (r.method, r.zone) == ("perks", "zone2")
    np.testing.assert_array_equal(r.ages, a)
    assert r.mu_hat.shape == a.shape and np.all(r.mu_hat > 0)
    assert set(r.params) == {"A", "B", "c", "D", "n_params"} and r.params["n_params"] == 4
    np.testing.assert_allclose(r.mu_hat, perks_mu(r.params["A"], r.params["B"], r.params["c"], r.params["D"], ages=a), rtol=1e-9)
    assert r.loglik == pytest.approx(poisson_loglik(d, e, r.mu_hat), rel=1e-14)
    assert isinstance(r.n_iter, int) and r.n_iter > 0 and "itérations" in r.message
    for got, was in zip((a, d, e), before):
        np.testing.assert_array_equal(got, was)


def test_the_config_defaults_of_the_spec_are_used_when_it_is_omitted():
    a, d, e = poisson_table(13)
    np.testing.assert_array_equal(fit_perks(a, d, e, None).mu_hat, fit_perks(a, d, e, C.SmootherConfig()).mu_hat)


def test_ages_keep_the_callers_order():
    a, d, e = poisson_table(14)
    order = np.random.default_rng(0).permutation(a.size)
    r1, r2 = fit_perks(a, d, e, None), fit_perks(a[order], d[order], e[order], None)
    np.testing.assert_allclose(r2.mu_hat, r1.mu_hat[order], rtol=1e-5)


@pytest.mark.parametrize("bad", ["exposure", "deaths"])
def test_an_age_without_exposure_or_with_missing_deaths_is_ignored_but_still_smoothed(bad):
    a, d, e = poisson_table(15)
    d2, e2 = d.copy(), e.copy()
    if bad == "exposure":
        e2[10] = 0.0
    else:
        d2[10] = np.nan
    keep = np.arange(a.size) != 10
    r, clean = fit_perks(a, d2, e2, None), fit_perks(a[keep], d[keep], e[keep], None)
    np.testing.assert_allclose(r.mu_hat[keep], clean.mu_hat, rtol=1e-4)
    assert np.all(np.isfinite(r.mu_hat)) and r.mu_hat[10] > 0 and "1 âge ignoré" in r.message


@pytest.mark.parametrize("args,msg", [
    (lambda a, d, e: (a, d[:-1], e), "même longueur"),
    (lambda a, d, e: (np.r_[a[:-1], a[0]], d, e), "dupliqués"),
    (lambda a, d, e: (a, np.r_[-1.0, d[1:]], e), "négatif"),
    (lambda a, d, e: (np.r_[np.nan, a[1:]], d, e), "non fini"),
])
def test_malformed_input_is_refused(args, msg):
    with pytest.raises(ValueError, match=msg):
        fit_perks(*args(*poisson_table(16)), None)


def test_four_ages_and_two_ages_with_deaths_are_needed():
    a, d, e = poisson_table(17)
    with pytest.raises(ValueError, match="au moins 4"):
        fit_perks(a[:3], d[:3], e[:3], None)
    one = np.zeros_like(d)
    one[30] = 50.0
    with pytest.raises(ValueError, match="au moins 4"):
        fit_perks(a, one, e, None)


# --------------------------------------------------------------------------
# Real data: France 2020 (HMD), the standard file shipped in sample_data/
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def france():
    df = L.load_data(ROOT / "sample_data" / "FRA_taux_bruts_moment_2020_female_male.csv")
    cfg = C.default_config(df["age"].unique())
    return {sex: C.zone_slice(df[df["sex"] == sex], cfg, "zone2") for sex in BOTH}, cfg


@pytest.mark.parametrize("sex", BOTH)
def test_france_2020_perks_reduces_to_makeham_and_a_global_search_agrees(france, sex):
    """The data accelerate beyond Makeham at the oldest ages (a steeper second exponential gains 400-900 in
    log-likelihood), so D >= 0 binds: not an optimiser failure, differential evolution finds the same optimum."""
    tables, cfg = france
    a, d, e = (tables[sex][c].to_numpy() for c in ("age", "Dx", "Ex"))
    r, m = fit_perks(a, d, e, cfg), fit_makeham(a, d, e, cfg)
    assert r.converged and r.params["D"] == 0.0
    assert r.loglik == pytest.approx(m.loglik, abs=1e-6)
    assert half_deviance(d, e, r.mu_hat) <= global_optimum(a, d, e) + 1e-4


@pytest.fixture(scope="module")
def smoothing_lab():
    sys.path.insert(0, str(POST3))
    pytest.importorskip("smoothing_lab.zone2_perks")
    from smoothing_lab import zone2_perks as mod
    return mod


@pytest.mark.parametrize("sex", BOTH)
def test_france_2020_agrees_with_the_smoothing_lab_implementation(smoothing_lab, france, sex):
    tables, cfg = france
    a, d, e = (tables[sex][c].to_numpy() for c in ("age", "Dx", "Ex"))
    r, lab = fit_perks(a, d, e, cfg), smoothing_lab.fit_perks(a, d, e)
    assert r.loglik == pytest.approx(poisson_loglik(d, e, lab.mu(a)), abs=1e-4)
    np.testing.assert_allclose(r.mu_hat, lab.mu(a), rtol=1e-5)
