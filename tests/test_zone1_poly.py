"""Module 3a - zone1_poly.py (poly3 and spline). Run:   python -m pytest tests/test_zone1_poly.py -v

Three kinds of evidence, none of them the code's own formulas replayed:
  * exact recovery: noise-free data generated from a known cubic / natural spline must come back exactly;
  * independent oracles: `np.polyfit` (weights sqrt(D)) for poly3, and the user's smoothing_lab (truncated-power
    basis, a different construction of the same spline space) when its folder is present;
  * the spec's rules as stated: D = 0 ages out of the fit and the knots but smoothed in the output, one knot
    every 5 ages, the FitResult protocol.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.interpolate import CubicSpline
from scipy.stats import poisson

import synthetic as S
from src import config as C
from src import loader as L
from src.zone1_poly import fit_poly3, fit_spline

FITS = {"poly3": fit_poly3, "spline": fit_spline}
BOTH = ("Female", "Male")
ROOT = Path(__file__).resolve().parent.parent
POST3 = Path(os.environ.get(
    "MORTALITY_POST3_DIR",
    r"C:\Users\simar\OneDrive\Bureau\Persos\LinkedIn\Life_Risks_Series\Post3-!Smoothing1"))


def zone(sex="Female", hi=20):
    """Synthetic Poisson table of one sex, ages 0..hi (every age has deaths)."""
    t = S.make_sex_table(sex).iloc[:hi + 1]
    assert (t["Dx"] > 0).all()
    return t["age"].to_numpy(), t["Dx"].to_numpy().astype(float), t["Ex"].to_numpy()


def exact(ages, ln_mu, e0=2.0e5):
    """Noise-free deaths: Dx = Ex * mu exactly (non-integer, which the fit accepts)."""
    ex = e0 * np.exp(-0.05 * ages)
    return ex * np.exp(ln_mu), ex


# --------------------------------------------------------------------------
# poly3
# --------------------------------------------------------------------------


@pytest.mark.parametrize("hi", [20, 59])
def test_poly3_recovers_an_exact_cubic_in_the_plain_age_basis(hi):
    """A zone ending at 59 has x^3 up to 2e5: the internal centring must not leak into the coefficients."""
    ages = np.arange(0, hi + 1, dtype=float)
    a = np.array([-6.0, -0.9, 0.04, -0.0004]) if hi == 59 else np.array([-6.0, -1.2, 0.1, -0.002])
    ln_mu = a[0] + a[1] * ages + a[2] * ages ** 2 + a[3] * ages ** 3
    d, e = exact(ages, ln_mu)
    r = fit_poly3(ages, d, e, None)
    np.testing.assert_allclose([r.params[f"a{j}"] for j in range(4)], a, rtol=1e-7)
    np.testing.assert_allclose(r.mu_hat, np.exp(ln_mu), rtol=1e-9)


def test_poly3_stays_exact_on_a_high_age_window_where_the_raw_normal_equations_lose_digits():
    """Ages 60..109: cond(A'WA) is about 1e18 in the plain-age basis (8 digits lost); the centred solve is not."""
    ages = np.arange(60, 110, dtype=float)
    a = np.array([-14.0, 0.3, 0.0005, -0.000005])
    d, e = exact(ages, a[0] + a[1] * ages + a[2] * ages ** 2 + a[3] * ages ** 3)
    r = fit_poly3(ages, d, e, None)
    np.testing.assert_allclose([r.params[f"a{j}"] for j in range(4)], a, rtol=1e-10)


def test_poly3_on_poisson_data_equals_an_independent_weighted_polyfit():
    """np.polyfit multiplies residuals by w, so w = sqrt(D) gives the weights D of the spec."""
    ages, d, e = zone("Male")
    r = fit_poly3(ages, d, e, None)
    oracle = np.polyfit(ages, np.log(d / e), 3, w=np.sqrt(d))[::-1]           # a0 first
    np.testing.assert_allclose([r.params[f"a{j}"] for j in range(4)], oracle, rtol=1e-6)
    np.testing.assert_allclose(r.mu_hat, np.exp(np.polyval(oracle[::-1], ages)), rtol=1e-9)


def test_poly3_is_weighted_not_ordinary_least_squares():
    ages, d, e = zone("Female")
    ols = np.polyfit(ages, np.log(d / e), 3)[::-1]
    got = np.array([fit_poly3(ages, d, e, None).params[f"a{j}"] for j in range(4)])
    assert np.max(np.abs(got - ols) / np.abs(ols)) > 1e-3


# --------------------------------------------------------------------------
# spline
# --------------------------------------------------------------------------


def test_spline_recovers_an_exact_natural_spline_and_its_knot_values():
    ages = np.arange(0, 21, dtype=float)
    knots = np.array([0.0, 5.0, 10.0, 15.0, 20.0])
    values = np.array([-5.5, -8.0, -9.3, -8.7, -7.5])
    ln_mu = CubicSpline(knots, values, bc_type="natural")(ages)
    d, e = exact(ages, ln_mu)
    r = fit_spline(ages, d, e, None)
    assert r.params["knots"] == knots.tolist()
    np.testing.assert_allclose(r.params["ln_mu_knots"], values, atol=1e-9)
    np.testing.assert_allclose(r.mu_hat, np.exp(ln_mu), rtol=1e-9)


def test_spline_with_one_observation_per_knot_is_the_interpolation_of_the_crude_rates():
    ages, d, e = np.array([0.0, 5.0, 10.0]), np.array([900.0, 40.0, 25.0]), np.array([3e5, 4e5, 4e5])
    r = fit_spline(ages, d, e, None)
    np.testing.assert_allclose(r.mu_hat, d / e, rtol=1e-10)
    assert r.params["n_params"] == 3


@pytest.mark.parametrize("lo,hi,step,expected", [
    (0, 20, 5, [0, 5, 10, 15, 20]),                 # the spec's rule: one knot every 5 ages
    (1, 21, 5, [1, 6, 11, 16, 21]),                 # counted from the first age of the zone
    (0, 11, 5, [0, 5, 11]),                         # last age 1 from the grid: it takes the grid point's place
    (0, 13, 5, [0, 5, 10, 13]),                     # last age 3 from the grid: it is added
    (0, 20, 4, [0, 4, 8, 12, 16, 20]),
])
def test_spline_knots_follow_one_every_five_ages_and_include_both_ends(lo, hi, step, expected):
    ages = np.arange(lo, hi + 1, dtype=float)
    d, e = exact(ages, -9.0 + 0.3 * np.sin(ages / 4))
    r = fit_spline(ages, d, e, None, knot_step=step)
    assert r.params["knots"] == [float(k) for k in expected]
    assert r.params["n_params"] == len(expected)


def test_spline_a_knot_age_without_deaths_is_dropped_and_still_smoothed():
    ages, d, e = zone("Female")
    d[10] = 0.0                                                         # age 10 is a knot target
    r = fit_spline(ages, d, e, None)
    assert r.params["knots"] == [0.0, 5.0, 15.0, 20.0]
    assert np.isfinite(r.mu_hat[10]) and r.mu_hat[10] > 0
    cs = CubicSpline(r.params["knots"], r.params["ln_mu_knots"], bc_type="natural")
    assert r.mu_hat[10] == pytest.approx(float(np.exp(cs(10.0))), rel=1e-12)


def test_spline_is_a_straight_line_in_ln_mu_beyond_the_boundary_knots():
    """Age 0 without deaths: the first knot is 5, so ages 0..5 must be collinear (natural = zero curvature)."""
    ages, d, e = zone("Female")
    d[0] = 0.0
    r = fit_spline(ages, d, e, None)
    assert r.params["knots"][0] == 5.0
    ln = np.log(r.mu_hat[:6])
    np.testing.assert_allclose(np.diff(ln, 2), 0.0, atol=1e-10)
    cs = CubicSpline(r.params["knots"], r.params["ln_mu_knots"], bc_type="natural")
    np.testing.assert_allclose(ln[5] - ln[4], float(cs(5.0, 1)), rtol=1e-9)     # the tangent at the first knot


# --------------------------------------------------------------------------
# Rules common to both methods
# --------------------------------------------------------------------------


@pytest.mark.parametrize("key", FITS)
def test_ages_without_deaths_are_left_out_of_the_fit_but_smoothed_in_the_output(key):
    ages, d, e = zone("Male")
    d0 = d.copy()
    d0[[7, 13]] = 0.0
    kept = np.setdiff1d(ages, [7, 13])
    full = FITS[key](ages, d0, e, None)
    without = FITS[key](kept, d0[kept.astype(int)], e[kept.astype(int)], None)
    np.testing.assert_allclose(full.mu_hat[kept.astype(int)], without.mu_hat, rtol=1e-10)
    assert len(full.mu_hat) == len(ages) and np.all(np.isfinite(full.mu_hat)) and np.all(full.mu_hat > 0)
    assert "2 âges exclus" in full.message
    if key == "poly3":
        a = [full.params[f"a{j}"] for j in range(4)]
        np.testing.assert_allclose(full.mu_hat[[7, 13]], np.exp(np.polyval(a[::-1], [7.0, 13.0])), rtol=1e-9)


@pytest.mark.parametrize("key", FITS)
def test_zero_exposure_is_treated_like_zero_deaths(key):
    ages, d, e = zone("Female")
    e0 = e.copy()
    d0 = d.copy()
    e0[3], d0[3] = 0.0, 0.0
    r = FITS[key](ages, d0, e0, None)
    assert np.all(np.isfinite(r.mu_hat)) and np.isfinite(r.loglik)


@pytest.mark.parametrize("key", FITS)
def test_an_age_with_deaths_but_no_exposure_is_a_data_error_left_out_not_a_nan(key):
    """E = 0 with D > 0 would give ln(D/E) = +inf; the loader flags such rows, the fit must not be poisoned."""
    ages, d, e = zone("Female")
    bad_e = e.copy()
    bad_e[3] = 0.0
    kept = np.setdiff1d(ages, [3]).astype(int)
    r = FITS[key](ages, d, bad_e, None)
    clean = FITS[key](ages[kept], d[kept], e[kept], None)
    np.testing.assert_allclose(r.mu_hat[kept], clean.mu_hat, rtol=1e-10)
    assert np.all(np.isfinite(r.mu_hat)) and np.isfinite(r.loglik)


@pytest.mark.parametrize("key", FITS)
def test_the_fitresult_protocol(key):
    ages, d, e = zone("Female")
    before = (ages.copy(), d.copy(), e.copy())
    r = FITS[key](ages, d, e, C.SmootherConfig())
    assert isinstance(r, C.FitResult)
    assert (r.method, r.zone) == (key, "zone1")
    np.testing.assert_array_equal(r.ages, ages)
    assert r.mu_hat.shape == ages.shape and np.all(r.mu_hat > 0)
    assert r.converged is True and r.n_iter == 0 and isinstance(r.message, str) and r.message
    assert isinstance(r.params["n_params"], int)
    assert r.params["n_params"] == (4 if key == "poly3" else len(r.params["knots"]))
    for got, was in zip((ages, d, e), before):
        np.testing.assert_array_equal(got, was)                          # the inputs are not modified


@pytest.mark.parametrize("key", FITS)
def test_loglik_is_the_full_poisson_loglik_at_the_fitted_rates(key):
    ages, d, e = zone("Male")
    r = FITS[key](ages, np.round(d), e, None)
    expected = poisson.logpmf(np.round(d), e * r.mu_hat).sum()
    assert r.loglik == pytest.approx(expected, rel=1e-10)


@pytest.mark.parametrize("key", FITS)
def test_loglik_counts_the_zero_death_ages_too(key):
    ages, d, e = zone("Female")
    d[8] = 0.0
    r = FITS[key](ages, d, e, None)
    assert r.loglik == pytest.approx(poisson.logpmf(d, e * r.mu_hat).sum(), rel=1e-10)


@pytest.mark.parametrize("key", FITS)
def test_the_config_is_not_needed_and_changes_nothing(key):
    ages, d, e = zone("Female")
    a = FITS[key](ages, d, e, None)
    b = FITS[key](ages, d, e, C.SmootherConfig(omega1=35))
    np.testing.assert_array_equal(a.mu_hat, b.mu_hat)


@pytest.mark.parametrize("key", FITS)
def test_ages_keep_the_callers_order(key):
    ages, d, e = zone("Female")
    order = np.random.default_rng(0).permutation(len(ages))
    a = FITS[key](ages, d, e, None)
    b = FITS[key](ages[order], d[order], e[order], None)
    np.testing.assert_allclose(b.mu_hat, a.mu_hat[order], rtol=1e-10)


# --------------------------------------------------------------------------
# Refusals (French messages, ValueError: the app shows them in the status table)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("key", FITS)
@pytest.mark.parametrize("what,args,msg", [
    ("lengths", lambda a, d, e: (a, d[:-1], e), "même longueur"),
    ("duplicate age", lambda a, d, e: (np.r_[a[:-1], a[0]], d, e), "dupliqués"),
    ("negative deaths", lambda a, d, e: (a, np.r_[-1.0, d[1:]], e), "négatif"),
    ("nan age", lambda a, d, e: (np.r_[np.nan, a[1:]], d, e), "non fini"),
])
def test_malformed_input_is_refused(key, what, args, msg):
    ages, d, e = zone("Female")
    with pytest.raises(ValueError, match=msg):
        FITS[key](*args(ages, d, e), None)


def test_poly3_needs_four_ages_with_deaths():
    ages, d, e = zone("Female", hi=9)
    d[:] = 0.0
    d[[2, 4, 6]] = 50.0
    with pytest.raises(ValueError, match="au moins 4"):
        fit_poly3(ages, d, e, None)
    d[8] = 50.0
    assert fit_poly3(ages, d, e, None).converged                          # four are enough (exact interpolation)


def test_spline_needs_three_knots():
    ages, d, e = zone("Female", hi=4)                                    # knots 0 and 4 only
    with pytest.raises(ValueError, match="au moins 3"):
        fit_spline(ages, d, e, None)
    with pytest.raises(ValueError, match="knot_step"):
        fit_spline(*zone("Female"), None, knot_step=0)
    a, dd, ee = zone("Female", hi=8)                                      # knots 0, 5, 8
    assert fit_spline(a, dd, ee, None).params["knots"] == [0.0, 5.0, 8.0]


# --------------------------------------------------------------------------
# Real data: France 2020 (HMD), the standard file shipped in sample_data/
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def france():
    df = L.load_data(ROOT / "sample_data" / "FRA_taux_bruts_moment_2020_female_male.csv")
    cfg = C.default_config(df["age"].unique())
    return {sex: C.zone_slice(df[df["sex"] == sex], cfg, "zone1") for sex in BOTH}, cfg


PINNED_POLY3 = {                       # measured on the sample file; agree with smoothing_lab to 1e-12
    "Female": [-5.833347716489525, -1.4146828254517347, 0.1321637783510149, -0.00344896694204864],
    "Male": [-5.6621876029723115, -1.3368468540871166, 0.12107079675517135, -0.002962042747056012],
}


@pytest.mark.parametrize("sex", BOTH)
def test_france_2020_poly3_coefficients(france, sex):
    tables, cfg = france
    t = tables[sex]
    assert (t["age"].min(), t["age"].max()) == (0, 20)
    r = fit_poly3(t["age"].to_numpy(), t["Dx"].to_numpy(), t["Ex"].to_numpy(), cfg)
    np.testing.assert_allclose([r.params[f"a{j}"] for j in range(4)], PINNED_POLY3[sex], rtol=1e-9)


@pytest.mark.parametrize("sex", BOTH)
def test_france_2020_the_curve_is_positive_finite_and_falls_from_birth_to_the_minimum(france, sex):
    tables, cfg = france
    t = tables[sex]
    for key, fit in FITS.items():
        mu = fit(t["age"].to_numpy(), t["Dx"].to_numpy(), t["Ex"].to_numpy(), cfg).mu_hat
        assert np.all(np.isfinite(mu)) and np.all(mu > 0)
        assert mu[0] > 2 * mu[1] and np.all(np.diff(mu[:4]) < 0), key            # infant spike, then the fall
        assert 4 <= np.argmin(mu) <= 12, key                                     # minimum in late childhood


@pytest.fixture(scope="module")
def smoothing_lab():
    sys.path.insert(0, str(POST3))
    pytest.importorskip("smoothing_lab.zone1_poly")
    from smoothing_lab import zone1_poly as mod
    return mod


@pytest.mark.parametrize("sex", BOTH)
def test_france_2020_agrees_with_the_smoothing_lab_implementations(smoothing_lab, france, sex):
    """poly3: same power-basis WLS. spline: the lab's truncated-power natural basis on the same knots spans the
    same space as the cardinal CubicSpline basis, so the fitted curves must coincide."""
    tables, cfg = france
    a, d, e = (tables[sex][c].to_numpy() for c in ("age", "Dx", "Ex"))
    p, s = fit_poly3(a, d, e, cfg), fit_spline(a, d, e, cfg)
    lab_p = smoothing_lab.fit_poly3(a, d, e)
    lab_s = smoothing_lab.fit_natural_spline(a, d, e, knots=np.array(s.params["knots"]))
    np.testing.assert_allclose([p.params[f"a{j}"] for j in range(4)], lab_p.coeffs, rtol=1e-8)
    np.testing.assert_allclose(p.mu_hat, lab_p.mu(a), rtol=1e-9)
    np.testing.assert_allclose(s.mu_hat, lab_s.mu(a), rtol=1e-8)
