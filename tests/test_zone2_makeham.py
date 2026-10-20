"""Module 3c - zone2_makeham.py. Run:   python -m pytest tests/test_zone2_makeham.py -v

Evidence that does not replay the code's own formulas:
  * exact recovery from noise-free expected counts (the score vanishes at the true theta);
  * the three score equations of the interior optimum, d ell / dA = d ell / dB = d ell / dc = 0;
  * independent optimisers: bounded L-BFGS-B on the (A, B) sub-problem (including the A = 0 and B = 0 walls),
    differential evolution and Nelder-Mead on the full 3-parameter problem, and the user's smoothing_lab;
  * nesting: Makeham can never be below Gompertz (A = 0), and collapses onto it when the data are Gompertz;
  * the validation test of spec section 8 (A > 0, c in [1.05, 1.15], AIC Makeham < AIC Gompertz).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import differential_evolution, minimize

import synthetic as S
from src import config as C
from src import loader as L
from src import zone2_makeham as M
from src.fit_common import half_deviance, poisson_loglik
from src.zone2_gompertz import fit_gompertz
from src.zone2_makeham import _solve_ab, _stays_on_wall, fit_makeham

BOTH = ("Female", "Male")
ROOT = Path(__file__).resolve().parent.parent
POST3 = Path(os.environ.get(
    "MORTALITY_POST3_DIR",
    r"C:\Users\simar\OneDrive\Bureau\Persos\LinkedIn\Life_Risks_Series\Post3-!Smoothing1"))

A_T, B_T, C_T = 2.0e-4, 1.5e-5, 1.105                        # truth: accident term, then 1.5e-5 * 1.105^x
AGES = np.arange(20, 86, dtype=float)
EXPO = 4.0e5 * np.exp(-0.06 * (AGES - 20))


def mu_true(a=A_T, b=B_T, c=C_T, ages=AGES):
    return a + b * c ** ages


def expected_table(**kw):
    return AGES.copy(), EXPO * mu_true(**kw), EXPO.copy()


def poisson_table(seed, **kw):
    d = np.random.default_rng(seed).poisson(EXPO * mu_true(**kw)).astype(float)
    return AGES.copy(), d, EXPO.copy()


def synthetic_zone2(sex="Female"):
    t = S.make_sex_table(sex).iloc[20:86]
    return t["age"].to_numpy(), t["Dx"].to_numpy().astype(float), t["Ex"].to_numpy()


def interior_scores(r, a, d, e):
    """Relative scores (d ell / d theta_i) * theta_i / sum(D) at the fitted (A, B, c)."""
    A, B, c = (r.params[k] for k in ("A", "B", "c"))
    mu = A + B * c ** a
    q = d / mu - e
    g = np.array([q.sum(), (q * c ** a).sum(), (q * B * a * c ** (a - 1)).sum()])
    return g * np.array([A, B, c]) / d.sum()


# --------------------------------------------------------------------------
# The sub-problem at a fixed c: (A, B) with A >= 0
# --------------------------------------------------------------------------


def oracle_ab(w, d, e):
    """Bounded L-BFGS-B on the half deviance, from several starts (scipy, independent of the module's Newton)."""
    best = None
    for a0, b0 in [(0.1 * d.sum() / e.sum(), 0.9 * d.sum() / (e * w).sum()), (1e-6, 0.5 * d.sum() / (e * w).sum()),
                   (d.sum() / e.sum(), 1e-9)]:
        res = minimize(lambda t: half_deviance(d, e, t[0] + t[1] * w) / 1e3, [a0, b0], method="L-BFGS-B",
                       bounds=[(0.0, None), (0.0, None)], options=dict(ftol=1e-15, gtol=1e-12, maxiter=2000))
        if best is None or res.fun < best.fun:
            best = res
    return best.x, best.fun * 1e3


@pytest.mark.parametrize("seed", range(12))
def test_the_ab_subproblem_reaches_the_bounded_optimum_an_independent_solver_finds(seed):
    """Random fixed-c problems, half of them with a truth whose A is NEGATIVE (the A >= 0 wall must bind)."""
    rng = np.random.default_rng(seed)
    c = rng.uniform(1.05, 1.15)
    xs = np.arange(30.0, 91.0)
    w = c ** (xs - xs.max())
    e = 3e5 * np.exp(-0.05 * (xs - 30))
    a_true = -4e-5 if seed % 2 else 3e-4
    b_true = 0.02 * rng.uniform(0.5, 1.5)
    d = rng.poisson(e * np.maximum(a_true + b_true * w, 1e-6)).astype(float)
    sol = _solve_ab(w, d, e, 200, 1e-10)
    x_oracle, dev_oracle = oracle_ab(w, d, e)
    assert sol.converged
    assert sol.dev <= dev_oracle + 1e-6 * (1 + dev_oracle)
    assert sol.dev == pytest.approx(dev_oracle, abs=1e-5)
    if seed % 2:
        assert sol.kind == "A=0" and sol.A == 0.0                       # the wall binds, in closed form, no iteration
        assert sol.n_iter == 0
    else:
        assert sol.kind == "interior" and sol.A > 0 and sol.n_iter > 0


def test_the_wall_test_accepts_a_boundary_only_when_leaving_it_cannot_gain_more_than_the_tolerance():
    """Gain of leaving the wall ~ g^2 / (2 S) with S = h_pp - h_pq^2 / h_qq (curvature once the other parameter adjusts)."""
    assert _stays_on_wall(-5.0, 1.0, 0.0, 1.0, 1e-8)                     # the score points into the wall: always stay
    assert _stays_on_wall(0.0, 1.0, 0.0, 1.0, 1e-8)
    assert _stays_on_wall(1e-3, 100.0, 0.0, 1.0, 1e-8)                   # 0.5e-6 / 100 = 5e-9 <= 1e-8
    assert not _stays_on_wall(1e-2, 100.0, 0.0, 1.0, 1e-8)               # 0.5e-4 / 100 = 5e-7 > 1e-8
    assert _stays_on_wall(1e-2, 100.0, 0.0, 1.0, 1e-6)                   # ... but not when the tolerance is looser
    # correlation with the other parameter removes curvature: S = 100 - 99^2 / 100 = 1.99, a 50x larger gain
    assert not _stays_on_wall(1e-3, 100.0, 99.0, 100.0, 1e-8) and _stays_on_wall(1e-3, 100.0, 0.0, 1.0, 1e-8)
    assert _stays_on_wall(1.0, 0.0, 0.0, 1.0, 1e-8) and _stays_on_wall(1.0, 1.0, 5.0, 1.0, 1e-8)     # flat: no gain possible
    assert _stays_on_wall(1.0, 1.0, 0.0, 0.0, 1e-8)                                                   # h_qq = 0: degenerate


def test_the_subproblem_gap_is_the_loglik_still_to_gain_and_maxiter_counts_the_steps():
    c = 1.1
    xs = np.arange(30.0, 91.0)
    w = c ** (xs - xs.max())
    e = 3e5 * np.exp(-0.05 * (xs - 30))
    d = np.random.default_rng(5).poisson(e * (3e-4 + 0.03 * w)).astype(float)
    star = _solve_ab(w, d, e, 200, 1e-13)
    assert star.kind == "interior" and star.n_iter > 2
    checked = 0
    for k in range(1, star.n_iter + 1):
        part = _solve_ab(w, d, e, k, 0.0)                                # stop after exactly k steps
        assert part.n_iter == k and part.kind in ("maxiter", "interior") and not part.converged
        if 1e-6 < part.gap < 1.0:
            assert part.dev - star.dev == pytest.approx(part.gap, rel=0.1)
            checked += 1
    assert checked >= 1 and star.gap <= 1e-13


def test_the_a_equals_zero_boundary_point_is_the_exact_gompertz_like_optimum():
    w = 1.1 ** (np.arange(30.0, 91.0) - 90)
    e = 2e5 * np.exp(-0.05 * np.arange(61))
    d = e * 0.03 * w                                                     # an exact A = 0 curve
    sol = _solve_ab(w, d, e, 200, 1e-10)
    assert (sol.kind, sol.A, sol.n_iter) == ("A=0", 0.0, 0)
    assert sol.b == pytest.approx(0.03, rel=1e-12) and sol.dev == pytest.approx(0.0, abs=1e-9)


def test_the_b_equals_zero_boundary_is_the_constant_force_when_deaths_do_not_rise_with_age():
    w = 1.1 ** (np.arange(30.0, 91.0) - 90)
    e = np.full(61, 1e5)
    d = e * 0.01 * np.exp(-0.02 * np.arange(61))                         # rates FALL with age
    sol = _solve_ab(w, d, e, 200, 1e-10)
    assert (sol.kind, sol.b) == ("B=0", 0.0)
    assert sol.A == pytest.approx(d.sum() / e.sum(), rel=1e-12)


def test_newton_halving_keeps_the_rate_positive_from_the_feasible_start():
    """Ages 30..109 at c = 1.15: w spans 11 orders of magnitude, the case that broke a plain Newton in the lab."""
    xs = np.arange(30.0, 110.0)
    w = 1.15 ** (xs - xs.max())
    e = 3e5 * np.exp(-0.07 * (xs - 30))
    d = np.random.default_rng(3).poisson(e * (2e-4 + 0.3 * w)).astype(float)
    sol = _solve_ab(w, d, e, 200, 1e-10)
    _, dev_oracle = oracle_ab(w, d, e)
    assert sol.converged and sol.A > 0 and sol.dev <= dev_oracle + 1e-5


# --------------------------------------------------------------------------
# The full fit
# --------------------------------------------------------------------------


def test_exact_recovery_from_noise_free_expected_counts():
    r = fit_makeham(*expected_table(), None)
    assert r.params["A"] == pytest.approx(A_T, rel=1e-6)
    assert r.params["B"] == pytest.approx(B_T, rel=1e-6)
    assert r.params["c"] == pytest.approx(C_T, rel=1e-8)
    np.testing.assert_allclose(r.mu_hat, mu_true(), rtol=1e-8)
    assert r.converged


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_the_three_score_equations_hold_at_the_interior_optimum(seed):
    """dl/dA = dl/dB = dl/dc = 0, relative to the deaths: the refined c is a true stationary point, which a
    grid-only c (up to 5e-4 off) would not be."""
    a, d, e = poisson_table(seed)
    r = fit_makeham(a, d, e, None)
    assert r.params["A"] > 0
    np.testing.assert_allclose(interior_scores(r, a, d, e), 0.0, atol=2e-6)


@pytest.mark.parametrize("seed", [4, 5])
def test_it_reaches_the_global_optimum_found_by_differential_evolution(seed):
    a, d, e = poisson_table(seed)
    r = fit_makeham(a, d, e, None)
    sA = d.sum() / e.sum()

    def obj(t):                                                          # (A / sA, ln b at x = 50, ln ln c)
        mu = sA * t[0] + np.exp(t[1]) * np.exp(np.exp(t[2]) * (a - 50.0))
        return half_deviance(d, e, mu)

    de = differential_evolution(obj, [(0.0, 3.0), (-8.0, -1.0), (np.log(np.log(1.05)), np.log(np.log(1.15)))],
                                seed=seed, popsize=40, tol=1e-13, maxiter=3000, polish=True)
    assert half_deviance(d, e, r.mu_hat) <= de.fun + 1e-6


def test_makeham_is_never_below_gompertz_and_collapses_onto_it_when_the_data_are_gompertz():
    for a, d, e in [poisson_table(6), synthetic_zone2("Female"), synthetic_zone2("Male")]:
        assert fit_makeham(a, d, e, None).loglik >= fit_gompertz(a, d, e, None).loglik - 1e-7
    a = AGES.copy()
    d = EXPO * np.exp(-10.8 + 0.095 * a)                                 # a pure Gompertz law, noise-free
    r, g = fit_makeham(a, d, EXPO.copy(), None), fit_gompertz(a, d, EXPO.copy(), None)
    assert r.params["A"] == 0.0 and "A = 0 (borne)" in r.message
    assert r.params["c"] == pytest.approx(g.params["c"], rel=1e-7) and r.params["B"] == pytest.approx(g.params["B"], rel=1e-6)
    assert r.loglik == pytest.approx(g.loglik, abs=1e-6)


def test_a_curve_whose_deaths_fall_with_age_reports_b_equal_zero():
    a = AGES.copy()
    r = fit_makeham(a, EXPO * 0.01 * np.exp(-0.02 * (a - 20)), EXPO.copy(), None)
    assert r.params["B"] == 0.0 and "B = 0 (borne)" in r.message and r.params["A"] > 0
    np.testing.assert_allclose(r.mu_hat, r.params["A"])


@pytest.mark.parametrize("c_true,bound", [(1.2, 1.15), (1.02, 1.05)])
def test_c_is_kept_inside_the_spec_range_and_the_message_says_when_it_sits_on_an_end(c_true, bound):
    a = AGES.copy()
    d = EXPO * (1e-4 + 1e-5 * c_true ** (a - 20) * (1.105 / c_true) ** 0)
    r = fit_makeham(a, d, EXPO.copy(), None)
    assert r.params["c"] == pytest.approx(bound, abs=1e-9)
    assert "borne de [1.05, 1.15]" in r.message


def test_the_c_refinement_improves_on_the_grid_and_can_be_switched_off(monkeypatch):
    a, d, e = poisson_table(7)
    refined = fit_makeham(a, d, e, None)
    monkeypatch.setattr(M, "REFINE_C", False)
    grid_only = fit_makeham(a, d, e, None)
    grid = np.linspace(1.05, 1.15, 100)
    assert np.min(np.abs(grid - grid_only.params["c"])) < 1e-12              # the spec's literal behaviour: c on the grid
    assert refined.loglik > grid_only.loglik + 1e-4
    assert "affiné" in refined.message and "affiné" not in grid_only.message


def test_spec_validation_a_positive_c_in_range_and_aic_below_gompertz():
    """Spec section 8, J3 morning: 'Parametre A > 0, c in [1.05, 1.15]. AIC Makeham < AIC Gompertz.'"""
    for a, d, e in [synthetic_zone2("Female"), synthetic_zone2("Male"), poisson_table(8)]:
        r, g = fit_makeham(a, d, e, None), fit_gompertz(a, d, e, None)
        assert r.params["A"] > 0 and 1.05 <= r.params["c"] <= 1.15 and r.converged
        assert 2 * 3 - 2 * r.loglik < 2 * 2 - 2 * g.loglik


def test_cfg_nr_maxiter_caps_the_newton_steps_and_the_failure_is_reported():
    a, d, e = poisson_table(9)
    r = fit_makeham(a, d, e, C.SmootherConfig(nr_maxiter=1))
    assert r.converged is False and "non convergé" in r.message
    assert np.all(np.isfinite(r.mu_hat)) and np.isfinite(r.loglik)


def test_cfg_nr_tol_is_the_remaining_loglik_gap_of_each_subproblem():
    a, d, e = poisson_table(10)
    loose = fit_makeham(a, d, e, C.SmootherConfig(nr_tol=1e12))
    assert loose.converged and loose.n_iter == 0                         # every start already meets the tolerance
    tight = fit_makeham(a, d, e, C.SmootherConfig(nr_tol=1e-12))
    assert tight.loglik >= loose.loglik and tight.n_iter > 0


def test_one_grid_subproblem_that_does_not_converge_makes_the_fit_not_converged(monkeypatch):
    """The final (A, B) solve converges, but the sub-problem of the first grid point did not: the profile maximum
    cannot be trusted, and the message says how many sub-problems failed."""
    real, calls = M._solve_ab, [0]

    def flaky(w, d, e, maxiter, tol):
        calls[0] += 1
        sol = real(w, d, e, maxiter, tol)
        return sol._replace(converged=False, kind="maxiter") if calls[0] == 1 else sol

    monkeypatch.setattr(M, "_solve_ab", flaky)
    r = fit_makeham(*poisson_table(19), None)
    assert r.converged is False and "1 sous-problème(s) (A, B) non convergé(s)" in r.message
    assert np.all(np.isfinite(r.mu_hat)) and np.isfinite(r.loglik)


def test_n_iter_counts_the_newton_steps_of_all_the_subproblems():
    a, d, e = poisson_table(11)
    r = fit_makeham(a, d, e, None)
    assert r.n_iter >= 100                                               # at least a couple of steps per grid point
    assert f"{r.n_iter} pas au total" in r.message


# --------------------------------------------------------------------------
# The FitResult protocol and the rules shared with the other zone modules
# --------------------------------------------------------------------------


def test_the_fitresult_protocol():
    a, d, e = poisson_table(12)
    before = (a.copy(), d.copy(), e.copy())
    r = fit_makeham(a, d, e, C.SmootherConfig())
    assert isinstance(r, C.FitResult) and (r.method, r.zone) == ("makeham", "zone2")
    np.testing.assert_array_equal(r.ages, a)
    assert r.mu_hat.shape == a.shape and np.all(r.mu_hat > 0)
    assert set(r.params) == {"A", "B", "c", "n_params"} and r.params["n_params"] == 3
    np.testing.assert_allclose(r.mu_hat, r.params["A"] + r.params["B"] * r.params["c"] ** a, rtol=1e-10)
    assert r.loglik == pytest.approx(poisson_loglik(d, e, r.mu_hat), rel=1e-14)
    assert r.converged is True and isinstance(r.n_iter, int) and r.message
    for got, was in zip((a, d, e), before):
        np.testing.assert_array_equal(got, was)


def test_the_config_defaults_of_the_spec_are_used_when_it_is_omitted():
    a, d, e = poisson_table(13)
    np.testing.assert_array_equal(fit_makeham(a, d, e, None).mu_hat, fit_makeham(a, d, e, C.SmootherConfig()).mu_hat)


def test_ages_keep_the_callers_order():
    a, d, e = poisson_table(14)
    order = np.random.default_rng(0).permutation(a.size)
    r1, r2 = fit_makeham(a, d, e, None), fit_makeham(a[order], d[order], e[order], None)
    np.testing.assert_allclose(r2.mu_hat, r1.mu_hat[order], rtol=1e-8)


@pytest.mark.parametrize("bad", ["exposure", "deaths"])
def test_an_age_without_exposure_or_with_missing_deaths_is_ignored_but_still_smoothed(bad):
    a, d, e = poisson_table(15)
    d2, e2 = d.copy(), e.copy()
    if bad == "exposure":
        e2[10] = 0.0
    else:
        d2[10] = np.nan
    keep = np.arange(a.size) != 10
    r, clean = fit_makeham(a, d2, e2, None), fit_makeham(a[keep], d[keep], e[keep], None)
    for k in ("A", "B", "c"):
        assert r.params[k] == pytest.approx(clean.params[k], rel=1e-6)
    assert np.all(np.isfinite(r.mu_hat)) and r.mu_hat[10] > 0 and "1 âge ignoré" in r.message


def test_ages_without_deaths_stay_in_the_likelihood():
    a, d, e = poisson_table(16)
    d[-8:] = 0.0
    full, without = fit_makeham(a, d, e, None), fit_makeham(a[:-8], d[:-8], e[:-8], None)
    assert abs(full.params["c"] - without.params["c"]) > 1e-4 and np.all(np.isfinite(full.mu_hat))


@pytest.mark.parametrize("args,msg", [
    (lambda a, d, e: (a, d[:-1], e), "même longueur"),
    (lambda a, d, e: (np.r_[a[:-1], a[0]], d, e), "dupliqués"),
    (lambda a, d, e: (a, np.r_[-1.0, d[1:]], e), "négatif"),
    (lambda a, d, e: (np.r_[np.nan, a[1:]], d, e), "non fini"),
])
def test_malformed_input_is_refused(args, msg):
    with pytest.raises(ValueError, match=msg):
        fit_makeham(*args(*poisson_table(17)), None)


def test_three_ages_and_two_ages_with_deaths_are_needed():
    a, d, e = poisson_table(18)
    with pytest.raises(ValueError, match="au moins 3"):
        fit_makeham(a[:2], d[:2], e[:2], None)
    one = np.zeros_like(d)
    one[30] = 50.0
    with pytest.raises(ValueError, match="au moins 3"):
        fit_makeham(a, one, e, None)
    assert fit_makeham(a[:3], d[:3] + 5, e[:3], None).converged is not None      # three ages: allowed (saturated)


# --------------------------------------------------------------------------
# Real data: France 2020 (HMD), the standard file shipped in sample_data/
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def france():
    df = L.load_data(ROOT / "sample_data" / "FRA_taux_bruts_moment_2020_female_male.csv")
    cfg = C.default_config(df["age"].unique())
    return {sex: C.zone_slice(df[df["sex"] == sex], cfg, "zone2") for sex in BOTH}, cfg


PINNED = {                       # measured on the sample file, ages 20..85
    "Female": (1.573552441860897e-04, 9.535060440471356e-06, 1.1052390989051302),
    "Male": (2.2909363183779227e-04, 3.5895235120522215e-05, 1.0949455209919055),
}


@pytest.mark.parametrize("sex", BOTH)
def test_france_2020_zone_two_estimates_and_the_spec_validation(france, sex):
    tables, cfg = france
    a, d, e = (tables[sex][c].to_numpy() for c in ("age", "Dx", "Ex"))
    r, g = fit_makeham(a, d, e, cfg), fit_gompertz(a, d, e, cfg)
    assert (r.params["A"], r.params["B"], r.params["c"]) == pytest.approx(PINNED[sex], rel=1e-6)
    assert r.params["A"] > 0 and 1.05 <= r.params["c"] <= 1.15 and r.converged            # spec J3
    assert 2 * 3 - 2 * r.loglik < 2 * 2 - 2 * g.loglik - 300                              # AIC gap measured: ~380-420


@pytest.fixture(scope="module")
def smoothing_lab():
    sys.path.insert(0, str(POST3))
    pytest.importorskip("smoothing_lab.zone2_makeham")
    from smoothing_lab import zone2_makeham as mod
    return mod


@pytest.mark.parametrize("sex", BOTH)
def test_france_2020_agrees_with_the_smoothing_lab_implementation(smoothing_lab, france, sex):
    """The lab solves the (A, B) sub-problem with L-BFGS-B (tolerance ~1e-7), this module with Newton."""
    tables, cfg = france
    a, d, e = (tables[sex][c].to_numpy() for c in ("age", "Dx", "Ex"))
    r, lab = fit_makeham(a, d, e, cfg), smoothing_lab.fit_makeham_profile(a, d, e)
    np.testing.assert_allclose([r.params["A"], r.params["B"], r.params["c"]], [lab.A, lab.B, lab.c], rtol=1e-5)
    assert r.loglik == pytest.approx(poisson_loglik(d, e, lab.mu(a)), abs=1e-4)
