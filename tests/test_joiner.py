"""Module 4 - joiner.py. Run:   python -m pytest tests/test_joiner.py -v

Evidence that does not replay the code's own formulas:
  * zones that are CONSTANTS (so the blend has a closed form, written here from the spec's tanh), with the weights summing to 1;
  * each of the nine methods' curves, evaluated from `params`, against the method's own `mu_hat`;
  * "each zone as is outside the window": array equality with the stored `mu_hat`;
  * a life table rebuilt with plain loops, a constant-force closed form, and the user's lab table_builder;
  * real France 2020 fits: the assembled curve reproduces the life expectancy of the crude rates.
"""
from __future__ import annotations

import math
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import config as C
from src import joiner as J
from src import loader as L
from src.joiner import blend_weight, build_life_table, evaluate_curve, join_zones, mu_to_q
from src.spline_all import fit_spline_all
from src.zone1_poly import fit_poly3, fit_spline
from src.zone2_gompertz import fit_gompertz
from src.zone2_makeham import fit_makeham
from src.zone2_makeham2 import fit_makeham2
from src.zone2_perks import fit_perks
from src.zone3 import fit_beard, fit_coale_kisker, fit_kannisto

BOTH = ("Female", "Male")
ROOT = Path(__file__).resolve().parent.parent
POST3 = Path(os.environ.get(
    "MORTALITY_POST3_DIR",
    r"C:\Users\simar\OneDrive\Bureau\Persos\LinkedIn\Life_Risks_Series\Post3-!Smoothing1"))

CFG = C.SmootherConfig(omega1=20, omega2=85, omega3=110, sigma_join=3.0)


def fake(zone, mu, ages, method="gompertz", params=None):
    """A FitResult whose curve is the constant `mu` (Gompertz with beta = 0), stored on `ages`."""
    ages = np.asarray(ages, dtype=float)
    p = params if params is not None else {"alpha": math.log(mu), "beta": 0.0, "B": mu, "c": 1.0, "n_params": 2}
    return C.FitResult(method=method, zone=zone, ages=ages, mu_hat=np.full(ages.size, float(mu)), params=p,
                       loglik=0.0, converged=True, n_iter=0, message="fake")


def constants(m1, m2, m3, cfg=CFG):
    return (fake("zone1", m1, np.arange(0, cfg.omega1 + 1)), fake("zone2", m2, np.arange(cfg.omega1, cfg.omega2 + 1)),
            fake("zone3", m3, np.arange(cfg.omega2, cfg.omega3 + 1)))


def spec_weight(x, omega, sigma):
    """The spec's weight written here from scratch: (1 + tanh((omega - x) / sigma)) / 2."""
    return (1.0 + np.tanh((omega - np.asarray(x, float)) / sigma)) / 2.0


@pytest.fixture(params=[True, False], ids=["continuous-edges", "literal-edges"])
def edges(request, monkeypatch):
    monkeypatch.setattr(J, "CONTINUOUS_EDGES", request.param)
    return request.param


# --------------------------------------------------------------------------
# Step 3: the weights
# --------------------------------------------------------------------------


def test_weight_is_one_below_the_window_zero_above_and_half_at_omega(edges):
    x = np.array([0.0, 14.0, 20.0, 26.0, 40.0])
    w = blend_weight(x, 20.0, 3.0)
    np.testing.assert_allclose(w[[0, 2, 4]], [1.0, 0.5, 0.0], atol=1e-15)
    assert w[1] == pytest.approx(1.0 if edges else 0.5 * (1 + math.tanh(2)), rel=1e-12)       # x = omega - 2 sigma
    assert w[3] == pytest.approx(0.0 if edges else 0.5 * (1 - math.tanh(2)), abs=1e-12)       # x = omega + 2 sigma


def test_weight_is_decreasing_and_antisymmetric_about_omega(edges):
    x = np.linspace(0, 40, 401)
    w = blend_weight(x, 20.0, 3.0)
    assert np.all(np.diff(w) <= 1e-15) and np.all((w >= 0) & (w <= 1))
    np.testing.assert_allclose(w + blend_weight(40.0 - x, 20.0, 3.0), 1.0, atol=1e-12)        # w(omega + d) = 1 - w(omega - d)


def test_the_documented_defaults():
    """Window of 2 sigma (spec 4.2), continuous edges, radix 100 000 - and with no override the weight is exactly 1 / 0
    at the window edges."""
    assert (J.WINDOW, J.CONTINUOUS_EDGES, J.RADIX) == (2.0, True, 100_000.0)
    assert blend_weight(14.0, 20.0, 3.0) == 1.0 and blend_weight(26.0, 20.0, 3.0) == 0.0
    assert build_life_table(np.full(5, 0.1))["lx"].iloc[0] == 100_000.0


def test_literal_weight_is_the_spec_formula_inside_the_window_and_steps_at_its_edges(monkeypatch):
    monkeypatch.setattr(J, "CONTINUOUS_EDGES", False)
    inside = np.linspace(14.0, 26.0, 49)
    np.testing.assert_allclose(blend_weight(inside, 20.0, 3.0), spec_weight(inside, 20.0, 3.0), rtol=1e-13)
    just_out = blend_weight(np.array([13.999, 26.001]), 20.0, 3.0)
    assert just_out[0] == 1.0 and just_out[1] == 0.0                         # outside: the zone alone
    assert 1.0 - spec_weight(14.0, 20.0, 3.0) == pytest.approx((1 - math.tanh(2)) / 2)       # a 1.8% step in the weight


def test_continuous_weight_has_no_step_at_the_window_edges(monkeypatch):
    monkeypatch.setattr(J, "CONTINUOUS_EDGES", True)
    eps = 1e-9
    for edge, side in ((14.0, 1.0), (26.0, 0.0)):
        before, after = blend_weight(np.array([edge - eps, edge + eps]), 20.0, 3.0)
        assert abs(before - after) < 1e-7 and abs(before - side) < 1e-7
    inside = np.linspace(14.0, 26.0, 49)                                     # same tanh shape, only rescaled
    assert np.max(np.abs(blend_weight(inside, 20.0, 3.0) - spec_weight(inside, 20.0, 3.0))) < 0.02


@pytest.mark.parametrize("sigma", [0.0, -1.0, np.nan])
def test_weight_needs_a_positive_sigma(sigma):
    with pytest.raises(ValueError, match="σ"):
        blend_weight(np.arange(5.0), 2.0, sigma)


# --------------------------------------------------------------------------
# Steps 1-6: the join
# --------------------------------------------------------------------------


def test_join_of_constants_is_the_spec_blend_written_out_by_hand(edges):
    """mu = w1 m1 + (1 - w1)(w2 m2 + (1 - w2) m3) with the weights of the spec's tanh, window cut at 2 sigma."""
    m1, m2, m3 = 0.001, 0.02, 0.4
    mu = join_zones(*constants(m1, m2, m3), CFG)
    x = np.arange(0, 111, dtype=float)

    def weight(omega):
        w = spec_weight(x, omega, 3.0)
        if edges:
            w = 0.5 * (1.0 + np.tanh((omega - x) / 3.0) / math.tanh(2))
        return np.where(omega - x > 6, 1.0, np.where(omega - x < -6, 0.0, w))

    w1, w2 = weight(20.0), weight(85.0)
    expected = w1 * m1 + (1 - w1) * (w2 * m2 + (1 - w2) * m3)
    assert mu.shape == (111,)
    np.testing.assert_allclose(mu, expected, rtol=1e-13)


def test_the_weights_of_the_three_zones_sum_to_one_even_when_the_windows_overlap(edges):
    """Three equal constants must come out equal, for any sigma: here omega2 - omega1 = 10 < 4 sigma = 40."""
    cfg = C.SmootherConfig(omega1=20, omega2=30, omega3=60, sigma_join=10.0)
    mu = join_zones(*constants(0.01, 0.01, 0.01, cfg), cfg)
    np.testing.assert_allclose(mu, 0.01, rtol=1e-13)
    mu_tight = join_zones(*constants(0.01, 0.01, 0.01), C.SmootherConfig(sigma_join=1.0))
    np.testing.assert_allclose(mu_tight, 0.01, rtol=1e-13)


def test_outside_the_windows_each_zone_is_used_as_is_bit_for_bit(edges):
    """Real fitted zones, not constants: x <= omega1 - 2 sigma is zone I's mu_hat exactly, and so on."""
    z1 = fit_poly3(np.arange(0, 21), np.full(21, 50.0), np.full(21, 1e5), CFG)
    z2 = fit_gompertz(np.arange(20, 86), 10.0 * 1.09 ** np.arange(0, 66), np.full(66, 1e5), CFG)
    z3 = fit_kannisto(np.arange(85, 110), 4000.0 * 1.12 ** np.arange(0, 25) / 10, np.full(25, 1e4), CFG)
    mu = join_zones(z1, z2, z3, CFG)
    x = np.arange(0, 111)
    # the window [omega - 6, omega + 6] is closed: with the literal weight its edges still blend (0.982 / 0.018),
    # with the continuous weight the edge weight is exactly 1 / 0, so one more age is "as is"
    k = 0 if edges else 1
    np.testing.assert_array_equal(mu[x <= 13 + (1 - k)], z1.mu_hat[:14 + (1 - k)])             # zone I, below its window
    mid = (x >= 26 + k) & (x <= 79 - k)
    np.testing.assert_array_equal(mu[mid], z2.mu_hat[6 + k:60 - k])                             # zone II between the windows
    np.testing.assert_array_equal(mu[x >= 91 + k], z3.mu_hat[6 + k:])                           # zone III, extrapolation included


def test_at_a_boundary_the_two_zones_weigh_the_same():
    mu = join_zones(*constants(0.001, 0.02, 0.4), CFG)
    assert mu[20] == pytest.approx(0.5 * 0.001 + 0.5 * 0.02, rel=1e-12)      # omega2 = 85 is far above 20 + 6
    assert mu[85] == pytest.approx(0.5 * 0.02 + 0.5 * 0.4, rel=1e-12)


def test_with_a_wide_window_the_literal_formula_leaves_a_kink_at_each_edge_and_the_continuous_one_far_less(monkeypatch):
    """sigma = 10, omega1 = 30: the window is [10, 50]. Constants 0.001 -> 0.02. The literal weight is 0.982 / 0.018 at the
    edges, so the second difference of the joined curve there is 1.8% of the gap (the largest irregularity of the whole
    table: the middle is gentle at this sigma); the continuous weight cuts it to 0.4%. Measured, see the module docstring."""
    cfg = C.SmootherConfig(omega1=30, omega2=85, omega3=110, sigma_join=10.0)
    gap = 0.02 - 0.001

    def edge_kinks(continuous):
        monkeypatch.setattr(J, "CONTINUOUS_EDGES", continuous)
        mu = join_zones(*constants(0.001, 0.02, 0.4, cfg), cfg)
        second = np.abs(np.diff(mu, 2))                                        # second[i] is centred on age i + 1
        return second[8:12].max() / gap, second[48:52].max() / gap, second[:60].max() / gap

    literal_lo, literal_hi, literal_max = edge_kinks(False)
    smooth_lo, smooth_hi, smooth_max = edge_kinks(True)
    assert literal_lo > 0.015 and literal_hi > 0.015 and literal_max == pytest.approx(max(literal_lo, literal_hi), rel=1e-9)
    assert smooth_lo < 0.006 and smooth_hi < 0.006


def test_a_zone_is_only_evaluated_where_its_weight_is_positive():
    """A cubic that overflows far from its zone (inf at 110) must not poison the result with 0 * inf = NaN."""
    exploding = fake("zone1", 0.001, np.arange(0, 21), method="poly3",
                     params={"a0": -7.0, "a1": 0.0, "a2": 0.0, "a3": 1e-3, "n_params": 4})
    assert np.isinf(evaluate_curve(exploding, [110.0])[0])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        mu = join_zones(exploding, *constants(0.001, 0.02, 0.4)[1:], CFG)
    assert np.all(np.isfinite(mu))


def test_zone_values_reads_the_stored_curve_and_evaluates_the_formula_only_where_nothing_is_stored():
    stored = fake("zone2", 0.01, np.arange(20, 31))
    stored.mu_hat[:] = 0.5                                                  # deliberately NOT what the formula gives (0.01)
    np.testing.assert_array_equal(J.zone_values(stored, np.array([20.0, 25.0, 30.0])), [0.5, 0.5, 0.5])
    mixed = J.zone_values(stored, np.array([25.0, 40.0, 41.0]))              # 25 is stored, 40 and 41 need the formula
    assert mixed[0] == 0.5
    np.testing.assert_allclose(mixed[1:], 0.01, rtol=1e-12)
    unusable = C.FitResult("gompertz", "zone2", np.arange(20.0, 31.0), np.full(11, 0.02), {"k": 1.0}, 0.0, True, 0, "")
    np.testing.assert_array_equal(J.zone_values(unusable, np.arange(20.0, 31.0)), np.full(11, 0.02))        # nothing to evaluate


def test_ages_below_the_first_age_of_zone_one_are_nan_not_invented():
    cfg = C.SmootherConfig(omega1=30, omega2=85, omega3=110)
    z1 = fake("zone1", 0.001, np.arange(20, 31))
    z2, z3 = fake("zone2", 0.02, np.arange(30, 86)), fake("zone3", 0.4, np.arange(85, 111))
    mu = join_zones(z1, z2, z3, cfg)
    assert mu.shape == (111,) and np.all(np.isnan(mu[:20])) and np.all(np.isfinite(mu[20:]))
    assert mu[20] == 0.001


def test_sigma_and_the_bounds_come_from_the_config():
    narrow = join_zones(*constants(0.001, 0.02, 0.4), C.SmootherConfig(sigma_join=1.0))
    wide = join_zones(*constants(0.001, 0.02, 0.4), C.SmootherConfig(sigma_join=8.0))
    assert narrow[16] == 0.001 and wide[16] != 0.001                         # age 16: outside the 2-sigma window of sigma = 1
    cfg = C.SmootherConfig(omega1=10, omega2=60, omega3=70)
    mu = join_zones(*constants(0.001, 0.02, 0.4, cfg), cfg)
    assert mu.shape == (71,) and mu[10] == pytest.approx(0.5 * 0.001 + 0.5 * 0.02)


def test_the_inputs_are_not_modified():
    zones = constants(0.001, 0.02, 0.4)
    before = [(z.mu_hat.copy(), z.ages.copy(), dict(z.params)) for z in zones]
    join_zones(*zones, CFG)
    for z, (m, a, p) in zip(zones, before):
        np.testing.assert_array_equal(z.mu_hat, m)
        np.testing.assert_array_equal(z.ages, a)
        assert z.params == p


@pytest.mark.parametrize("order", [(1, 0, 2), (2, 1, 0), (0, 0, 2)])
def test_each_result_must_be_of_the_zone_it_is_passed_as(order):
    zones = constants(0.001, 0.02, 0.4)
    with pytest.raises(ValueError, match="Raccord"):
        join_zones(*(zones[i] for i in order), CFG)


def test_bounds_and_sigma_are_checked():
    zones = constants(0.001, 0.02, 0.4)
    with pytest.raises(ValueError, match="ω₁ < ω₂ < ω₃"):
        join_zones(*zones, C.SmootherConfig(omega1=90, omega2=85, omega3=110))
    with pytest.raises(ValueError, match="σ"):
        join_zones(*zones, C.SmootherConfig(sigma_join=0.0))
    with pytest.raises(ValueError, match="aucun âge"):
        join_zones(C.FitResult("poly3", "zone1", np.array([]), np.array([]), {}, 0.0, True, 0, ""), *zones[1:], CFG)


# --------------------------------------------------------------------------
# Step 4: evaluating a zone's curve from its parameters
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def france():
    df = L.load_data(ROOT / "sample_data" / "FRA_taux_bruts_moment_2020_female_male.csv")
    cfg = C.default_config(df["age"].unique())
    return df, cfg


def zone_rows(france, sex, zone):
    df, cfg = france
    sub = C.zone_slice(df[df["sex"] == sex], cfg, zone)
    return tuple(sub[c].to_numpy() for c in ("age", "Dx", "Ex"))


@pytest.fixture(scope="module")
def all_nine(france):
    """All nine methods fitted to France 2020 women, each on its own zone."""
    _, cfg = france
    z1, z2, z3 = (zone_rows(france, "Female", z) for z in ("zone1", "zone2", "zone3"))
    return {
        "poly3": fit_poly3(*z1, cfg), "spline": fit_spline(*z1, cfg),
        "gompertz": fit_gompertz(*z2, cfg), "makeham": fit_makeham(*z2, cfg), "makeham2": fit_makeham2(*z2, cfg),
        "perks": fit_perks(*z2, cfg),
        "kannisto": fit_kannisto(*z3, cfg), "beard": fit_beard(*z3, cfg), "coale_kisker": fit_coale_kisker(*z3, cfg),
    }


@pytest.mark.parametrize("key", list(C.METHODS))
def test_every_method_evaluated_from_its_params_reproduces_its_own_mu_hat(all_nine, key):
    """The guard against drift between a zone module's formula and the joiner's copy of it."""
    r = all_nine[key]
    np.testing.assert_allclose(evaluate_curve(r, r.ages), r.mu_hat, rtol=1e-8)


@pytest.mark.parametrize("key", list(C.METHODS))
def test_every_method_can_be_read_beyond_its_own_ages_without_a_warning(all_nine, key):
    r = all_nine[key]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        mu = evaluate_curve(r, np.arange(0, 111))
    assert mu.shape == (111,) and not np.any(np.isnan(mu)) and np.all(mu >= 0)             # a cubic may underflow to 0, never NaN
    assert np.all(np.isfinite(mu[10:100]))


def test_perks_with_a_positive_d_is_evaluated_as_its_formula_says():
    """On France 2020 Perks collapses to D = 0, which hides the denominator: a fixed D > 0 checks it, by hand."""
    p = {"A": 1.0e-4, "B": 1.0e-5, "c": 1.1, "D": 2.0e-4, "n_params": 4}
    r = fake("zone2", 0.01, np.arange(20, 86), method="perks", params=p)
    x = np.array([20.0, 50.0, 80.0, 110.0])
    expected = (p["A"] + p["B"] * p["c"] ** x) / (1.0 + p["D"] * p["c"] ** x)
    np.testing.assert_allclose(evaluate_curve(r, x), expected, rtol=1e-13)
    assert evaluate_curve(r, [300.0])[0] == pytest.approx(p["B"] / p["D"], rel=1e-6)        # the plateau B / D


def test_coale_kisker_is_capped_below_q_equal_one_far_beyond_omega3(all_nine):
    """q = q_omega2 e^(k (x - omega2)) passes 1 well beyond the target age; the rate stays finite (q capped at 1 - 1e-15)."""
    r = all_nine["coale_kisker"]
    far = evaluate_curve(r, np.array([110.0, 150.0, 400.0]))
    assert np.all(np.isfinite(far)) and far[1] == pytest.approx(-np.log1p(-(1.0 - 1e-15)), rel=1e-12) == far[2]
    assert far[0] < far[1]


def test_the_spline_continues_as_a_straight_line_in_ln_mu_beyond_its_end_knots(all_nine):
    r = all_nine["spline"]
    beyond = np.log(evaluate_curve(r, np.arange(20, 31)))
    np.testing.assert_allclose(np.diff(beyond, 2), 0.0, atol=1e-10)


def test_coale_kisker_is_evaluated_back_to_q_exactly(all_nine):
    r = all_nine["coale_kisker"]
    x = np.arange(85, 111)
    q = r.params["q_omega2"] * np.exp(r.params["k"] * (x - 85))
    np.testing.assert_allclose(mu_to_q(evaluate_curve(r, x)), q, rtol=1e-12)


def test_an_unknown_method_is_refused_in_french():
    bad = C.FitResult("lee_carter", "zone2", np.arange(3.0), np.ones(3), {}, 0.0, True, 0, "")
    with pytest.raises(ValueError, match="méthode inconnue"):
        evaluate_curve(bad, [1.0])


# --------------------------------------------------------------------------
# Spec 4.4: q and the life table
# --------------------------------------------------------------------------


def test_mu_to_q_is_one_minus_exp_minus_mu_and_keeps_precision_for_tiny_rates():
    mu = np.array([0.0, 1e-18, 0.01, 0.5, 3.0, 50.0])
    np.testing.assert_allclose(mu_to_q(mu), 1.0 - np.exp(-mu), rtol=1e-12, atol=1e-18)
    assert mu_to_q(1e-18) == pytest.approx(1e-18, rel=1e-9, abs=0)           # 1 - exp(-1e-18) would give 0.0
    assert mu_to_q(np.array([0.3, 0.4])).shape == (2,)


def naive_life_table(mu, radix=100_000.0, open_tail=True):
    """The table with plain Python loops, no vectorised shortcut."""
    n = len(mu)
    l = [radix]
    for i in range(n):
        l.append(l[-1] * math.exp(-mu[i]))
    ex = []
    for i in range(n):
        t = 0.0
        for j in range(i, n):
            d = l[j] - l[j + 1]
            t += l[j + 1] + 0.5 * d
        if open_tail and mu[-1] > 0:
            t += l[n] / mu[-1]
        ex.append(t / l[i])
    return l, ex


def test_the_life_table_matches_a_loop_by_loop_construction_and_has_the_spec_columns():
    mu = 0.0005 * 1.1 ** np.arange(0, 111)
    lt = build_life_table(mu)
    assert list(lt.columns) == ["age", "mu", "qx", "px", "lx", "dx", "ex"]
    assert lt.shape == (111, 7) and lt["age"].dtype == np.int64 and (lt["age"] == np.arange(111)).all()
    l, ex = naive_life_table(mu)
    np.testing.assert_allclose(lt["lx"], l[:-1], rtol=1e-12)
    np.testing.assert_allclose(lt["ex"], ex, rtol=1e-12)
    np.testing.assert_allclose(lt["qx"], 1 - np.exp(-mu), rtol=1e-12)
    np.testing.assert_allclose(lt["px"] + lt["qx"], 1.0, rtol=1e-14)
    np.testing.assert_allclose(lt["dx"], lt["lx"] * lt["qx"], rtol=1e-14)
    assert lt["lx"].iloc[0] == 100_000.0 and np.all(np.diff(lt["lx"]) < 0)
    assert lt["dx"].sum() == pytest.approx(100_000.0 - l[-1], rel=1e-12)    # the survivors beyond the last age are not in dx


def test_the_open_tail_adds_the_years_the_survivors_beyond_the_last_age_still_have_to_live():
    mu = np.full(5, 0.4)
    closed, opened = build_life_table(mu, open_tail=False), build_life_table(mu)
    p = math.exp(-0.4)
    assert closed["ex"].iloc[-1] == pytest.approx((1 + p) / 2, rel=1e-12)                  # one year, deaths mid-year
    assert opened["ex"].iloc[-1] == pytest.approx((1 + p) / 2 + p / 0.4, rel=1e-12)        # + 1 / mu each, for the fraction p
    np.testing.assert_allclose(closed["lx"], opened["lx"], rtol=0)
    l, ex = naive_life_table(mu, open_tail=False)
    np.testing.assert_allclose(closed["ex"], ex, rtol=1e-12)


def test_a_constant_force_gives_the_known_expectation_with_the_open_tail():
    """mu constant, table long enough: e_x -> sum_k p^k ... computed in closed form, UDD within each year."""
    mu, n = 0.1, 400
    lt = build_life_table(np.full(n, mu))
    p = math.exp(-mu)
    closed = (1 + p) / 2 + p / mu                                            # the open last interval
    for _ in range(3):
        closed = (1 + p) / 2 + p * closed                                    # e_x = (1 + p)/2 + p e_{x+1} for every earlier x
    # e_x is the same at every age for a constant force: the fixed point of e = (1 + p)/2 + p e
    assert lt["ex"].iloc[0] == pytest.approx(((1 + p) / 2) / (1 - p), rel=1e-9)
    assert lt["ex"].iloc[-1] == pytest.approx((1 + p) / 2 + p / mu, rel=1e-12)


def test_radix_scales_the_table_and_nothing_else():
    mu = 0.0005 * 1.1 ** np.arange(0, 100)
    a, b = build_life_table(mu), build_life_table(mu, radix=1.0)
    np.testing.assert_allclose(a["lx"], b["lx"] * 1e5, rtol=1e-12)
    for col in ("qx", "px", "ex"):
        np.testing.assert_allclose(a[col], b[col], rtol=1e-12)
    with pytest.raises(ValueError, match="radix"):
        build_life_table(mu, radix=0.0)


def test_leading_nan_are_dropped_so_the_table_starts_at_the_first_age_with_a_rate():
    mu = np.r_[np.full(20, np.nan), 0.001 * 1.1 ** np.arange(0, 91)]
    lt = build_life_table(mu)
    assert lt["age"].iloc[0] == 20 and lt["age"].iloc[-1] == 110 and len(lt) == 91
    assert lt["lx"].iloc[0] == 100_000.0
    np.testing.assert_allclose(lt["ex"], build_life_table(mu[20:])["ex"], rtol=0)


@pytest.mark.parametrize("bad,match", [
    (lambda m: np.full(5, np.nan), "aucun taux fini"),
    (lambda m: np.r_[m[:10], np.nan, m[11:]], "trou"),
    (lambda m: np.r_[m[:10], np.inf, m[11:]], "trou"),
    (lambda m: np.r_[m[:10], -0.01, m[11:]], "négatif"),
    (lambda m: np.array([]), "aucun taux fini"),
])
def test_a_curve_with_a_gap_a_negative_rate_or_nothing_is_refused(bad, match):
    with pytest.raises(ValueError, match=match):
        build_life_table(bad(0.001 * 1.1 ** np.arange(0, 40)))


# --------------------------------------------------------------------------
# Real data: France 2020 (HMD)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("sex", BOTH)
def test_france_2020_the_assembled_table_reproduces_the_life_expectancy_of_the_crude_rates(france, sex):
    """Independent: e0 from the crude rates (D / E, ages 0..109, same table construction) against e0 from the three
    smoothed zones joined. Measured: 85.15 vs 85.19 (women), 79.12 vs 79.20 (men)."""
    df, cfg = france
    z1, z2, z3 = (zone_rows(france, sex, z) for z in ("zone1", "zone2", "zone3"))
    r1, r2, r3 = fit_spline(*z1, cfg), fit_makeham(*z2, cfg), fit_kannisto(*z3, cfg)
    mu = join_zones(r1, r2, r3, cfg)
    assert mu.shape == (111,) and np.all(np.isfinite(mu)) and np.all(mu > 0)
    lt = build_life_table(mu)
    t = df[df["sex"] == sex].sort_values("age")
    crude = build_life_table((t["Dx"] / t["Ex"]).to_numpy()[:110])
    assert abs(lt["ex"].iloc[0] - crude["ex"].iloc[0]) < 0.15
    assert abs(lt["ex"].iloc[65] - crude["ex"].iloc[65]) < 0.3             # Makeham undershoots 70-85: 0.19 y measured (women)
    assert lt["lx"].iloc[-1] * lt["px"].iloc[-1] < 20                         # almost nobody is left beyond 110
    assert lt["dx"].sum() == pytest.approx(100_000.0 - lt["lx"].iloc[-1] * lt["px"].iloc[-1], rel=1e-12)


def test_france_2020_a_curve_without_visible_steps_at_omega1_and_omega2(france, monkeypatch):
    """Spec section 8, J4 morning: 'Courbe assemblee sans discontinuite visible a omega1 et omega2.' Measured on the
    log-rate: no adjacent-age change in the windows is larger than the steepest change of the zones themselves."""
    _, cfg = france
    rows = [zone_rows(france, "Female", z) for z in ("zone1", "zone2", "zone3")]
    r1, r2, r3 = fit_poly3(*rows[0], cfg), fit_makeham(*rows[1], cfg), fit_kannisto(*rows[2], cfg)
    monkeypatch.setattr(J, "CONTINUOUS_EDGES", True)
    mu = join_zones(r1, r2, r3, cfg)
    jumps = np.abs(np.diff(np.log(mu)))
    steepest_zone = max(np.abs(np.diff(np.log(r.mu_hat))).max() for r in (r1, r2, r3))
    for omega in (cfg.omega1, cfg.omega2):
        window = jumps[omega - 8: omega + 8]
        assert window.max() <= steepest_zone + 1e-12


@pytest.fixture(scope="module")
def smoothing_lab():
    sys.path.insert(0, str(POST3))
    pytest.importorskip("smoothing_lab.table_builder")
    from smoothing_lab import joiner as lab_joiner, table_builder
    return table_builder, lab_joiner


def test_the_life_table_agrees_with_the_smoothing_lab_table_builder(smoothing_lab):
    """The lab's finite construction (L_last = l_next + a d, nothing after) is `open_tail=False`."""
    table_builder, _ = smoothing_lab
    mu = np.minimum(0.0004 * 1.105 ** np.arange(0, 111), 3.0)             # the lab's 1 - q loses precision once p < 1e-3
    mine = build_life_table(mu, open_tail=False)
    lab = table_builder.from_mu(np.arange(111), mu).frame
    for mine_col, lab_col in (("lx", "lx"), ("dx", "dx"), ("qx", "qx"), ("ex", "ex")):
        np.testing.assert_allclose(mine[mine_col], lab[lab_col], rtol=1e-10)


def test_the_literal_weight_is_the_labs_tanh_weight(smoothing_lab, monkeypatch):
    _, lab_joiner = smoothing_lab
    monkeypatch.setattr(J, "CONTINUOUS_EDGES", False)
    x = np.linspace(14.0, 26.0, 25)
    np.testing.assert_allclose(blend_weight(x, 20.0, 3.0), lab_joiner._tanh_weight(x, 20.0, 3.0), rtol=1e-13)


# --------------------------------------------------------------------------
# The global spline (zone = 'all') in the join
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def global_spline(france):
    _, cfg = france
    return fit_spline_all(*zone_rows(france, "Female", "all"), cfg)


def test_the_global_spline_evaluated_from_its_params_reproduces_its_own_mu_hat(global_spline):
    assert (global_spline.method, global_spline.zone) == ("spline_all", "all")
    np.testing.assert_allclose(evaluate_curve(global_spline, global_spline.ages), global_spline.mu_hat, rtol=1e-8)


def test_the_global_spline_chosen_in_all_three_zones_is_the_global_curve(france, global_spline):
    _, cfg = france
    mu = join_zones(global_spline, global_spline, global_spline, cfg)           # weights sum to 1: w1 S + (1-w1)(w2 S + (1-w2) S)
    n = len(global_spline.mu_hat)
    np.testing.assert_allclose(mu[:n], global_spline.mu_hat, rtol=1e-12)
    np.testing.assert_allclose(mu[n:], evaluate_curve(global_spline, np.arange(n, cfg.omega3 + 1.0)), rtol=1e-12)


def test_the_global_spline_in_one_zone_is_that_zone_as_is_outside_the_windows_and_the_others_keep_theirs(france, all_nine, global_spline):
    _, cfg = france                                                              # omega 20 / 85, sigma 3: windows 14..26 and 79..91
    mu = join_zones(all_nine["poly3"], global_spline, all_nine["kannisto"], cfg)
    np.testing.assert_array_equal(mu[:14], all_nine["poly3"].mu_hat[:14])
    np.testing.assert_array_equal(mu[27:79], global_spline.mu_hat[27:79])
    np.testing.assert_array_equal(mu[92:110], all_nine["kannisto"].mu_hat[92 - 85:110 - 85])


def test_a_global_result_is_accepted_in_any_place_but_a_zone_result_only_in_its_own(france, all_nine, global_spline):
    _, cfg = france
    for place in range(3):
        zones = [all_nine["poly3"], all_nine["makeham"], all_nine["kannisto"]]
        zones[place] = global_spline
        assert np.isfinite(join_zones(*zones, cfg)[:110]).all()
    with pytest.raises(ValueError, match="zone1 est de la zone3"):
        join_zones(all_nine["kannisto"], all_nine["makeham"], all_nine["kannisto"], cfg)
