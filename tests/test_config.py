"""Module 2 - config.py. Run:   python -m pytest tests/test_config.py -v

Several tests pin the spec literally (field names, labels, defaults): if the spec
changes, they should fail loudly rather than drift.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from src import config as C
from src.config import FitResult, SmootherConfig


def table(lo=0, hi=109):
    ages = np.arange(lo, hi + 1)
    return pd.DataFrame({"age": ages, "Dx": 5.0, "Ex": 1000.0})


# --------------------------------------------------------------------------
# The spec, literally
# --------------------------------------------------------------------------


def test_smoother_config_fields_are_those_of_spec_2_3_plus_the_global_methods():
    names = [f.name for f in dataclasses.fields(SmootherConfig)]
    assert names[:10] == [
        "omega1", "omega2", "omega3", "z1_methods", "z2_methods", "z3_methods",
        "sigma_join", "nr_maxiter", "nr_tol", "grid_n"]
    assert names[10:] == ["all_methods"]                    # the one addition to the spec: global methods, off by default
    assert SmootherConfig().all_methods == []


def test_defaults_are_those_of_spec_2_4():
    c = SmootherConfig()
    assert (c.omega1, c.omega2, c.omega3) == (20, 85, 110)
    assert (c.sigma_join, c.nr_maxiter, c.nr_tol, c.grid_n) == (3, 200, 1e-8, 100)


def test_fit_result_fields_are_those_of_the_common_convention():
    assert [f.name for f in dataclasses.fields(FitResult)] == [
        "method", "zone", "ages", "mu_hat", "params", "loglik", "converged", "n_iter", "message"]


def test_method_keys_and_labels_are_those_of_spec_2_2_and_2_3():
    c = SmootherConfig()
    assert c.z1_methods == ["poly3", "spline"]
    assert c.z2_methods == ["gompertz", "makeham", "makeham2", "perks"]
    assert c.z3_methods == ["kannisto", "beard", "coale_kisker"]
    labels = {z: [m.label for m in C.methods_of(z)] for z in C.ZONES}
    assert labels == {
        "zone1": ["Polynôme deg 3", "Splines naturelles"],
        "zone2": ["Gompertz", "Makeham", "Makeham 2e ordre", "Perks"],
        "zone3": ["Kannisto", "Beard", "Coale-Kisker"]}


def test_methods_live_in_the_files_of_the_architecture_table():
    """The spec's table puts Makeham 2nd order in zone2_perks.py; it has its own zone2_makeham2.py at the user's request."""
    mods = {k: m.module for k, m in C.METHODS.items()}
    assert mods == {"poly3": "zone1_poly", "spline": "zone1_poly",
                    "gompertz": "zone2_gompertz", "makeham": "zone2_makeham",
                    "makeham2": "zone2_makeham2", "perks": "zone2_perks",
                    "kannisto": "zone3", "beard": "zone3", "coale_kisker": "zone3"}
    assert C.METHODS["makeham2"].fit_name == "fit_makeham2"


def test_sigma_slider_range_is_1_to_10():
    assert C.SIGMA_JOIN_RANGE == (1.0, 10.0)


# --------------------------------------------------------------------------
# SmootherConfig behaviour
# --------------------------------------------------------------------------


def test_two_configs_never_share_their_method_lists():
    a, b = SmootherConfig(), SmootherConfig()
    a.z1_methods.remove("poly3")
    assert "poly3" in b.z1_methods


def test_selected_methods_come_zone_by_zone_in_registry_order():
    c = SmootherConfig(z1_methods=["spline"], z2_methods=["perks", "gompertz"], z3_methods=["beard"])
    assert [m.key for m in c.selected_methods()] == ["spline", "gompertz", "perks", "beard"]
    assert [m.key for m in SmootherConfig().selected_methods()] == list(C.METHODS)


def test_zone_slice_includes_both_bounds_so_zones_share_their_border_age():
    df, c = table(), SmootherConfig()
    z1, z2, z3 = (C.zone_slice(df, c, z) for z in C.ZONES)
    assert (z1["age"].min(), z1["age"].max()) == (0, 20)
    assert (z2["age"].min(), z2["age"].max()) == (20, 85)
    assert (z3["age"].min(), z3["age"].max()) == (85, 109)        # table stops before omega3 = 110


# --------------------------------------------------------------------------
# default_config: the spec defaults, fitted to the data range
# --------------------------------------------------------------------------


@pytest.mark.parametrize("lo,hi,want", [
    (0, 109, (20, 85, 110)),          # HMD 1x1 without the open group
    (0, 104, (20, 85, 110)),          # smoothing_lab's measured ceiling
    (0, 120, (20, 85, 120)),          # data beyond 110: omega3 follows
    (40, 95, (51, 81, 110)),          # recommended boundaries fall outside: 20% / 75%
    (0, 30, (6, 22, 110)),
])
def test_default_config_fits_the_age_range(lo, hi, want):
    c = C.default_config(np.arange(lo, hi + 1))
    assert (c.omega1, c.omega2, c.omega3) == want
    assert lo < c.omega1 < c.omega2 <= hi or hi - lo < 3


def test_default_config_is_valid_on_the_data_it_was_made_for():
    for lo, hi in ((0, 109), (0, 104), (40, 95), (0, 30)):
        ages = np.arange(lo, hi + 1)
        assert C.validate_config(C.default_config(ages), ages).ok


# --------------------------------------------------------------------------
# validate_config
# --------------------------------------------------------------------------


def test_extrapolating_zone_iii_is_flagged_only_beyond_the_hmd_convention():
    """HMD tables stop at 109 and omega3 = 110 by convention (spec 2.4): no noise there.
    A table stopping at 104 (smoothing_lab's measured ceiling) extrapolates 6 ages."""
    res = C.validate_config(SmootherConfig(), np.arange(0, 110))       # ages 0..109
    assert res.ok and res.warnings == []
    res = C.validate_config(SmootherConfig(), np.arange(0, 105))       # ages 0..104
    assert res.ok and any("extrapolée" in w and "6 ans" in w for w in res.warnings)


def test_boundaries_must_be_ordered():
    for bad in (dict(omega1=85, omega2=20), dict(omega1=20, omega2=20), dict(omega2=110),
                dict(omega1=-1)):
        res = C.validate_config(SmootherConfig(**bad))
        assert not res.ok and "ω₁ < ω₂ < ω₃" in res.errors[0]


def test_boundaries_must_lie_inside_the_data():
    res = C.validate_config(SmootherConfig(omega1=20, omega2=85), np.arange(0, 60))
    assert any("ω₂ = 85" in e for e in res.errors)
    res = C.validate_config(SmootherConfig(omega1=20, omega2=85), np.arange(30, 110))
    assert any("ω₁ = 20" in e for e in res.errors)


def test_nothing_ticked_is_an_error_but_an_empty_zone_is_a_warning():
    res = C.validate_config(SmootherConfig(z1_methods=[], z2_methods=[], z3_methods=[]))
    assert "Cochez au moins une méthode." in res.errors
    res = C.validate_config(SmootherConfig(z2_methods=[]))
    assert res.ok and any("Zone II" in w for w in res.warnings)


def test_unknown_or_misplaced_method_keys_are_errors():
    assert any("inconnue" in e for e in C.validate_config(SmootherConfig(z1_methods=["poly9"])).errors)
    assert any("inconnue" in e for e in C.validate_config(SmootherConfig(z1_methods=["gompertz"])).errors)


@pytest.mark.parametrize("field,value", [("sigma_join", 0), ("sigma_join", -1), ("nr_maxiter", 0),
                                         ("nr_tol", 0), ("nr_tol", -1e-8), ("grid_n", 1)])
def test_non_positive_numeric_settings_are_errors(field, value):
    assert not C.validate_config(SmootherConfig(**{field: value})).ok


def test_overlapping_junction_windows_are_a_warning():
    res = C.validate_config(SmootherConfig(omega1=20, omega2=30, sigma_join=3))      # 10 < 4 * 3
    assert res.ok and any("chevauchent" in w for w in res.warnings)
    assert not any("chevauchent" in w for w in C.validate_config(SmootherConfig()).warnings)


def test_a_zone_without_more_ages_than_parameters_is_flagged():
    ages = np.arange(0, 110)
    res = C.validate_config(SmootherConfig(omega1=3, omega2=85), ages)               # zone I: ages 0..3
    assert any("Zone I" in w and "Polynôme deg 3" in w for w in res.warnings)
    res = C.validate_config(SmootherConfig(omega1=20, omega2=24), ages)              # zone II: 5 ages
    flagged = [w for w in res.warnings if w.startswith("Zone II")]
    assert any("Makeham 2e ordre" in w for w in flagged)                             # 5 ages <= 5 params
    assert not any("Gompertz" in w for w in flagged)


# --------------------------------------------------------------------------
# Global methods (asked for after the specs): the spline over every age
# --------------------------------------------------------------------------


def global_result(knots, zone="all"):
    ages = np.arange(0, 110, dtype=float)
    return FitResult("spline_all", zone, ages, np.full(110, 0.01), {"knots": list(knots), "ln_mu_knots": [0.0] * len(knots),
                                                                       "n_params": len(knots)}, -1.0, True, 0, "")


def test_the_global_spline_is_registered_apart_from_the_nine_zone_methods():
    assert list(C.GLOBAL_METHODS) == ["spline_all"] and "spline_all" not in C.METHODS
    m = C.GLOBAL_METHODS["spline_all"]
    assert (m.zone, m.module, m.fit_name, m.label) == ("all", "spline_all", "fit_spline_all", "Spline globale (tous âges)")
    assert [x.key for x in C.every_method()] == [*C.METHODS, "spline_all"]
    assert C.methods_of("all") == [m] and C.methods_of("zone1") == [x for x in C.METHODS.values() if x.zone == "zone1"]
    assert C.method_spec("spline_all") is m and C.method_spec("gompertz") is C.METHODS["gompertz"]
    with pytest.raises(KeyError):
        C.method_spec("nothing")
    assert C.zones_served(m) == C.ZONES and C.zones_served(C.METHODS["beard"]) == ("zone3",)
    assert C.ZONE_LABELS["all"] == "Tous âges" and C.ZONES == ("zone1", "zone2", "zone3")      # the zone loop is untouched


def test_the_global_spline_is_off_by_default_and_comes_last_when_ticked():
    assert "spline_all" not in [m.key for m in SmootherConfig().selected_methods()]
    c = SmootherConfig(z1_methods=["poly3"], z2_methods=["gompertz"], z3_methods=[], all_methods=["spline_all"])
    assert [m.key for m in c.selected_methods()] == ["poly3", "gompertz", "spline_all"]
    assert c.methods_in("all") == ["spline_all"]


def test_the_global_zone_runs_from_the_first_age_to_omega3():
    c = SmootherConfig(omega1=30, omega2=70, omega3=105)
    assert c.zone_bounds("all") == (None, 105)
    df = table(0, 109)
    sub = C.zone_slice(df, c, "all")
    assert (sub["age"].min(), sub["age"].max()) == (0, 105) and len(sub) == 106


def test_a_global_method_covers_the_three_zones_in_the_validation():
    ages = np.arange(0, 110)
    none_ticked = SmootherConfig(z1_methods=[], z2_methods=[], z3_methods=[])
    assert len(C.validate_config(none_ticked, ages).warnings) >= 3 or not C.validate_config(none_ticked, ages).ok
    covered = SmootherConfig(z1_methods=[], z2_methods=[], z3_methods=[], all_methods=["spline_all"])
    issues = C.validate_config(covered, ages)
    assert issues.ok and not [w for w in issues.warnings if "Aucune méthode cochée" in w]
    only_zone1 = SmootherConfig(z1_methods=["poly3"], z2_methods=[], z3_methods=[])
    assert len([w for w in C.validate_config(only_zone1, ages).warnings if "Aucune méthode cochée" in w]) == 2


def test_validation_names_an_unknown_global_method_and_a_table_too_short_for_its_spline():
    bad = SmootherConfig(all_methods=["poly3"])                       # a zone method in the global list
    assert any("Tous âges" in e and "inconnue" in e for e in C.validate_config(bad, np.arange(0, 110)).errors)
    short = SmootherConfig(omega1=1, omega2=2, omega3=110, z1_methods=[], z2_methods=[], z3_methods=[], all_methods=["spline_all"])
    warns = C.validate_config(short, np.arange(0, 4)).warnings        # 4 ages for a method with 4 parameters
    assert any(w.startswith("Tous âges :") and "non identifiable" in w for w in warns)


KNOTS = [*range(0, 105 + 1, 5), 109]                                  # the rule of the spline on ages 0..109


def test_a_zone_sees_the_global_curve_with_the_knots_that_fall_inside_it():
    cfg, g = SmootherConfig(), global_result(KNOTS)
    views = {z: C.zone_view(g, z, cfg) for z in C.ZONES}
    assert [views[z].params["n_params"] for z in C.ZONES] == [5, 14, 6]      # [0,20] -> 0,5,10,15,20 ; [20,85] -> 20..85 ; [85,110]
    for z, v in views.items():
        assert v.zone == z and v.method == "spline_all"
        assert v.mu_hat is g.mu_hat and v.ages is g.ages and v.params["knots"] == KNOTS       # the same whole curve
    assert g.zone == "all" and g.params["n_params"] == len(KNOTS) == 23     # the original is untouched


def test_a_zone_with_no_knot_is_charged_one_parameter_and_a_zone_result_comes_back_as_is():
    two_knots = global_result([0, 100])
    assert C.zone_view(two_knots, "zone2", SmootherConfig()).params["n_params"] == 1
    poly = FitResult("poly3", "zone1", np.arange(3.0), np.ones(3), {"n_params": 4}, 0.0, True, 0, "")
    assert C.zone_view(poly, "zone1", SmootherConfig()) is poly
    assert C.zone_view(poly, "zone3", SmootherConfig()) is poly       # not its business: only a global result is relabelled
    with pytest.raises(KeyError):
        C.zone_view(global_result(KNOTS), "all", SmootherConfig())


def test_adapt_config_copies_everything_and_never_shares_a_list():
    src = SmootherConfig(omega1=25, omega2=70, omega3=105, z2_methods=["perks"], all_methods=["spline_all"], sigma_join=5.0,
                         nr_maxiter=50, nr_tol=1e-6, grid_n=40)
    out = C.adapt_config(src, np.arange(0, 110))
    assert out == src and out is not src
    for name in ("z1_methods", "z2_methods", "z3_methods", "all_methods"):
        assert getattr(out, name) is not getattr(src, name)
    out.z2_methods.append("makeham")
    out.all_methods.clear()
    assert src.z2_methods == ["perks"] and src.all_methods == ["spline_all"]


@pytest.mark.parametrize("w1,w2,w3,lo,hi,want", [
    (20, 85, 110, 0, 59, (20, 59, 110)),          # omega2 beyond the last age -> the last age
    (90, 95, 110, 0, 59, (58, 59, 110)),          # both beyond: omega1 at last - 1, omega2 just above
    (20, 30, 110, 40, 95, (40, 41, 110)),         # both below the first age
    (20, 85, 41, 40, 95, (40, 85, 42)),           # omega3 under first + 2
    (20, 85, 110, 0, 109, (20, 85, 110)),         # same range: nothing moves
])
def test_adapt_config_brings_the_boundaries_inside_the_other_sexs_ages(w1, w2, w3, lo, hi, want):
    out = C.adapt_config(SmootherConfig(omega1=w1, omega2=w2, omega3=w3), np.arange(lo, hi + 1))
    assert (out.omega1, out.omega2, out.omega3) == want
    assert lo <= out.omega1 < out.omega2 <= hi
