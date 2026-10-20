"""Module 7 - app.py skeleton, driven headlessly with streamlit.testing.v1.AppTest.

Run:   python -m pytest tests/test_app.py -v

AppTest runs the real script (the code `streamlit run app.py` runs), with real uploads, real
clicks and the real loader: end-to-end tests of tabs 1 and 2, no browser needed.
"""
from __future__ import annotations

import base64
import io
import json
import os
import re
import sys
import types
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import streamlit as st
from streamlit.delta_generator import DeltaGenerator
from streamlit.testing.v1 import AppTest

import synthetic as S
from src import config as C
from src import evaluator as EV
from src import exporter as X
from src import joiner as J

APP = str(Path(__file__).resolve().parent.parent / "app.py")
MODE_C = "C — Un fichier de données élémentaires"
MODE_A = "A — Taux (ou décès) + exposition, sexes en colonnes"
MODE_B = "B — Un fichier par sexe"
PAGES = {"chargement": "views/chargement.py", "parametrage": "views/parametrage.py",
         "evaluation": "views/evaluation.py", "export": "views/export.py"}
SIDEBAR_LABELS = ["1 · Chargement", "2 · Paramétrage", "3 · Évaluation", "4 · Export", "Accueil"]
BOTH = ("Female", "Male")


@pytest.fixture(scope="module")
def truth():
    return {(s, y): S.make_sex_table(s, y) for s in BOTH for y in (2020, 2021)}


@pytest.fixture
def files(tmp_path, truth):
    by_year = S.hmd_tables((2020, 2021))
    by_cohort = S.hmd_tables((1950, 1951))
    return {
        "dir": tmp_path,
        "female_csv": S.write_per_sex_csv(tmp_path / "female.csv", truth[("Female", 2020)].drop(columns="mu_true")),
        "male_csv": S.write_per_sex_csv(tmp_path / "male.csv", truth[("Male", 2020)].drop(columns="mu_true")),
        "deaths": S.write_hmd(tmp_path / "Deaths_1x1.txt", "Dx", by_year),
        "exposure": S.write_hmd(tmp_path / "Exposures_1x1.txt", "Ex", by_year),
        "rates": S.write_hmd(tmp_path / "Mx_1x1.txt", "mx", by_year),
        "c_deaths": S.write_hmd(tmp_path / "cDeaths_1x1.txt", "Dx", by_cohort, key_name="Cohort"),
        "c_exposure": S.write_hmd(tmp_path / "cExposures_1x1.txt", "Ex", by_cohort, key_name="Cohort"),
    }


def goto(at, page):
    """Show another page, as a click on the sidebar would (AppTest does not follow links): the session state is kept,
    but a widget of a page that is not shown loses its state, as in the browser."""
    at.switch_page(PAGES[page]).run()
    assert not at.exception, [e.value for e in at.exception]
    at.page = page
    return at


def ensure(at, page):
    """`goto` unless the page is already the one shown."""
    return at if getattr(at, "page", None) == page else goto(at, page)


def start(mode: str = MODE_B, view: str = "period") -> AppTest:
    """The loading page, on option B (one file per sex) unless told otherwise; MODE_C is the page's own default.
    `view` is what the home page would have chosen ('period' or 'cohort')."""
    at = AppTest.from_file(APP, default_timeout=60)
    at.session_state["view_choice"] = view
    goto(at, "chargement")
    if mode != MODE_C:
        at.segmented_control[0].set_value(mode).run()
    return at


def upload(at, key, path):
    at.file_uploader(key=key).set_value((path.name, path.read_bytes(), "text/plain"))


def select(at, label, value):
    [w for w in at.selectbox if w.label == label][0].select(value)


def seg(at, label):
    return [w for w in at.segmented_control if w.label == label][0]


def load(at):
    ensure(at, "chargement")
    at.button(key="btn_load").click().run()
    assert not at.exception, [e.value for e in at.exception]


def option_a(files, *keys):
    at = start()
    at.segmented_control[0].set_value(MODE_A).run()
    for key in keys:
        upload(at, f"up_{key}", files[key])
    at.run()
    return at


def loaded_option_b(files, page="parametrage"):
    """Both sexes loaded from two files; the settings page is shown unless `page` says otherwise."""
    at = start()
    upload(at, "up_female", files["female_csv"])
    upload(at, "up_male", files["male_csv"])
    at.run()
    load(at)
    return at if page == "chargement" else goto(at, page)


def rows(df, sex):
    return df[df["sex"] == sex].reset_index(drop=True)


def labels(elements):
    return [e.label for e in elements]


# --------------------------------------------------------------------------
# Skeleton and state (spec 7.1, 7.2)
# --------------------------------------------------------------------------


def sidebar_html(at):
    return "".join(h.proto.body for h in at.sidebar.get("html"))


def test_the_app_opens_on_a_home_page_that_has_the_sidebar_and_nothing_loaded():
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception, [e.value for e in at.exception]
    assert [b.key for b in at.button] == ["home_period", "home_cohort"]
    assert [p.proto.label for p in at.sidebar.get("page_link")] == SIDEBAR_LABELS      # the sidebar is there on the home page too
    assert "Risk" in sidebar_html(at) and "rdl-chip" not in sidebar_html(at)           # no table type to recall before a choice
    for key in ("df_raw", "config", "fit_results", "mu_joint", "life_table",
                "selected_z1", "selected_z2", "selected_z3"):                  # spec 7.2
        assert key in at.session_state
    assert at.session_state["df_raw"] is None and at.session_state["view_choice"] == "period"
    assert at.session_state["config"] == {} and at.session_state["fit_results"] == {}


def test_two_sessions_never_share_the_mutable_defaults_of_the_state():
    """The defaults live at module level: each session must get its own copy (a dict shared across sessions leaks one user's
    configuration, estimations and choices into another's)."""
    first = AppTest.from_file(APP, default_timeout=60).run()
    for key in ("ui", "config", "fit_results", "export_params"):
        first.session_state[key]["marker"] = 1
    first.session_state["fit_status"].append("marker")
    second = AppTest.from_file(APP, default_timeout=60).run()
    for key in ("ui", "config", "fit_results", "export_params"):
        assert "marker" not in second.session_state[key], key
    assert second.session_state["fit_status"] == []


@pytest.mark.parametrize("button,view,phrase", [("home_period", "period", "tables du moment"),
                                                ("home_cohort", "cohort", "tables générationnelles")])
def test_choosing_the_kind_of_table_on_the_home_page_opens_the_loading_page_and_its_sidebar(button, view, phrase):
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.button(key=button).click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.session_state["view_choice"] == view
    assert [w.label for w in at.segmented_control] == ["Disposition des fichiers"]          # the loading page is shown
    assert any(f"Type de table : **{phrase}**" in m.value for m in at.main.markdown)
    assert [p.proto.label for p in at.sidebar.get("page_link")] == SIDEBAR_LABELS


def test_a_different_choice_on_the_home_page_drops_what_was_loaded_and_the_same_one_keeps_it(files):
    at = loaded_option_b(files, page="chargement")
    assert at.session_state["df_raw"] is not None
    home = AppTest.from_file(APP, default_timeout=60)
    home.session_state["view_choice"] = "period"
    home.session_state["df_raw"] = at.session_state["df_raw"]
    home.run()                                                                   # the home page, with something loaded
    home.button(key="home_period").click().run()
    assert home.session_state["df_raw"] is not None                              # same kind: kept
    home2 = AppTest.from_file(APP, default_timeout=60)
    home2.session_state["view_choice"] = "period"
    home2.session_state["df_raw"] = at.session_state["df_raw"]
    home2.run()
    home2.button(key="home_cohort").click().run()
    assert home2.session_state["view_choice"] == "cohort" and home2.session_state["df_raw"] is None


def test_the_sidebar_ticks_the_pages_that_are_done(files):
    at = loaded_option_b(files, page="chargement")
    labels = lambda: [p.proto.label for p in at.sidebar.get("page_link")]
    assert labels() == ["1 · Chargement ✓", "2 · Paramétrage", "3 · Évaluation", "4 · Export", "Accueil"]
    launch(at)
    goto(at, "evaluation")
    assert labels() == ["1 · Chargement ✓", "2 · Paramétrage ✓", "3 · Évaluation", "4 · Export", "Accueil"]
    validate(at)
    assert labels() == ["1 · Chargement ✓", "2 · Paramétrage ✓", "3 · Évaluation ✓", "4 · Export", "Accueil"]


def test_the_pages_without_their_input_point_back_to_the_previous_page():
    at = AppTest.from_file(APP, default_timeout=60)
    for page, previous in (("parametrage", "chargement"), ("evaluation", "parametrage"), ("export", "evaluation")):
        at.switch_page(PAGES[page]).run()
        assert not at.exception and at.info
        assert any(p.proto.page == previous for p in at.main.get("page_link"))


def test_the_app_opens_on_option_c_and_each_option_has_its_own_uploaders():
    at = start(MODE_C)
    assert at.segmented_control[0].value == MODE_C and at.segmented_control[0].options == [MODE_C, MODE_A, MODE_B]
    assert labels(at.file_uploader) == ["Fichier de données élémentaires"]
    at.segmented_control[0].set_value(MODE_B).run()
    assert labels(at.file_uploader) == ["Fichier Féminin", "Fichier Masculin"]
    at.segmented_control[0].set_value(MODE_A).run()
    assert labels(at.file_uploader) == ["Deaths (Dx)", "Exposure (Ex)", "Rates (mx)"]


# --------------------------------------------------------------------------
# Tab 1, mode A: one all-in-one file per sex
# --------------------------------------------------------------------------


def test_mode_a_two_files_load_as_one_table_with_both_sexes(files, truth):
    at = start()
    assert not [b for b in at.button if b.key == "btn_load"]                   # nothing uploaded yet
    upload(at, "up_female", files["female_csv"])
    upload(at, "up_male", files["male_csv"])
    at.run()
    choice = seg(at, "Sexe(s) à traiter")
    assert choice.options == ["Les deux", "Féminin", "Masculin"] and choice.value == "both"

    load(at)
    df = at.session_state["df_raw"]
    assert list(df.columns) == ["sex", "age", "Dx", "Ex", "mx_raw", "source"]
    assert df.attrs["sexes"] == ["Female", "Male"] and len(df) == 202
    for s in BOTH:
        np.testing.assert_array_equal(rows(df, s)["Dx"], truth[(s, 2020)]["Dx"])
    summary = [d for d in at.dataframe if list(d.value.columns) == ["Contrôle", "Féminin", "Masculin"]]
    assert len(summary) == 1 and not at.error


def test_mode_a_the_sex_button_keeps_one_sex_or_both(files, truth):
    at = start()
    upload(at, "up_female", files["female_csv"])
    upload(at, "up_male", files["male_csv"])
    at.run()
    seg(at, "Sexe(s) à traiter").set_value("Male")
    at.run()
    load(at)
    df = at.session_state["df_raw"]
    assert df.attrs["sexes"] == ["Male"] and set(df["sex"]) == {"Male"}
    np.testing.assert_array_equal(df["Dx"], truth[("Male", 2020)]["Dx"])


def test_mode_a_a_single_file_loads_without_a_sex_choice(files, truth):
    at = start()
    upload(at, "up_male", files["male_csv"])
    at.run()
    assert not [w for w in at.segmented_control if w.label == "Sexe(s) à traiter"]
    assert any("Sexe : masculin" in c.value for c in at.caption)
    load(at)
    df = at.session_state["df_raw"]
    assert df.attrs["sexes"] == ["Male"]
    np.testing.assert_array_equal(df["Dx"], truth[("Male", 2020)]["Dx"])


def test_loading_never_triggers_the_attrs_serialisation_warning(files):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        loaded_option_b(files)
    assert not [w for w in caught if "serialize" in str(w.message)]


def test_the_standard_file_with_both_sexes_is_offered_for_download(files):
    at = loaded_option_b(files, page="chargement")
    downloads = at.get("download_button")
    assert len(downloads) == 1 and "standardisée" in downloads[0].proto.label


def test_mode_a_unrecognised_columns_get_dropdowns_then_load(files):
    odd = files["dir"] / "odd.csv"
    t = S.make_sex_table("Female", 2020)
    pd.DataFrame({"a": t["age"], "b": t["Dx"], "c": t["Ex"]}).to_csv(odd, index=False)
    at = start()
    upload(at, "up_female", odd)
    at.run()
    maps = [w for w in at.selectbox if w.label.startswith("Colonne")]
    assert [w.label for w in maps] == ["Colonne d'âge", "Colonne Dx (décès)", "Colonne Ex (exposition)"]
    assert maps[0].options == ["—", "a", "b", "c"]
    load(at)                                                                    # nothing mapped yet
    assert at.session_state["df_raw"] is None
    assert any("colonne d'âge introuvable" in e.value for e in at.error)
    for label, col in zip(("Colonne d'âge", "Colonne Dx (décès)", "Colonne Ex (exposition)"), "abc"):
        select(at, label, col)                      # re-query: handles from before a run() are stale
    at.run()
    load(at)
    np.testing.assert_array_equal(at.session_state["df_raw"]["Dx"], t["Dx"])


def test_mode_a_rates_only_asks_for_a_reference_exposure(files):
    t = S.make_sex_table("Female", 2020)
    rates = files["dir"] / "rates_only.csv"
    pd.DataFrame({"Age": t["age"], "mx": t["Dx"] / t["Ex"]}).to_csv(rates, index=False)
    at = start()
    upload(at, "up_female", rates)
    at.run()
    assert not [w for w in at.selectbox if w.label.startswith("Colonne")]       # no Dx/Ex dropdowns to nag
    [w for w in at.number_input if w.label.startswith("Exposition de référence")][0].set_value(50_000.0)
    at.run()
    load(at)
    assert (at.session_state["df_raw"]["Ex"] == 50_000).all()


def test_mode_a_excel_sheet_and_header_row_are_configurable(files, truth):
    xlsx = files["dir"] / "f.xlsx"
    t = truth[("Female", 2020)].drop(columns="mu_true").rename(columns={"age": "Age", "Dx": "Deaths", "Ex": "Exposure"})
    with pd.ExcelWriter(xlsx) as w:
        pd.DataFrame({"readme": ["x"]}).to_excel(w, sheet_name="notes", index=False)
        t.to_excel(w, sheet_name="data", index=False, startrow=2)
    at = start()
    upload(at, "up_female", xlsx)
    at.run()
    select(at, "Feuille", "data")
    at.run()
    load(at)                                                                    # header: found by itself
    np.testing.assert_array_equal(at.session_state["df_raw"]["Dx"], truth[("Female", 2020)]["Dx"])
    [w for w in at.number_input if w.label.startswith("Ligne d'en-tête")][0].set_value(3)   # ... or told
    at.run()
    load(at)
    np.testing.assert_array_equal(at.session_state["df_raw"]["Dx"], truth[("Female", 2020)]["Dx"])


def test_old_xls_is_refused_with_the_loader_message(files):
    xls = files["dir"] / "old.xls"
    xls.write_bytes(b"\xd0\xcf\x11\xe0" + b"\x00" * 300)
    at = start()
    upload(at, "up_female", xls)
    at.run()
    assert any(".xlsx" in e.value for e in at.error)
    assert not [b for b in at.button if b.key == "btn_load" and not b.disabled]


# --------------------------------------------------------------------------
# Tab 1, mode B: one file per quantity, period tables
# --------------------------------------------------------------------------


def test_mode_b_year_and_sexes_then_load(files, truth):
    at = option_a(files, "deaths", "exposure", "rates")
    year = [w for w in at.selectbox if w.label == "Année"][0]
    assert year.value == 2021                                                   # latest year by default
    year.select(2020)
    assert seg(at, "Sexe(s) à traiter").options == ["Les deux", "Féminin", "Masculin"]
    assert any("Vision : tables du moment" in c.value for c in at.caption)      # one view offered: no control
    at.run()
    load(at)
    df = at.session_state["df_raw"]
    assert (df.attrs["view"], df.attrs["year"]) == ("period", 2020) and df.attrs["sexes"] == ["Female", "Male"]
    for s in BOTH:
        np.testing.assert_array_equal(rows(df, s)["Dx"], truth[(s, 2020)]["Dx"])
    assert df["source"].iloc[0] == "Deaths_1x1.txt + Exposures_1x1.txt + Mx_1x1.txt"


def test_mode_b_one_sex_by_choice(files, truth):
    at = option_a(files, "deaths", "exposure")
    select(at, "Année", 2021)
    seg(at, "Sexe(s) à traiter").set_value("Female")
    at.run()
    load(at)
    df = at.session_state["df_raw"]
    assert df.attrs["sexes"] == ["Female"]
    np.testing.assert_array_equal(df["Dx"], truth[("Female", 2021)]["Dx"])


def test_mode_b_exposure_and_rates_rebuild_dx_and_say_so(files):
    at = option_a(files, "exposure", "rates")
    select(at, "Année", 2020)
    at.run()
    load(at)
    assert any("Dx reconstruit" in i.value for i in at.info)


def test_mode_b_rates_alone_asks_for_a_reference_exposure(files):
    at = option_a(files, "rates")
    assert [w for w in at.number_input if w.label.startswith("Exposition de référence")]
    select(at, "Année", 2020)
    at.run()
    load(at)
    assert (at.session_state["df_raw"]["Ex"] == 100_000).all()


def test_mode_b_missing_exposure_is_reported_in_french_not_as_a_crash(files):
    at = option_a(files, "deaths")
    select(at, "Année", 2020)
    at.run()
    load(at)
    assert at.session_state["df_raw"] is None
    assert any("Ex introuvable" in e.value for e in at.error)


def test_mode_b_a_file_that_does_not_say_its_sex_forces_a_single_choice(files, truth):
    f = truth[("Female", 2020)]
    d, e = files["dir"] / "d.csv", files["dir"] / "e.csv"
    pd.DataFrame({"Age": f["age"], "Deaths": f["Dx"]}).to_csv(d, index=False)
    pd.DataFrame({"Age": f["age"], "Exposure": f["Ex"]}).to_csv(e, index=False)
    at = start()
    at.segmented_control[0].set_value(MODE_A).run()
    upload(at, "up_deaths", d)
    upload(at, "up_exposure", e)
    at.run()
    choice = seg(at, "Sexe(s) à traiter")
    assert choice.options == ["Féminin", "Masculin"]                            # no 'both': nothing to split on
    assert any("ne précise pas le sexe" in c.value for c in at.caption)
    choice.set_value("Female")
    at.run()
    load(at)
    df = at.session_state["df_raw"]
    assert df.attrs["sexes"] == ["Female"]
    np.testing.assert_array_equal(df["Ex"], f["Ex"])


def test_mode_b_an_exposure_for_both_sexes_combined_cannot_serve_one_sex(files, truth):
    expo = S.write_hmd(files["dir"] / "E_total.txt", "Ex", S.hmd_tables((2020, 2021)), columns=("Total",))
    at = start()
    at.segmented_control[0].set_value(MODE_A).run()
    upload(at, "up_deaths", files["deaths"])
    upload(at, "up_exposure", expo)
    at.run()
    select(at, "Année", 2020)
    seg(at, "Sexe(s) à traiter").set_value("Female")
    at.run()
    load(at)
    assert at.session_state["df_raw"] is None
    assert any("deux sexes confondus" in e.value for e in at.error)


def test_the_hmd_inputdb_is_refused_up_front(files):
    inputdb = S.write_inputdb(files["dir"] / "FRATNPdeath.txt", S.hmd_tables((2020,)))
    at = start()
    at.segmented_control[0].set_value(MODE_A).run()
    upload(at, "up_deaths", inputdb)
    at.run()
    assert any("InputDB" in e.value and "Mx_1x1" in e.value for e in at.error)
    assert not [b for b in at.button if b.key == "btn_load" and not b.disabled]


def test_a_blocking_quality_error_names_the_sex_lists_the_rows_and_loads_nothing(files):
    f = S.make_sex_table("Female", 2020).drop(columns="mu_true")
    m = S.make_sex_table("Male", 2020).drop(columns="mu_true")
    m.loc[60, ["Dx", "Ex"]] = [5, 0.0]
    bad = S.write_per_sex_csv(files["dir"] / "bad_m.csv", m)
    at = start()
    upload(at, "up_female", S.write_per_sex_csv(files["dir"] / "ok_f.csv", f))
    upload(at, "up_male", bad)
    at.run()
    load(at)
    assert at.session_state["df_raw"] is None
    assert any("Erreur bloquante" in e.value and "Male" in e.value and "60" in e.value for e in at.error)
    listed = [d for d in at.dataframe if "age" in d.value.columns and list(d.value["age"]) == [60]]
    assert listed, "the offending row should be listed (spec 1.5)"


def test_a_red_alert_on_one_sex_is_shown_but_the_table_is_still_loaded(files):
    f = S.make_sex_table("Female", 2020).drop(columns="mu_true")
    f.loc[40, "Dx"] = -3
    at = start()
    upload(at, "up_female", S.write_per_sex_csv(files["dir"] / "neg.csv", f))
    upload(at, "up_male", files["male_csv"])
    at.run()
    load(at)
    assert at.session_state["df_raw"] is not None
    assert any("Alerte rouge" in e.value and "40" in e.value for e in at.error)
    qc = at.session_state["df_raw"].attrs["qc"]
    assert not qc["Female"].can_proceed and qc["Male"].can_proceed


# --------------------------------------------------------------------------
# Tab 1, option C: one file of elementary data - and the sources beyond HMD
# --------------------------------------------------------------------------


def option_c(path):
    at = start(MODE_C)
    upload(at, "up_elementary", path)
    at.run()
    return at


def test_option_c_one_elementary_file_year_and_sexes_then_load(files, truth):
    p = S.write_elementary_csv(files["dir"] / "elementary.csv", S.hmd_tables((2020, 2021)))
    at = option_c(p)
    assert seg(at, "Sexe(s) à traiter").options == ["Les deux", "Féminin", "Masculin"]
    year = [w for w in at.selectbox if w.label == "Année"][0]
    assert year.value == 2021
    year.select(2020)
    at.run()
    load(at)
    df = at.session_state["df_raw"]
    assert df.attrs["sexes"] == ["Female", "Male"] and df.attrs["year"] == 2020 and len(df) == 202
    for s in BOTH:
        np.testing.assert_array_equal(rows(df, s)["Dx"], truth[(s, 2020)]["Dx"])
    assert any("mₓ" in c.value or "mx" in c.value.lower() or "tables du moment" in c.value for c in at.caption)


def test_option_c_a_segment_column_gets_its_own_selector(files, truth):
    p = S.write_elementary_csv(files["dir"] / "regions.csv", S.hmd_tables((2020,)),
                               segment={"Region": {"North": 1.0, "South": 0.5}})
    at = option_c(p)
    region = [w for w in at.selectbox if w.label == "Segment [Region]"][0]
    assert region.options == ["North", "South"]
    region.select("South")
    at.run()
    load(at)
    df = at.session_state["df_raw"]
    np.testing.assert_allclose(rows(df, "Male")["Ex"], truth[("Male", 2020)]["Ex"] * 0.5, atol=0.01)


def test_option_a_insee_deaths_with_the_field_to_choose(files, truth):
    insee = S.write_insee_deaths(files["dir"] / "T74BIS.csv", S.hmd_tables((2020, 2021)), fr_extra=5)
    at = start(MODE_A)
    upload(at, "up_deaths", insee)
    upload(at, "up_exposure", files["exposure"])
    at.run()
    assert any("format reconnu : INSEE" in c.value for c in at.caption)
    field = [w for w in at.selectbox if w.label == "Segment [CHAMP2]"][0]
    assert field.options == ["FM", "FR"]
    select(at, "Année", 2020)
    field.select("FR")
    at.run()
    load(at)
    df = at.session_state["df_raw"]
    np.testing.assert_array_equal(rows(df, "Female")["Dx"], truth[("Female", 2020)]["Dx"].iloc[:100] + 5)
    assert rows(df, "Male")["age"].max() == 99                           # AGE2_REVOLU: '100' is 100 and over


def test_option_a_eurostat_tsv_deaths_and_a_country_to_choose(files, truth):
    tsv = S.write_eurostat_tsv(files["dir"] / "demo_magec.tsv", "Dx", S.hmd_tables((2020, 2021)))
    at = start(MODE_A)
    upload(at, "up_deaths", tsv)
    upload(at, "up_exposure", files["exposure"])
    at.run()
    assert any("format reconnu : Eurostat" in c.value for c in at.caption)
    select(at, "Segment [geo]", "DE")
    select(at, "Année", 2020)
    at.run()
    load(at)
    df = at.session_state["df_raw"]
    np.testing.assert_allclose(rows(df, "Female")["Dx"], truth[("Female", 2020)]["Dx"].iloc[:100] + 7)


def test_option_c_the_header_is_found_below_title_lines(files, truth):
    t = truth[("Female", 2020)]
    p = files["dir"] / "titled.csv"
    p.write_text("Statistiques de mortalité 2020\nSource : test\n\nÂge;Décès;Exposition\n"
                 + "\n".join(f"{a};{d};{e}" for a, d, e in zip(t["age"], t["Dx"], t["Ex"])) + "\n", encoding="utf-8")
    at = option_c(p)
    assert seg(at, "Sexe(s) à traiter").options == ["Féminin", "Masculin"]      # the file names no sex: attribute one
    seg(at, "Sexe(s) à traiter").set_value("Female")
    at.run()
    load(at)
    np.testing.assert_array_equal(at.session_state["df_raw"]["Dx"], t["Dx"])


def test_option_c_january_population_checkbox_changes_the_exposure(files, truth):
    parts = []
    for y in (2020, 2021):
        for s in BOTH:
            t = truth[(s, y)]
            parts.append(pd.DataFrame({"Year": y, "Sex": s, "Age": t["age"], "Deaths": t["Dx"],
                                       "Population": truth[(s, 2020)]["Ex"] * (1 + 0.1 * (y - 2020))}))
    p = files["dir"] / "pop.csv"
    pd.concat(parts).to_csv(p, index=False)

    at = option_c(p)
    select(at, "Année", 2020)
    at.run()
    load(at)
    np.testing.assert_allclose(rows(at.session_state["df_raw"], "Male")["Ex"], truth[("Male", 2020)]["Ex"])
    assert any("lue comme exposition" in i.value for i in at.info)               # 'Population' read as the exposure, said so

    [c for c in at.checkbox if c.label.startswith("Exposition = population")][0].check()
    at.run()
    load(at)
    df = at.session_state["df_raw"]
    np.testing.assert_allclose(rows(df, "Male")["Ex"], truth[("Male", 2020)]["Ex"] * 1.05)
    assert any("1er janvier 2020 et 2021" in i.value for i in at.info)


def test_option_c_a_file_with_one_sex_column_less_still_loads_by_choice(files, truth):
    f = truth[("Female", 2020)].drop(columns="mu_true")
    p = S.write_per_sex_csv(files["dir"] / "one_sex.csv", f)
    at = option_c(p)
    assert seg(at, "Sexe(s) à traiter").options == ["Féminin", "Masculin"]
    seg(at, "Sexe(s) à traiter").set_value("Male")                               # the user says whose it is
    at.run()
    load(at)
    assert at.session_state["df_raw"].attrs["sexes"] == ["Male"]


def test_option_c_the_hmd_inputdb_is_refused_up_front(files):
    inputdb = S.write_inputdb(files["dir"] / "FRATNPdeath.txt", S.hmd_tables((2020,)))
    at = option_c(inputdb)
    assert any("InputDB" in e.value for e in at.error)
    assert not [b for b in at.button if b.key == "btn_load" and not b.disabled]


# --------------------------------------------------------------------------
# Tab 1: generation tables
# --------------------------------------------------------------------------


def test_cohort_files_are_loaded_as_generation_tables(files, truth):
    at = start(view="cohort")
    at.segmented_control[0].set_value(MODE_A).run()
    upload(at, "up_deaths", files["c_deaths"])
    upload(at, "up_exposure", files["c_exposure"])
    at.run()
    assert any("Vision : tables générationnelles" in c.value for c in at.caption)
    gen = [w for w in at.selectbox if w.label == "Génération"][0]
    assert gen.options == ["1950", "1951"] and not [w for w in at.selectbox if w.label == "Année"]
    gen.select(1950)
    at.run()
    load(at)
    df = at.session_state["df_raw"]
    assert (df.attrs["view"], df.attrs["cohort"], df.attrs["year"]) == ("cohort", 1950, None)
    assert any("génération 1950" in c.value for c in at.caption)


def test_a_file_with_both_a_year_and_a_cohort_column_follows_the_kind_chosen_on_the_home_page(files, truth):
    parts = []
    for yr in (2020, 2021):
        for s, label in (("Female", "f"), ("Male", "m")):
            t = truth[(s, yr)]
            parts.append(pd.DataFrame({"Year": yr, "Cohort": yr - t["age"], "Age": t["age"], "Sex": label,
                                       "Deaths": t["Dx"], "Exposure": t["Ex"]}))
    lexis = files["dir"] / "lexis.csv"
    pd.concat(parts).to_csv(lexis, index=False)
    for view, shown, hidden in (("period", "Année", "Génération"), ("cohort", "Génération", "Année")):
        at = start(view=view)
        upload(at, "up_female", lexis)
        at.run()
        assert not [w for w in at.segmented_control if w.label == "Vision"]            # decided on the home page
        assert [w for w in at.selectbox if w.label == shown] and not [w for w in at.selectbox if w.label == hidden]


def test_files_of_the_other_kind_than_the_one_chosen_are_refused_with_the_way_out(files):
    for view, keys, wanted, found in (("cohort", ("deaths", "exposure"), "tables générationnelles", "tables du moment"),
                                      ("period", ("c_deaths", "c_exposure"), "tables du moment", "tables générationnelles")):
        at = start(MODE_A, view=view)
        for key in keys:
            upload(at, f"up_{key.removeprefix('c_')}", files[key])
        at.run()
        assert any(f"Vous avez choisi les {wanted}" in e.value and found in e.value for e in at.error)
        assert at.button(key="btn_load").disabled
        assert any(p.proto.page == "" for p in at.main.get("page_link"))              # 'Changer de type' leads to the home page


def test_files_that_give_neither_a_year_nor_a_generation_follow_the_choice_without_complaint(files):
    at = start(view="cohort")
    upload(at, "up_female", files["female_csv"])
    upload(at, "up_male", files["male_csv"])
    at.run()
    assert not at.error and not at.button(key="btn_load").disabled
    assert any("ni année ni génération" in c.value for c in at.caption)


def test_a_period_file_and_a_cohort_file_cannot_be_loaded_together(files):
    at = start()
    at.segmented_control[0].set_value(MODE_A).run()
    upload(at, "up_deaths", S.write_hmd(files["dir"] / "D_2020.txt", "Dx", S.hmd_tables((2020,))))
    upload(at, "up_exposure", S.write_hmd(files["dir"] / "cE_1950.txt", "Ex", S.hmd_tables((1950,)), key_name="Cohort"))
    at.run()
    assert any("même vision" in e.value for e in at.error)
    assert at.button(key="btn_load").disabled


# --------------------------------------------------------------------------
# Tab 2: one configuration per sex (spec 2.2) and the launch
# --------------------------------------------------------------------------


def method_boxes(at):
    """The tab-2 checkboxes (one per smoothing method), not the 1 January box of tab 1."""
    names = {m.label for m in C.METHODS.values()}
    return [c for c in at.checkbox if c.label in names]


def slider(at, prefix):
    return [s for s in at.slider if s.label.startswith(prefix)][0]


def test_tab2_sliders_follow_the_loaded_age_range_and_the_spec_defaults(files):
    at = loaded_option_b(files)                                                    # ages 0..100, both sexes
    w1, w2, sigma = slider(at, "ω₁"), slider(at, "ω₂"), slider(at, "σ")
    assert (w1.min, w1.max, w1.value) == (0, 99, 20)                             # spec 2.2: range of the data; omega1 < last age
    assert (w2.min, w2.max, w2.value) == (21, 100, 85)                           # omega2 starts at omega1 + 1
    assert (sigma.min, sigma.max, sigma.value) == (1.0, 10.0, 3.0)               # spec: slider 1-10, default 3
    assert [c.label for c in method_boxes(at)] == ["Polynôme deg 3", "Splines naturelles", "Gompertz", "Makeham",
                                                   "Makeham 2e ordre", "Perks", "Kannisto", "Beard", "Coale-Kisker"]
    assert all(c.value for c in method_boxes(at))
    assert seg(at, "Jeu de données à paramétrer").options == ["Féminin", "Masculin"]


def test_tab2_has_no_dataset_switch_for_a_single_sex(files):
    at = start()
    upload(at, "up_male", files["male_csv"])
    at.run()
    load(at)
    goto(at, "parametrage")
    assert not [w for w in at.segmented_control if w.label == "Jeu de données à paramétrer"]
    assert list(at.session_state["config"]) == ["Male"]


def test_tab2_each_sex_keeps_its_own_configuration(files):
    at = loaded_option_b(files)
    slider(at, "ω₁").set_value(25)
    [c for c in at.checkbox if c.label == "Perks"][0].uncheck()
    at.run()
    seg(at, "Jeu de données à paramétrer").set_value("Masculin")
    at.run()
    assert slider(at, "ω₁").value == 20 and [c for c in at.checkbox if c.label == "Perks"][0].value   # Male: fresh
    slider(at, "ω₁").set_value(30)
    slider(at, "σ").set_value(5.0)
    at.run()
    seg(at, "Jeu de données à paramétrer").set_value("Féminin")
    at.run()
    assert slider(at, "ω₁").value == 25 and slider(at, "σ").value == 3.0                  # Female: as left
    assert not [c for c in at.checkbox if c.label == "Perks"][0].value

    cfg = at.session_state["config"]
    assert set(cfg) == {"Female", "Male"}
    assert (cfg["Female"].omega1, cfg["Female"].sigma_join, "perks" in cfg["Female"].z2_methods) == (25, 3.0, False)
    assert (cfg["Male"].omega1, cfg["Male"].sigma_join, "perks" in cfg["Male"].z2_methods) == (30, 5.0, True)


def test_tab2_advanced_options_reach_the_config(files):
    at = loaded_option_b(files)
    [n for n in at.number_input if n.label.startswith("ω₃")][0].set_value(105)
    [n for n in at.number_input if n.label.startswith("Itérations")][0].set_value(50)
    [n for n in at.number_input if n.label.startswith("Grille")][0].set_value(40)
    [s for s in at.select_slider if s.label.startswith("Critère")][0].set_value(1e-10)
    at.run()
    cfg = at.session_state["config"]["Female"]
    assert (cfg.omega3, cfg.nr_maxiter, cfg.grid_n, cfg.nr_tol) == (105, 50, 40, 1e-10)


def omega3_box(at):
    return [n for n in at.number_input if n.label.startswith("ω₃")][0]


def test_tab2_blocks_the_launch_on_inconsistent_boundaries(files):
    at = loaded_option_b(files)
    omega3_box(at).set_value(50)                                                 # omega3 below omega2 = 85
    at.run()
    assert any("ω₁ < ω₂ < ω₃" in e.value for e in at.error)
    assert at.button(key="btn_run").disabled


def test_tab2_an_invalid_configuration_on_the_other_sex_also_blocks_the_launch(files):
    at = loaded_option_b(files)
    omega3_box(at).set_value(50)                                                 # Female made invalid ...
    at.run()
    seg(at, "Jeu de données à paramétrer").set_value("Masculin")                 # ... then we look at Male
    at.run()
    assert any("Féminin : configuration invalide" in e.value for e in at.error)
    assert at.button(key="btn_run").disabled
    seg(at, "Jeu de données à paramétrer").set_value("Féminin")
    at.run()
    omega3_box(at).set_value(110)
    at.run()
    assert not at.button(key="btn_run").disabled


def test_tab2_blocks_the_launch_when_nothing_is_ticked(files):
    at = loaded_option_b(files)
    for c in method_boxes(at):
        c.uncheck()
    at.run()
    assert any("Cochez au moins une méthode" in e.value for e in at.error)
    assert at.button(key="btn_run").disabled


def test_tab2_an_unticked_zone_is_only_a_warning(files):
    at = loaded_option_b(files)
    for c in method_boxes(at):
        if c.label in ("Kannisto", "Beard", "Coale-Kisker"):
            c.uncheck()
    at.run()
    assert any("Aucune méthode cochée en Zone III" in w.value for w in at.warning)
    assert not at.button(key="btn_run").disabled


def test_a_new_age_range_gets_fresh_defaults_not_stale_sliders(files, truth):
    at = loaded_option_b(files)
    slider(at, "ω₁").set_value(30)
    at.run()
    short = S.write_per_sex_csv(files["dir"] / "short.csv", truth[("Female", 2020)].drop(columns="mu_true").iloc[:60])
    goto(at, "chargement")
    upload(at, "up_male", short)
    upload(at, "up_female", short)
    at.run()
    load(at)                                                                     # ages 0..59
    goto(at, "parametrage")
    w1, w2 = slider(at, "ω₁"), slider(at, "ω₂")
    assert (w1.min, w1.max) == (0, 58)
    assert (w1.value, w2.value) == (12, 44)       # 20% / 75% of 0..59 (85 is out of range) - not the 30 set before
    assert not at.exception


# --------------------------------------------------------------------------
# Launch: the zone modules are called as fit_<key>(ages, Dx, Ex, cfg), for each sex
# --------------------------------------------------------------------------


def launch(at):
    ensure(at, "parametrage")
    at.button(key="btn_run").click().run()
    assert not at.exception, [e.value for e in at.exception]
    return at.session_state["fit_status"]


def fake_fit(key, zone, calls=None, fail=False, wrong_type=False):
    def fit(ages, Dx, Ex, cfg):
        if calls is not None:
            calls.setdefault(key, []).append((ages.copy(), Dx.copy(), Ex.copy(), cfg))
        if fail:
            raise RuntimeError("Hessian singular")
        if wrong_type:
            return {"mu_hat": 1}
        return C.FitResult(method=key, zone=zone, ages=ages, mu_hat=Dx / Ex, params={"k": 1.0},
                           loglik=-123.4, converged=True, n_iter=7, message="fake fit")
    return fit


def install(monkeypatch, key, module, zone, **kw):
    mod = types.ModuleType(f"src.{module}")
    setattr(mod, f"fit_{key}", fake_fit(key, zone, **kw))
    monkeypatch.setitem(sys.modules, f"src.{module}", mod)


def block_zone_modules(monkeypatch):
    """Make every zone module unimportable (None in sys.modules raises ModuleNotFoundError), whatever
    has been built so far."""
    for module in {m.module for m in C.METHODS.values()}:
        monkeypatch.setitem(sys.modules, f"src.{module}", None)


def test_launch_with_no_zone_module_reports_every_method_of_every_sex_as_not_implemented(files, monkeypatch):
    block_zone_modules(monkeypatch)
    at = loaded_option_b(files)
    status = launch(at)
    assert len(status) == 2 * len(C.METHODS)
    assert [r["sex"] for r in status[:len(C.METHODS)]] == ["Féminin"] * len(C.METHODS)
    assert [r["sex"] for r in status[len(C.METHODS):]] == ["Masculin"] * len(C.METHODS)
    assert {r["status"] for r in status} == {"non implémenté"}
    assert at.session_state["fit_results"] == {"Female": {}, "Male": {}}
    assert any("non implémenté" in c.value for c in at.caption)


def test_launch_runs_the_real_zone_one_modules_for_both_sexes(files, truth, monkeypatch):
    """No stand-in: poly3 and spline from src/zone1_poly.py, on each sex's own rows 0..omega1."""
    for module in {m.module for m in C.METHODS.values()} - {"zone1_poly"}:
        monkeypatch.setitem(sys.modules, f"src.{module}", None)
    at = loaded_option_b(files)
    status = launch(at)
    zone1 = [r for r in status if r["method"] in ("poly3", "spline")]
    assert [(r["sex"], r["method"], r["status"]) for r in zone1] == [
        ("Féminin", "poly3", "ok"), ("Féminin", "spline", "ok"),
        ("Masculin", "poly3", "ok"), ("Masculin", "spline", "ok")]
    assert all(r["converged"] and r["n_iter"] == 0 and np.isfinite(r["loglik"]) and r["detail"] for r in zone1)
    assert {r["status"] for r in status if r["method"] not in ("poly3", "spline")} == {"non implémenté"}
    res = at.session_state["fit_results"]
    for sex in BOTH:
        assert sorted(res[sex]) == ["poly3", "spline"]
        for key, r in res[sex].items():
            assert (r.method, r.zone) == (key, "zone1")
            np.testing.assert_array_equal(r.ages, np.arange(0, 21))                      # omega1 = 20, both bounds
            assert r.mu_hat.shape == (21,) and np.all(r.mu_hat > 0)


def test_launch_runs_the_real_gompertz_on_zone_two_for_both_sexes(files, truth, monkeypatch):
    """Zone I and Gompertz are the real modules; each sex is fitted on its own rows omega1..omega2 and the
    rows handed over are the ones the loader produced (the app's table, not a copy of the truth)."""
    for module in {m.module for m in C.METHODS.values()} - {"zone1_poly", "zone2_gompertz"}:
        monkeypatch.setitem(sys.modules, f"src.{module}", None)
    at = loaded_option_b(files)
    status = launch(at)
    gomp = [r for r in status if r["method"] == "gompertz"]
    assert [(r["sex"], r["status"], r["converged"]) for r in gomp] == [("Féminin", "ok", True), ("Masculin", "ok", True)]
    assert all(r["n_iter"] < 50 and np.isfinite(r["loglik"]) and "convergé" in r["detail"] for r in gomp)
    assert {r["status"] for r in status if r["method"] in ("makeham", "makeham2", "perks", "kannisto", "beard",
                                                           "coale_kisker")} == {"non implémenté"}
    res = at.session_state["fit_results"]
    for sex in BOTH:
        r = res[sex]["gompertz"]
        assert (r.method, r.zone) == ("gompertz", "zone2")
        np.testing.assert_array_equal(r.ages, np.arange(20, 86))                          # 20..85, both bounds
        assert r.mu_hat.shape == (66,) and r.params["c"] > 1.0


def test_launch_runs_the_four_real_zone_two_modules_for_both_sexes(files, monkeypatch):
    """Gompertz, Makeham, Makeham 2nd order and Perks, each from its own file, on each sex's rows omega1..omega2,
    all through the app's launch button; only zone III is still missing."""
    monkeypatch.setitem(sys.modules, "src.zone3", None)
    at = loaded_option_b(files)
    status = launch(at)
    keys = ("gompertz", "makeham", "makeham2", "perks")
    zone2 = [r for r in status if r["method"] in keys]
    assert [(r["sex"], r["method"], r["status"]) for r in zone2] == [
        (sex, key, "ok") for sex in ("Féminin", "Masculin") for key in keys]
    assert all(r["converged"] and np.isfinite(r["loglik"]) and r["detail"] for r in zone2)
    assert {r["status"] for r in status if r["method"] in ("kannisto", "beard", "coale_kisker")} == {"non implémenté"}
    res = at.session_state["fit_results"]
    for sex in BOTH:
        assert [k for k in res[sex] if k in keys] == list(keys)
        for key in keys:
            r = res[sex][key]
            assert (r.method, r.zone) == (key, "zone2") and r.params["n_params"] == C.METHODS[key].n_params
            np.testing.assert_array_equal(r.ages, np.arange(20, 86))
            assert r.mu_hat.shape == (66,) and np.all(r.mu_hat > 0)
        assert res[sex]["makeham"].loglik >= res[sex]["gompertz"].loglik - 1e-6          # nested models, through the app
        assert res[sex]["makeham2"].loglik >= res[sex]["makeham"].loglik - 1e-6
        assert res[sex]["perks"].loglik >= res[sex]["makeham"].loglik - 1e-6


def test_launch_runs_all_nine_real_methods_for_both_sexes(files):
    """No stand-in, nothing blocked: every method of every zone, each sex on its own rows. The app's table stops at
    age 100, so zone III is fitted on 85..100 and must still cover 85..omega3 = 110 (the extrapolation it exists for)."""
    at = loaded_option_b(files)
    status = launch(at)
    assert len(status) == 2 * len(C.METHODS)
    assert {r["status"] for r in status} == {"ok"}, [(r["sex"], r["method"], r["detail"]) for r in status if r["status"] != "ok"]
    assert all(r["converged"] and np.isfinite(r["loglik"]) for r in status)
    res = at.session_state["fit_results"]
    spans = {"zone1": (0, 20), "zone2": (20, 85), "zone3": (85, 110)}
    for sex in BOTH:
        assert sorted(res[sex]) == sorted(C.METHODS)
        for key, r in res[sex].items():
            lo, hi = spans[C.METHODS[key].zone]
            assert (r.method, r.zone) == (key, C.METHODS[key].zone) and r.params["n_params"] > 0
            np.testing.assert_array_equal(r.ages, np.arange(lo, hi + 1))
            assert r.mu_hat.shape == (hi - lo + 1,) and np.all(np.isfinite(r.mu_hat)) and np.all(r.mu_hat > 0)
        assert res[sex]["beard"].loglik >= res[sex]["kannisto"].loglik - 1e-6          # nested, through the app


def test_launch_calls_each_sex_with_its_own_rows_and_its_own_config(files, truth, monkeypatch):
    block_zone_modules(monkeypatch)                                              # only the stand-in below exists
    calls = {}
    install(monkeypatch, "gompertz", "zone2_gompertz", "zone2", calls=calls)
    at = loaded_option_b(files)
    slider(at, "ω₁").set_value(25)                                               # Female: zone II is 25..85
    at.run()
    seg(at, "Jeu de données à paramétrer").set_value("Masculin")
    at.run()                                                                      # Male keeps the defaults: 20..85
    status = launch(at)

    assert [r["status"] for r in status if r["method"] == "gompertz"] == ["ok", "ok"]
    (f_ages, f_dx, _, f_cfg), (m_ages, m_dx, _, m_cfg) = calls["gompertz"]       # Female first, then Male
    assert (f_ages.min(), f_ages.max(), f_cfg.omega1) == (25, 85, 25)
    assert (m_ages.min(), m_ages.max(), m_cfg.omega1) == (20, 85, 20)
    np.testing.assert_array_equal(f_dx, truth[("Female", 2020)]["Dx"].iloc[25:86])
    np.testing.assert_array_equal(m_dx, truth[("Male", 2020)]["Dx"].iloc[20:86])
    res = at.session_state["fit_results"]
    assert list(res["Female"]) == ["gompertz"] and list(res["Male"]) == ["gompertz"]
    assert isinstance(res["Male"]["gompertz"], C.FitResult)


def test_a_failing_method_is_reported_and_the_others_still_run(files, monkeypatch):
    block_zone_modules(monkeypatch)
    install(monkeypatch, "makeham", "zone2_makeham", "zone2", fail=True)
    install(monkeypatch, "gompertz", "zone2_gompertz", "zone2")
    install(monkeypatch, "poly3", "zone1_poly", "zone1", wrong_type=True)
    at = loaded_option_b(files)
    status = launch(at)
    for sex in ("Féminin", "Masculin"):
        got = {r["method"]: r for r in status if r["sex"] == sex}
        assert got["makeham"]["status"] == "erreur" and "Hessian singular" in got["makeham"]["detail"]
        assert got["poly3"]["status"] == "erreur" and "FitResult" in got["poly3"]["detail"]
        assert got["gompertz"]["status"] == "ok"
    assert {s: list(r) for s, r in at.session_state["fit_results"].items()} == {
        "Female": ["gompertz"], "Male": ["gompertz"]}


def test_a_fitresult_that_names_the_wrong_method_is_rejected(files, monkeypatch):
    mod = types.ModuleType("src.zone2_gompertz")
    mod.fit_gompertz = fake_fit("makeham", "zone2")                              # copy-paste slip
    monkeypatch.setitem(sys.modules, "src.zone2_gompertz", mod)
    at = loaded_option_b(files)
    row = [r for r in launch(at) if r["method"] == "gompertz"][0]
    assert row["status"] == "erreur" and "incohérent" in row["detail"]


def test_a_module_without_the_fit_function_counts_as_not_implemented(files, monkeypatch):
    monkeypatch.setitem(sys.modules, "src.zone2_gompertz", types.ModuleType("src.zone2_gompertz"))
    at = loaded_option_b(files)
    row = [r for r in launch(at) if r["method"] == "gompertz"][0]
    assert row["status"] == "non implémenté" and "fit_gompertz" in row["detail"]


def test_a_new_load_discards_what_was_computed_from_the_previous_one(files, monkeypatch):
    install(monkeypatch, "gompertz", "zone2_gompertz", "zone2")
    at = loaded_option_b(files)
    launch(at)
    assert at.session_state["fit_results"]["Female"]
    at.session_state["selected_z2"] = "gompertz"
    at.run()
    goto(at, "chargement")
    upload(at, "up_female", files["female_csv"])
    upload(at, "up_male", files["male_csv"])
    at.run()
    load(at)                                                                     # 'Charger' again
    assert at.session_state["fit_results"] == {} and at.session_state["fit_status"] == []
    assert at.session_state["selected_z2"] is None
    assert at.session_state["config"] == {}                   # reset; the settings page rebuilds it when it is shown


# --------------------------------------------------------------------------
# The real France HMD files through the app (skipped if they are absent)
# --------------------------------------------------------------------------

_POST3 = Path(os.environ.get(
    "MORTALITY_POST3_DIR",
    r"C:\Users\simar\OneDrive\Bureau\Persos\LinkedIn\Life_Risks_Series\Post3-!Smoothing1"))
_EXPOSURE, _RATES = _POST3 / "HMDB" / "FRA_Exposures_1x1.txt", _POST3 / "HMDB" / "FRATNP_Mx_1x1.txt"
_INPUTDB = _POST3 / "HMDB" / "FRATNPdeath.txt"


@pytest.mark.skipif(not (_EXPOSURE.exists() and _RATES.exists()), reason="France HMD files not found")
def test_france_hmd_through_the_app_both_sexes_with_the_spec_defaults_and_no_noise():
    """Exposure + Rates, 209 years in the files, ages 0-109 once the open group is out: both sexes
    in one table, omega1 = 20 / omega2 = 85 / omega3 = 110 (spec 2.4) for each, no warning."""
    at = start()
    at.segmented_control[0].set_value(MODE_A).run()
    upload(at, "up_exposure", _EXPOSURE)
    upload(at, "up_rates", _RATES)
    at.run()
    year = [w for w in at.selectbox if w.label == "Année"][0]
    assert year.value == 2024 and len(year.options) == 209
    year.select(2020)
    at.run()
    load(at)
    df = at.session_state["df_raw"]
    assert df.attrs["sexes"] == ["Female", "Male"] and len(df) == 220 and df.attrs["year"] == 2020
    assert (df["age"].min(), df["age"].max()) == (0, 109)
    goto(at, "parametrage")

    for label, sex in (("Féminin", "Female"), ("Masculin", "Male")):       # open each dataset, as a user does
        seg(at, "Jeu de données à paramétrer").set_value(label)
        at.run()
        cfg = at.session_state["config"][sex]
        assert (cfg.omega1, cfg.omega2, cfg.omega3) == (20, 85, 110)
        assert not any("extrapolée" in w.value for w in at.warning)
        assert not at.button(key="btn_run").disabled


@pytest.mark.skipif(not (_INPUTDB.exists() and _EXPOSURE.exists()), reason="France HMD files not found")
def test_france_inputdb_is_refused_with_the_files_to_use_instead():
    at = start()
    at.segmented_control[0].set_value(MODE_A).run()
    upload(at, "up_deaths", _INPUTDB)
    at.run()
    assert any("InputDB" in e.value and "Exposures_1x1" in e.value for e in at.error)


# --------------------------------------------------------------------------
# Tab 3: evaluation, selection, assembly (spec 5.3, 7.3 step 4)
# --------------------------------------------------------------------------


def evaluated(files):
    """Option B files loaded and every method estimated, for both sexes."""
    at = loaded_option_b(files)
    launch(at)
    return goto(at, "evaluation")


def trace_values(arr):
    """A plotly trace array as numbers: the JSON of a figure stores numpy arrays as {dtype, bdata}."""
    if isinstance(arr, dict):
        return np.frombuffer(base64.b64decode(arr["bdata"]), dtype=np.dtype(arr["dtype"]))
    return np.asarray(arr)


def tab3_table(at):
    return at.main.dataframe[0].value


def validate(at):
    ensure(at, "evaluation")
    at.button(key="btn_validate").click().run()
    assert not at.exception, [e.value for e in at.exception]


def evaluator_table(at, sex):
    ss = at.session_state
    df = ss["df_raw"]
    return EV.evaluate_fits(ss["fit_results"][sex], df[df["sex"] == sex], ss["fit_config"][sex])


def test_tab3_asks_to_launch_the_estimations_first(files):
    at = loaded_option_b(files, page="evaluation")
    assert any("Lancez d'abord les estimations" in i.value for i in at.main.info)
    assert not at.main.dataframe and not at.main.button and not at.main.selectbox
    assert at.session_state["validated_selection"] is None and at.session_state["fit_config"] == {}


def test_tab3_shows_the_table_of_every_criterion_for_each_sex_and_one_suggestion_per_zone(files):
    at = evaluated(files)
    t = tab3_table(at)
    assert list(t.columns) == [EV.CRITERIA_LABELS[c] for c in EV.TABLE_COLUMNS if c not in ("method",)]
    assert len(t) == 9 and list(t["Méthode"]) == [m.label for m in C.METHODS.values()]
    assert t.groupby("Zone")["Suggérée"].sum().eq(1).all() and t.groupby("Zone")["ΔAIC"].min().eq(0).all()
    np.testing.assert_allclose(t["AIC"], -2 * t["Log-vraisemblance"] + 2 * t["p"], rtol=1e-12)
    expected = evaluator_table(at, "Female")                                     # what the evaluator says about these fits
    np.testing.assert_allclose(t["AIC"], expected["aic"], rtol=1e-12)
    assert list(t["Suggérée"]) == list(expected["suggested"])

    seg(at, "Jeu de données").set_value("Masculin").run()
    male = tab3_table(at)
    np.testing.assert_allclose(male["AIC"], evaluator_table(at, "Male")["aic"], rtol=1e-12)
    assert not np.allclose(male["AIC"], t["AIC"])


def test_tab3_draws_one_residual_line_and_one_fitted_curve_per_method(files):
    at = evaluated(files)
    specs = [json.loads(c.proto.spec) for c in at.main.get("plotly_chart")]
    assert [sp["layout"]["title"]["text"] for sp in specs] == ["Résidus de déviance", "Taux bruts et taux lissés"]
    assert [d["name"] for d in specs[0]["data"]] == [m.label for m in C.METHODS.values()]
    assert [d["name"] for d in specs[1]["data"]] == ["Taux bruts"] + [m.label for m in C.METHODS.values()]
    assert specs[1]["layout"]["yaxis"]["type"] == "log"                          # spec 5.3: log scale
    zone3 = [d for d in specs[1]["data"] if d["name"] == "Kannisto"][0]
    ages = trace_values(zone3["x"])
    assert ages[0] == 85 and ages[-1] == 110 and len(ages) == 26                 # the curve runs to omega3, past the data


def test_tab3_menus_start_on_the_suggestion_and_validation_assembles_both_sexes(files):
    at = evaluated(files)
    boxes = at.main.selectbox
    assert [b.label for b in boxes] == ["Méthode finale — Zone I", "Méthode finale — Zone II", "Méthode finale — Zone III"]
    suggested = {m.zone: m.key for m, ok in zip(C.METHODS.values(), evaluator_table(at, "Female")["suggested"]) if ok}
    assert [b.value for b in boxes] == [suggested[z] for z in C.ZONES]
    assert at.session_state["mu_joint"] is None and at.session_state["life_table"] is None

    validate(at)
    ss = at.session_state
    assert set(ss["mu_joint"]) == set(ss["life_table"]) == set(ss["validated_selection"]) == set(BOTH)
    for sex in BOTH:
        pick = ss["validated_selection"][sex]
        want = J.join_zones(*(ss["fit_results"][sex][pick[z]] for z in C.ZONES), ss["fit_config"][sex])
        np.testing.assert_array_equal(ss["mu_joint"][sex], want)                   # the joiner, on the chosen results
        pd.testing.assert_frame_equal(ss["life_table"][sex], J.build_life_table(want))
        assert ss["mu_joint"][sex].shape == (111,) and list(ss["life_table"][sex].columns) == J.LIFE_TABLE_COLUMNS
    male_suggestion = {m.zone: m.key for m, ok in zip(C.METHODS.values(), evaluator_table(at, "Male")["suggested"]) if ok}
    assert ss["validated_selection"]["Male"] == male_suggestion                  # a sex never opened here: the suggestion
    assert ss["selected_z1"] == {"Female": suggested["zone1"]} and ss["selected_z3"] == {"Female": suggested["zone3"]}
    assert any("Sélection validée" in m.value and "féminin" in m.value for m in at.main.success)


def test_tab3_a_changed_menu_flags_the_stale_table_and_validating_again_rebuilds_it(files):
    at = evaluated(files)
    validate(at)
    before = at.session_state["mu_joint"]["Female"].copy()
    at.main.selectbox[1].select("gompertz").run()                             # zone II: Gompertz instead of the suggestion
    assert any("ne correspondent plus" in w.value for w in at.main.warning)
    assert not any("Sélection validée" in m.value for m in at.main.success)
    assert at.session_state["selected_z2"]["Female"] == "gompertz"
    validate(at)
    assert at.session_state["validated_selection"]["Female"]["zone2"] == "gompertz"
    assert not np.array_equal(at.session_state["mu_joint"]["Female"], before)
    assert not any("ne correspondent plus" in w.value for w in at.main.warning)
    assert any("Sélection validée" in m.value for m in at.main.success)


def test_tab3_the_evaluation_and_the_assembly_use_the_settings_of_the_estimation_not_tab_2_sliders(files):
    at = evaluated(files)
    n_zone1 = int(tab3_table(at)["n"].iloc[0])
    goto(at, "parametrage")
    slider(at, "ω₁").set_value(30)
    at.run()                                                                      # tab 2 now says omega1 = 30 for the women
    goto(at, "evaluation")
    ss = at.session_state
    assert ss["config"]["Female"].omega1 == 30 and ss["fit_config"]["Female"].omega1 == 20
    assert int(tab3_table(at)["n"].iloc[0]) == n_zone1                           # the evaluation still reads zone I as 0..20
    validate(at)
    pick = ss["validated_selection"]["Female"]
    want = J.join_zones(*(ss["fit_results"]["Female"][pick[z]] for z in C.ZONES), ss["fit_config"]["Female"])
    np.testing.assert_array_equal(ss["mu_joint"]["Female"], want)


def test_tab3_a_zone_without_an_estimated_method_blocks_only_that_sex(files):
    at = loaded_option_b(files)
    for key in ("kannisto", "beard", "coale_kisker"):                            # the women get no zone III method
        [c for c in at.checkbox if c.key == f"use_Female_{key}"][0].set_value(False)
    at.run()
    launch(at)
    goto(at, "evaluation")
    assert any("Aucune méthode estimée pour la Zone III" in w.value for w in at.main.warning)
    assert len(at.main.selectbox) == 2
    validate(at)
    ss = at.session_state
    assert set(ss["mu_joint"]) == {"Male"} and set(ss["validated_selection"]) == {"Male"}
    assert any("Féminin : aucune méthode estimée pour Zone III" in e.value for e in at.main.error)


def test_tab3_a_new_estimation_discards_the_assembled_table(files):
    at = evaluated(files)
    validate(at)
    assert at.session_state["life_table"] is not None
    launch(at)
    goto(at, "evaluation")
    ss = at.session_state
    assert ss["mu_joint"] is None and ss["life_table"] is None and ss["validated_selection"] is None
    assert not any("Sélection validée" in m.value for m in at.main.success)


def test_tab3_the_whittaker_henderson_h_reaches_the_table(files):
    at = evaluated(files)
    base = tab3_table(at)
    at.number_input(key="wh_h").set_value(100.0).run()
    again = tab3_table(at)
    np.testing.assert_allclose(again["WH fidélité"], base["WH fidélité"], rtol=1e-12)
    np.testing.assert_allclose(again["WH total"] - again["WH fidélité"], 100.0 * again["WH rugosité"], rtol=1e-9)
    assert (again["WH total"] > base["WH total"]).any()


def test_tab3_the_choice_made_for_one_sex_is_kept_while_the_other_sex_is_viewed(files):
    at = evaluated(files)
    at.main.selectbox[1].select("gompertz").run()                             # women, zone II
    seg(at, "Jeu de données").set_value("Masculin").run()
    at.main.selectbox[1].select("perks").run()                                # men, zone II
    seg(at, "Jeu de données").set_value("Féminin").run()
    assert at.main.selectbox[1].value == "gompertz"                           # the women's menu is where they left it
    assert at.session_state["selected_z2"] == {"Female": "gompertz", "Male": "perks"}
    validate(at)
    assert at.session_state["validated_selection"]["Female"]["zone2"] == "gompertz"
    assert at.session_state["validated_selection"]["Male"]["zone2"] == "perks"


def test_tab3_the_residual_chart_is_cut_at_plus_or_minus_8_and_the_hover_keeps_the_true_value(files):
    """The synthetic men: Coale-Kisker reaches +31 and Gompertz +11 (the women stay within +-8, nothing to cut)."""
    at = evaluated(files)
    seg(at, "Jeu de données").set_value("Masculin").run()
    spec = json.loads(at.main.get("plotly_chart")[0].proto.spec)
    for name, beyond in (("Coale-Kisker", 30.0), ("Gompertz", 10.0)):
        trace = [d for d in spec["data"] if d["name"] == name][0]
        shown, true = trace_values(trace["y"]), trace_values(trace["customdata"])
        assert true.max() > beyond and np.all(np.abs(shown) <= 8.0)               # the figure is cut, the hover is not
        np.testing.assert_allclose(shown, np.clip(true, -8.0, 8.0))
    within = [d for d in spec["data"] if d["name"] == "Makeham"][0]
    np.testing.assert_allclose(trace_values(within["y"]), trace_values(within["customdata"]))   # within +-8: untouched


# --------------------------------------------------------------------------
# Tab 4: final chart, preview and downloads (spec 6.3)
# --------------------------------------------------------------------------


@pytest.fixture
def downloads(monkeypatch):
    """Replace the download buttons by a recorder: AppTest cannot click a download, so the test calls the deferred builder
    itself, which is what Streamlit does when the user downloads. {label: {data, file_name, mime, disabled}}."""
    calls = {}

    def spy(self, label, data=None, file_name=None, mime=None, **kw):
        calls[label] = {"data": data, "file_name": file_name, "mime": mime, "disabled": kw.get("disabled", False)}
        return False

    monkeypatch.setattr(DeltaGenerator, "download_button", spy)
    return calls


def export_sex(at):
    """The sex switch of tab 4 (tab 3 has one with the same label, so it is found by its key)."""
    return [w for w in at.segmented_control if w.key == "export_sex"][0]


def validated(files):
    """Both sexes estimated and the suggested selection validated; the export page is shown."""
    at = evaluated(files)
    validate(at)
    return goto(at, "export")


def test_tab4_asks_to_validate_the_selection_first(files):
    at = goto(evaluated(files), "export")
    assert any("Validez d'abord la sélection" in i.value for i in at.main.info)
    assert not at.main.get("plotly_chart") and not at.main.get("download_button")


def test_tab4_shows_the_final_chart_the_chosen_methods_and_the_first_and_last_fifty_rows(files):
    at = validated(files)
    ss, t4 = at.session_state, at.main
    pick = ss["validated_selection"]["Female"]
    assert t4.caption[0].value == "Méthodes retenues — " + " · ".join(
        f"{C.ZONE_LABELS[z]} : {C.METHODS[pick[z]].label}" for z in C.ZONES)

    spec = json.loads(t4.get("plotly_chart")[0].proto.spec)
    assert [d["name"] for d in spec["data"]] == ["Taux bruts", "Lissé — Zone I", "Lissé — Zone II", "Lissé — Zone III"]
    mu, cfg = ss["mu_joint"]["Female"], ss["fit_config"]["Female"]
    for trace, (lo, hi) in zip(spec["data"][1:], [(0, 20), (20, 85), (85, 110)]):
        ages = trace_values(trace["x"])
        assert ages[0] == lo and ages[-1] == hi                                   # each segment includes its two bounds
        np.testing.assert_allclose(trace_values(trace["y"]), mu[ages.astype(int)], rtol=0)
    assert [trace["line"]["color"] for trace in spec["data"][1:]] == [X.ZONE_COLORS[z] for z in C.ZONES]
    assert sorted(shape["x0"] for shape in spec["layout"]["shapes"]) == [cfg.omega1, cfg.omega2]     # omega1, omega2 dashed
    assert spec["layout"]["yaxis"]["type"] == "log"

    first, last = (d.value for d in t4.dataframe)
    table = ss["life_table"]["Female"]
    assert list(first["age"]) == list(range(50)) and list(last["age"]) == list(range(61, 111))
    pd.testing.assert_frame_equal(first.reset_index(drop=True), table.head(50).reset_index(drop=True))
    pd.testing.assert_frame_equal(last.reset_index(drop=True), table.tail(50).reset_index(drop=True))
    assert [c.value for c in t4.caption[1:3]] == ["50 premières lignes", "50 dernières lignes"]


def test_tab4_the_three_downloads_hold_the_table_in_each_format(files, downloads):
    at = validated(files)
    table = at.session_state["life_table"]["Female"]
    assert set(downloads) == {"Excel (.xlsx)", "CSV", "Texte (format HMD)"}
    assert [downloads[k]["file_name"] for k in downloads] == ["table_lissee_feminin.xlsx", "table_lissee_feminin.csv", "table_lissee_feminin.txt"]
    assert downloads["Excel (.xlsx)"]["mime"].endswith("spreadsheetml.sheet") and downloads["CSV"]["mime"] == "text/csv"
    assert not any(d["disabled"] for d in downloads.values())

    csv = pd.read_csv(io.BytesIO(downloads["CSV"]["data"]()), float_precision="round_trip")
    pd.testing.assert_frame_equal(csv, table, check_dtype=False)
    txt = downloads["Texte (format HMD)"]["data"]().decode("utf-8")
    assert "smoothed life table (period 1x1)" in txt.split("\n")[0] and txt.split("\n")[2].split() == ["Age", "mu", "qx", "px", "lx", "dx", "ex"]
    values = np.loadtxt(io.StringIO(txt), skiprows=3)
    assert values.shape == (111, 7) and np.array_equal(values[:, 0], table["age"])

    from openpyxl import load_workbook
    wb = load_workbook(io.BytesIO(downloads["Excel (.xlsx)"]["data"]()))
    assert wb.sheetnames == ["Table_lissée", "Taux_bruts", "Paramètres", "Critères", "Graphique"]
    assert wb["Table_lissée"].max_row == 112 and len(wb["Graphique"]._images) == 1
    crit = pd.DataFrame(list(wb["Critères"].values)[1:], columns=list(wb["Critères"].values)[0])
    assert len(crit) == 9 and set(crit["Zone"]) == {"Zone I", "Zone II", "Zone III"}          # the same table as tab 3, labelled
    np.testing.assert_allclose(crit["AIC"], evaluator_table(at, "Female")["aic"], rtol=1e-9)
    params = pd.DataFrame(list(wb["Paramètres"].values)[1:], columns=list(wb["Paramètres"].values)[0])
    assert set(params.loc[params["Retenue"], "Clé"]) == set(at.session_state["validated_selection"]["Female"].values())


def test_tab4_the_controls_reach_the_files(files, downloads):
    at = validated(files)
    [m for m in at.multiselect if m.key == "export_cols_Female"][0].set_value(["qx", "ex"])
    seg(at, "Séparateur du CSV").set_value(";")
    seg(at, "Marque décimale du CSV").set_value(",")
    at.checkbox(key="export_multi_Female").set_value(True)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    table = at.session_state["life_table"]["Female"]
    csv = downloads["CSV"]["data"]().decode("utf-8")
    assert csv.splitlines()[0] == "age;qx;ex"
    back = pd.read_csv(io.StringIO(csv), sep=";", decimal=",", float_precision="round_trip")
    np.testing.assert_array_equal(back["qx"], table["qx"])
    assert np.loadtxt(io.StringIO(downloads["Texte (format HMD)"]["data"]().decode("utf-8")), skiprows=3).shape == (111, 3)

    from openpyxl import load_workbook
    wb = load_workbook(io.BytesIO(downloads["Excel (.xlsx)"]["data"]()))
    assert [c.value for c in wb["Table_lissée"][1]] == ["age", "qx", "ex"]                    # the choice reaches the table sheet
    assert wb.sheetnames[5:] == [f"Méthode_{k}" for k in C.METHODS]                           # the multi-method option


def test_tab4_the_excel_criteria_use_the_h_chosen_in_tab_3(files, downloads):
    at = validated(files)
    goto(at, "evaluation")
    at.number_input(key="wh_h").set_value(50.0).run()
    goto(at, "export")                                  # h is remembered although its widget is not drawn on this page
    from openpyxl import load_workbook
    wb = load_workbook(io.BytesIO(downloads["Excel (.xlsx)"]["data"]()))
    rows = list(wb["Critères"].values)
    crit = pd.DataFrame(rows[1:], columns=rows[0])
    np.testing.assert_allclose(crit["WH total"] - crit["WH fidélité"], 50.0 * crit["WH rugosité"], rtol=1e-9)


def test_tab4_a_bad_choice_is_explained_and_the_buttons_that_cannot_work_are_disabled(files, downloads):
    at = validated(files)
    [m for m in at.multiselect if m.key == "export_cols_Female"][0].set_value([])
    at.run()
    assert any("au moins une colonne" in e.value for e in at.main.error)
    assert all(d["disabled"] for d in downloads.values())
    [m for m in at.multiselect if m.key == "export_cols_Female"][0].set_value(["qx"])
    seg(at, "Séparateur du CSV").set_value(",")
    seg(at, "Marque décimale du CSV").set_value(",")
    at.run()
    assert any("ne peuvent pas être identiques" in e.value for e in at.main.error)
    assert downloads["CSV"]["disabled"] and not downloads["Excel (.xlsx)"]["disabled"] and not downloads["Texte (format HMD)"]["disabled"]


def test_tab4_the_real_download_buttons_accept_the_deferred_builders(files):
    at = validated(files)
    assert not at.exception, [e.value for e in at.exception]
    assert len(at.main.get("download_button")) == 3


def test_tab4_each_sex_has_its_own_table_and_file_name(files, downloads):
    at = validated(files)
    female = downloads["CSV"]["data"]()
    export_sex(at).set_value("Masculin").run()
    assert downloads["CSV"]["file_name"] == "table_lissee_masculin.csv" and downloads["CSV"]["data"]() != female
    male = pd.read_csv(io.BytesIO(downloads["CSV"]["data"]()), float_precision="round_trip")
    pd.testing.assert_frame_equal(male, at.session_state["life_table"]["Male"], check_dtype=False)
    assert at.main.caption[0].value.startswith("Méthodes retenues")

    df = at.session_state["df_raw"]
    raw_male = df[df["sex"] == "Male"].sort_values("age")
    spec = json.loads(at.main.get("plotly_chart")[0].proto.spec)                # the chart's points are the men's
    points = [d for d in spec["data"] if d["name"] == "Taux bruts"][0]
    positive = raw_male[raw_male["mx_raw"] > 0]
    np.testing.assert_allclose(trace_values(points["y"]), positive["mx_raw"].to_numpy())
    from openpyxl import load_workbook                                             # ... and so are the workbook's raw rates
    wb = load_workbook(io.BytesIO(downloads["Excel (.xlsx)"]["data"]()))
    np.testing.assert_allclose([r[1] for r in list(wb["Taux_bruts"].values)[1:]], raw_male["Dx"].to_numpy())
    crit = pd.DataFrame(list(wb["Critères"].values)[1:], columns=list(wb["Critères"].values)[0])
    np.testing.assert_allclose(crit["AIC"], evaluator_table(at, "Male")["aic"], rtol=1e-9)


def test_tab4_flags_a_selection_that_changed_after_the_validation(files):
    at = validated(files)
    assert not at.main.warning
    goto(at, "evaluation")
    at.main.selectbox[1].select("gompertz").run()
    goto(at, "export")
    assert any("ont changé depuis la validation" in w.value for w in at.main.warning)
    validate(at)
    goto(at, "export")
    assert not any("ont changé depuis la validation" in w.value for w in at.main.warning)


@pytest.mark.skipif(not (_EXPOSURE.exists() and _RATES.exists()), reason="France HMD files not found")
def test_france_2020_from_the_hmd_files_to_the_downloads(downloads):
    """The whole chain on the real tables: load, estimate the nine methods for both sexes, validate the suggested
    selection, export. The life expectancy at birth must be the one of France 2020 (HMD: 85.2 women, 79.2 men)."""
    at = start()
    at.segmented_control[0].set_value(MODE_A).run()
    upload(at, "up_exposure", _EXPOSURE)
    upload(at, "up_rates", _RATES)
    at.run()
    [w for w in at.selectbox if w.label == "Année"][0].select(2020)
    at.run()
    load(at)
    launch(at)
    validate(at)
    goto(at, "export")
    ss = at.session_state
    for sex, e0 in (("Female", 85.2), ("Male", 79.2)):
        assert ss["life_table"][sex]["ex"].iloc[0] == pytest.approx(e0, abs=0.2) and ss["mu_joint"][sex].shape == (111,)
    assert downloads["CSV"]["file_name"] == "table_lissee_feminin_2020.csv"
    txt = downloads["Texte (format HMD)"]["data"]().decode("utf-8")
    assert txt.split("\n")[2].split() == ["Year", "Age", "mu", "qx", "px", "lx", "dx", "ex"] and txt.split("\n")[3].split()[0] == "2020"
    export_sex(at).set_value("Masculin").run()
    assert downloads["Excel (.xlsx)"]["file_name"] == "table_lissee_masculin_2020.xlsx"
    from openpyxl import load_workbook
    wb = load_workbook(io.BytesIO(downloads["Excel (.xlsx)"]["data"]()))
    assert len(list(wb["Taux_bruts"].values)) == 111 and len(wb["Graphique"]._images) == 1


# --------------------------------------------------------------------------
# Asked for after the seven modules: omega2 >= omega1 + 1, the global spline, the link between the sexes
# --------------------------------------------------------------------------

GLOBAL_LABEL = "Spline globale (tous âges)"


def global_box(at):
    return [c for c in at.checkbox if c.label == GLOBAL_LABEL][0]


def perks_box(at):
    return [c for c in at.checkbox if c.label == "Perks"][0]


def test_tab2_omega2_starts_one_above_omega1_follows_it_and_keeps_its_value_otherwise(files):
    at = loaded_option_b(files)
    assert (slider(at, "ω₂").min, slider(at, "ω₂").value) == (21, 85)
    slider(at, "ω₁").set_value(90).run()
    w2 = slider(at, "ω₂")
    assert (w2.min, w2.max, w2.value) == (91, 100, 91)                          # omega1 pushed omega2
    cfg = at.session_state["config"]["Female"]
    assert (cfg.omega1, cfg.omega2) == (90, 91) and not at.error and not at.button(key="btn_run").disabled
    slider(at, "ω₁").set_value(30).run()
    w2 = slider(at, "ω₂")
    assert (w2.min, w2.value) == (31, 91)                                        # the floor moved down, omega2 stayed where it was
    slider(at, "ω₂").set_value(50).run()
    cfg = at.session_state["config"]["Female"]
    assert (cfg.omega1, cfg.omega2) == (30, 50)
    slider(at, "ω₁").set_value(49).run()
    assert (slider(at, "ω₂").min, slider(at, "ω₂").value) == (50, 50)           # omega2 may sit right on the floor
    assert not at.exception


def test_tab2_the_global_spline_is_a_checkbox_off_by_default_that_reaches_the_config(files):
    at = loaded_option_b(files)
    assert not global_box(at).value and GLOBAL_LABEL not in [c.label for c in method_boxes(at)]
    assert at.session_state["config"]["Female"].all_methods == []
    global_box(at).check()
    at.run()
    assert at.session_state["config"]["Female"].all_methods == ["spline_all"]
    assert not [w for w in at.warning if "Aucune méthode cochée" in w.value]
    global_box(at).uncheck()
    at.run()
    assert at.session_state["config"]["Female"].all_methods == []


def test_tab2_a_global_spline_alone_covers_the_three_zones_without_a_warning(files):
    at = loaded_option_b(files)
    for c in method_boxes(at):
        c.uncheck()
    at.run()
    assert any("Cochez au moins une méthode" in e.value for e in at.error)
    global_box(at).check()
    at.run()
    assert not at.error and not [w for w in at.warning if "Aucune méthode cochée" in w.value]
    assert not at.button(key="btn_run").disabled


def test_launch_fits_the_global_spline_once_on_all_the_ages_for_the_sex_that_ticked_it(files):
    at = loaded_option_b(files)
    global_box(at).check()
    at.run()
    status = launch(at)
    res = at.session_state["fit_results"]
    assert "spline_all" in res["Female"] and "spline_all" not in res["Male"]
    g = res["Female"]["spline_all"]
    assert (g.method, g.zone) == ("spline_all", "all")
    np.testing.assert_array_equal(g.ages, np.arange(0, 101))                         # every loaded age, not a zone
    assert g.params["n_params"] == len(g.params["knots"]) and g.converged
    rows_ = [r for r in status if r["method"] == "spline_all"]
    assert [(r["sex"], r["zone"], r["label"], r["status"]) for r in rows_] == [("Féminin", "Tous âges", GLOBAL_LABEL, "ok")]
    assert len(status) == 10 + 9


def test_tab2_linking_the_sexes_copies_the_configuration_and_keeps_the_two_together(files):
    at = loaded_option_b(files)
    slider(at, "ω₁").set_value(30)
    slider(at, "σ").set_value(5.0)
    perks_box(at).uncheck()
    global_box(at).check()
    at.run()
    assert set(at.session_state["config"]) == {"Female"}                              # Male not visited yet
    at.checkbox(key="link_cfg").check()
    at.run()
    cfg = at.session_state["config"]
    assert set(cfg) == {"Female", "Male"} and cfg["Male"] == cfg["Female"] and cfg["Male"] is not cfg["Female"]
    assert (cfg["Male"].omega1, cfg["Male"].sigma_join, "perks" in cfg["Male"].z2_methods, cfg["Male"].all_methods) == (30, 5.0, False, ["spline_all"])
    assert cfg["Male"].z2_methods is not cfg["Female"].z2_methods

    seg(at, "Jeu de données à paramétrer").set_value("Masculin").run()              # the men's widgets show what was copied
    assert slider(at, "ω₁").value == 30 and slider(at, "σ").value == 5.0
    assert not perks_box(at).value and global_box(at).value
    slider(at, "ω₁").set_value(40)                                                    # and a change on the men reaches the women
    [c for c in at.checkbox if c.label == "Gompertz"][0].uncheck()
    at.run()
    cfg = at.session_state["config"]
    assert cfg["Female"].omega1 == 40 and "gompertz" not in cfg["Female"].z2_methods and cfg["Female"] == cfg["Male"]

    at.checkbox(key="link_cfg").uncheck()
    at.run()
    slider(at, "ω₁").set_value(50)
    at.run()
    cfg = at.session_state["config"]
    assert (cfg["Male"].omega1, cfg["Female"].omega1) == (50, 40)                     # independent again
    assert not at.exception


def test_tab2_a_single_sex_has_nothing_to_link(files):
    at = start()
    upload(at, "up_male", files["male_csv"])
    at.run()
    load(at)
    goto(at, "parametrage")
    assert not [c for c in at.checkbox if c.key == "link_cfg"]
    assert list(at.session_state["config"]) == ["Male"]


def test_tab2_linking_never_copies_a_configuration_onto_a_sex_that_has_fewer_than_three_ages(files, truth):
    tiny = S.write_per_sex_csv(files["dir"] / "tiny_male.csv", truth[("Male", 2020)].drop(columns="mu_true").iloc[:2])
    at = start()
    upload(at, "up_female", files["female_csv"])
    upload(at, "up_male", tiny)
    at.run()
    load(at)
    assert sorted(at.session_state["df_raw"].attrs["sexes"]) == ["Female", "Male"]
    goto(at, "parametrage")
    at.checkbox(key="link_cfg").check()
    at.run()
    assert not at.exception and set(at.session_state["config"]) == {"Female"}         # no zones can be defined for two ages
    seg(at, "Jeu de données à paramétrer").set_value("Masculin").run()
    assert any("Moins de 3 âges" in e.value for e in at.error) and not at.exception


def test_tab2_linked_sexes_are_both_estimated_with_the_copied_settings(files):
    at = loaded_option_b(files)
    global_box(at).check()
    perks_box(at).uncheck()
    at.checkbox(key="link_cfg").check()
    at.run()
    launch(at)
    res = at.session_state["fit_results"]
    for sex in BOTH:
        assert "spline_all" in res[sex] and "perks" not in res[sex] and len(res[sex]) == 9
    assert at.session_state["fit_config"]["Male"] == at.session_state["fit_config"]["Female"]


def tab3_global_rows(at):
    t = tab3_table(at)
    return t[t["Méthode"] == GLOBAL_LABEL]


def test_tab3_the_global_spline_has_a_row_per_zone_and_is_a_choice_in_each_menu(files):
    at = loaded_option_b(files)
    global_box(at).check()
    at.checkbox(key="link_cfg").check()
    at.run()
    launch(at)
    goto(at, "evaluation")
    t = tab3_table(at)
    assert len(t) == 12 and list(tab3_global_rows(at)["Zone"]) == ["Zone I", "Zone II", "Zone III"]
    assert t.groupby("Zone")["Suggérée"].sum().eq(1).all() and t.groupby("Zone")["ΔAIC"].min().eq(0).all()
    ks = list(tab3_global_rows(at)["p"])
    knots = at.session_state["fit_results"]["Female"]["spline_all"].params["knots"]
    assert ks == [sum(lo <= k <= hi for k in knots) for lo, hi in ((0, 20), (20, 85), (85, 100))]
    for box in at.main.selectbox:
        assert GLOBAL_LABEL in box.options
    specs = [json.loads(c.proto.spec) for c in at.main.get("plotly_chart")]
    assert [d["name"] for d in specs[0]["data"]].count(GLOBAL_LABEL) == 1            # one line per method, not one per zone
    assert [d["name"] for d in specs[1]["data"]].count(GLOBAL_LABEL) == 1


def test_tab3_the_global_spline_chosen_in_the_three_zones_gives_the_global_curve_as_the_final_table(files):
    at = loaded_option_b(files)
    global_box(at).check()
    at.checkbox(key="link_cfg").check()
    at.run()
    launch(at)
    goto(at, "evaluation")
    for box in at.main.selectbox:
        box.select("spline_all")
    at.run()
    validate(at)
    ss = at.session_state
    assert ss["validated_selection"]["Female"] == {z: "spline_all" for z in C.ZONES}
    g = ss["fit_results"]["Female"]["spline_all"]
    np.testing.assert_allclose(ss["mu_joint"]["Female"][:101], g.mu_hat, rtol=1e-12)
    goto(at, "export")
    assert at.main.caption[0].value == "Méthodes retenues — " + " · ".join(f"{C.ZONE_LABELS[z]} : {GLOBAL_LABEL}" for z in C.ZONES)
    assert not at.exception


def test_tab3_linking_the_sexes_copies_the_picked_methods_in_both_directions(files):
    at = evaluated(files)
    assert not at.checkbox(key="link_pick").value
    at.checkbox(key="link_pick").check()
    at.run()
    ss = at.session_state
    for i in (1, 2, 3):
        assert ss[f"selected_z{i}"]["Male"] == ss[f"selected_z{i}"]["Female"]       # ticking copies what the women's menus say
    at.main.selectbox[1].select("gompertz")
    at.run()
    assert ss["selected_z2"] == {"Female": "gompertz", "Male": "gompertz"}
    export_view = seg(at, "Jeu de données")
    export_view.set_value("Masculin").run()
    assert at.main.selectbox[1].value == "gompertz"                                 # the men's menu shows the copied choice
    at.main.selectbox[2].select("beard")
    at.run()
    assert ss["selected_z3"] == {"Female": "beard", "Male": "beard"}
    validate(at)
    assert ss["validated_selection"]["Female"] == ss["validated_selection"]["Male"]
    assert ss["validated_selection"]["Male"]["zone2"] == "gompertz" and ss["validated_selection"]["Male"]["zone3"] == "beard"

    at.checkbox(key="link_pick").uncheck()
    at.main.selectbox[1].select("perks")
    at.run()
    assert ss["selected_z2"] == {"Female": "gompertz", "Male": "perks"}               # independent again


def test_tab3_a_method_the_other_sex_did_not_estimate_is_not_copied_and_the_page_says_so(files):
    at = loaded_option_b(files)
    global_box(at).check()                                                              # the women only
    at.run()
    launch(at)
    goto(at, "evaluation")
    assert "spline_all" in at.session_state["fit_results"]["Female"] and "spline_all" not in at.session_state["fit_results"]["Male"]
    at.checkbox(key="link_pick").check()
    for box in at.main.selectbox:
        box.select("spline_all")
    at.run()
    ss = at.session_state
    assert all(ss[f"selected_z{i}"]["Female"] == "spline_all" for i in (1, 2, 3))
    assert all(ss[f"selected_z{i}"].get("Male") != "spline_all" for i in (1, 2, 3))
    assert any("Non reporté" in c.value and "masculin" in c.value for c in at.main.caption)
    validate(at)
    assert ss["validated_selection"]["Female"] == {z: "spline_all" for z in C.ZONES}
    assert "spline_all" not in ss["validated_selection"]["Male"].values()


def test_tab3_the_global_spline_picked_for_one_sex_is_copied_when_both_sexes_estimated_it(files):
    at = loaded_option_b(files)
    global_box(at).check()
    at.checkbox(key="link_cfg").check()
    at.run()
    launch(at)
    goto(at, "evaluation")
    at.checkbox(key="link_pick").check()
    at.main.selectbox[1].select("spline_all")
    at.run()
    ss = at.session_state
    assert ss["selected_z2"] == {"Female": "spline_all", "Male": "spline_all"}
    assert not any("Non reporté" in c.value for c in at.main.caption)


def cols_box(at, sex):
    return [m for m in at.multiselect if m.key == f"export_cols_{sex}"][0]


def test_tab4_the_export_options_belong_to_each_sex_until_they_are_linked(files, downloads):
    at = validated(files)
    cols_box(at, "Female").set_value(["qx", "ex"])
    at.run()
    assert at.session_state["export_params"]["Female"]["cols"] == ["qx", "ex"]
    assert downloads["CSV"]["data"]().decode("utf-8").splitlines()[0] == "age,qx,ex"
    export_sex(at).set_value("Masculin").run()
    assert cols_box(at, "Male").value == X.TABLE_COLUMNS[1:]                          # the men were not touched
    assert downloads["CSV"]["data"]().decode("utf-8").splitlines()[0] == "age,mu,qx,px,lx,dx,ex"
    export_sex(at).set_value("Féminin").run()
    assert cols_box(at, "Female").value == ["qx", "ex"]                               # and the women kept theirs


def test_tab4_linking_copies_the_options_to_the_other_sex_and_keeps_them_together(files, downloads):
    at = validated(files)
    cols_box(at, "Female").set_value(["qx", "ex"])
    seg(at, "Séparateur du CSV").set_value(";")
    seg(at, "Marque décimale du CSV").set_value(",")
    at.checkbox(key="export_multi_Female").set_value(True)
    at.run()
    at.checkbox(key="link_export").check()
    at.run()
    want = {"cols": ["qx", "ex"], "sep": ";", "decimal": ",", "multi": True}
    assert at.session_state["export_params"] == {"Female": want, "Male": want}
    export_sex(at).set_value("Masculin").run()
    assert cols_box(at, "Male").value == ["qx", "ex"] and at.checkbox(key="export_multi_Male").value
    assert seg(at, "Séparateur du CSV").value == ";" and seg(at, "Marque décimale du CSV").value == ","
    csv = downloads["CSV"]["data"]().decode("utf-8")
    assert downloads["CSV"]["file_name"] == "table_lissee_masculin.csv" and csv.splitlines()[0] == "age;qx;ex"
    back = pd.read_csv(io.StringIO(csv), sep=";", decimal=",", float_precision="round_trip")
    np.testing.assert_array_equal(back["qx"], at.session_state["life_table"]["Male"]["qx"])

    cols_box(at, "Male").set_value(["dx"])                                            # a change on the men reaches the women
    at.run()
    assert at.session_state["export_params"]["Female"]["cols"] == ["dx"]
    at.checkbox(key="link_export").uncheck()
    cols_box(at, "Male").set_value(["mu"])
    at.run()
    assert at.session_state["export_params"]["Male"]["cols"] == ["mu"] and at.session_state["export_params"]["Female"]["cols"] == ["dx"]
    assert not at.exception


def test_tab4_a_single_sex_has_nothing_to_link(files):
    at = start()
    upload(at, "up_male", files["male_csv"])
    at.run()
    load(at)
    launch(at)
    validate(at)
    goto(at, "export")
    assert not [c for c in at.checkbox if c.key in ("link_export", "link_pick", "link_cfg")]
    assert cols_box(at, "Male").value == X.TABLE_COLUMNS[1:]


# --------------------------------------------------------------------------
# Pages: what a page remembers, the banners
# --------------------------------------------------------------------------


def banner_svg(at):
    """The SVG of the first banner of the page shown, decoded from its data: URI."""
    html = at.main.get("html")[0].proto.body
    return base64.b64decode(re.search(r'base64,([A-Za-z0-9+/=]+)"', html).group(1)).decode("utf-8")


def test_every_page_opens_on_a_banner_with_its_step_and_title(files):
    at = loaded_option_b(files, page="chargement")
    for page, step, title in (("chargement", 1, "Chargement et contrôle des données"), ("parametrage", 2, "Paramétrage et lancement"),
                              ("evaluation", 3, "Évaluation et sélection"), ("export", 4, "Visualisation et export")):
        goto(at, page)
        body = at.main.get("html")[0].proto.body
        assert 'class="rdl-header"' in body and f"Étape {step} sur 4" in body and title in body


def test_the_home_page_shows_the_brand_and_the_two_choices():
    at = AppTest.from_file(APP, default_timeout=60).run()
    body = at.main.get("html")[0].proto.body
    assert "Mortality Table Smoother" in body and "Exploring risk through data" in body and "Data Lab" in body
    assert [b.label for b in at.button] == ["Lisser des tables du moment", "Lisser des tables générationnelles"]


def test_each_of_the_four_steps_of_the_home_page_has_its_own_pictogram():
    at = AppTest.from_file(APP, default_timeout=60).run()
    strip = [h.proto.body for h in at.main.get("html") if "rdl-steps" in h.proto.body][0]
    tiles = strip.split('class="rdl-step-tile"')[1:]
    assert len(tiles) == 4 and all(t.count("<img") == 1 for t in tiles)
    for tile, name in zip(tiles, ("Chargement", "Paramétrage", "Évaluation", "Export")):
        assert name in tile and tile.index("<img") < tile.index(name)                   # the drawing comes first, then the step
    images = [re.search(r'base64,([A-Za-z0-9+/=]+)"', t).group(1) for t in tiles]
    assert len(set(images)) == 4                                                    # four different drawings


def test_using_the_home_page_raises_the_flag_that_lets_the_sidebar_recall_the_choice():
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert at.session_state["view_chosen"] is False
    at.button(key="home_cohort").click().run()
    assert at.session_state["view_chosen"] is True and at.session_state["view_choice"] == "cohort"


def test_the_home_sidebar_recalls_the_table_type_once_one_was_chosen():
    at = AppTest.from_file(APP, default_timeout=60)
    at.session_state["view_choice"], at.session_state["view_chosen"] = "cohort", True
    at.run()
    assert "Tables générationnelles" in sidebar_html(at)
    at.button(key="home_cohort").click().run()                                    # choosing sets the flag for good
    assert at.session_state["view_chosen"] is True and "Tables générationnelles" in sidebar_html(at)


def test_the_settings_banner_draws_the_boundaries_the_sliders_say(files):
    at = loaded_option_b(files)
    slider(at, "ω₁").set_value(30).run()                                        # one slider at a time, as a user moves them
    slider(at, "ω₂").set_value(70).run()
    xs = [float(x) for x in re.findall(r'<line x1="([\d.]+)"', banner_svg(at))]
    omega3 = at.session_state["config"]["Female"].omega3
    assert xs == [pytest.approx(10 + w / omega3 * 320, abs=0.06) for w in (30, 70)]


def test_a_page_remembers_its_choices_while_another_page_is_shown(files):
    at = loaded_option_b(files)                                                 # settings page
    seg(at, "Jeu de données à paramétrer").set_value("Masculin")
    at.checkbox(key="link_cfg").check()
    at.run()
    goto(at, "evaluation")
    goto(at, "parametrage")
    assert seg(at, "Jeu de données à paramétrer").value == "Masculin" and at.checkbox(key="link_cfg").value

    launch(at)
    goto(at, "evaluation")
    seg(at, "Jeu de données").set_value("Masculin")
    at.checkbox(key="link_pick").check()
    at.number_input(key="wh_h").set_value(50.0)
    at.run()
    goto(at, "export")
    goto(at, "evaluation")
    assert seg(at, "Jeu de données").value == "Masculin" and at.checkbox(key="link_pick").value
    assert at.number_input(key="wh_h").value == 50.0

    validate(at)
    goto(at, "export")
    export_sex(at).set_value("Masculin").run()
    at.checkbox(key="link_export").check()
    at.run()
    goto(at, "evaluation")
    goto(at, "export")
    assert export_sex(at).value == "Masculin" and at.checkbox(key="link_export").value


def test_the_loading_page_remembers_the_layout_of_the_files():
    at = start(MODE_A)
    goto(at, "parametrage")
    goto(at, "chargement")
    assert at.segmented_control[0].value == MODE_A and [u.key for u in at.file_uploader] == ["up_deaths", "up_exposure", "up_rates"]


# --------------------------------------------------------------------------
# Charts: the legend is a column beside the plot, never on the title
# --------------------------------------------------------------------------


def figure_specs(at):
    return [json.loads(c.proto.spec) for c in at.main.get("plotly_chart")]


def test_the_evaluation_charts_keep_their_legend_beside_the_plot_whatever_the_number_of_methods(files):
    at = loaded_option_b(files)
    global_box(at).check()
    at.checkbox(key="link_cfg").check()
    at.run()
    launch(at)
    goto(at, "evaluation")
    specs = figure_specs(at)
    assert [len(sp["data"]) for sp in specs] == [10, 11]                          # residuals: 10 methods; fits: the crude rates too
    for spec in specs:
        layout, names = spec["layout"], [d["name"] for d in spec["data"]]
        assert layout["legend"]["orientation"] == "v" and layout["legend"]["x"] > 1 and layout["legend"]["y"] == 1
        assert layout["margin"]["t"] == 70 and layout["margin"]["r"] >= int(56 + 7.2 * max(len(n) for n in names))
        assert layout["height"] - layout["margin"]["t"] - layout["margin"]["b"] >= 20 * len(names)           # the column fits beside the plot


def test_the_two_evaluation_charts_are_one_under_the_other_not_in_columns(files):
    at = evaluated(files)
    assert len(at.main.get("plotly_chart")) == 2
    assert not [chart for column in at.columns for chart in column.get("plotly_chart")]              # full width: the legend needs room


def test_the_final_chart_has_the_legend_beside_the_plot_too(files):
    at = validated(files)
    (spec,) = figure_specs(at)
    assert spec["layout"]["legend"]["orientation"] == "v" and spec["layout"]["legend"]["x"] > 1
    assert spec["layout"]["margin"]["r"] >= int(56 + 7.2 * len("Lissé — Zone III"))
