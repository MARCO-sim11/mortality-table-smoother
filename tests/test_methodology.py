"""views/methodologie.py and ui/methodology.py - the methodology page. Run:   python -m pytest tests/test_methodology.py -v

The page shows the documents of docs/methodes/ and fits any method, on demand, to the France 2020 sample. The curves, the
parameters and the log-likelihoods it shows are compared with the zone modules called directly.
"""
from __future__ import annotations

import base64
import importlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from src import config as C
from src import loader as L
from ui import methodology as M

ROOT = Path(__file__).resolve().parent.parent
APP = str(ROOT / "app.py")
KEYS = [d.key for d in M.all_docs() if d.key]
LABELS = {"Female": "Féminin", "Male": "Masculin"}


def page() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=120)
    at.switch_page("views/methodologie.py").run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def values(arr) -> np.ndarray:
    """A plotly array as numbers (the JSON of a figure stores numpy arrays as {dtype, bdata})."""
    if isinstance(arr, dict):
        return np.frombuffer(base64.b64decode(arr["bdata"]), dtype=np.dtype(arr["dtype"]))
    return np.asarray(arr, dtype=float)


def direct_fit(key: str, sex: str):
    """The method fitted by calling its module, on the rows the app would give it: the oracle of the page."""
    df = L.load_data(ROOT / "sample_data" / "FRA_taux_bruts_moment_2020_female_male.csv")
    cfg = C.default_config(df["age"].unique())
    spec = C.method_spec(key)
    sub = C.zone_slice(df[df["sex"] == sex], cfg, spec.zone)
    fit = getattr(importlib.import_module(f"src.{spec.module}"), spec.fit_name)
    return fit(sub["age"].to_numpy(), sub["Dx"].to_numpy(), sub["Ex"].to_numpy(), cfg), cfg, df[df["sex"] == sex]


# --------------------------------------------------------------------------
# The page
# --------------------------------------------------------------------------


def test_the_page_has_a_tab_per_group_and_a_sub_tab_per_method():
    at = page()
    expected = []
    for name, docs in M.GROUPS:
        expected += [name] + ([d.label for d in docs] if len(docs) > 1 else [])
    assert [t.label for t in at.tabs] == expected
    assert [d.label for d in M.all_docs() if d.key] == [C.method_spec(k).label for k in KEYS]        # tab = the label used elsewhere in the app


def test_the_page_shows_every_document_as_it_is_on_disk_and_in_the_page_order():
    at = page()
    assert [m.value for m in at.main.markdown] == [M.page_text(d.slug).strip() for d in M.all_docs()]      # Streamlit trims the ends
    assert len(at.main.markdown) == 14
    assert not any("<!--" in m.value for m in at.main.markdown)             # the markers of the generated tables are not printed


def test_the_page_opens_on_a_reference_banner_not_a_step():
    body = page().main.get("html")[0].proto.body
    assert 'class="rdl-header"' in body and "Référence" in body and "Méthodologie" in body and "Étape" not in body


def test_the_sidebar_links_the_methodology_under_the_four_steps_and_above_the_home_page():
    at = page()
    links = [(p.proto.label, p.proto.page) for p in at.sidebar.get("page_link")]
    assert links[-3:] == [("4 · Export", "export"), ("Méthodologie", "methodologie"), ("Accueil", "")]
    assert len(links) == 6


def test_the_home_page_points_to_the_methodology():
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert any(p.proto.page == "methodologie" and "Méthodologie" in p.proto.label for p in at.main.get("page_link"))


def test_the_illustration_is_off_until_asked_for():
    at = page()
    assert not at.main.get("plotly_chart") and not at.metric
    assert [t.value for t in at.toggle] == [False] * len(KEYS) and len(at.toggle) == len(KEYS)


# --------------------------------------------------------------------------
# The live illustration
# --------------------------------------------------------------------------


@pytest.mark.parametrize("key", KEYS)
def test_a_method_fitted_on_the_page_is_the_fit_of_its_module_on_the_france_2020_rows(key):
    at = page()
    at.toggle(key=f"meth_show_{key}").set_value(True).run()
    assert not at.exception, [e.value for e in at.exception]
    result, cfg, df_sex = direct_fit(key, "Female")
    (chart,) = at.main.get("plotly_chart")
    spec = json.loads(chart.proto.spec)
    assert [d["name"] for d in spec["data"]] == ["Taux bruts", C.method_spec(key).label]
    low, high = M.zone_span(key, cfg, int(df_sex["age"].min()))
    keep = (result.ages >= low) & (result.ages <= high)
    np.testing.assert_allclose(values(spec["data"][1]["x"]), result.ages[keep])
    np.testing.assert_allclose(values(spec["data"][1]["y"]), result.mu_hat[keep], rtol=1e-9)
    crude = df_sex[(df_sex["age"] >= low) & (df_sex["age"] <= high) & (df_sex["mx_raw"] > 0)]
    np.testing.assert_allclose(values(spec["data"][0]["y"]), crude["mx_raw"].to_numpy())
    assert spec["layout"]["yaxis"]["type"] == "log" and spec["layout"]["xaxis"]["range"] == [low - 1, high + 1]
    assert [m.value for m in at.metric] == [f"{result.loglik:,.1f}", str(result.params["n_params"]), "oui" if result.converged else "non",
                                            str(result.n_iter)]
    shown = at.main.get("dataframe")[-1].value
    assert list(shown["Paramètre"]) == list(result.params)


def test_the_sex_switch_of_a_method_changes_the_fit_to_the_men():
    at = page()
    at.toggle(key="meth_show_gompertz").set_value(True).run()
    women = json.loads(at.main.get("plotly_chart")[0].proto.spec)
    at.segmented_control(key="meth_sex_gompertz").set_value("Masculin").run()
    men = json.loads(at.main.get("plotly_chart")[0].proto.spec)
    assert "féminin" in women["layout"]["title"]["text"] and "masculin" in men["layout"]["title"]["text"]
    result, _, _ = direct_fit("gompertz", "Male")
    np.testing.assert_allclose(values(men["data"][1]["y"]), result.mu_hat, rtol=1e-9)
    assert not np.allclose(values(men["data"][1]["y"]), values(women["data"][1]["y"]))


def test_each_method_has_its_own_toggle_so_that_fitting_one_does_not_fit_the_others():
    at = page()
    at.toggle(key="meth_show_poly3").set_value(True).run()
    assert len(at.main.get("plotly_chart")) == 1 and len(at.metric) == 4
    at.toggle(key="meth_show_beard").set_value(True).run()
    assert len(at.main.get("plotly_chart")) == 2 and len(at.metric) == 8


def test_a_missing_sample_file_is_explained_not_a_crash(monkeypatch):
    monkeypatch.setattr(M, "SAMPLE", Path("no_such_directory") / "FRA.csv")
    at = page()
    notes = [i.value for i in at.main.info if "introuvable" in i.value]
    assert len(notes) == len(KEYS) and not at.toggle


# --------------------------------------------------------------------------
# The helpers
# --------------------------------------------------------------------------


@pytest.mark.parametrize("key,span", [("poly3", (0, 20)), ("spline", (0, 20)), ("gompertz", (20, 85)), ("perks", (20, 85)),
                                      ("kannisto", (85, 110)), ("coale_kisker", (85, 110)), ("spline_all", (0, 110))])
def test_the_illustration_shows_the_zone_of_the_method_or_everything(key, span):
    assert M.zone_span(key, C.SmootherConfig(), 0) == span


def test_a_table_that_does_not_start_at_zero_starts_the_global_illustration_at_its_first_age():
    assert M.zone_span("spline_all", C.SmootherConfig(), 40) == (40, 110)
    assert M.zone_span("poly3", C.SmootherConfig(), 40) == (40, 20)


def test_the_parameter_table_writes_a_list_parameter_on_one_line():
    result, _, _ = direct_fit("spline", "Female")
    table = M.parameter_table(result)
    assert list(table.columns) == ["Paramètre", "Valeur"] and list(table["Paramètre"]) == ["knots", "ln_mu_knots", "n_params"]
    assert table.loc[0, "Valeur"] == "0, 5, 10, 15, 20" and table.loc[2, "Valeur"] == "5"
    assert len(table.loc[1, "Valeur"].split(", ")) == 5


def test_the_sample_table_is_the_loader_s_without_its_report():
    df = M.sample_table()
    ref = L.load_data(ROOT / "sample_data" / "FRA_taux_bruts_moment_2020_female_male.csv")
    pd.testing.assert_frame_equal(df, ref.reset_index(drop=True), check_like=False)
    assert df.attrs == {} and ref.attrs                                    # the report is left out of the cached copy


def test_every_document_slug_is_a_file_and_every_group_is_a_tab_label():
    assert all((M.DOC_DIR / f"{d.slug}.md").is_file() for d in M.all_docs())
    assert [name for name, _ in M.GROUPS] == ["Cadre général", "Zone I", "Zone II", "Zone III", "Tous âges", "Raccord et évaluation"]
