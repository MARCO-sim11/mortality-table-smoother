"""Methodology page helpers: the method documents of docs/methodes/ and a live illustration of each method.

The documents are plain Markdown files (formulas in LaTeX between dollar signs, which GitHub and GitLab render as well as Streamlit);
the page only reads and displays them, so the files stay the single source. A document that describes a method carries a button
to fit that method, on demand, to the France 2020 table of `sample_data/` with the app's default settings: the page then shows the curve
against the crude rates, the estimated parameters and what the fit reported.
"""
from __future__ import annotations

import importlib
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import config as C
from src import loader as L
from ui.common import SEX_COLOR, SEX_LABEL
from ui.evaluation import _style_figure
from ui.loading import plain

ROOT = Path(__file__).resolve().parent.parent
DOC_DIR = ROOT / "docs" / "methodes"
SAMPLE = ROOT / "sample_data" / "FRA_taux_bruts_moment_2020_female_male.csv"
SEXES = ("Female", "Male")


@dataclass(frozen=True)
class Doc:
    """One document: its file (`<slug>.md` in docs/methodes/), its tab label, and the method it describes (None for the
    general ones: such a document has no illustration)."""
    slug: str
    label: str
    key: str | None = None


def _method(slug: str, key: str) -> Doc:
    return Doc(slug, C.method_spec(key).label, key)


# (tab of the page, the documents under it), in the order of the page
GROUPS: tuple = (
    ("Cadre général", (Doc("00-cadre-general", "Cadre général"),)),
    ("Zone I", (_method("01-polynome-degre-3", "poly3"), _method("02-spline-naturelle", "spline"))),
    ("Zone II", (_method("03-gompertz", "gompertz"), _method("04-makeham", "makeham"),
                 _method("05-makeham-2e-ordre", "makeham2"), _method("06-perks", "perks"))),
    ("Zone III", (_method("07-kannisto", "kannisto"), _method("08-beard", "beard"), _method("09-coale-kisker", "coale_kisker"))),
    ("Tous âges", (_method("10-spline-globale", "spline_all"),)),
    ("Raccord et évaluation", (Doc("11-raccord", "Raccord des zones"), Doc("12-evaluation", "Évaluation et sélection"),
                               Doc("13-table-de-vie", "Table de vie"))),
)


def all_docs() -> list:
    return [doc for _, docs in GROUPS for doc in docs]


def read_doc(slug: str) -> str:
    """The text of a document, as in the file (read at each run: the files are small and may be edited while the app runs)."""
    return (DOC_DIR / f"{slug}.md").read_text(encoding="utf-8")


def page_text(slug: str) -> str:
    """The text of a document as the page shows it: without the HTML comments that mark the generated tables. GitHub and GitLab
    hide such comments, Streamlit's Markdown prints them."""
    return re.sub(r"<!--.*?-->\n?", "", read_doc(slug), flags=re.S)


# --------------------------------------------------------------------------
# The live illustration
# --------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def sample_table() -> pd.DataFrame:
    """France 2020, both sexes, as the loader reads the file of sample_data/ (without the loader's report, which is not displayable)."""
    return plain(L.load_data(SAMPLE))


@st.cache_data(show_spinner="Ajustement en cours…")
def sample_fit(key: str, sex: str) -> tuple:
    """(FitResult, SmootherConfig) of method `key` fitted to one sex of the sample with the default settings (omega 20 / 85 / 110).

    The global spline is fitted on all the ages, a zone method on the rows of its zone, exactly as the app's launch does."""
    df = sample_table()
    cfg = C.default_config(df["age"].unique())
    spec = C.method_spec(key)
    sub = C.zone_slice(df[df["sex"] == sex], cfg, spec.zone)
    fit = getattr(importlib.import_module(f"src.{spec.module}"), spec.fit_name)
    return fit(sub["age"].to_numpy(), sub["Dx"].to_numpy(), sub["Ex"].to_numpy(), cfg), cfg


def zone_span(key: str, cfg: C.SmootherConfig, first_age: int) -> tuple:
    """(low, high) ages the illustration shows for method `key`: its zone, or everything for the global spline."""
    low, high = cfg.zone_bounds(C.method_spec(key).zone)
    return (first_age if low is None else low), high


def illustration_figure(key: str, sex: str, result: C.FitResult, cfg: C.SmootherConfig, df_sex: pd.DataFrame) -> go.Figure:
    """The crude rates of the method's zone (points, log scale) and its fitted curve (line)."""
    spec = C.method_spec(key)
    low, high = zone_span(key, cfg, int(df_sex["age"].min()))
    crude = df_sex[(df_sex["age"] >= low) & (df_sex["age"] <= high) & (df_sex["mx_raw"] > 0)]
    color, symbol = SEX_COLOR[sex]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=crude["age"], y=crude["mx_raw"], mode="markers", name="Taux bruts",
                             marker=dict(size=6, color=color, symbol=symbol, line=dict(width=1, color="white"))))
    ages, mu = pd.Series(result.ages), pd.Series(result.mu_hat)
    keep = (ages >= low) & (ages <= high) & (mu > 0) & mu.notna()
    fig.add_trace(go.Scatter(x=ages[keep], y=mu[keep], mode="lines", name=spec.label, line=dict(color="#134A7C", width=2.4)))
    fig = _style_figure(fig, cfg, f"{spec.label} — {SEX_LABEL[sex].lower()}, France 2020", "μₓ (échelle logarithmique)")
    fig.update_yaxes(type="log")
    fig.update_xaxes(range=[low - 1, high + 1])
    return fig


def parameter_table(result: C.FitResult) -> pd.DataFrame:
    """The estimated parameters as two columns, a list parameter (the spline's knots) written on one line."""
    rows = []
    for name, value in result.params.items():
        if isinstance(value, (list, tuple)):
            rows.append((name, ", ".join(f"{v:.5g}" for v in value)))
        else:
            rows.append((name, f"{value:.6g}"))
    return pd.DataFrame(rows, columns=["Paramètre", "Valeur"])
