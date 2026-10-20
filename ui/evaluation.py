"""Evaluation page helpers: evaluation table, charts, assembly (specs 5.3 and 4)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import config as C
from src import evaluator as EV
from src import joiner as J
from ui.common import SEX_COLOR, SEX_LABEL

# --------------------------------------------------------------------------
# Tab 3 helpers: evaluation table, charts, assembly (specs 5.3 and 4)
# --------------------------------------------------------------------------

METHOD_COLORS = dict(zip([m.key for m in C.every_method()],
                         ["#E69F00", "#56B4E9", "#009E73", "#D55E00", "#CC79A7",
                          "#0072B2", "#B8A000", "#000000", "#7F7F7F", "#6A3D9A"]))     # Okabe-Ito, one per method (+ a purple)
RESIDUAL_CLIP = 8.0                                   # residual chart: drawn within +-8, true value on hover
EVAL_FORMATS = {                                      # column label -> st.column_config format (numbers stay sortable)
    "p": "%d", "n": "%d", "ddl": "%d", "Runs": "%d", "Log-vraisemblance": "%.2f", "χ² Pearson": "%.1f",
    "p(χ²)": "%.3g", "Déviance": "%.1f", "AIC": "%.2f", "ΔAIC": "%.2f", "BIC": "%.2f", "p(runs)": "%.3f",
    "|r| > 2": "percent", "WH fidélité": "%.3e", "WH rugosité": "%.3e", "WH total": "%.3e"}


def evaluation_tables(sex: str, h: float):
    """(comparison table, residual series) of one sex: the fits of the last run, read with the settings they were run with."""
    df = st.session_state.df_raw
    df_sex = df[df["sex"] == sex]
    results, cfg = st.session_state.fit_results[sex], st.session_state.fit_config[sex]
    return EV.evaluate_fits(results, df_sex, cfg, h), EV.residual_series(results, df_sex, cfg)


def evaluation_labelled(table: pd.DataFrame) -> pd.DataFrame:
    """The evaluator's table as one reads it: French column labels, zone names, no method key."""
    return table.drop(columns=["method"]).assign(zone=table["zone"].map(C.ZONE_LABELS)).rename(columns=EV.CRITERIA_LABELS)


def evaluation_display(table: pd.DataFrame):
    """The table as shown (see `evaluation_labelled`) and its column_config."""
    show = evaluation_labelled(table)
    config = {name: st.column_config.NumberColumn(name, format=fmt) for name, fmt in EVAL_FORMATS.items()
              if name in show.columns}
    return show, config


TITLE_MARGIN = 70                                     # px above the plot: the title only
BOTTOM_MARGIN = 60                                    # px under the plot: the tick labels and the x-axis title
LEGEND_ROW = 20                                       # px of one entry of the legend
LEGEND_CHAR = 7.2                                     # px per character of an entry name, at the default size
LEGEND_PAD = 56                                       # px of an entry besides its name: the line, the marker, the gaps


def legend_width(names) -> int:
    """Pixels the legend needs for the entries `names`: the longest name, plus the line and the marker. 0 for no entry."""
    names = list(names)
    return int(LEGEND_PAD + LEGEND_CHAR * max(len(name) for name in names)) if names else 0


def _style_figure(fig: go.Figure, cfg: C.SmootherConfig, title: str, ytitle: str, plot_height: int = 320) -> go.Figure:
    """Title above, the plot, and the legend as a column to the RIGHT of the plot, so that however many methods it lists it can never
    reach the title or the axes: the right margin is as wide as the longest method name needs, and the plot is at least tall enough
    for the column (one entry per `LEGEND_ROW` px). A horizontal legend over or under a chart a few hundred pixels wide wrapped
    one entry per row, past any margin that could be guessed in advance."""
    for omega in (cfg.omega1, cfg.omega2):
        fig.add_vline(x=omega, line=dict(color="grey", dash="dash", width=1))
    names = [trace.name for trace in fig.data if trace.showlegend is not False and trace.name]
    plot_height = max(plot_height, LEGEND_ROW * len(names))
    fig.update_layout(title=dict(text=title, x=0, xanchor="left", font=dict(size=15)), xaxis_title="Âge",
                      yaxis_title=ytitle, height=TITLE_MARGIN + plot_height + BOTTOM_MARGIN,
                      margin=dict(l=10, r=max(10, legend_width(names)), t=TITLE_MARGIN, b=BOTTOM_MARGIN),
                      legend=dict(orientation="v", x=1.01, xanchor="left", y=1.0, yanchor="top"))
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.2)")
    return fig


def residual_figure(rs: pd.DataFrame, cfg: C.SmootherConfig) -> go.Figure:
    """Spec 5.3: the deviance residuals by age, one line per method; the +-2 band of the methodology; the zone
    boundaries dashed. A residual beyond +-8 (Coale-Kisker on a big table) is drawn at the edge, its value on hover."""
    fig = go.Figure()
    for key in rs["method"].unique():
        g = rs[rs["method"] == key]
        label = g["label"].iloc[0]
        fig.add_trace(go.Scatter(
            x=g["age"], y=np.clip(g["residual"], -RESIDUAL_CLIP, RESIDUAL_CLIP), mode="lines+markers", name=label,
            line=dict(color=METHOD_COLORS[key], width=1.6), marker=dict(size=5), customdata=g["residual"],
            hovertemplate=f"{label} · âge %{{x}}<br>résidu = %{{customdata:.2f}}<extra></extra>"))
    for y in (-EV.RESIDUAL_BOUND, EV.RESIDUAL_BOUND):
        fig.add_hline(y=y, line=dict(color="grey", dash="dot", width=1))
    fig = _style_figure(fig, cfg, "Résidus de déviance", "Résidu de déviance")
    fig.update_yaxes(range=[-RESIDUAL_CLIP - 0.5, RESIDUAL_CLIP + 0.5])
    return fig


def fit_figure(results: dict, df_sex: pd.DataFrame, cfg: C.SmootherConfig, sex: str) -> go.Figure:
    """Spec 5.3: the crude rates (points) over the smoothed rates of every method (lines), log scale. A method's
    line covers the ages of its FitResult (zone III runs to omega3, past the last row of data)."""
    fig = go.Figure()
    pos = df_sex[df_sex["mx_raw"] > 0]
    color, symbol = SEX_COLOR[sex]
    fig.add_trace(go.Scatter(x=pos["age"], y=pos["mx_raw"], mode="markers", name="Taux bruts",
                             marker=dict(size=6, color=color, symbol=symbol, line=dict(width=1, color="white"))))
    for key, r in results.items():
        mu = np.asarray(r.mu_hat, dtype=float)
        ok = np.isfinite(mu) & (mu > 0)
        fig.add_trace(go.Scatter(x=np.asarray(r.ages)[ok], y=mu[ok], mode="lines", name=C.method_spec(key).label,
                                 line=dict(color=METHOD_COLORS[key], width=1.6)))
    fig = _style_figure(fig, cfg, "Taux bruts et taux lissés", "μₓ (échelle logarithmique)")
    fig.update_yaxes(type="log")
    return fig


def offered_in_zone(results: dict, zone: str) -> list:
    """The estimated methods of one sex that a zone's menu offers: those of the zone, and the global ones."""
    return [k for k in results if C.method_spec(k).zone in (zone, C.ZONE_ALL)]


def assemble_tables(results: dict, cfgs: dict, picks: dict):
    """Spec 7.3 step 4, per sex: `join_zones` on the three chosen results, then `build_life_table`.

    `picks` is {sex: {zone: method_key or None}}. Returns (mu_joint, life_table, validated, problems): the first
    three are {sex: ...} for the sexes that could be assembled, `problems` the French messages of those that could not
    (a zone with no estimated method, or a ValueError of the joiner)."""
    mu_joint, life_table, validated, problems = {}, {}, {}, []
    for sex, pick in picks.items():
        missing = [C.ZONE_LABELS[z] for z in C.ZONES if not pick.get(z)]
        if missing:
            problems.append(f"{SEX_LABEL[sex]} : aucune méthode estimée pour {', '.join(missing)}.")
            continue
        try:
            mu = J.join_zones(*(results[sex][pick[z]] for z in C.ZONES), cfgs[sex])
            life_table[sex] = J.build_life_table(mu)
        except ValueError as exc:
            problems.append(f"{SEX_LABEL[sex]} : {exc}")
            continue
        mu_joint[sex], validated[sex] = mu, dict(pick)
    return mu_joint, life_table, validated, problems
