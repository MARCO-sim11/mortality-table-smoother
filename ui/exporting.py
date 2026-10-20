"""Export page helpers: final chart, downloads (spec 6)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src import config as C
from src import evaluator as EV
from src import exporter as X
from ui.common import SEX_COLOR
from ui.evaluation import _style_figure, evaluation_labelled

# --------------------------------------------------------------------------
# Tab 4 helpers: final chart, downloads (spec 6)
# --------------------------------------------------------------------------

SEX_FILE = {"Female": "feminin", "Male": "masculin"}
PREVIEW_ROWS = 50                                    # spec 6.3: the 50 first and the 50 last lines of the table
EXPORT_DEFAULTS = {"cols": X.TABLE_COLUMNS[1:], "sep": ",", "decimal": ".", "multi": False}   # options of a sex not seen yet
PREVIEW_FORMATS = {"age": "%d", "mu": "%.4e", "qx": "%.4e", "px": "%.6f", "lx": "%.2f", "dx": "%.2f", "ex": "%.3f"}


def final_figure(df_sex: pd.DataFrame, mu_joint, cfg: C.SmootherConfig, sex: str) -> go.Figure:
    """Spec 6.3: the crude rates (points), the assembled curve coloured by zone (I to omega1, II to omega2, III beyond;
    each segment includes its two boundary ages, so the colours meet) and the boundaries omega1, omega2 dashed. Log scale."""
    mu = np.asarray(mu_joint, dtype=float)
    ages = np.arange(mu.size)
    fig = go.Figure()
    pos = df_sex[df_sex["mx_raw"] > 0]
    color, symbol = SEX_COLOR[sex]
    fig.add_trace(go.Scatter(x=pos["age"], y=pos["mx_raw"], mode="markers", name="Taux bruts",
                             marker=dict(size=6, color=color, symbol=symbol, line=dict(width=1, color="white"))))
    for zone, lo, hi in (("zone1", -1, cfg.omega1), ("zone2", cfg.omega1, cfg.omega2), ("zone3", cfg.omega2, mu.size)):
        keep = np.isfinite(mu) & (ages >= lo) & (ages <= hi)
        fig.add_trace(go.Scatter(x=ages[keep], y=mu[keep], mode="lines", name=f"Lissé — {C.ZONE_LABELS[zone]}",
                                 line=dict(color=X.ZONE_COLORS[zone], width=2.4)))
    fig = _style_figure(fig, cfg, "Taux bruts et table lissée", "μₓ (échelle logarithmique)", plot_height=360)
    fig.update_yaxes(type="log")
    return fig


def csv_payload(table: pd.DataFrame, columns, sep: str, decimal: str) -> bytes:
    return X.to_csv(table, columns, sep, decimal).encode("utf-8")


def hmd_payload(table: pd.DataFrame, columns, title: str, year, cohort) -> bytes:
    return X.to_hmd_txt(table, columns, title=title, year=year, cohort=cohort).encode("utf-8")


def excel_payload(df_sex, cfg, fits, selection, table, mu, sex, columns, multi, title, h) -> bytes:
    """The workbook, built when the download is asked for (criteria included, recomputed from the fits)."""
    criteria = evaluation_labelled(EV.evaluate_fits(fits, df_sex, cfg, h))
    return X.final_workbook(df_sex, cfg, fits, selection, table, mu, sex, criteria, columns=columns,
                            include_methods=multi, title=title)
