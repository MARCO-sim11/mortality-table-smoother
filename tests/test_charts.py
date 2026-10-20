"""ui/evaluation.py - the layout of the evaluation charts. Run:   python -m pytest tests/test_charts.py -v

The legend of a chart of ten methods used to open upward over the title, then (moved under the plot) wrapped one entry per row in a
chart a few hundred pixels wide. It is a column to the right of the plot now; the margins are the proof.
"""
from __future__ import annotations

import plotly.graph_objects as go
import pytest

from src import config as C
from ui import evaluation as UE

CFG = C.SmootherConfig()


def figure(n_traces: int, hidden: int = 0, name: str = "Méthode") -> go.Figure:
    fig = go.Figure()
    for i in range(n_traces):
        label = f"{name} {i}" + (" with a very long name that no legend shows" if i < hidden else "")
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name=label, showlegend=i >= hidden))
    return fig


def test_the_legend_width_follows_the_longest_name_and_is_nothing_without_entries():
    assert UE.legend_width([]) == 0
    assert UE.legend_width(["a", "bb"]) == int(UE.LEGEND_PAD + UE.LEGEND_CHAR * 2)
    assert UE.legend_width(["Beard", "Spline globale (tous âges)"]) == int(UE.LEGEND_PAD + UE.LEGEND_CHAR * len("Spline globale (tous âges)"))


@pytest.mark.parametrize("n", [1, 4, 10, 12, 20])
def test_the_legend_is_a_column_right_of_the_plot_and_never_reaches_the_title(n):
    lay = UE._style_figure(figure(n), CFG, "Titre", "Axe").layout
    assert lay.legend.orientation == "v" and lay.legend.xanchor == "left" and lay.legend.x > 1.0       # outside the plot, on the right
    assert lay.legend.yanchor == "top" and lay.legend.y == 1.0                                          # from the top of the plot down
    assert lay.margin.t == UE.TITLE_MARGIN >= 60 and lay.title.text == "Titre"                           # the title has the band above
    plot = lay.height - lay.margin.t - lay.margin.b
    assert plot >= UE.LEGEND_ROW * n and plot >= 320                                                     # the column fits beside the plot
    assert lay.margin.r >= UE.legend_width(f"Méthode {i}" for i in range(n))                             # and the margin holds its width


def test_a_longer_name_widens_the_right_margin():
    short = UE._style_figure(figure(3, name="A"), CFG, "T", "A").layout
    long = UE._style_figure(figure(3, name="Spline naturelle globale sur tous les âges"), CFG, "T", "A").layout
    assert long.margin.r > short.margin.r


def test_the_plot_grows_when_there_are_more_entries_than_its_default_height_holds():
    few = UE._style_figure(figure(5), CFG, "T", "A").layout
    many = UE._style_figure(figure(30), CFG, "T", "A").layout
    assert few.height == UE.TITLE_MARGIN + 320 + UE.BOTTOM_MARGIN
    assert many.height == UE.TITLE_MARGIN + UE.LEGEND_ROW * 30 + UE.BOTTOM_MARGIN > few.height


def test_traces_that_are_not_in_the_legend_do_not_count():
    some_hidden = UE._style_figure(figure(8, hidden=5, name="Long method name"), CFG, "T", "A").layout
    assert some_hidden.margin.r == UE.legend_width(["Long method name 5", "Long method name 6", "Long method name 7"])
    nothing = go.Figure()
    nothing.add_trace(go.Scatter(x=[0, 1], y=[0, 1], showlegend=False))
    assert UE._style_figure(nothing, CFG, "T", "A").layout.margin.r == 10


def test_the_boundaries_are_drawn_dashed_and_the_axes_titled():
    fig = UE._style_figure(figure(2), CFG, "T", "μ")
    xs = sorted(shape.x0 for shape in fig.layout.shapes)
    assert xs == [CFG.omega1, CFG.omega2] and fig.layout.xaxis.title.text == "Âge" and fig.layout.yaxis.title.text == "μ"
