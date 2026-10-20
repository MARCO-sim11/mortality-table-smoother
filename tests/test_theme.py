"""ui/theme.py - the Risk Data Lab look. Run:   python -m pytest tests/test_theme.py -v

The drawings are SVG strings: they are parsed as XML and their coordinates are checked against the age and the log scale they stand for.
"""
from __future__ import annotations

import base64
import re
import xml.etree.ElementTree as ET

from pathlib import Path

import numpy as np
import pytest

from src import loader as L
from ui import common, theme

ROOT = Path(__file__).resolve().parent.parent

NS = {"s": "http://www.w3.org/2000/svg"}


def parse(svg: str):
    return ET.fromstring(svg)


def points(piece) -> list:
    return [tuple(map(float, pair.split(","))) for pair in piece.get("points").split()]


def test_the_logo_and_the_pictograms_are_well_formed_svg():
    for svg in (theme.logo_svg(40), theme._pictogram_svg("period"), theme._pictogram_svg("cohort")):
        root = parse(svg)
        assert root.tag == "{http://www.w3.org/2000/svg}svg" and root.get("viewBox")
    logo = parse(theme.logo_svg(40))
    assert len(logo.findall("s:rect", NS)) == 3 and len(logo.findall("s:circle", NS)) == 4     # three bars; the disc and three dots


def test_an_svg_goes_into_an_img_with_a_data_uri_that_decodes_back():
    svg = theme.logo_svg(40)
    html = theme.svg_img(svg, "width:10px", "alt text")
    data = re.search(r'src="data:image/svg\+xml;base64,([^"]+)"', html).group(1)
    assert base64.b64decode(data).decode("utf-8") == svg and 'alt="alt text"' in html and 'style="width:10px"' in html


def test_the_curve_has_three_coloured_pieces_cut_at_the_boundaries_on_a_log_scale():
    root = parse(theme.zone_curve_svg(20, 85, 110, width=560, height=170, labels=False))
    pieces = root.findall("s:polyline", NS)
    assert [p.get("stroke") for p in pieces] == list(theme.ZONE_TONES)
    assert [len(points(p)) for p in pieces] == [21, 66, 26]               # ages 0..20, 20..85, 85..110: boundary ages are shared
    xs = np.array([x for p in pieces for x, _ in points(p)])
    assert xs.min() == pytest.approx(10.0) and xs.max() == pytest.approx(550.0)            # 10 px margins either side
    log_mu = np.log10(theme.mortality_mu(np.arange(0, 111)))
    expected = 14 + (log_mu.max() - log_mu) / (log_mu.max() - log_mu.min()) * (170 - 14 - 10)
    drawn = np.array([y for p in pieces for _, y in points(p)])
    ages = np.array([*range(0, 21), *range(20, 86), *range(85, 111)])
    np.testing.assert_allclose(drawn, expected[ages], atol=0.06)           # every vertex at its own age on the log scale
    boundaries = [float(line.get("x1")) for line in root.findall("s:line", NS)]
    assert boundaries == [pytest.approx(10 + 20 / 110 * 540, abs=0.06), pytest.approx(10 + 85 / 110 * 540, abs=0.06)]


def test_the_curve_is_always_drawable_whatever_the_boundaries():
    for w1, w2, w3 in ((0, 1, 110), (200, 300, 110), (50, 40, 110), (5, 6, 7), (20, 85, 2)):
        pieces = parse(theme.zone_curve_svg(w1, w2, w3)).findall("s:polyline", NS)
        assert len(pieces) == 3 and all(len(points(p)) >= 2 for p in pieces)


def test_the_labels_and_the_dots_are_optional():
    plain = parse(theme.zone_curve_svg(labels=False))
    assert not plain.findall("s:text", NS) and not plain.findall("s:circle", NS)
    labelled = parse(theme.zone_curve_svg(labels=True, dots=2.0))
    assert [t.text for t in labelled.findall("s:text", NS)] == ["Zone I", "Zone II", "Zone III"]
    assert len(labelled.findall("s:circle", NS)) == 110                                    # ages 0..109 have a crude rate, 110 has none
    assert len(parse(theme.zone_curve_svg(omega3=130, dots=2.0)).findall("s:circle", NS)) == 110    # none is invented beyond the data
    assert len(parse(theme.zone_curve_svg(omega3=60, dots=2.0)).findall("s:circle", NS)) == 61


def test_the_dots_are_the_crude_rates_placed_on_the_log_scale_that_also_holds_the_curve():
    width, height, labels = 560, 170, True
    root = parse(theme.zone_curve_svg(20, 85, 110, width=width, height=height, labels=labels, dots=2.0))
    circles = root.findall("s:circle", NS)
    cy = np.array([float(c.get("cy")) for c in circles])
    cx = np.array([float(c.get("cx")) for c in circles])
    log_rates = np.log10(theme.CRUDE_RATES)
    log_curve = np.log10(theme.mortality_mu(np.arange(0, 111)))
    high, low = max(log_rates.max(), log_curve.max()), min(log_rates.min(), log_curve.min())
    top, bottom = 14, 26
    np.testing.assert_allclose(cy, top + (high - log_rates) / (high - low) * (height - top - bottom), atol=0.06)
    np.testing.assert_allclose(cx, 10 + np.arange(110) / 110 * (width - 20), atol=0.06)
    assert cy.min() >= top - 0.06 and cy.max() <= height - bottom + 0.06                  # every dot inside the drawing area


def test_the_crude_rates_of_the_illustration_are_the_france_2020_ones_of_the_sample_file():
    df = L.load_data(ROOT / "sample_data" / "FRA_taux_bruts_moment_2020_female_male.csv")
    total = df.groupby("age")[["Dx", "Ex"]].sum().loc[range(110)]
    np.testing.assert_allclose(theme.CRUDE_RATES, (total["Dx"] / total["Ex"]).to_numpy(), rtol=6e-3)       # 3 significant digits
    assert len(theme.CRUDE_RATES) == 110 and min(theme.CRUDE_RATES) > 0


def test_the_mortality_curve_has_the_shape_of_a_life_table_with_its_young_adult_hump():
    ages = np.arange(0, 111)
    mu = theme.mortality_mu(ages)
    low = int(np.argmin(mu))
    assert 5 <= low <= 11 and mu[0] > 20 * mu[low]                         # the infant fall, then a minimum in childhood
    assert mu[22] > 3 * mu[low]                                            # the hump: back up to several times the minimum by 22
    growth_hump, growth_old = mu[30] / mu[20], mu[70] / mu[60]
    assert growth_hump < 0.75 * growth_old                                 # ... and a shoulder: 20 to 30 rises much less than old age does
    assert (np.diff(mu[30:]) > 0).all() and mu[110] < 1.2                  # then it rises for good, and stays below one


def test_the_law_follows_the_crude_rates_it_was_fitted_to_hump_included():
    mu = theme.mortality_mu(np.arange(0, 101))
    residual = np.log10(mu) - np.log10(theme.CRUDE_RATES[:101])
    assert np.sqrt(np.mean(residual ** 2)) < 0.1                           # a tenth of a decade, about 25 %, all over 0..100
    for lo, hi in ((0, 5), (5, 15), (15, 30), (30, 60), (60, 100)):        # and no stretch is off by more than 0.25 decade on average
        assert abs(residual[lo:hi + 1].mean()) < 0.25, (lo, hi)
    crude = np.array(theme.CRUDE_RATES)
    assert crude[22] > 3 * crude[:12].min()                                # the hump is in the data itself


def test_the_brand_palette_is_the_one_of_the_validated_charts():
    assert (common.NAVY, common.BLUE, common.GREEN, common.SKY) == ("#134A7C", "#1B6CA8", "#3EB553", "#35A8E0")
    assert common.SEX_COLOR["Female"][0] == common.BLUE and common.SEX_COLOR["Male"][0] == common.GREEN


def test_the_kind_of_table_reads_as_a_phrase():
    assert theme.view_phrase("period") == "tables du moment" and theme.view_phrase("cohort") == "tables générationnelles"


def test_every_step_has_a_well_formed_and_different_pictogram():
    kinds = ("load", "settings", "evaluate", "export")
    drawings = [theme._step_svg(k) for k in kinds]
    assert len(set(drawings)) == 4
    for svg in drawings:
        assert parse(svg).get("viewBox") == "0 0 64 44"
    assert all(theme.step_pictogram(k).startswith("<img") for k in kinds)

