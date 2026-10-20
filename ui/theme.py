"""Risk Data Lab look: CSS, the logo, the mortality-curve illustration, page banners, the sidebar.

Design "A - tableau de bord marine": a marine sidebar (on every page), a navy banner on top of each page with the title and
a mortality curve in three zones, light content. Everything is drawn here, as SVG strings, so the app ships no image file.

The illustration is a real mortality curve: France 2020, both sexes, from the Human Mortality Database (the crude rates of
`sample_data/`, kept here as `CRUDE_RATES`) and a smooth law fitted to them (`LAW`, see `mortality_mu`). It shows what the tool is for:
the infant fall, the childhood minimum, the young-adult hump, then the exponential rise of old age, cut at omega1 and omega2 into the
three zones. It is decoration: the colours are the logo's light tones, which read on navy, and differ from the CVD-checked zone colours
of the data charts (`exporter.ZONE_COLORS`).
"""
from __future__ import annotations

import base64

import numpy as np
import streamlit as st

from ui.common import GREEN, GREEN_DEEP, INK_SOFT, LIME, NAVY, NAVY_DEEP, NAVY_NIGHT, RULE, SKY, VIEW_LABEL

ZONE_TONES = (SKY, GREEN, LIME)                     # zone I, II, III on a navy background
APP_NAME = "Mortality Table Smoother"
TAGLINE = "Exploring risk through data"

# --------------------------------------------------------------------------
# Drawings
# --------------------------------------------------------------------------


def svg_img(svg: str, css: str = "", alt: str = "") -> str:
    """An SVG string as an <img> with a data: URI. st.html sanitises inline <svg> away; an <img> goes through."""
    data = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return f'<img alt="{alt}" src="data:image/svg+xml;base64,{data}" style="{css}">'


def logo_svg(size: int = 40) -> str:
    """The Risk Data Lab mark: navy disc, three green bars, a white trend line with its three dots."""
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 40 40">'
            f'<circle cx="20" cy="20" r="19" fill="{NAVY}"/>'
            f'<rect x="10" y="24" width="5" height="9" fill="{GREEN_DEEP}"/><rect x="17.5" y="20" width="5" height="13" fill="{GREEN}"/>'
            f'<rect x="25" y="15" width="5" height="18" fill="{LIME}"/>'
            f'<path d="M7 27 Q17 16 33 11" stroke="#fff" stroke-width="1.6" fill="none"/>'
            f'<circle cx="9" cy="26" r="2.6" fill="{SKY}"/><circle cx="20" cy="18.5" r="2.2" fill="#fff"/>'
            f'<circle cx="33" cy="11" r="2.8" fill="{LIME}"/></svg>')


def logo_mark(size: int = 40) -> str:
    """The mark as an <img> (see `svg_img`)."""
    return svg_img(logo_svg(size), f"width:{size}px;height:{size}px;display:block", "Risk Data Lab")


def wordmark(on_dark: bool = True, size: str = "1.05rem") -> str:
    """'Risk Data Lab' on one line, as the logo spells it (white and green on dark, navy and green on light)."""
    risk = "#FFFFFF" if on_dark else NAVY
    lab = GREEN if on_dark else GREEN_DEEP
    return f'<span style="font-size:{size};font-weight:600;letter-spacing:.01em;color:{risk}">Risk <span style="color:{lab}">Data Lab</span></span>'


# France 2020, both sexes (HMD): crude rates D / E at ages 0..109, three significant digits. Source of the dots of the illustration.
CRUDE_RATES = (
    0.00352, 0.0002, 0.000165, 8.46e-05, 9.71e-05, 6.61e-05, 6.06e-05, 5.98e-05, 6.77e-05, 6.65e-05,
    5.5e-05, 7.15e-05, 8.74e-05, 9.69e-05, 0.000107, 0.000128, 0.00016, 0.000197, 0.000331, 0.000318,
    0.000358, 0.000378, 0.000384, 0.000385, 0.000359, 0.000421, 0.00041, 0.000429, 0.000481, 0.000491,
    0.000505, 0.000579, 0.000506, 0.000662, 0.000691, 0.000732, 0.000772, 0.000787, 0.00087, 0.000958,
    0.00105, 0.00115, 0.00126, 0.00141, 0.00152, 0.00174, 0.0018, 0.00208, 0.00233, 0.00257,
    0.00281, 0.00314, 0.00338, 0.00363, 0.00395, 0.00431, 0.00471, 0.00528, 0.00567, 0.00619,
    0.00684, 0.00731, 0.00814, 0.00863, 0.00931, 0.01, 0.0108, 0.0113, 0.0124, 0.0132,
    0.0143, 0.0151, 0.0165, 0.0178, 0.0198, 0.022, 0.0242, 0.0268, 0.0295, 0.0339,
    0.037, 0.0421, 0.0472, 0.0536, 0.0614, 0.0708, 0.0811, 0.0929, 0.107, 0.121,
    0.14, 0.159, 0.179, 0.2, 0.231, 0.254, 0.285, 0.31, 0.346, 0.372,
    0.419, 0.478, 0.494, 0.51, 0.516, 1.34, 0.798, 0.862, 0.952, 1.04,
)

# The smooth law fitted to them on the log scale (equal weights, ages 0..100, bounded least squares).
LAW = {"ln_a1": -5.8339, "t1": 0.3, "ln_a2": -8.5536, "t2": 2.9795, "ln_c": -13.816, "ln_h": -8.3803, "x0": 24.032, "w": 5.3022,
       "ln_g": -10.803, "k": 0.096738}


def mortality_mu(ages) -> np.ndarray:
    """The illustration's mu_x, the sum of five terms (parameters in `LAW`)::

        a1 exp(-x / t1)                   the infant fall, almost all of it at age 0
        a2 exp(-x / t2)                   the decline through childhood, to a minimum of 6e-5 around age 8
        c                                 a constant (negligible here)
        h exp(-((x - x0) / w)^2 / 2)      the young-adult hump: a rise from 1e-4 at 15 to a shoulder of 3-5e-4 from 20 to 30
        g e^(kx)                          Gompertz: old-age mortality rising by 10 % a year of age

    Fitted to the crude rates of `CRUDE_RATES` (cost 1.2 on the log scale, i.e. about 0.1 per age). A fifth, saturating
    term for the very old was tried and dropped: the fit left it switched off."""
    x = np.asarray(ages, dtype=float)
    q = LAW
    return (np.exp(q["ln_a1"]) * np.exp(-x / q["t1"]) + np.exp(q["ln_a2"]) * np.exp(-x / q["t2"]) + np.exp(q["ln_c"])
            + np.exp(q["ln_h"]) * np.exp(-0.5 * ((x - q["x0"]) / q["w"]) ** 2) + np.exp(q["ln_g"] + q["k"] * x))


def zone_curve_svg(omega1: int = 20, omega2: int = 85, omega3: int = 110, width: int = 560, height: int = 170,
                   tones: tuple = ZONE_TONES, stroke: float = 3.2, labels: bool = True, ink: str = "#9FC3E0",
                   dots: float = 0.0) -> str:
    """The mortality curve (log scale, ages 0..omega3) in three coloured pieces, boundaries dashed, zones named.

    `dots` > 0 draws the crude rates of `CRUDE_RATES` (France 2020) as dots of that radius under the curve (ages 0..109): smoothing,
    in one picture. The vertical scale then covers the dots as well, whose oldest ages exceed the curve.

    `omega1` and `omega2` are clipped to 1 <= omega1 < omega2 < omega3 so that any setting draws something sensible."""
    omega3 = max(int(omega3), 6)
    omega1 = min(max(int(omega1), 1), omega3 - 3)
    omega2 = min(max(int(omega2), omega1 + 1), omega3 - 1)
    ages = np.arange(0, omega3 + 1)
    log_mu = np.log10(mortality_mu(ages))
    shown = [int(a) for a in ages if dots and a < len(CRUDE_RATES) and CRUDE_RATES[a] > 0]          # ages whose crude rate is drawn
    log_dots = np.log10([CRUDE_RATES[a] for a in shown])
    low = min(log_mu.min(), log_dots.min()) if shown else log_mu.min()
    high = max(log_mu.max(), log_dots.max()) if shown else log_mu.max()
    left, right, top, bottom = 10, 10, 14, 26 if labels else 10
    xs = left + ages / omega3 * (width - left - right)

    def y_of(log_value):
        return top + (high - log_value) / (high - low) * (height - top - bottom)

    ys = y_of(log_mu)
    parts = [f'<circle cx="{xs[a]:.1f}" cy="{y_of(v):.1f}" r="{dots}" fill="#FFFFFF" fill-opacity=".45"/>' for a, v in zip(shown, log_dots)]
    for (a, b), tone in zip(((0, omega1), (omega1, omega2), (omega2, omega3)), tones):
        pts = " ".join(f"{xs[i]:.1f},{ys[i]:.1f}" for i in range(a, b + 1))
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{tone}" stroke-width="{stroke}" stroke-linecap="round" stroke-linejoin="round"/>')
    for omega in (omega1, omega2):
        parts.append(f'<line x1="{xs[omega]:.1f}" y1="{top}" x2="{xs[omega]:.1f}" y2="{height - bottom}" stroke="{ink}" stroke-opacity=".55" '
                     f'stroke-width="1" stroke-dasharray="3 4"/>')
    if labels:
        base = height - 8
        for (a, b), name, tone in zip(((0, omega1), (omega1, omega2), (omega2, omega3)), ("I", "II", "III"), tones):
            parts.append(f'<text x="{(xs[a] + xs[b]) / 2:.1f}" y="{base}" text-anchor="middle" font-size="{12 + height // 90}" fill="{tone}" '
                         f'font-family="Segoe UI,Calibri,Arial,sans-serif">Zone {name}</text>')
    return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">{"".join(parts)}</svg>'


def zone_curve_img(*args, css: str = "width:100%;height:100%;object-fit:contain", **kwargs) -> str:
    """`zone_curve_svg` as an <img>."""
    return svg_img(zone_curve_svg(*args, **kwargs), css, "Courbe de mortalité en trois zones d'âge")


def _pictogram_svg(kind: str) -> str:
    head = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 96 56" width="96" height="56">'
    if kind == "cohort":
        return (f'{head}<path d="M8 52H92M8 52V6" stroke="{RULE}" stroke-width="2" fill="none"/>'
                f'<path d="M12 48L84 10" stroke="{NAVY}" stroke-width="3" fill="none" stroke-linecap="round"/>'
                f'<circle cx="12" cy="48" r="5" fill="{SKY}"/><circle cx="48" cy="29" r="4" fill="#fff" stroke="{NAVY}" stroke-width="2"/>'
                f'<circle cx="84" cy="10" r="5" fill="{LIME}"/></svg>')
    return (f'{head}<rect x="10" y="8" width="76" height="42" rx="5" fill="none" stroke="{NAVY}" stroke-width="3"/>'
            f'<path d="M10 20H86M30 3V13M66 3V13" stroke="{NAVY}" stroke-width="3" stroke-linecap="round"/>'
            f'<rect x="20" y="28" width="12" height="14" fill="{GREEN_DEEP}"/><rect x="38" y="25" width="12" height="17" fill="{GREEN}"/>'
            f'<rect x="56" y="31" width="12" height="11" fill="{LIME}"/></svg>')


def pictogram(kind: str) -> str:
    """Small line pictograms for the two kinds of table: a calendar year (moment) and a diagonal cohort (generation)."""
    return svg_img(_pictogram_svg(kind), "width:96px;height:56px;display:block")


def _step_svg(kind: str) -> str:
    head = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 44" width="64" height="44">'
    if kind == "load":                                         # a sheet and an arrow going up into it
        return (f'{head}<path d="M18 3H38L48 13V41H18Z" fill="none" stroke="{NAVY}" stroke-width="3" stroke-linejoin="round"/>'
                f'<path d="M38 3V13H48" fill="none" stroke="{NAVY}" stroke-width="3" stroke-linejoin="round"/>'
                f'<path d="M33 35V22M27 28L33 21L39 28" fill="none" stroke="{GREEN_DEEP}" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/></svg>')
    if kind == "settings":                                     # three sliders
        return (f'{head}<path d="M8 11H56M8 22H56M8 33H56" stroke="{NAVY}" stroke-width="3" stroke-linecap="round"/>'
                f'<circle cx="22" cy="11" r="5" fill="{SKY}" stroke="#fff" stroke-width="2"/><circle cx="42" cy="22" r="5" fill="{GREEN}" stroke="#fff" stroke-width="2"/>'
                f'<circle cx="30" cy="33" r="5" fill="{LIME}" stroke="#fff" stroke-width="2"/></svg>')
    if kind == "evaluate":                                     # bars, one of them ticked
        return (f'{head}<path d="M6 40H58" stroke="{NAVY}" stroke-width="3" stroke-linecap="round"/>'
                f'<rect x="10" y="26" width="10" height="14" fill="{GREEN_DEEP}"/><rect x="24" y="16" width="10" height="24" fill="{GREEN}"/>'
                f'<rect x="38" y="22" width="10" height="18" fill="{LIME}"/>'
                f'<circle cx="52" cy="11" r="8" fill="{NAVY}"/><path d="M48 11L51 14L56 8" fill="none" stroke="#fff" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"/></svg>')
    return (f'{head}<path d="M10 28V40H54V28" fill="none" stroke="{NAVY}" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>'
            f'<path d="M32 4V27M23 18L32 27L41 18" fill="none" stroke="{GREEN_DEEP}" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/></svg>')


def step_pictogram(kind: str) -> str:
    """The small pictogram of one of the four steps: 'load', 'settings', 'evaluate' or 'export'."""
    return svg_img(_step_svg(kind), "width:64px;height:44px;display:block")


# --------------------------------------------------------------------------
# CSS
# --------------------------------------------------------------------------

BLUE_RULE = "#1B6CA8"
CSS = f"""
<style>
[data-testid="stAppDeployButton"], [data-testid="stMainMenu"] {{ display: none; }}
header[data-testid="stHeader"] {{ background: transparent; }}
[data-testid="stMainBlockContainer"] {{ padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1320px; }}
h1, h2, h3 {{ color: {NAVY}; }}
[data-testid="stSidebar"] {{ background: {NAVY_DEEP}; }}
[data-testid="stSidebar"] * {{ color: #E6F1FB; }}
[data-testid="stSidebar"] hr {{ border-color: {BLUE_RULE}; }}
[data-testid="stSidebar"] [data-testid="stPageLink-NavLink"] {{ border-radius: 8px; padding: .55rem .8rem; border-left: 3px solid transparent; }}
[data-testid="stSidebar"] [data-testid="stPageLink-NavLink"]:hover {{ background: #134A7C; }}
[data-testid="stSidebar"] [data-testid="stPageLink-NavLink"][aria-current="page"] {{ background: {NAVY}; border-left: 3px solid {GREEN}; font-weight: 600; }}
.rdl-brand {{ display: flex; align-items: center; gap: .65rem; padding: .2rem .2rem .9rem .2rem; }}
.rdl-brand small {{ display: block; color: #9FC3E0 !important; font-size: .78rem; margin-top: .1rem; }}
.rdl-chip {{ display: inline-block; background: #134A7C; color: #E6F1FB !important; border-radius: 999px; padding: .15rem .7rem; font-size: .8rem; margin: 0 0 .8rem .2rem; }}
.rdl-header {{ position: relative; display: flex; align-items: center; justify-content: space-between; gap: 1.5rem; overflow: hidden;
  background: {NAVY}; border-radius: 14px; padding: 1.1rem 1.6rem; margin: 0 0 1rem 0; min-height: 110px; }}
.rdl-header .rdl-step {{ color: {LIME}; font-size: .85rem; letter-spacing: .06em; text-transform: uppercase; margin: 0 0 .15rem 0; }}
.rdl-header h1 {{ color: #fff !important; font-size: 1.6rem; line-height: 1.2; margin: 0; padding: 0; }}
.rdl-header p {{ color: #CFE3F4 !important; margin: .3rem 0 0 0; font-size: .98rem; }}
.rdl-header > div:first-child {{ min-width: 0; flex: 1 1 auto; }}
.rdl-header .rdl-art {{ flex: 0 1 300px; min-width: 0; height: 96px; opacity: .95; }}
.rdl-hero {{ position: relative; overflow: hidden; background: {NAVY_NIGHT}; border-radius: 18px; padding: 2.2rem 2.6rem 2rem 2.6rem; margin: 0 0 1.4rem 0; min-height: 300px; }}
.rdl-hero .rdl-art {{ position: absolute; right: 1.2rem; bottom: .6rem; width: 56%; height: 78%; opacity: .9; }}
.rdl-hero .rdl-text {{ position: relative; max-width: 560px; }}
.rdl-hero h1 {{ color: #fff !important; font-size: 2.6rem; line-height: 1.1; margin: 1.2rem 0 .6rem 0; padding: 0; }}
.rdl-hero p {{ color: #CFE3F4 !important; font-size: 1.08rem; margin: 0 0 .9rem 0; }}
.rdl-hero .rdl-tag {{ color: {LIME} !important; font-size: 1rem; font-weight: 600; letter-spacing: .02em; }}
.rdl-card-title {{ font-size: 1.25rem; font-weight: 600; color: {NAVY}; margin: .4rem 0 .2rem 0; }}
.rdl-card-text {{ color: {INK_SOFT}; font-size: .95rem; min-height: 4.8rem; margin: 0 0 .4rem 0; }}
.rdl-badge {{ display: inline-block; background: #E3EEF8; color: {NAVY}; border-radius: 999px; padding: .05rem .6rem; font-size: .75rem; font-weight: 600; margin-left: .5rem; vertical-align: middle; }}
.rdl-steps {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: .8rem; margin: .4rem 0 0 0; }}
.rdl-step-tile {{ background: #fff; border: 1px solid {RULE}; border-radius: 12px; padding: .8rem 1rem; }}
.rdl-step-tile img {{ margin: 0 0 .55rem 0; }}
.rdl-step-tile b {{ display: block; color: {NAVY}; margin-bottom: .15rem; }}
.rdl-step-tile span {{ color: {INK_SOFT}; font-size: .9rem; }}
.rdl-step-num {{ display: inline-block; width: 1.5rem; height: 1.5rem; line-height: 1.5rem; text-align: center; border-radius: 50%; background: {NAVY}; color: #fff !important; font-size: .85rem; margin-right: .4rem; }}
</style>
"""


def inject_css() -> None:
    """Install the stylesheet (once per run, at the top of the entry script)."""
    st.html(CSS)


# --------------------------------------------------------------------------
# Blocks
# --------------------------------------------------------------------------


def page_header(title: str, step: int, subtitle: str = "", omegas: tuple = (20, 85, 110)) -> None:
    """The navy band that opens a page: 'ETAPE n SUR 4', the title, a sentence, and the zone curve on the right."""
    st.html(f'<div class="rdl-header"><div><p class="rdl-step">Étape {step} sur 4</p><h1>{title}</h1>'
            f'{f"<p>{subtitle}</p>" if subtitle else ""}</div>'
            f'<div class="rdl-art">{zone_curve_img(*omegas, width=340, height=96, stroke=2.4, labels=False, dots=1.4)}</div></div>')


def hero() -> None:
    """The home page's dark band: logo, name, tagline, and the big zone curve behind."""
    st.html(f'<div class="rdl-hero"><div class="rdl-art">{zone_curve_img(width=700, height=280, stroke=4, dots=2.4)}</div>'
            f'<div class="rdl-text"><div style="display:flex;align-items:center;gap:.7rem">{logo_mark(54)}{wordmark(True, "1.35rem")}</div>'
            f'<h1>{APP_NAME}</h1>'
            f"<p>Lissage paramétrique de tables de mortalité par zone d'âge, avec des méthodes codées from scratch.</p>"
            f'<div class="rdl-tag">{TAGLINE}</div></div></div>')


def choice_card(kind: str, title: str, text: str, badge: str = "") -> None:
    """Pictogram, title and sentence of one home-page choice (the button sits under it, in the page)."""
    st.html(f'<div>{pictogram(kind)}<div class="rdl-card-title">{title}'
            f'{f"<span class=rdl-badge>{badge}</span>" if badge else ""}</div><p class="rdl-card-text">{text}</p></div>')


def steps_strip() -> None:
    """The four steps of the app, on the home page."""
    tiles = (("load", "Chargement", "Déposez vos fichiers HMD, INSEE, CSV ou Excel et contrôlez-les."),
             ("settings", "Paramétrage", "Choisissez les frontières de zones et les méthodes à tester."),
             ("evaluate", "Évaluation", "Comparez les méthodes zone par zone et validez votre sélection."),
             ("export", "Export", "Visualisez la table lissée et téléchargez-la en Excel, CSV ou texte."))
    st.html('<div class="rdl-steps">' + "".join(
        f'<div class="rdl-step-tile">{step_pictogram(kind)}<b><span class="rdl-step-num">{i}</span>{name}</b><span>{text}</span></div>'
        for i, (kind, name, text) in enumerate(tiles, start=1)) + "</div>")


def view_phrase(view: str) -> str:
    """'tables du moment' / 'tables générationnelles'."""
    return VIEW_LABEL[view].split(" (")[0].lower()


def render_sidebar(pages: list, home, show_choice: bool = True) -> None:
    """The marine sidebar, on every page including the home page: brand, the table type chosen at the home page (when there is one
    to show), the four pages, the way back home.

    `pages` is [(st.Page, label, done)]; `done` pages get a tick."""
    with st.sidebar:
        chip = f'<div class="rdl-chip">{view_phrase(st.session_state.view_choice).capitalize()}</div>' if show_choice else ""
        st.html(f'<div class="rdl-brand">{logo_mark(40)}<div>{wordmark(True)}<small>{APP_NAME}</small></div></div>{chip}')
        for page, label, done in pages:
            st.page_link(page, label=f"{label} ✓" if done else label)
        st.divider()
        st.page_link(home, label="Accueil", icon=":material/home:")
