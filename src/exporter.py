"""exporter.py - Module 6: the final smoothed table as Excel, CSV and HMD-style text, plus the final chart.

Spec 6.1: "exporter la table de mortalite lissee finale dans les formats requis. Produire les visualisations finales."
Everything here works on ONE sex's results (the app calls it once per sex); it takes tables and results and returns
text or bytes, it never touches Streamlit, so each piece can be tested on its own. The steps of an export::

    1. COLUMNS     the table of `joiner.build_life_table` (age, mu, qx, px, lx, dx, ex): `select_columns` keeps a chosen
                   subset (spec 6.3: "selecteur de colonnes"); `age` is always kept, a table without it is not a table;
    2. CSV         `to_csv`: the chosen columns, separator "," or ";" (spec 6.2), decimal "." or "," for a French Excel;
    3. TXT (HMD)   `to_hmd_txt`: a title line, a blank line, a header, then fixed-width columns separated by spaces
                   (spec 6.2: "meme format que les fichiers HMD Deaths, entete standardise, espace comme separateur");
    4. EXCEL       `build_excel` (spec 6.4): sheet 1 'Table_lissée' (the table), 2 'Taux_bruts' (age, Dx, Ex, mx_raw), 3
                   'Paramètres' (`parameter_table`: the estimated parameters by zone and method), 4 'Critères' (the
                   evaluator's table: AIC, BIC, chi2...); and, beyond the spec's four, 'Graphique' with the final chart;
                   `final_workbook` assembles all of it for one sex from the results the app holds;
    5. MULTI       `method_curves` + `build_excel(methods=...)`: spec 6.3's "export multi-methodes", one more sheet per
                   method that was tested (age, mu, qx of its own curve), in the same workbook;
    6. CHART       `chart_image`: the crude rates and the assembled curve coloured by zone, the boundaries omega1 and
                   omega2 dashed, as a PNG.

Decisions the specs leave open or that cannot be followed to the letter, flagged to the user:

  * CHART IMAGE WITH MATPLOTLIB. Spec 6.2 says "graphique Plotly exporte en image". Plotly writes static images through
    `kaleido`, which is not among the allowed dependencies (numpy, scipy, pandas, streamlit, matplotlib, plotly,
    openpyxl) and is not installed. The same chart is drawn with matplotlib (allowed) and embedded with openpyxl. The
    chart of tab 4 on screen is Plotly, as the spec says.
  * A FIFTH SHEET, 'Graphique', since spec 6.2 asks for "table complete + feuille graphique" while 6.4 lists four sheets.
  * `age` ALWAYS EXPORTED, and the CSV `decimal` option, which the spec does not mention (a ";" file read by a French
    Excel needs a decimal comma).
  * HMD STYLE. HMD's Deaths files have Year, Age and one column per sex; a life table has neither sexes in columns nor
    the same columns, so the TXT keeps HMD's layout (title, blank line, header, right-aligned fixed-width columns,
    spaces) with `Year` (or `Cohort`, for a generation table) and `Age` first, then the chosen columns. The Year column is
    left out when the year is not known.
"""
from __future__ import annotations

import io
from datetime import date

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                  # noqa: E402
import numpy as np                                                # noqa: E402
import pandas as pd                                               # noqa: E402
from openpyxl.drawing.image import Image as XlImage               # noqa: E402
from openpyxl.styles import Font                                  # noqa: E402
from openpyxl.utils import get_column_letter                      # noqa: E402

from src import config as C                                       # noqa: E402
from src.config import FitResult, SmootherConfig                  # noqa: E402
from src.joiner import LIFE_TABLE_COLUMNS, mu_to_q                # noqa: E402

TABLE_COLUMNS = list(LIFE_TABLE_COLUMNS)                          # age, mu, qx, px, lx, dx, ex (spec 4.4, 6.2)
RAW_COLUMNS = ["age", "Dx", "Ex", "mx_raw"]                       # spec 6.4, sheet 2
SHEET_TABLE, SHEET_RAW, SHEET_PARAMS, SHEET_CRITERIA, SHEET_CHART = "Table_lissée", "Taux_bruts", "Paramètres", "Critères", "Graphique"
SEPARATORS = (",", ";")
DECIMALS = (".", ",")
ZONE_COLORS = {"zone1": "#D55E00", "zone2": "#009E73", "zone3": "#CC79A7"}     # one colour per zone (Okabe-Ito)
SEX_COLORS = {"Female": "#1B6CA8", "Male": "#3EB553"}                           # Risk Data Lab brand
HMD_WIDTHS = {"age": 5, "mu": 17, "qx": 17, "px": 17, "lx": 14, "dx": 14, "ex": 12}
HMD_DECIMALS = {"mu": 12, "qx": 12, "px": 12, "lx": 4, "dx": 4, "ex": 4}


# --------------------------------------------------------------------------
# Step 1: columns
# --------------------------------------------------------------------------


def select_columns(table: pd.DataFrame, columns=None) -> pd.DataFrame:
    """The table restricted to `columns`, in the table's own order, with `age` always first.

    `columns=None` keeps everything. `age` is added when it was not asked for. Raises ValueError for a name that is not
    a column of the table (the message lists the columns there are) and for a request that leaves only `age`."""
    if columns is None:
        return table.copy()
    columns = list(columns)
    unknown = [c for c in columns if c not in table.columns]
    if unknown:
        raise ValueError(f"Colonne(s) inconnue(s) : {', '.join(map(str, unknown))} (colonnes disponibles : "
                         f"{', '.join(table.columns)}).")
    keep = [c for c in table.columns if c in columns or c == "age"]
    if keep == ["age"]:
        raise ValueError("Export : choisissez au moins une colonne autre que l'âge.")
    return table[keep].copy()


# --------------------------------------------------------------------------
# Steps 2 and 3: CSV and HMD-style text
# --------------------------------------------------------------------------


def to_csv(table: pd.DataFrame, columns=None, sep: str = ",", decimal: str = ".") -> str:
    """The table as CSV text (spec 6.2): the chosen columns, a header line, `sep` between fields, `decimal` as the decimal
    mark, full precision, no index. Raises ValueError for a separator other than "," or ";", a decimal other than "." or
    ",", or "," used for both (the file could not be read back)."""
    if sep not in SEPARATORS:
        raise ValueError(f"Séparateur CSV : « , » ou « ; » (reçu : {sep!r}).")
    if decimal not in DECIMALS:
        raise ValueError(f"Marque décimale : « . » ou « , » (reçu : {decimal!r}).")
    if sep == decimal:
        raise ValueError("Le séparateur et la marque décimale ne peuvent pas être identiques (« , » et « , »).")
    return select_columns(table, columns).to_csv(index=False, sep=sep, decimal=decimal, lineterminator="\n")


def to_hmd_txt(table: pd.DataFrame, columns=None, title: str = "", year=None, cohort=None, modified: date | None = None) -> str:
    """The table as an HMD-style text file (spec 6.2).

    Layout, as in HMD's files::

        <title>, smoothed life table (period 1x1)<TAB>Last modified: 19 Oct 2026, Mortality Table Smoother
        <blank line>
          Year  Age              mu              qx ...
          2020    0  0.003208000000  0.003202859650 ...

    Fields are right-aligned in fixed-width columns and separated by spaces, so `numpy.loadtxt(skiprows=3)` reads the data
    back. A `Year` column (or `Cohort`, when `cohort` is given, and the title says "cohort 1x1") comes first when the year
    is known; `age` is `Age`. Rates and probabilities are written with 12 decimals (a rate of 3e-5 keeps 7 digits), `lx`
    and `dx` with 4, `ex` with 4."""
    t = select_columns(table, columns)
    label = f"{title}, " if title else ""
    kind = "cohort" if cohort is not None else "period"
    stamp = (modified or date.today()).strftime("%d %b %Y")
    lines = [f"{label}smoothed life table ({kind} 1x1)\tLast modified: {stamp}, Mortality Table Smoother", ""]
    key_name, key_value = ("Cohort", cohort) if cohort is not None else ("Year", year)
    names = ([key_name] if key_value is not None else []) + ["Age"] + [c for c in t.columns if c != "age"]
    widths = ([6] if key_value is not None else []) + [HMD_WIDTHS["age"] + 1] + [HMD_WIDTHS[c] for c in t.columns if c != "age"]
    lines.append("".join(f"{n:>{w}}" for n, w in zip(names, widths)))
    for row in t.itertuples(index=False):
        cells = ([f"{int(key_value):>6d}"] if key_value is not None else []) + [f"{int(row.age):>{HMD_WIDTHS['age'] + 1}d}"]
        for c in t.columns:
            if c != "age":
                cells.append(f"{getattr(row, c):>{HMD_WIDTHS[c]}.{HMD_DECIMALS[c]}f}")
        lines.append("".join(cells))
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# Tables that go to the Excel sheets
# --------------------------------------------------------------------------


def raw_table(df_sex: pd.DataFrame) -> pd.DataFrame:
    """Sheet 'Taux_bruts' (spec 6.4): the loaded rows of one sex as age, Dx, Ex, mx_raw, by age."""
    return df_sex.sort_values("age")[RAW_COLUMNS].reset_index(drop=True)


def parameter_table(fit_results: dict, selection: dict | None = None) -> pd.DataFrame:
    """Sheet 'Paramètres' (spec 6.4): the estimated parameters, one row per parameter, by zone and method.

    Columns: Zone, Méthode, Clé (method key), Paramètre, Valeur, Retenue (True for the method chosen for its zone, per
    `selection` = {zone: method_key}; all False without it; a global method is chosen when any zone picked it, and its
    Zone reads 'Tous âges'). A list parameter (the spline's knots and knot values) gets one
    row per element, named `name[i]`. Methods follow the registry order; methods absent from `fit_results` are left out."""
    rows = []
    for method in C.every_method():
        result = fit_results.get(method.key)
        if result is None:
            continue
        kept = bool(selection) and any(selection.get(z) == method.key for z in C.zones_served(method))
        for name, value in result.params.items():
            values = list(value) if isinstance(value, (list, tuple, np.ndarray)) else [value]
            for i, v in enumerate(values):
                rows.append({"Zone": C.ZONE_LABELS[method.zone], "Méthode": method.label, "Clé": method.key,
                             "Paramètre": f"{name}[{i}]" if len(values) > 1 or isinstance(value, (list, tuple, np.ndarray)) else name,
                             "Valeur": float(v), "Retenue": kept})
    return pd.DataFrame(rows, columns=["Zone", "Méthode", "Clé", "Paramètre", "Valeur", "Retenue"])


def method_curves(fit_results: dict) -> dict:
    """{method_key: DataFrame(age, mu, qx)} for every tested method: its own curve on the ages of its FitResult (zone III
    runs to omega3). The sheets of the multi-method export (spec 6.3)."""
    return {key: pd.DataFrame({"age": np.asarray(r.ages, dtype=int), "mu": np.asarray(r.mu_hat, dtype=float),
                               "qx": mu_to_q(r.mu_hat)})
            for key in (m.key for m in C.every_method()) if (r := fit_results.get(key)) is not None}


# --------------------------------------------------------------------------
# Step 6: the final chart
# --------------------------------------------------------------------------


def chart_image(raw: pd.DataFrame, mu_joint, cfg: SmootherConfig, sex: str = "Female", title: str = "", dpi: int = 130) -> bytes:
    """The final chart of spec 6.3 as PNG bytes (matplotlib: see the module docstring): crude rates `raw` (points, the sex's
    brand colour), the assembled `mu_joint` (one curve coloured by zone: I to omega1, II to omega2, III beyond), the
    boundaries omega1 and omega2 dashed, log scale. Ages with no crude rate above 0 are not drawn."""
    mu = np.asarray(mu_joint, dtype=float)
    ages = np.arange(mu.size)
    fig, ax = plt.subplots(figsize=(9.0, 5.4))
    pos = raw[(raw["age"].notna()) & (raw["mx_raw"] > 0)]
    ax.scatter(pos["age"], pos["mx_raw"], s=14, color=SEX_COLORS.get(sex, "#555555"), alpha=0.75, label="Taux bruts", zorder=2)
    edges = [(None, cfg.omega1, "zone1"), (cfg.omega1, cfg.omega2, "zone2"), (cfg.omega2, None, "zone3")]
    for lo, hi, zone in edges:
        keep = np.isfinite(mu) & ((ages >= lo) if lo is not None else True) & ((ages <= hi) if hi is not None else True)
        ax.plot(ages[keep], mu[keep], color=ZONE_COLORS[zone], lw=2.2, label=f"Lissé — {C.ZONE_LABELS[zone]}", zorder=3)
    for omega, name in ((cfg.omega1, "ω₁"), (cfg.omega2, "ω₂")):
        ax.axvline(omega, color="grey", ls="--", lw=1)
        ax.text(omega, 1.0, f" {name} = {omega}", transform=ax.get_xaxis_transform(), va="top", fontsize=8, color="grey")
    ax.set_yscale("log")
    ax.set_xlabel("Âge")
    ax.set_ylabel("μₓ (échelle logarithmique)")
    ax.set_title(title or "Taux bruts et table lissée", loc="left", fontsize=12)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    plt.close(fig)
    return buf.getvalue()


# --------------------------------------------------------------------------
# Steps 4 and 5: the workbook
# --------------------------------------------------------------------------


def _write_sheet(writer, df: pd.DataFrame, name: str) -> None:
    df.to_excel(writer, sheet_name=name, index=False)
    ws = writer.sheets[name]
    ws.freeze_panes = "A2"
    for cell in ws[1]:
        cell.font = Font(bold=True)
    for i, col in enumerate(df.columns, start=1):
        width = max([len(str(col))] + [len(f"{v:.6g}") if isinstance(v, float) else len(str(v)) for v in df[col].head(60)])
        ws.column_dimensions[get_column_letter(i)].width = min(max(width + 2, 8), 48)


def build_excel(table: pd.DataFrame, raw: pd.DataFrame, params: pd.DataFrame, criteria: pd.DataFrame,
                columns=None, chart_png: bytes | None = None, methods: dict | None = None) -> bytes:
    """The final workbook (spec 6.4) as .xlsx bytes, written with openpyxl.

    Sheets, in order: 'Table_lissée' (`table`, the chosen `columns`), 'Taux_bruts' (`raw`), 'Paramètres' (`params`),
    'Critères' (`criteria`), then 'Graphique' when `chart_png` is given (the image, anchored at A1), then one sheet
    'Méthode_<key>' per entry of `methods` ({key: DataFrame}, see `method_curves`) when the multi-method export is asked.
    The tables are written as they are, with a bold header, a frozen first row and column widths fitted to the content."""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        _write_sheet(writer, select_columns(table, columns), SHEET_TABLE)
        _write_sheet(writer, raw, SHEET_RAW)
        _write_sheet(writer, params, SHEET_PARAMS)
        _write_sheet(writer, criteria, SHEET_CRITERIA)
        if chart_png is not None:
            ws = writer.book.create_sheet(SHEET_CHART)
            ws.add_image(XlImage(io.BytesIO(chart_png)), "A1")
        for key, curve in (methods or {}).items():
            _write_sheet(writer, curve, f"Méthode_{key}")
    return buf.getvalue()


def final_workbook(df_sex: pd.DataFrame, cfg: SmootherConfig, fit_results: dict, selection: dict, table: pd.DataFrame,
                   mu_joint, sex: str, criteria: pd.DataFrame, columns=None, include_methods: bool = False,
                   title: str = "") -> bytes:
    """The workbook of spec 6.4 for ONE sex, from what the app holds: `df_sex` (its loaded rows), `cfg` (the settings its
    fits were made with), `fit_results` ({method_key: FitResult}) and `selection` ({zone: method_key} behind the table),
    `table` (the life table) and `mu_joint` (the assembled curve), `criteria` (the evaluator's table, already labelled for
    reading). `columns` is the column choice of the table sheet; `include_methods` adds one sheet per tested method.
    Returns the .xlsx bytes: sheets 'Table_lissée', 'Taux_bruts', 'Paramètres', 'Critères', 'Graphique' and, if asked,
    the 'Méthode_<key>' sheets."""
    raw = raw_table(df_sex)
    return build_excel(table, raw, parameter_table(fit_results, selection), criteria, columns=columns,
                       chart_png=chart_image(raw, mu_joint, cfg, sex, title),
                       methods=method_curves(fit_results) if include_methods else None)
