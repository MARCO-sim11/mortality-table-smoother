"""Synthetic mortality data: a KNOWN hazard, Poisson deaths, and file writers.

Used by tests/test_loader.py and notebooks/01_loader_demo.ipynb. The truth
(`mu_true`) is returned next to Dx and Ex, so later modules (Gompertz, Makeham,
...) can reuse the same generator to check that they recover the law.

    mu(x) = A + B * c^x + 0.004 * exp(-x / 1.2)     Makeham + an infant term
    Ex    = base * exp(-0.045 x)                     rounded to 2 decimals
    Dx    ~ Poisson(Ex * mu(x))

Ex is rounded to the precision the writers print, so a file round-trip is
exact and tests can assert equality rather than closeness.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

AGES = np.arange(0, 101)

_LAW = {                                   # sex -> (A, B, c, exposure at age 0, seed)
    "Female": (0.0003, 1.5e-5, 1.105, 6.5e5, 11),
    "Male":   (0.0008, 3.0e-5, 1.105, 6.3e5, 22),
}


def true_mu(ages, sex: str) -> np.ndarray:
    A, B, c, _, _ = _LAW[sex]
    ages = np.asarray(ages, dtype=float)
    return A + B * c ** ages + 0.004 * np.exp(-ages / 1.2)


def make_sex_table(sex: str, year: int = 2020, ages=AGES) -> pd.DataFrame:
    """One sex, one year: columns age, Dx (int), Ex (float), mu_true. Deterministic."""
    _, _, _, e0, seed = _LAW[sex]
    rng = np.random.default_rng(seed * 1000 + year)
    ages = np.asarray(ages)
    mu = true_mu(ages, sex)
    ex = np.round(e0 * np.exp(-0.045 * ages), 2)
    dx = rng.poisson(ex * mu)
    return pd.DataFrame({"age": ages.astype("int64"), "Dx": dx.astype("int64"),
                         "Ex": ex, "mu_true": mu})


# --------------------------------------------------------------------------
# Mode A writers: one all-in-one file per sex
# --------------------------------------------------------------------------


def _num(x: float, decimals: int, decimal: str, thousands: str | None) -> str:
    s = f"{x:,.{decimals}f}"
    if thousands is None:
        s = s.replace(",", "")
    else:
        s = s.replace(",", "\0")
    s = s.replace(".", decimal)
    return s.replace("\0", thousands or "")


def write_per_sex_csv(path, table: pd.DataFrame, *, sep: str = ",", decimal: str = ".",
                      names=("Age", "Deaths", "Exposure"), encoding: str = "utf-8",
                      thousands: str | None = None, with_mx: bool = False) -> Path:
    """Write age / Dx / Ex (and optionally mx) as a CSV in the requested dialect."""
    path = Path(path)
    header = list(names) + (["mx"] if with_mx else [])
    lines = [sep.join(header)]
    for r in table.itertuples():
        row = [str(r.age), _num(r.Dx, 0, decimal, thousands), _num(r.Ex, 2, decimal, thousands)]
        if with_mx:
            row.append(_num(r.Dx / r.Ex, 6, decimal, None))
        lines.append(sep.join(row))
    path.write_bytes(("\n".join(lines) + "\n").encode(encoding))
    return path


# --------------------------------------------------------------------------
# Mode B writers: HMD 1x1 layout, one file per quantity
# --------------------------------------------------------------------------

_HMD = {
    "Dx": ("Deaths (period 1x1)", "{:.2f}"),
    "Ex": ("Exposure to risk (period 1x1)", "{:.2f}"),
    "mx": ("Death rates (period 1x1)", "{:.6f}"),
}


def _by_sex(quantity: str, f: pd.Series, m: pd.Series) -> tuple:
    if quantity == "Dx":
        return f.Dx, m.Dx, f.Dx + m.Dx
    if quantity == "Ex":
        return f.Ex, m.Ex, f.Ex + m.Ex
    return f.Dx / f.Ex, m.Dx / m.Ex, (f.Dx + m.Dx) / (f.Ex + m.Ex)


def hmd_tables(years=(2020, 2021)) -> dict:
    """{year: {'Female': table, 'Male': table}} - the input of `write_hmd`."""
    return {y: {s: make_sex_table(s, y) for s in ("Female", "Male")} for y in years}


def write_hmd(path, quantity: str, by_year: dict, *, title: str | None = None,
              missing=(), open_group: bool = True, comment_header: bool = False,
              columns=("Female", "Male", "Total"), key_name: str = "Year") -> Path:
    """Write an HMD-style file: title, blank line, 'Year Age Female Male Total', CRLF.

    quantity   'Dx' | 'Ex' | 'mx'  (decides the title and the number format)
    by_year    {year: {'Female': table, 'Male': table}}; the Total column is
               derived the way the HMD derives it
    missing    {(year, age, sex), ...} printed as '.', the HMD's missing marker
    open_group add a '110+' row per year (the HMD always has one)
    comment_header  '#'-prefix the title lines (the variant the spec describes)
    columns    which sex columns to print: all three (default), ('Female', 'Male') for a file
               without Total, ('Total',) for a file with the sexes combined only
    key_name   'Year' (period files) or 'Cohort' (generation files); the keys of `by_year`
               are then birth years
    """
    label, fmt = _HMD[quantity]
    title = title or (f"Synthland, Total population, {label}, \tLast modified: 01 Jan 2026;  "
                      "Methods Protocol: v6 (2017)")
    columns = tuple(columns)
    lines = [("# " + title) if comment_header else title,
             "#" if comment_header else "",
             f"  {key_name:<10}    Age" + "".join(f"{c:>16}" for c in columns)]
    for year, sexes in sorted(by_year.items()):
        F, M = sexes["Female"].set_index("age"), sexes["Male"].set_index("age")
        for age in F.index:
            vals = dict(zip(("Female", "Male", "Total"), _by_sex(quantity, F.loc[age], M.loc[age])))
            cells = ["." if (year, int(age), sx) in missing else fmt.format(vals[sx]) for sx in columns]
            lines.append(f"{year:6d}{int(age):12d}" + "".join(f"{c:>16}" for c in cells))
        if open_group:
            open_vals = {"Female": "3.00", "Male": "1.00", "Total": "4.00"}
            lines.append(f"{year:6d}{'110+':>12}" + "".join(f"{open_vals[c]:>16}" for c in columns))
    path = Path(path)
    path.write_bytes(("\r\n".join(lines) + "\r\n").encode("utf-8"))
    return path


_LONG_NAME = {"Dx": "Deaths", "Ex": "Exposure", "mx": "Rate"}
_LONG_SEX = {"Female": "f", "Male": "m", "Total": "t"}


def write_long_csv(path, quantity: str, by_year: dict, *, sexes=("Female", "Male"),
                   sep: str = ",", key_name: str = "Year") -> Path:
    """One row per (year, sex, age) and a single value column - the 'one column for both
    sexes' layout, with a Sex column (f / m / t) telling the rows apart. `sexes` may
    include 'Total' to write the Total rows as well."""
    lines = [sep.join([key_name, "Sex", "Age", _LONG_NAME[quantity]])]
    fmt = _HMD[quantity][1]
    for year, tables in sorted(by_year.items()):
        F, M = tables["Female"].set_index("age"), tables["Male"].set_index("age")
        for sx in sexes:
            for age in F.index:
                v = dict(zip(("Female", "Male", "Total"), _by_sex(quantity, F.loc[age], M.loc[age])))[sx]
                lines.append(sep.join([str(year), _LONG_SEX[sx], str(int(age)), fmt.format(v)]))
    path = Path(path)
    path.write_bytes(("\n".join(lines) + "\n").encode("utf-8"))
    return path


_INPUTDB_HEADER = ("PopName,Area,Year,YearReg,YearInterval,Sex,Age,AgeInterval,Lexis,RefCode,Access,"
                   "Deaths,NoteCode1,NoteCode2,NoteCode3,LDB")


def write_inputdb(path, by_year: dict, *, rr_years=(), top_open: bool = True,
                  ldb0_duplicate: bool = True, extra_rows=()) -> Path:
    """The HMD InputDB deaths layout (FRATNPdeath.txt): one row per (year, sex, age, Lexis
    triangle), with the quirks of the real file - a TOT line per year and sex, TL + TU rows
    summing to the deaths of the age, RR (rectangle) rows instead for `rr_years`, and at the
    top age a single-age TL beside a TU that is the OPEN group ('100+'). A stray LDB = 0 row
    (value 99999, age 10) must be ignored. `extra_rows` are appended verbatim."""
    def row(year, sex, age, interval, lexis, deaths, ldb=1):
        return f"SYN,20,{year},{year},1,{sex},{age},{interval},{lexis},130,O,{deaths},.,.,.,{ldb}"

    lines = [_INPUTDB_HEADER]
    for year, tables in sorted(by_year.items()):
        for label, sx in (("Female", "f"), ("Male", "m")):
            t = tables[label]
            last = int(t["age"].iloc[-1])
            lines.append(f"SYN,20,{year},{year},1,{sx},TOT,.,.,130,O,{int(t['Dx'].sum())},.,.,.,1")
            for age, dx in zip(t["age"], t["Dx"]):
                age, dx = int(age), int(dx)
                if year in rr_years:
                    lines.append(row(year, sx, age, 1, "RR", dx))
                    continue
                tl = int(round(dx * 0.7))
                lines.append(row(year, sx, age, 1, "TL", tl))
                if top_open and age == last:
                    lines.append(row(year, sx, age, "+", "TU", dx - tl + 500))     # the open group
                else:
                    lines.append(row(year, sx, age, 1, "TU", dx - tl))
            if ldb0_duplicate:
                lines.append(row(year, sx, 10, 1, "TL", 99999, ldb=0))
    lines += list(extra_rows)
    path = Path(path)
    path.write_bytes(("\n".join(lines) + "\n").encode("utf-8"))
    return path


# --------------------------------------------------------------------------
# Other sources. INSEE and Eurostat are modelled on the real files' layout (INSEE T74BIS.csv: header
# CHAMP2;ANNEE_1975_2024;SEXE;AGE2_REVOLU;NBDEC; Eurostat bulk TSV: 'unit,sex,age,geo\TIME_PERIOD', one
# column per year, 'value flag' cells, ':' for missing). The population files are generic fixtures.
# --------------------------------------------------------------------------


def write_elementary_csv(path, by_year: dict, *, sep: str = ",", segment: dict | None = None) -> Path:
    """Option C: one file, one row per (year, sex, age) - Year, Sex, Age, Deaths, Exposure.
    `segment={'Region': {'North': 1.0, 'South': 0.5}}` repeats every cell once per value, scaling its
    deaths and exposure by the weight (a second dimension the loader must be told to choose in)."""
    cols = ["Year", "Sex", "Age", "Deaths", "Exposure"]
    seg_name, seg_values = next(iter((segment or {"": {"": 1.0}}).items()))
    rows = [sep.join(([seg_name] if segment else []) + cols)]
    for year, tables in sorted(by_year.items()):
        for sex in ("Female", "Male"):
            for r in tables[sex].itertuples():
                for seg, w in seg_values.items():
                    rows.append(sep.join(([seg] if segment else []) + [
                        str(year), sex, str(int(r.age)), str(int(round(r.Dx * w))), f"{r.Ex * w:.2f}"]))
    path = Path(path)
    path.write_bytes(("\n".join(rows) + "\n").encode("utf-8"))
    return path


def write_insee_deaths(path, by_year: dict, *, fr_extra: int = 0) -> Path:
    """INSEE T74BIS layout: deaths by sex (F / H / E = both) and age in completed years, for two
    fields (FM = metropolitan France, FR = France; FR gets `fr_extra` more deaths a cell). Per year and
    sex: a TOT line, ages 0..99, and '100' = 100 AND OVER (deaths of age 100 + 50 here)."""
    rows = ["CHAMP2;ANNEE_1975_2024;SEXE;AGE2_REVOLU;NBDEC"]
    for champ, extra in (("FM", 0), ("FR", fr_extra)):
        for year, tables in sorted(by_year.items()):
            f, m = tables["Female"], tables["Male"]
            n = int((f["age"] <= 100).sum())
            for code, sx in (("E", None), ("F", f), ("H", m)):
                def d(i):
                    base = (f["Dx"].iloc[i] + m["Dx"].iloc[i]) if sx is None else sx["Dx"].iloc[i]
                    return int(base) + extra + (50 if i == n - 1 else 0)
                total = sum(d(i) for i in range(n))
                rows.append(f"{champ};{year};{code};TOT;{total}")
                rows += [f"{champ};{year};{code};{i};{d(i)}" for i in range(n)]
    path = Path(path)
    path.write_bytes(("\n".join(rows) + "\n").encode("utf-8"))
    return path


def write_eurostat_tsv(path, quantity: str, by_year: dict, *, geos=("FR", "DE"), missing=(),
                       provisional_year=None) -> Path:
    """Eurostat bulk TSV: header 'unit,sex,age,geo\\TIME_PERIOD' then one tab-separated column per
    year; each cell is 'value flag' (flag 'p' for the provisional year, else a trailing space), ':' for
    missing; ages Y_LT1, Y1..Y99 and the open Y_OPEN; sexes F, M, T. `missing` = {(sex code, age code, year)}."""
    years = sorted(by_year)
    unit = "NR"
    rows = ["unit,sex,age,geo\\TIME_PERIOD\t" + "\t".join(str(y) for y in years)]
    for geo in geos:
        extra = 7 if geo != geos[0] else 0
        for code, label in (("F", "Female"), ("M", "Male"), ("T", None)):
            n = int((by_year[years[0]]["Female"]["age"] <= 100).sum())
            for i in range(n):
                age = "Y_LT1" if i == 0 else ("Y_OPEN" if i == n - 1 else f"Y{i}")
                cells = []
                for y in years:
                    f, m = by_year[y]["Female"], by_year[y]["Male"]
                    key = quantity if quantity in ("Dx", "Ex") else "Dx"
                    v = (f[key].iloc[i] + m[key].iloc[i]) if label is None else by_year[y][label][key].iloc[i]
                    v = v + extra
                    cell = ":" if (code, age, y) in missing else (
                        f"{v:.2f} p" if y == provisional_year else f"{v:.2f} ")
                    cells.append(cell)
                rows.append(f"{unit},{code},{age},{geo}\t" + "\t".join(cells))
    path = Path(path)
    path.write_bytes(("\n".join(rows) + "\n").encode("utf-8"))
    return path


def write_wide_years_xlsx(path, quantity: str, by_year: dict, sex: str, *, title_rows: int = 2) -> Path:
    """An Excel table the way national offices and historical actuarial files give it: title lines,
    then 'Âge | 2019 | 2020 ...' with French age labels ('0 an', '85 ans', '100 ans ou plus') and
    no sex column (the sheet is one sex). The open last age is labelled so."""
    years = sorted(by_year)
    n = int((by_year[years[0]][sex]["age"] <= 100).sum())
    key = quantity if quantity in ("Dx", "Ex") else "Dx"
    labels = [("0 an" if i == 0 else f"{i} ans") if i < n - 1 else f"{i} ans ou plus" for i in range(n)]
    table = pd.DataFrame({"Âge": labels, **{str(y): by_year[y][sex][key].iloc[:n].to_numpy() for y in years}})
    path = Path(path)
    with pd.ExcelWriter(path) as w:
        table.to_excel(w, sheet_name="Données", index=False, startrow=title_rows)
        ws = w.sheets["Données"]
        ws["A1"] = f"Table synthétique - {quantity} - {sex}"
        if title_rows > 1:
            ws["A2"] = "Source : générateur de test"
    return path


def write_tab_with_footnotes(path, by_year: dict, year: int = 2020) -> Path:
    """A quoted tab-delimited export with notes UNDER the data (rows with more fields than the header),
    age labels like '< 1 year' / '100+ years', and a suppressed cell - the shape of database exports."""
    f, m = by_year[year]["Female"], by_year[year]["Male"]
    n = int((f["age"] <= 100).sum())
    head = ["Notes", "Age", "Gender", "Year", "Deaths", "Population"]
    q = lambda xs: "\t".join(f'"{x}"' for x in xs)
    rows = [q(head)]
    for sex_label, t in (("Female", f), ("Male", m)):
        for i in range(n):
            age = "< 1 year" if i == 0 else (f"{i}+ years" if i == n - 1 else f"{i} years")
            deaths = "Suppressed" if (sex_label == "Male" and i == 3) else int(t["Dx"].iloc[i])
            rows.append(q(["", age, sex_label, year, deaths, f"{t['Ex'].iloc[i]:.0f}"]))
    rows += ['"---"', q(["Dataset: synthetic", "", "", "", "", "", "extra field"]), q(["Query parameters:", "", "", "", "", "", "x", "y"])]
    path = Path(path)
    path.write_bytes(("\n".join(rows) + "\n").encode("utf-8"))
    return path


def write_sample_data(folder) -> dict:
    """Write the sample files used by the notebook and by the Streamlit uploader, one set per option:
    C (elementary file), A (HMD-style Deaths / Exposure / Rates, plus INSEE- and Eurostat-style
    deaths), B (a CSV per sex). 2020 and 2021, no defect. Returns their paths."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    by_year = hmd_tables()
    return {
        "elementary": write_elementary_csv(folder / "synthetic_elementary_2020_2021.csv", by_year),
        "deaths": write_hmd(folder / "synthetic_Deaths_1x1.txt", "Dx", by_year),
        "exposure": write_hmd(folder / "synthetic_Exposures_1x1.txt", "Ex", by_year),
        "rates": write_hmd(folder / "synthetic_Mx_1x1.txt", "mx", by_year),
        "insee_deaths": write_insee_deaths(folder / "synthetic_T74BIS_style.csv", by_year, fr_extra=5),
        "eurostat_deaths": write_eurostat_tsv(folder / "synthetic_demo_magec_style.tsv", "Dx", by_year),
        "female_csv": write_per_sex_csv(folder / "synthetic_female_2020.csv",
                                        by_year[2020]["Female"].drop(columns="mu_true")),
        "male_csv": write_per_sex_csv(folder / "synthetic_male_2020.csv",
                                      by_year[2020]["Male"].drop(columns="mu_true")),
    }
