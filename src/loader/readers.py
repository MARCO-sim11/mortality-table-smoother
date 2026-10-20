"""Bytes -> a table of raw cells. HMD text, CSV / TSV and Excel; the row holding the column names is
looked for (title lines, notes above the data), not assumed to be the first.

An input is described by a `Source`: the file and how to read it. Anything that is a path, an
uploaded file (Streamlit's `UploadedFile`) or a file object is accepted wherever a Source is.
"""
from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .columns import ROLE_ALIASES, SEX_COLUMN_ALIASES, is_header_like
from .errors import FileFormatError, LoaderError
from .layouts import detect_origin, normalize
from .parsing import norm

FORMATS = ("auto", "hmd", "csv", "excel")
_OLE_MAGIC = b"\xd0\xcf\x11\xe0"            # legacy .xls container


@dataclass
class Source:
    """One input file and how to read it.

    sheet, header   Excel sheet (name or index); the row holding the column names (None: found)
    col_map         hand-made mapping {'age': 'Âge', 'Dx': 'Décès', ...} (also year, cohort, sex, mx,
                    Ex, value, Female, Male)
    fmt             'auto' | 'hmd' | 'csv' | 'excel'
    where           {column: value}: keep the rows of one segment (INSEE {'CHAMP2': 'FM'},
                    Eurostat {'geo': 'FR'}); needed only when a segment column holds several values
    open_from       the age code from which the age column is an open group, to be left out
                    (INSEE's AGE2_REVOLU is detected by name: its '100' means 100 and over)
    exposure_basis  'central' (default): the exposure column is the exposure. 'jan1_mean': it is a
                    population at 1 January - exposure of year t = mean of the populations at 1
                    January t and t+1 (trapezoid rule; the loader's one computed quantity)
    """
    file: Any
    sheet: int | str = 0
    header: int | None = None
    col_map: Mapping | None = None
    fmt: str = "auto"
    where: Mapping | None = None
    open_from: int | None = None
    exposure_basis: str = "central"


def as_source(x) -> Source:
    return x if isinstance(x, Source) else Source(x)


@dataclass
class Table:
    name: str
    fmt: str
    frame: pd.DataFrame            # raw cells, long form, column names stripped
    title: str = ""                # lines above an HMD header
    sheets: list | None = None
    origin: str | None = None      # 'HMD' | 'INSEE' | 'Eurostat' | None
    notes: list = field(default_factory=list)


def read_source(src) -> tuple:
    if isinstance(src, (str, Path)):
        p = Path(src)
        if not p.is_file():
            raise FileNotFoundError(f"Fichier introuvable : {p}")
        return p.read_bytes(), p.name
    if hasattr(src, "getvalue"):                     # BytesIO, Streamlit UploadedFile
        data = src.getvalue()
    elif hasattr(src, "read"):
        if hasattr(src, "seek"):
            src.seek(0)
        data = src.read()
    else:
        raise LoaderError(f"Source non reconnue : {type(src).__name__} (chemin ou fichier attendu).")
    if isinstance(data, str):
        data = data.encode("utf-8")
    return data, Path(getattr(src, "name", "upload")).name


def decode(data: bytes) -> str:
    for enc in ("utf-8-sig", "cp1252"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("latin-1")


def find_hmd_header(lines: list) -> int | None:
    """Index of the 'Year Age Female Male Total' line (or 'Cohort Age ...', 'Age Female Male ...').

    A line qualifies when its first two whitespace tokens are known column names
    - a title like 'France, Total population, ...' never is. '#' lines and
    blanks before it are skipped, which covers the spec's '#'-headed variant.
    """
    first = ROLE_ALIASES["age"] | ROLE_ALIASES["year"] | ROLE_ALIASES["cohort"]
    second = (first | ROLE_ALIASES["dx"] | ROLE_ALIASES["ex"] | ROLE_ALIASES["mx"]
              | set().union(*SEX_COLUMN_ALIASES.values()))
    for i, ln in enumerate(lines[:60]):
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        tok = s.split()
        if len(tok) >= 2 and norm(tok[0]) in first and norm(tok[1]) in second:
            return i
    return None


def detect_format(name: str, data: bytes) -> str:
    suffix = Path(name).suffix.lower()
    if suffix in (".xlsx", ".xlsm", ".xls") or data[:4] == _OLE_MAGIC:
        return "excel"
    if suffix in (".csv", ".tsv"):
        return "csv"
    return "hmd" if find_hmd_header(decode(data).splitlines()) is not None else "csv"


def sniff_sep(line: str, *, tabs: bool = False) -> str | None:
    """The delimiter of a line. Eurostat's TSV header ('unit,sex,age,geo\\TIME_PERIOD<TAB>2019<TAB>...')
    holds commas inside its first cell, so a tab wins there; `tabs` says the file is a .tsv."""
    if "\t" in line and (tabs or "\\" in line.split("\t", 1)[0]):
        return "\t"
    counts = {s: line.count(s) for s in (";", "\t", ",", "|")}
    best = max(counts, key=counts.get)
    if counts[best] > 0:
        return best
    return r"\s+" if len(line.split()) > 1 else None


def _split(line: str, sep: str) -> list:
    return line.split() if sep == r"\s+" else next(csv.reader([line], delimiter=sep))


def _csv_header_index(head: list, header, tabs: bool) -> int:
    """Index (among the non-blank lines) of the row holding the column names."""
    if header is not None:
        return header
    for i, ln in enumerate(head[:60]):
        sep = sniff_sep(ln, tabs=tabs)
        if sep and is_header_like(_split(ln, sep)):
            return i
    return 0


def drop_blank(df: pd.DataFrame) -> pd.DataFrame:
    """Drop empty rows and the empty 'Unnamed: n' columns trailing separators create."""
    def blank(v) -> bool:
        return v is None or (isinstance(v, float) and np.isnan(v)) or (
            isinstance(v, str) and not v.strip())

    empty = df.apply(lambda col: col.map(blank))
    df = df.loc[~empty.all(axis=1)]
    keep = [c for c in df.columns
            if not (str(c).startswith("Unnamed:") and empty.loc[df.index, c].all())]
    return df[keep]


_INPUTDB_COLUMNS = {"year", "sex", "age", "ageinterval", "lexis", "deaths"}


def is_inputdb(cols) -> bool:
    """The HMD InputDB deaths layout (FRATNPdeath.txt): one row per age and Lexis triangle."""
    return _INPUTDB_COLUMNS <= {norm(c) for c in cols}


def _read_csv(text: str, sep: str, header_idx: int) -> tuple:
    """-> (frame, notes). Footnotes under the data (CDC WONDER...) can have more fields than the header:
    they are set aside and counted, not allowed to stop the read."""
    kw = dict(sep=sep, header=header_idx, dtype=str, keep_default_na=False, skip_blank_lines=True)
    try:
        return pd.read_csv(io.StringIO(text), **kw), []
    except pd.errors.ParserError:
        bad: list = []
        frame = pd.read_csv(io.StringIO(text), engine="python", on_bad_lines=lambda row: bad.append(row), **kw)
        return frame, [f"{len(bad)} ligne(s) mal formée(s) ignorée(s) (notes de bas de tableau ?)"]


def read_table(src: Source) -> Table:
    if src.fmt not in FORMATS:
        raise LoaderError(f"Format inconnu : {src.fmt!r} (admis : {', '.join(FORMATS)}).")
    data, name = read_source(src.file)
    if not data.strip():
        raise FileFormatError(f"{name} : fichier vide.")
    eff = detect_format(name, data) if src.fmt == "auto" else src.fmt

    title, sheets, notes = "", None, []
    try:
        if eff == "excel":
            if Path(name).suffix.lower() == ".xls" or data[:4] == _OLE_MAGIC:
                raise FileFormatError(
                    f"{name} : le format .xls n'est pas pris en charge (xlrd n'est pas dans les "
                    "dépendances autorisées). Réenregistrez le fichier en .xlsx ou en .csv.")
            xl = pd.ExcelFile(io.BytesIO(data), engine="openpyxl")
            sheets = list(xl.sheet_names)
            target = sheets[src.sheet] if isinstance(src.sheet, int) else src.sheet
            header = src.header
            if header is None:
                scan = xl.parse(target, header=None, nrows=60, dtype=object).fillna("").astype(str)
                header = next((i for i, r in enumerate(scan.values.tolist()) if is_header_like(r)), 0)
            frame = xl.parse(target, header=header)
        else:
            text = decode(data)
            lines = text.splitlines()
            if eff == "hmd":
                i = find_hmd_header(lines)
                if i is None:
                    raise FileFormatError(
                        f"{name} : en-tête HMD introuvable (attendu : 'Year Age Female Male Total').")
                title = " ".join(ln.strip().lstrip("#").strip() for ln in lines[:i] if ln.strip())
                frame = pd.read_csv(io.StringIO("\n".join(lines[i:])), sep=r"\s+", dtype=str,
                                    keep_default_na=False)
            else:
                head = [ln for ln in lines if ln.strip()]
                tabs = Path(name).suffix.lower() == ".tsv"
                idx = _csv_header_index(head, src.header, tabs)
                if idx >= len(head):
                    raise FileFormatError(f"{name} : pas de ligne d'en-tête n°{idx}.")
                sep = sniff_sep(head[idx], tabs=tabs)
                if sep is None:
                    raise FileFormatError(f"{name} : séparateur introuvable (attendu : ; , tabulation).")
                frame, notes = _read_csv(text, sep, idx)
    except LoaderError:
        raise
    except (IndexError, KeyError, ValueError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise FileFormatError(f"{name} : lecture impossible ({exc}).") from exc

    frame.columns = [str(c).strip() for c in frame.columns]
    frame = drop_blank(frame).reset_index(drop=True)
    if frame.empty:
        raise FileFormatError(f"{name} : aucune ligne de données.")
    if is_inputdb(frame.columns):
        raise FileFormatError(
            f"{name} : fichier InputDB de HMD (une ligne par âge et par triangle de Lexis TL / TU / RR). "
            "Ce sont des comptages bruts à agréger, ce qui sort du rôle du loader (importer des taux bruts "
            "ou leurs entrées directes). Utilisez les fichiers de sortie HMD : Deaths_1x1 + Exposures_1x1, "
            "ou Mx_1x1 + Exposures_1x1 - ou agrégez d'abord ce fichier (smoothing_lab/io_hmd.py).")
    frame, origin, more = normalize(frame, name)
    origin = origin or detect_origin(frame.columns, title, eff)
    return Table(name=name, fmt=eff, frame=frame, title=title, sheets=sheets, origin=origin,
                 notes=notes + more)
