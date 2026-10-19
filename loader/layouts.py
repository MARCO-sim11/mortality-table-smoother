"""Reshaping the raw table into one row per observation, whatever the source's layout.

Two layouts need it, both common in national statistics:
  * Eurostat's bulk TSV: the first column holds the dimension codes joined by commas, its header is
    'unit,sex,age,geo\\TIME_PERIOD', and there is one column per year;
  * years as columns (national offices, historical actuarial tables in Excel): Age | 1990 | 1991 | ...
Both become: dimension columns + `year` + `value`. Status flags ('98.5 p') and missing markers (':')
stay in the cells; `parsing.to_numeric` deals with them.
"""
from __future__ import annotations

import re

import pandas as pd

from .columns import role_of
from .errors import FileFormatError
from .parsing import norm

_YEAR_COLUMN = re.compile(r"^(?:1[6-9]\d\d|20\d\d)$")


def normalize(frame: pd.DataFrame, name: str) -> tuple:
    """-> (long-form frame, origin or None, notes). Frames that are already long come back unchanged."""
    origin = None
    first = str(frame.columns[0])
    if "\\" in first:                                              # Eurostat series keys
        dims = [d.strip() for d in first.split("\\", 1)[0].split(",")]
        keys = frame[frame.columns[0]].astype(str).str.split(",", expand=True)
        if keys.shape[1] != len(dims):
            raise FileFormatError(f"{name} : clés Eurostat illisibles (en-tête : {first!r}).")
        keys.columns = dims
        frame = pd.concat([keys, frame.drop(columns=[frame.columns[0]])], axis=1)
        origin = "Eurostat"

    year_cols = [c for c in frame.columns if _YEAR_COLUMN.match(str(c).strip())]
    already_year = any(role_of(norm(c)) == "year" for c in frame.columns)
    if len(year_cols) >= 2 and not already_year:                   # years as columns -> rows
        ids = [c for c in frame.columns if c not in year_cols]
        long = frame.melt(id_vars=ids, value_vars=year_cols, var_name="year", value_name="value")
        long["year"] = long["year"].astype(str).str.strip()
        frame = long
    return frame, origin, []


def detect_origin(columns, title: str, fmt: str) -> str | None:
    """A label for the notes: which family of file this looks like (None when unknown)."""
    if "methods protocol" in title.lower():
        return "HMD"
    names = {norm(c) for c in columns}
    if any(re.fullmatch(r"annee\d{8}", n) for n in names) or "nbdec" in names:
        return "INSEE"
    return None
