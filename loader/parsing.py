"""Cells to numbers, and age labels to integer ages - the two things every source writes differently.

Numbers: French exports ('1 234,5'), flags ('98.5 p' Eurostat, '12 (p)' INSEE) and every missing marker
met in the wild (HMD '.', Eurostat ':', 'x', 'Suppressed', ...).
Ages: '85', '85 ans', 'Y85' (Eurostat), 'D85' (INSEE codes), '< 1 year' and 'Y_LT1' (age 0), and the
groups that are NOT single ages: open ('110+', '100 ans ou plus', '105_PL', 'Y_OPEN'), totals ('TOT',
'Ensemble') and bands ('0-4').
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .errors import LoaderError


def norm(s: Any) -> str:
    """'Décès' -> 'deces', 'm_x' -> 'mx', 'Exposure to risk' -> 'exposuretorisk'."""
    t = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]", "", t.lower())


# --------------------------------------------------------------------------
# Numbers
# --------------------------------------------------------------------------

NA_TOKENS = {"", ".", "..", "...", "-", "--", "—", "–", ":", "x", "na", "n/a", "nan", "null", "none",
             "nd", "n.d.", "s", "c", "ns", "not applicable", "suppressed", "unreliable", "missing"}
_FLAG = re.compile(r"(?:\s+[A-Za-z]{1,3}|\s*\([A-Za-z]{1,3}\))$")      # '98.5 p', '12 (p)'


def to_numeric(col: pd.Series, label: str) -> pd.Series:
    """Float series, NaN for missing cells.

    Reads French exports too: '1 234,5' -> 1234.5 (a lone comma is a decimal comma; with both
    separators the right-most one is the decimal mark), and drops a trailing status flag ('98.5 p',
    '12 (p)'). A cell that is neither a number nor a missing marker raises instead of becoming a
    silent NaN - that would hide a real data problem as a "missing value".
    """
    vals = col.tolist()
    out = np.full(len(vals), np.nan)
    bad: list = []
    for i, v in enumerate(vals):
        if v is None:
            continue
        if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool):
            x = float(v)
        else:
            t = str(v).replace(" ", "").replace(" ", "").strip()
            t = _FLAG.sub("", t).strip()
            if t.lower() in NA_TOKENS:
                continue
            t = t.replace(" ", "")
            if "," in t:
                if "." in t:
                    t = (t.replace(".", "").replace(",", ".") if t.rfind(",") > t.rfind(".")
                         else t.replace(",", ""))
                else:
                    t = t.replace(",", ".")
            try:
                x = float(t)
            except ValueError:
                bad.append(str(v))
                continue
        if np.isnan(x):
            continue
        if np.isinf(x):
            bad.append(str(v))
            continue
        out[i] = x
    if bad:
        raise LoaderError(f"{label} : valeur(s) non numérique(s) {bad[:5]}"
                          + (f" (+{len(bad) - 5})" if len(bad) > 5 else "") + ".")
    return pd.Series(out, index=col.index)


# --------------------------------------------------------------------------
# Ages
# --------------------------------------------------------------------------

_TOTAL = {"tot", "total", "totale", "ensemble", "ens", "all", "allages", "tous ages", "tous âges", "all ages"}
_UNKNOWN = {"unk", "unknown", "inconnu", "inconnue", "nd", "n.d.", "not stated", "non precise", "non précisé"}
_OPEN = re.compile(r"(\+|\bou plus\b|\bet plus\b|\bet \+|\bor more\b|\band over\b|\band older\b|_pl$|^y_open$|^y_ge\d+|^open$)")
_LT1 = re.compile(r"^(<\s*1|moins de 1|less than 1|under 1)(?:\s*(?:an|ans|year|years))?$")
_PLAIN = re.compile(r"^\d+(?:[.,]0+)?$")
_FRAC = re.compile(r"^\d+[.,]\d*[1-9]\d*$")
_LABELLED = re.compile(r"^(?:âge|age)?\s*(\d+)\s*(?:ans?|years?|yrs?)?(?:\s*(?:révolus?|revolus?|old))?$")
_BAND = re.compile(r"^y?\d+\s*(?:-|–|à|to)\s*\d+")


def age_from_label(raw: Any) -> tuple:
    """-> (kind, age): ('age', 85) | ('open', None) | ('total', None) | ('group', None) |
    ('fraction', None) | ('blank', None) | ('other', None)."""
    t = re.sub(r"\s+", " ", str(raw).replace(" ", " ").strip().lower())
    if not t:
        return "blank", None
    if _PLAIN.match(t):
        return "age", int(float(t.replace(",", ".")))
    if _FRAC.match(t):
        return "fraction", None
    if t in _TOTAL or t.startswith(("tot ", "total ", "ensemble ")):
        return "total", None
    if t in _UNKNOWN:
        return "other", None
    if _OPEN.search(t):
        return "open", None
    if t == "y_lt1" or _LT1.match(t):
        return "age", 0
    m = re.fullmatch(r"y(\d+)", t) or re.fullmatch(r"d(\d+)", t) or _LABELLED.match(t)
    if m:
        return "age", int(m.group(1))
    if _BAND.match(t):
        return "group", None
    return "other", None


@dataclass
class ParsedAges:
    ages: pd.Series                                  # float, NaN where the cell is not a single age
    excluded: list = field(default_factory=list)     # labels left out (open groups, totals, bands...)
    n_group: int = 0                                 # of which age bands ('0-4')


def parse_ages(col: pd.Series, label: str, open_from: int | None = None) -> ParsedAges:
    """Single ages of a column of age cells (numbers, or labels - see `age_from_label`).

    Everything that is not one age is left out and reported: open groups, totals, bands, blanks.
    `open_from` declares the code from which the age column is an open group (INSEE's AGE2_REVOLU
    stops at '100' meaning 100 and over). A fractional age (2.5) is an error: this tool smooths
    integer-age tables."""
    out = np.full(len(col), np.nan)
    excluded: list = []
    fractional: list = []
    n_group = 0
    for i, v in enumerate(col.tolist()):
        if isinstance(v, (int, np.integer)) and not isinstance(v, bool):
            kind, age = ("age", int(v)) if v >= 0 else ("other", None)
        elif isinstance(v, (float, np.floating)):
            if np.isnan(v):
                kind, age = "blank", None
            elif float(v).is_integer() and v >= 0:
                kind, age = "age", int(v)
            elif v >= 0:
                kind, age = "fraction", None
            else:
                kind, age = "other", None
        else:
            kind, age = age_from_label(v)
        if kind == "age":
            out[i] = age
        elif kind == "fraction":
            fractional.append(str(v))
        else:
            excluded.append("(vide)" if kind == "blank" else str(v).strip())
            n_group += kind == "group"
    if fractional:
        raise LoaderError(f"{label} : âges non entiers {fractional[:5]} - "
                          "l'outil lisse des tables par âge entier.")
    if open_from is not None:
        top = out >= open_from
        excluded += [f"{int(a)}+" for a in dict.fromkeys(out[top])]
        out[top] = np.nan
    return ParsedAges(ages=pd.Series(out, index=col.index), excluded=excluded, n_group=n_group)
