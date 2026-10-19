"""Which column is which: the vocabulary of the sources the loader reads (HMD, INSEE, Eurostat, national
offices, hand-made actuarial files, French and English).

Matching is on the whole normalised name - never a substring guess - so a column is only mapped
automatically when there is no doubt. qx is deliberately absent from mx: a probability is not a rate.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Mapping

from .constants import Q_KEY
from .errors import ColumnMappingError, LoaderError
from .parsing import norm

ROLE_ALIASES = {
    "age": {"age", "ages", "x", "agex", "singleyearages", "singleyearagescode", "agecode"},
    "year": {"year", "years", "annee", "annees", "yr", "calendaryear", "anneecalendaire", "period",
             "periode", "timeperiod", "time", "anneeref", "referenceyear"},
    "cohort": {"cohort", "cohorts", "cohorte", "cohortes", "generation", "generations",
               "birthyear", "yearofbirth", "anneedenaissance"},
    "sex": {"sex", "sexe", "gender", "genre", "gendercode", "sexcode", "sexecode"},
    "dx": {"dx", "d", "deaths", "death", "deces", "nbdeces", "nbdec", "nombredeces", "nombrededeces",
           "observeddeaths", "deathcount", "decesobserves"},
    "ex": {"ex", "e", "exposure", "exposures", "exposuretorisk", "exposurerisk", "centralexposure",
           "expo", "exposition", "expositioncentrale", "expositionaurisque", "personyears",
           "anneespersonnes", "py", "pyears", "populationexposee",
           "population", "pop", "populationmoyenne", "popmoyenne", "midyearpopulation"},
    "mx": {"mx", "m", "rate", "rates", "deathrate", "deathrates", "mortalityrate", "mortalityrates",
           "taux", "tauxbrut", "tauxbruts", "tauxdemortalite", "tauxdedeces", "mux", "muxraw", "mxraw",
           "crudemx", "crude", "mubrut"},
    "value": {"obsvalue", "value", "values", "valeur", "valeurs", "nombre", "nb", "effectif", "effectifs",
              "count", "number"},
}
# a value column that holds a population, read as the exposure: worth telling the user
POPULATION_LIKE = {"population", "pop", "populationmoyenne", "popmoyenne", "midyearpopulation"}

# columns that split the rows into sub-populations (a geography, a nationality...): one value has to
# be chosen, or the same age would appear twice. INSEE: CHAMP / CHAMP2 (FM = metropolitan France, FR = France).
SEGMENT_ALIASES = {"champ", "champ2", "geo", "geography", "area", "region", "country", "countrycode",
                   "pays", "territoire", "popname", "nationality", "nationalite", "departement", "dept",
                   "race", "segment", "categorie", "category"}

SEX_VALUE_ALIASES = {
    "Female": {"f", "female", "females", "femme", "femmes", "feminin", "women", "woman"},
    "Male": {"m", "male", "males", "h", "homme", "hommes", "masculin", "men", "man"},
    "Total": {"t", "total", "tot", "both", "bothsexes", "ensemble", "allsexes", "mixte"},
}
# As a COLUMN NAME a single letter is too ambiguous ('m' is also a rate). A Total column is
# recognised only so that it is not mistaken for a value column - Total is never used.
SEX_COLUMN_ALIASES = {k: {a for a in v if len(a) > 1} for k, v in SEX_VALUE_ALIASES.items()}
SEX_KEYS = ("female", "male", "total")
ROLE_KEYS = tuple(ROLE_ALIASES)
MAP_KEYS = set(ROLE_KEYS) | set(SEX_KEYS)


def role_of(n: str) -> str | None:
    """Role of a normalised column name. Beyond the exact aliases: INSEE's ANNEE_1975_2024 is a year,
    AGE2_REVOLU / AGE_ATTEINT / AGE_REVOLU_R are ages."""
    for role, aliases in ROLE_ALIASES.items():
        if n in aliases:
            return role
    if re.fullmatch(r"(annee|year)\d*", n):
        return "year"
    if re.fullmatch(r"age\d*(revolu\w*|atteint\w*)", n):
        return "age"
    return None


def canon_sex(label) -> str | None:
    n = norm(label)
    for canon, aliases in SEX_VALUE_ALIASES.items():
        if n in aliases:
            return canon
    return None


@dataclass
class Detected:
    roles: dict = field(default_factory=dict)        # 'age' -> column ...
    sex_cols: dict = field(default_factory=dict)     # 'female' / 'male' / 'total' -> column (wide layout)
    segments: list = field(default_factory=list)     # columns that split the rows


def detect_columns(cols) -> Detected:
    out = Detected()
    for c in cols:
        n = norm(c)
        role = role_of(n)
        if role:
            out.roles.setdefault(role, c)
        elif n in SEGMENT_ALIASES:
            out.segments.append(c)
        else:
            for canon, aliases in SEX_COLUMN_ALIASES.items():
                if n in aliases:
                    out.sex_cols.setdefault(canon.lower(), c)
    return out


def is_header_like(cells) -> bool:
    """A row of cells that looks like column names: at least two known ones (a year column counts)."""
    score = 0
    for c in cells:
        n = norm(c)
        if not n:
            continue
        known = (role_of(n) or n in SEGMENT_ALIASES or n in set().union(*SEX_COLUMN_ALIASES.values())
                 or re.fullmatch(r"(1[6-9]|20)\d\d", n))
        score += bool(known)
    return score >= 2


def apply_col_map(cols: list, col_map: Mapping | None) -> dict:
    """Validate the user's mapping; keys are case-insensitive roles
    (age, year, cohort, sex, Dx, Ex, mx, value, Female, Male), values column names.
    An empty value means 'not mapped' (a dropdown left on its blank entry)."""
    out: dict = {}
    if not col_map:
        return out
    lookup = {str(c).strip(): c for c in cols}
    for key, col in col_map.items():
        k = norm(key)
        if k not in MAP_KEYS:
            raise LoaderError(f"col_map : clé inconnue {key!r} "
                              f"(admises : age, year, cohort, sex, Dx, Ex, mx, value, Female, Male).")
        if col is None or str(col).strip() == "":
            continue
        if str(col).strip() not in lookup:
            raise ColumnMappingError(f"col_map : colonne {col!r} introuvable.",
                                     missing=[key], available=cols)
        out[k] = lookup[str(col).strip()]
    return out


def canon_quantity(q: str | None) -> str | None:
    if q is None:
        return None
    for canon in Q_KEY:
        if norm(q) == canon.lower():
            return canon
    raise LoaderError(f"quantity inconnue : {q!r} (admises : Dx, Ex, mx).")


def guess_quantity(title: str, name: str) -> str | None:
    """Which quantity an HMD-style file holds, from its title line or file name."""
    t = f"{title} {name}".lower()
    if "expos" in t:
        return "Ex"
    if re.search(r"death\s*rates?|mortality\s*rates?|taux|(?<![a-z])c?mx(?![a-z])", t):
        return "mx"
    if re.search(r"death|d[eé]c[eè]s|(?<![a-z])dx(?![a-z])", t):
        return "Dx"
    return None
