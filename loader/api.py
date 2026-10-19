"""The loader's public functions: three ways to bring crude mortality data in, one table out.

    Option A  load_separate   one file of rates (or deaths) and one of exposure, sexes in columns
    Option B  load_by_sex     one file per sex
    Option C  load_data       one file of elementary data: age, sex, year, deaths, exposure   (recommended)

All three end in the same pipeline (`_load`): read, slice (view, year or cohort, segment), split by sex,
resolve Dx / Ex / mx, check - and return `sex | age | Dx | Ex | mx_raw | source`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import pandas as pd

from .columns import canon_quantity, canon_sex, detect_columns, guess_quantity
from .constants import Q_KEY, SEXES, VIEWS
from .conventions import apply_exposure_basis
from .errors import DataQualityError, LoaderError
from .parsing import norm, to_numeric
from .qc import QCItem, check_quality
from .readers import Source, as_source, read_table
from .select import extract, merge_inputs, prepare
from .standardize import build_standard

_BOTH = {"both", "deux", "lesdeux", "2", "mf", "fm"}


def _sexes_wanted(sex) -> tuple:
    """'both' -> (Female, Male); 'Female' / 'f' / 'femmes' ... -> (Female,)."""
    if norm(sex) in _BOTH:
        return SEXES
    canon = canon_sex(sex)
    if canon in SEXES:
        return (canon,)
    raise LoaderError(f"Sexe inconnu : {sex!r} (attendu : Female, Male ou both).")


def _check_view_args(view, year, cohort) -> None:
    if view not in ("auto", *VIEWS):
        raise LoaderError(f"Vision inconnue : {view!r} (admises : auto, period, cohort).")
    if year is not None and cohort is not None:
        raise LoaderError("`year` et `cohort` sont exclusifs : une table du moment ou une génération.")
    if view == "period" and cohort is not None:
        raise LoaderError("`cohort` n'a de sens qu'en vision générationnelle (view='cohort').")
    if view == "cohort" and year is not None:
        raise LoaderError("`year` n'a de sens qu'en vision du moment (view='period').")


def _load(inputs: list, sexes: tuple, *, view, year, cohort, E_ref, merge: bool, hint=None) -> pd.DataFrame:
    """The one pipeline behind the three loaders.

    inputs  [(Source, quantities, sexes asked of this input)]
    merge   True: the inputs are one file per quantity, joined on age for each sex;
            False: each sex comes from a single input (an all-in-one file, or one file per sex).
    """
    parts, names, keys, views, notes, available = [], [], set(), set(), [], []
    for src, quantities, asked in inputs:
        tbl = read_table(src)
        prep = prepare(tbl, src, view, year, cohort)
        part = extract(prep, asked, quantities, hint=hint, single_ok=merge)
        part, note = apply_exposure_basis(tbl, src, prep, part, asked, quantities, hint=hint, single_ok=merge)
        parts.append(part)
        names.append(prep.name)
        views.add(prep.view)
        notes += prep.items + ([note] if note else [])
        available += prep.columns
        if prep.key is not None:
            keys.add(prep.key)
    if len(views) > 1:
        raise LoaderError("Les fichiers ne sont pas dans la même vision (tables du moment / "
                          "générationnelles) : l'un porte une colonne Year, l'autre Cohort.")
    if len(keys) > 1:
        what = "années" if "period" in views else "générations"
        raise LoaderError(f"Les fichiers portent sur des {what} différentes : {sorted(keys)}.")

    source = " + ".join(dict.fromkeys(names))
    reports, frames = {}, []
    for s in sexes:
        pieces = [p[s] for p in parts if s in p]
        frame = merge_inputs(pieces, names) if merge else pieces[0]
        items: list[QCItem] = []
        df_s, n_missing = build_standard(frame, E_ref=E_ref, source=source, items=items,
                                         available=sorted(set(available)))
        report = check_quality(df_s, n_missing=n_missing, source=source)
        report.items.extend(i for i in items if i.level != "info")        # this sex's findings
        notes += [i for i in items if i.level == "info"]                   # facts about the input
        df_s.insert(0, "sex", s)
        reports[s] = report
        frames.append(df_s)

    df = pd.concat(frames, ignore_index=True)
    view_used = next(iter(views))
    key = next(iter(keys), None)
    df.attrs.update(view=view_used, year=key if view_used == "period" else None,
                    cohort=key if view_used == "cohort" else None, source=source,
                    sexes=list(sexes), qc=reports,
                    notes=list({(i.check, i.message): i for i in notes}.values()))
    if any(r.blocking for r in reports.values()):
        raise DataQualityError(reports, df)
    return df


# --------------------------------------------------------------------------
# Option C - one file of elementary data
# --------------------------------------------------------------------------


def load_data(filepath, fmt: str = "auto", sex: str = "both", year: int | None = None,
              col_map: Mapping | None = None, E_ref: float | None = None, *,
              cohort: int | None = None, view: str = "auto", quantity: str | None = None,
              sheet: int | str = 0, header: int | None = None, where: Mapping | None = None,
              open_from: int | None = None, exposure_basis: str = "central") -> pd.DataFrame:
    """Option C (spec 1.4): ONE file of elementary data - age, sex, year, deaths, exposure (a long file
    with one row per cell, as INSEE, Eurostat or a database export give it). The loader slices it,
    splits it by sex, checks it and computes mx_raw = Dx / Ex.

    sex       'both' (Female and Male), 'Female' or 'Male' (also f / m / h, French labels). A file that
              does not say which sex it holds is attributed to the single sex asked for.
    year / cohort / view   period tables take `year=`, generation tables `cohort=`. Required when the
              file holds several (AmbiguousSelectionError lists them).
    where     {column: value} to keep one segment (INSEE {'CHAMP2': 'FM'}, Eurostat {'geo': 'FR'}).
    fmt, col_map, sheet, header, open_from, exposure_basis   how to read the file - or pass a `Source`
              as `filepath`. header=None finds the row holding the column names.
    E_ref     reference exposure when only mx is available (spec 1.3).
    quantity  for a file whose values are in one unnamed column (Eurostat OBS_VALUE) or in an
              HMD-style file whose quantity cannot be read from its title: 'Dx' | 'Ex' | 'mx'.

    Returns the standard table; raises FileFormatError, ColumnMappingError, AmbiguousSelectionError,
    DataQualityError (a blocking control failed).
    """
    sexes, qty = _sexes_wanted(sex), canon_quantity(quantity)        # bad arguments fail first
    _check_view_args(view, year, cohort)
    src = filepath if isinstance(filepath, Source) else Source(
        filepath, sheet, header, col_map, fmt, where, open_from, exposure_basis)
    return _load([(src, tuple(Q_KEY), sexes)], sexes, view=view, year=year, cohort=cohort,
                 E_ref=E_ref, merge=False, hint=qty)


# --------------------------------------------------------------------------
# Option B - one file per sex
# --------------------------------------------------------------------------


def load_by_sex(*, female=None, male=None, view: str = "auto", year: int | None = None,
                cohort: int | None = None, E_ref: float | None = None) -> pd.DataFrame:
    """Option B: one file per sex, each with its deaths, exposures or rates by age -> one table
    holding the sexes given.

    Each input is a path, an uploaded file or a `Source`. The slot says the sex: a file with no
    Female/Male column or Sex column is attributed to its slot. Both files must be on the same
    year (or cohort)."""
    given = {s: as_source(f) for s, f in (("Female", female), ("Male", male)) if f is not None}
    if not given:
        raise LoaderError("Fournissez au moins un fichier (female= et/ou male=).")
    _check_view_args(view, year, cohort)
    return _load([(src, tuple(Q_KEY), (s,)) for s, src in given.items()], tuple(given),
                 view=view, year=year, cohort=cohort, E_ref=E_ref, merge=False)


# --------------------------------------------------------------------------
# Option A - one file per quantity
# --------------------------------------------------------------------------


def load_separate(*, deaths=None, exposure=None, rates=None, sex: str = "both",
                  view: str = "auto", year: int | None = None, cohort: int | None = None,
                  E_ref: float | None = None) -> pd.DataFrame:
    """Option A: a file of rates (or deaths) and a file of exposure, sexes in columns (the HMD layout,
    historical actuarial tables, Excel sheets).

    Each input is a path, an uploaded file or a `Source`. A file lays its sexes out as one column
    per sex (Female / Male; a Total column is ignored) or as rows with a Sex column; one that does
    not say is attributed to the single sex asked for. Ages are matched across files: an age missing
    from one counts as a missing value and is excluded (and reported); duplicate ages inside a file
    cannot be paired and raise LoaderError. All files must be on the same year (or cohort).
    Enough inputs: Dx + Ex; Ex + mx (Dx = mx * Ex - rates alone do not give the exact deaths without
    the exposure and its convention); mx alone with `E_ref`.
    """
    sexes = _sexes_wanted(sex)
    _check_view_args(view, year, cohort)
    inputs = [(as_source(src), (q,), sexes)
              for q, src in (("Dx", deaths), ("Ex", exposure), ("mx", rates)) if src is not None]
    if not inputs:
        raise LoaderError("Aucun fichier fourni (deaths=, exposure= et/ou rates=).")
    return _load(inputs, sexes, view=view, year=year, cohort=cohort, E_ref=E_ref, merge=True)


# --------------------------------------------------------------------------
# The standard file, and a peek at a file before loading it
# --------------------------------------------------------------------------


def standard_table(df: pd.DataFrame) -> pd.DataFrame:
    """The loaded table as a plain tidy file: [year | cohort], sex, age, Dx, Ex, mx_raw.

    Both sexes in one table, self-describing (the first column says which view and which
    year / generation), and readable by `load_data` again - it is an Option C file."""
    out = df[["sex", "age", "Dx", "Ex", "mx_raw"]].copy()
    cohort, year = df.attrs.get("cohort"), df.attrs.get("year")
    if cohort is not None:
        out.insert(0, "cohort", cohort)
    elif year is not None:
        out.insert(0, "year", year)
    return out


@dataclass
class FileInfo:
    name: str
    fmt: str                       # 'hmd' | 'csv' | 'excel'
    layout: str                    # 'sex_columns' (Female/Male columns) | 'roles' (age / Dx / Ex ...)
    columns: list
    auto_map: dict                 # role -> column found automatically
    quantity: str | None           # for 'sex_columns' files: 'Dx' | 'Ex' | 'mx' | None
    views: list                    # views the file can serve: 'period' (Year), 'cohort' (Cohort)
    years: list
    cohorts: list
    sexes: list                    # Female / Male, as found; [] = the file does not say
    sheets: list | None            # Excel sheet names
    n_rows: int
    segments: dict                 # segment column -> its values, when it has several (INSEE CHAMP2: FM, FR)
    origin: str | None             # 'HMD' | 'INSEE' | 'Eurostat' | None


def inspect_file(src, fmt: str = "auto", *, sheet: int | str = 0, header: int | None = None) -> FileInfo:
    """What the UI needs before calling a loader: the views, the years or cohorts to offer, the
    sexes, the segments to choose in, the Excel sheets, the columns and what was auto-mapped
    (anything missing from `auto_map` gets a dropdown, spec 1.3). `src` is a path, an uploaded
    file or a `Source`."""
    s = src if isinstance(src, Source) else Source(src, sheet, header, None, fmt)
    tbl = read_table(s)
    cols = list(tbl.frame.columns)
    det = detect_columns(cols)
    roles, sex_cols = det.roles, det.sex_cols

    def values(role):
        if role not in roles:
            return []
        v = to_numeric(tbl.frame[roles[role]], f"{tbl.name} [{roles[role]}]")
        return sorted({int(x) for x in v.dropna() if float(x).is_integer()})

    if any(k in sex_cols for k in ("female", "male")):
        sexes = [x for x in SEXES if x.lower() in sex_cols]
    elif "sex" in roles:
        found = {lab for lab in tbl.frame[roles["sex"]].map(canon_sex) if lab}
        sexes = [x for x in SEXES if x in found]
    else:
        sexes = []
    segments = {}
    for col in det.segments:
        distinct = [v for v in dict.fromkeys(tbl.frame[col].astype(str).str.strip()) if v]
        if len(distinct) > 1:
            segments[col] = distinct[:200]
    return FileInfo(
        name=tbl.name, fmt=tbl.fmt, layout="sex_columns" if sex_cols else "roles",
        columns=cols, auto_map={**roles, **sex_cols},
        quantity=guess_quantity(tbl.title, tbl.name) if sex_cols else None,
        views=[v for v, r in (("period", "year"), ("cohort", "cohort")) if r in roles],
        years=values("year"), cohorts=values("cohort"), sexes=sexes,
        sheets=tbl.sheets, n_rows=len(tbl.frame), segments=segments, origin=tbl.origin)
