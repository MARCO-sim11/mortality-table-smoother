"""From a raw table to the rows that matter: one slice (a year or a cohort, one segment) and, for each
sex, the quantities found. Nothing is computed here - rows are chosen and columns are named.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .columns import (POPULATION_LIKE, ROLE_KEYS, SEX_KEYS, apply_col_map, canon_sex, detect_columns,
                      guess_quantity)
from .constants import Q_KEY, SEXES
from .errors import AmbiguousSelectionError, ColumnMappingError, LoaderError
from .parsing import norm, parse_ages, to_numeric
from .qc import QCItem, age_list, fmt_ages
from .readers import Source, Table

# INSEE's AGE2_REVOLU codes 0..99 then '100' = "100 ans ou plus" (metadata of the state-civil tables).
AUTO_OPEN_FROM = {"age2revolu": 100}


@dataclass
class Prepared:
    name: str
    title: str
    frame: pd.DataFrame            # the selected slice + '__age__' (int64)
    roles: dict                    # role -> column  (explicit mapping applied)
    explicit: dict                 # what the caller mapped by hand
    sex_cols: dict                 # 'female'/'male'/'total' -> column (wide layout)
    view: str                      # 'period' | 'cohort'
    key: int | None                # the year or cohort selected (None: the file has no such column)
    items: list                    # QCItems produced while selecting
    columns: list
    origin: str | None = None


def prepare(tbl: Table, src: Source, view: str, year, cohort) -> Prepared:
    cols = list(tbl.frame.columns)
    explicit = apply_col_map(cols, src.col_map)
    det = detect_columns(cols)
    roles, sex_cols = det.roles, det.sex_cols
    roles.update({k: v for k, v in explicit.items() if k in ROLE_KEYS})
    sex_cols.update({k: v for k, v in explicit.items() if k in SEX_KEYS})
    if "age" not in roles:
        raise ColumnMappingError(f"{tbl.name} : colonne d'âge introuvable.",
                                 missing=["age"], available=cols)

    if view == "auto":
        wants_cohort = cohort is not None or ("year" not in roles and "cohort" in roles)
        view = "cohort" if wants_cohort else "period"
    role, want, noun, param = (("year", year, "années", "year") if view == "period"
                               else ("cohort", cohort, "générations", "cohort"))
    if view == "cohort" and "cohort" not in roles:
        raise ColumnMappingError(
            f"{tbl.name} : pas de colonne de génération (Cohort, Génération, année de naissance) : "
            "ce fichier ne porte pas de tables générationnelles.", missing=["cohort"], available=cols)
    if view == "cohort" and year is not None:
        raise LoaderError(f"{tbl.name} porte des générations (colonne [{roles['cohort']}]), pas des années : "
                          "utilisez `cohort=` et non `year=`.")
    if view == "period" and "year" not in roles and "cohort" in roles:
        raise ColumnMappingError(
            f"{tbl.name} : ce fichier porte des générations (colonne [{roles['cohort']}]) et aucune année : "
            "lisez-le en vision générationnelle (view='cohort').", missing=["year"], available=cols)

    items = [QCItem("input_note", "info", f"{tbl.name} : {n}") for n in tbl.notes]
    frame, key = tbl.frame, None
    if role in roles:                                # slice first, so every note concerns it only
        vals = to_numeric(frame[roles[role]], f"{tbl.name} [{roles[role]}]")
        avail = sorted({int(v) for v in vals.dropna() if float(v).is_integer()})
        if want is None:
            if len(avail) > 1:
                raise AmbiguousSelectionError(
                    f"{tbl.name} contient {len(avail)} {noun} ({avail[0]}-{avail[-1]}) : "
                    f"précisez `{param}`.", what=role, options=avail)
            key = avail[0] if avail else None
        else:
            if int(want) not in avail:
                raise AmbiguousSelectionError(
                    f"{tbl.name} : {param} {want} absent(e) (disponibles : {avail[0]}-{avail[-1]}).",
                    what=role, options=avail)
            key = int(want)
        if key is not None:
            frame = frame.loc[vals.eq(key)]
    if frame.empty:
        raise LoaderError(f"{tbl.name} : aucune ligne pour {param}={key!r}.")

    frame = _select_segments(frame, det.segments, src.where, tbl.name, cols)

    age_col = roles["age"]
    open_from = src.open_from if src.open_from is not None else AUTO_OPEN_FROM.get(norm(age_col))
    parsed = parse_ages(frame[age_col], f"{tbl.name} [{age_col}]", open_from=open_from)
    keep = parsed.ages.notna()
    if not keep.any():
        if parsed.n_group:
            raise LoaderError(f"{tbl.name} : les âges de [{age_col}] sont des tranches ({parsed.excluded[0]!r}...) : "
                              "le lissage exige des âges individuels (0, 1, 2, ...).")
        raise LoaderError(f"{tbl.name} : aucun âge individuel exploitable dans [{age_col}].")
    if parsed.excluded:
        counts = pd.Series(parsed.excluded).value_counts()
        shown = ", ".join(f"'{k}'" + (f" x{n}" if n > 1 else "") for k, n in counts.head(6).items())
        items.append(QCItem("non_single_ages", "info",
                            f"{len(parsed.excluded)} ligne(s) hors âge individuel exclue(s) "
                            f"(groupe ouvert, total, ...) : {shown}"))
    ex_col = roles.get("ex")
    if ex_col and norm(ex_col) in POPULATION_LIKE and src.exposure_basis == "central":
        items.append(QCItem("population_as_exposure", "info",
                            f"La colonne [{ex_col}] est lue comme exposition (population moyenne). "
                            "Une population au 1er janvier se déclare avec exposure_basis='jan1_mean'."))
    frame = frame.loc[keep].copy()
    frame["__age__"] = parsed.ages[keep].astype("int64")
    return Prepared(name=tbl.name, title=tbl.title, frame=frame, roles=roles, explicit=explicit,
                    sex_cols=sex_cols, view=view, key=key, items=items, columns=cols, origin=tbl.origin)


def _select_segments(frame: pd.DataFrame, segments: list, where, name: str, cols: list) -> pd.DataFrame:
    """Keep one value of each segment column (INSEE CHAMP2 = FM or FR; Eurostat geo). Asked for by
    `where`; required only when the column still holds several values after the year was chosen."""
    lookup = {str(c).strip().lower(): c for c in cols}
    wanted = {}
    for k, v in (where or {}).items():
        col = lookup.get(str(k).strip().lower())
        if col is None:
            raise ColumnMappingError(f"{name} : colonne {k!r} introuvable pour `where`.",
                                     missing=[str(k)], available=cols)
        wanted[col] = str(v).strip()
    for col in dict.fromkeys([*segments, *wanted]):
        vals = frame[col].astype(str).str.strip()
        distinct = [v for v in dict.fromkeys(vals) if v != ""]
        if col in wanted:
            if wanted[col] not in distinct:
                raise AmbiguousSelectionError(
                    f"{name} : [{col}] = {wanted[col]!r} absent (disponibles : {distinct}).",
                    what="segment", options=distinct, column=col)
            frame = frame.loc[vals.eq(wanted[col])]
        elif len(distinct) > 1:
            raise AmbiguousSelectionError(
                f"{name} contient plusieurs valeurs de [{col}] ({', '.join(distinct[:6])}"
                f"{'...' if len(distinct) > 6 else ''}) : choisissez-en une.",
                what="segment", options=distinct, column=col)
    return frame


def value_column(prep: Prepared, q: str, *, single_ok: bool) -> str | None:
    """The column holding quantity `q`: mapped by hand, aliased, or (single_ok) the file's one value column."""
    key = Q_KEY[q]
    if key in prep.explicit:
        return prep.explicit[key]
    if key in prep.roles:
        return prep.roles[key]
    if single_ok:
        if "value" in prep.roles:
            return prep.roles["value"]
        used = set(prep.roles.values()) | set(prep.sex_cols.values()) | {"__age__"}
        rest = [c for c in prep.frame.columns if c not in used]
        if len(rest) == 1:
            return rest[0]
    return None


def extract(prep: Prepared, asked: tuple, quantities: tuple, *, hint=None, single_ok=False) -> dict:
    """Split one input by sex -> {sex: DataFrame(age, <the quantities found>)} for the sexes asked.

    Three layouts: one column per sex (the file holds ONE quantity, named by its title / by the
    caller), a Sex column (rows of each sex), or neither - then the file is attributed to the
    single sex asked for. Total columns / rows are ignored.
    """
    frame, name = prep.frame, prep.name
    ages = frame["__age__"].to_numpy()
    slot_sex = asked[0] if len(asked) == 1 else None
    out: dict = {}

    wide = prep.sex_cols and not any(Q_KEY[q] in prep.explicit for q in Q_KEY)
    if wide:
        q = quantities[0] if len(quantities) == 1 else (hint or guess_quantity(prep.title, name))
        if q not in Q_KEY:
            raise ColumnMappingError(
                f"{name} : fichier à colonnes de sexe dont la grandeur (Dx, Ex ou mx) n'est pas "
                "identifiable - précisez quantity=.", missing=["quantity"], available=prep.columns)
        for s in asked:
            col = prep.sex_cols.get(s.lower())
            if col is None:
                have = [c.capitalize() for c in prep.sex_cols]
                why = (" Cette grandeur n'est fournie que pour les deux sexes confondus (Total) : "
                       "elle ne peut pas servir à un sexe seul." if have == ["Total"] else "")
                raise ColumnMappingError(
                    f"{name} : pas de colonne {s} (colonnes de sexe : {have}).{why}",
                    missing=[s], available=prep.columns)
            out[s] = pd.DataFrame({"age": ages, q: to_numeric(frame[col], f"{name} [{col}]").to_numpy()})
        return out

    cols = {q: c for q in quantities if (c := value_column(prep, q, single_ok=single_ok))}
    if not cols and "value" in prep.roles:                  # a lone 'value' column: its quantity is the caller's
        q = quantities[0] if len(quantities) == 1 else hint
        if q is None:
            raise ColumnMappingError(
                f"{name} : colonne de valeurs [{prep.roles['value']}] sans grandeur identifiable - "
                "précisez quantity= (Dx, Ex ou mx).", missing=["quantity"], available=prep.columns)
        cols[q] = prep.roles["value"]
    if not cols:
        raise ColumnMappingError(
            f"{name} : aucune colonne {' / '.join(quantities)} identifiée - associez-la via « Paramètres ».",
            missing=list(quantities), available=prep.columns)
    values = {q: to_numeric(frame[c], f"{name} [{c}]").to_numpy() for q, c in cols.items()}

    if "sex" in prep.roles:
        labels = frame[prep.roles["sex"]].map(canon_sex).to_numpy()
        present = [s for s in SEXES if s in set(labels)]
        for s in asked:
            if s not in present:
                raise AmbiguousSelectionError(
                    f"{name} : sexe {s!r} absent de la colonne [{prep.roles['sex']}] "
                    f"(disponibles : {present}).", what="sex", options=present)
            m = labels == s
            out[s] = pd.DataFrame({"age": ages[m], **{q: v[m] for q, v in values.items()}})
        return out

    if slot_sex is None:
        raise ColumnMappingError(
            f"{name} : ce fichier n'indique pas le sexe (colonnes Female / Male, ou colonne Sexe). "
            "Choisissez un seul sexe pour le lui attribuer.", missing=["sex"], available=prep.columns)
    out[slot_sex] = pd.DataFrame({"age": ages, **values})
    return out


def merge_inputs(pieces: list, names: list) -> pd.DataFrame:
    """Join the one-quantity frames of several files on age (a file's ages must be unique)."""
    indexed = []
    for f, name in zip(pieces, names):
        dup = age_list(f["age"][f["age"].duplicated(keep=False)])
        if dup:
            raise LoaderError(f"{name} : âges dupliqués {fmt_ages(dup)} (après sélection) - "
                              "appariement avec les autres fichiers impossible.")
        indexed.append(f.set_index("age"))
    return pd.concat(indexed, axis=1).rename_axis("age").reset_index()
