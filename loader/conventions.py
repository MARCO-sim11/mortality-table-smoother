"""The one place where the loader computes a quantity instead of reading it.

Official population series (INSEE, Eurostat, ONS...) give the population at 1 January, not the
central exposure the Poisson likelihood needs. The usual approximation is the trapezoid rule:

    E(x, t) = ( P(x, t) + P(x, t+1) ) / 2          P = population aged x at 1 January

It is off by default (`exposure_basis='central'`: a column is taken as the exposure it is called)
and, when asked for, it is written in the notes of the result so that nothing about it is hidden.
"""
from __future__ import annotations

from .errors import AmbiguousSelectionError, LoaderError
from .qc import QCItem
from .readers import Source, Table
from .select import Prepared, extract, prepare


def apply_exposure_basis(tbl: Table, src: Source, prep: Prepared, part: dict, asked: tuple,
                         quantities: tuple, *, hint, single_ok) -> tuple:
    """-> (part with `Ex` converted, note or None). `part` is `extract`'s {sex: frame} for year t."""
    if src.exposure_basis == "central":
        return part, None
    if src.exposure_basis != "jan1_mean":
        raise LoaderError(f"exposure_basis inconnu : {src.exposure_basis!r} (admis : central, jan1_mean).")
    if prep.view != "period" or prep.key is None:
        raise LoaderError(f"{prep.name} : exposure_basis='jan1_mean' suppose une table du moment "
                          "datée (colonne année).")
    t = prep.key
    try:
        nxt = prepare(tbl, src, "period", t + 1, None)
    except AmbiguousSelectionError as exc:
        raise LoaderError(f"{prep.name} : la population au 1er janvier {t + 1} manque, or l'exposition "
                          f"de {t} en est la moyenne avec celle de {t + 1} ({exc}).") from exc
    part_next = extract(nxt, asked, quantities, hint=hint, single_ok=single_ok)

    out = {}
    for sex, f in part.items():
        g = part_next[sex]
        if "Ex" not in f.columns:
            raise LoaderError(f"{prep.name} : exposure_basis='jan1_mean' s'applique à une exposition "
                              "(population) : aucune colonne Ex dans ce fichier.")
        if f["age"].duplicated().any() or g["age"].duplicated().any():
            raise LoaderError(f"{prep.name} : âges dupliqués, la moyenne des populations est impossible.")
        merged = f.merge(g[["age", "Ex"]].rename(columns={"Ex": "Ex_next"}), on="age", how="left")
        merged["Ex"] = (merged["Ex"] + merged["Ex_next"]) / 2          # a missing t+1 stays missing
        out[sex] = merged.drop(columns="Ex_next")
    note = QCItem("exposure_convention", "info",
                  f"{prep.name} : Ex = moyenne des populations au 1er janvier {t} et {t + 1} "
                  "(règle du trapèze) - approximation de l'exposition centrale.")
    return out, note
