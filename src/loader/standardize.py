"""From the quantities found (Dx, Ex, mx) to a standard per-sex table: age | Dx | Ex | mx_raw | source."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .constants import MX_ATOL, MX_RTOL
from .errors import ColumnMappingError, LoaderError
from .qc import QCItem, age_list, fmt_ages


def build_standard(frame: pd.DataFrame, *, E_ref, source: str, items: list,
                    available=()) -> tuple[pd.DataFrame, int]:
    """frame has `age` plus any of Dx / Ex / mx. Returns (standard df, n_missing)."""
    f = frame.copy()
    has_d, has_e, has_m = "Dx" in f, "Ex" in f, "mx" in f
    given_mx = f["mx"].copy() if has_m else None

    if has_d and has_e:
        pass
    elif has_e and has_m:
        f["Dx"] = f["mx"] * f["Ex"]
        items.append(QCItem("dx_reconstructed", "info",
                            "Dx reconstruit = mx x Ex (valeur non entière, au pas d'arrondi de mx près) ; "
                            "mx_raw reste exactement le mx fourni."))
    elif has_m and not has_d and not has_e:
        if E_ref is None:
            raise ColumnMappingError(
                "Seul mx est disponible : renseignez une exposition de référence E_ref (> 0).",
                missing=["Dx", "Ex"], available=available)
        if not (np.isfinite(E_ref) and E_ref > 0):
            raise LoaderError(f"E_ref doit être un nombre > 0 (reçu : {E_ref!r}).")
        f["Ex"] = float(E_ref)
        f["Dx"] = f["mx"] * float(E_ref)
        items.append(QCItem("ex_reference", "info",
                            f"Seul mx disponible : Ex = E_ref = {E_ref:g} pour tous les âges et "
                            "Dx = mx x E_ref. Les tests statistiques (chi2, AIC/BIC) dépendent de "
                            "cette valeur arbitraire."))
    elif has_d and has_m:
        raise ColumnMappingError(
            "Ex introuvable : Dx et mx ne permettent pas de reconstruire Ex aux âges où Dx = 0.",
            missing=["Ex"], available=available)
    elif has_d:
        raise ColumnMappingError("Ex introuvable (seul Dx est fourni).", missing=["Ex"],
                                 available=available)
    elif has_e:
        raise ColumnMappingError("Dx introuvable (seul Ex est fourni) : fournissez Dx ou mx.",
                                 missing=["Dx"], available=available)
    else:
        raise ColumnMappingError("Aucune des grandeurs Dx / Ex / mx n'a été identifiée.",
                                 missing=["Dx", "Ex"], available=available)

    miss = f["Dx"].isna() | f["Ex"].isna()
    n_missing = int(miss.sum())
    if n_missing:
        a = age_list(f.loc[miss, "age"])
        which = " et ".join(q for q in ("Dx", "Ex") if f.loc[miss, q].isna().any())
        items.append(QCItem("missing_values", "orange",
                            f"{n_missing} valeur(s) manquante(s) ({which} absent) : "
                            f"âge(s) exclu(s) : {fmt_ages(a)}", a))
        f = f.loc[~miss]

    noinfo = (f["Ex"] == 0) & (f["Dx"] == 0)
    if noinfo.any():
        a = age_list(f.loc[noinfo, "age"])
        items.append(QCItem("no_exposure", "orange",
                            f"Ex = 0 et Dx = 0 (aucune information, mx = 0/0) : âge(s) exclu(s) : "
                            f"{fmt_ages(a)}", a))
        f = f.loc[~noinfo]
    if f.empty:
        raise LoaderError("Aucune ligne exploitable après exclusion des valeurs manquantes.")

    D = f["Dx"].to_numpy(float)
    E = f["Ex"].to_numpy(float)
    mx_raw = np.divide(D, E, out=np.full(len(f), np.nan), where=E > 0)

    if has_d and has_e and has_m:
        g = given_mx.loc[f.index].to_numpy(float)
        comparable = (E > 0) & ~np.isnan(g)
        bad = comparable & ~np.isclose(mx_raw, g, rtol=MX_RTOL, atol=MX_ATOL)
        if bad.any():
            a = age_list(f["age"].to_numpy()[bad])
            items.append(QCItem("mx_consistency", "orange",
                                f"mx fourni incohérent avec Dx/Ex à {len(a)} âge(s) : {fmt_ages(a)} "
                                "(mx_raw = Dx/Ex est retenu).", a))

    df = pd.DataFrame({"age": f["age"].to_numpy().astype("int64"),
                       "Dx": D, "Ex": E, "mx_raw": mx_raw, "source": source})
    df = df.sort_values("age", kind="mergesort").reset_index(drop=True)
    return df, n_missing
