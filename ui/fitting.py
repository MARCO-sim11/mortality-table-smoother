"""Settings page helper: call the zone modules (spec 3 convention)."""
from __future__ import annotations

import importlib

import pandas as pd

from src import config as C

# --------------------------------------------------------------------------
# Tab 2 helper: call the zone modules (spec 3 convention)
# --------------------------------------------------------------------------


def run_estimations(df: pd.DataFrame, cfg: C.SmootherConfig, progress=None,
                    import_module=importlib.import_module) -> tuple:
    """Call `fit_<key>(ages, Dx, Ex, cfg)` for every ticked method, on ONE sex's table.

    Returns ({method_key: FitResult}, status rows). One method failing never stops
    the others; a module or function that does not exist yet is "non implémenté".
    `progress(fraction, text)` is called before each method and once at the end.
    """
    chosen = cfg.selected_methods()
    results, status = {}, []
    for i, m in enumerate(chosen):
        if progress:
            progress(i / len(chosen), f"{m.label} ({C.ZONE_LABELS[m.zone]})…")
        row = {"zone": C.ZONE_LABELS[m.zone], "method": m.key, "label": m.label,
               "status": "ok", "converged": None, "loglik": None, "n_iter": None, "detail": ""}
        status.append(row)
        try:
            module = import_module(f"src.{m.module}")
        except ModuleNotFoundError as exc:
            if exc.name == f"src.{m.module}":
                row.update(status="non implémenté", detail=f"src/{m.module}.py n'existe pas encore")
            else:
                row.update(status="erreur", detail=f"import de src/{m.module}.py : {exc}")
            continue
        except Exception as exc:                                    # a real bug in an existing module
            row.update(status="erreur", detail=f"import de src/{m.module}.py : {type(exc).__name__}: {exc}")
            continue
        fit = getattr(module, m.fit_name, None)
        if fit is None:
            row.update(status="non implémenté", detail=f"{m.fit_name}() absente de src/{m.module}.py")
            continue
        sub = C.zone_slice(df, cfg, m.zone)
        try:
            res = fit(sub["age"].to_numpy(), sub["Dx"].to_numpy(), sub["Ex"].to_numpy(), cfg)
        except Exception as exc:
            row.update(status="erreur", detail=f"{type(exc).__name__}: {exc}")
            continue
        if not isinstance(res, C.FitResult):
            row.update(status="erreur", detail=f"{m.fit_name}() doit retourner un FitResult "
                                               f"(reçu : {type(res).__name__}).")
        elif (res.method, res.zone) != (m.key, m.zone):
            row.update(status="erreur", detail=f"FitResult incohérent : method={res.method!r}, "
                                               f"zone={res.zone!r} (attendu : {m.key!r}, {m.zone!r}).")
        else:
            results[m.key] = res
            row.update(converged=bool(res.converged), loglik=float(res.loglik),
                       n_iter=int(res.n_iter), detail=res.message)
    if progress:
        progress(1.0, "Terminé")
    return results, status
