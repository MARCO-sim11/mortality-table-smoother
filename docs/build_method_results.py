"""Rebuild the results tables of the method documents (docs/methodes/*.md) from the code.

    python docs/build_method_results.py          # from the project root: rewrites the tables in place
    python docs/build_method_results.py --check  # only says which tables are out of date (exit code 1 if any)

Each document that shows numbers holds ONE block between the markers

    <!-- resultats:debut -->  ...  <!-- resultats:fin -->

and the block is produced here: France 2020 (the file of sample_data/, both sexes), the default settings of the app
(omega1 = 20, omega2 = 85, omega3 = 110, sigma = 3, h = 1), every method fitted with the code of src/. tests/test_method_docs.py
runs the same functions and fails when a document disagrees with the code, so the numbers of the documentation cannot drift.
"""
from __future__ import annotations

import importlib
import re
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import config as C            # noqa: E402
from src import evaluator as EV        # noqa: E402
from src import joiner as J            # noqa: E402
from src import loader as L            # noqa: E402

DOC_DIR = ROOT / "docs" / "methodes"
SAMPLE = ROOT / "sample_data" / "FRA_taux_bruts_moment_2020_female_male.csv"
START, END = "<!-- resultats:debut -->", "<!-- resultats:fin -->"
SEX_NAME = {"Female": "Féminin", "Male": "Masculin"}
PARAM_LABELS = {
    "poly3": [("a0", "a₀"), ("a1", "a₁"), ("a2", "a₂"), ("a3", "a₃")],
    "gompertz": [("alpha", "α"), ("beta", "β"), ("B", "B"), ("c", "c")],
    "makeham": [("A", "A"), ("B", "B"), ("c", "c")],
    "makeham2": [("A", "A"), ("B", "B"), ("c", "c"), ("D", "D"), ("d", "d")],
    "perks": [("A", "A"), ("B", "B"), ("c", "c"), ("D", "D")],
    "kannisto": [("a", "a"), ("b", "b")],
    "beard": [("B", "B"), ("c", "c"), ("K", "K")],
    "coale_kisker": [("k", "k"), ("q_omega2", "q(ω₂)"), ("q_target", "q cible")],
}


def num(value: float, digits: int = 4) -> str:
    """Four significant digits, the way every table of the documents prints a number."""
    return f"{value:.{digits}g}"


@lru_cache(maxsize=1)
def sample_table() -> pd.DataFrame:
    return L.load_data(SAMPLE)


@lru_cache(maxsize=1)
def fits() -> tuple:
    """(cfg, {sex: {key: FitResult}}): the ten methods fitted on France 2020 with the app's default settings."""
    df = sample_table()
    cfg = C.default_config(df["age"].unique())
    cfg.all_methods = ["spline_all"]
    out = {}
    for sex in ("Female", "Male"):
        d = df[df["sex"] == sex]
        res = {}
        for method in cfg.selected_methods():
            sub = C.zone_slice(d, cfg, method.zone)
            fit = getattr(importlib.import_module(f"src.{method.module}"), method.fit_name)
            res[method.key] = fit(sub["age"].to_numpy(), sub["Dx"].to_numpy(), sub["Ex"].to_numpy(), cfg)
        out[sex] = res
    return cfg, out


def evaluation(sex: str, with_global: bool = False) -> pd.DataFrame:
    cfg, res = fits()
    df = sample_table()
    chosen = {k: r for k, r in res[sex].items() if with_global or k != "spline_all"}
    return EV.evaluate_fits(chosen, df[df["sex"] == sex], cfg, EV.WH_H)


def _params_text(key: str, result: C.FitResult) -> str:
    if key == "spline":
        knots = result.params["knots"]
        return f"{len(knots)} nœuds : " + ", ".join(f"{k:g}" for k in knots)
    return ", ".join(f"{label} = {num(result.params[name])}" for name, label in PARAM_LABELS[key])


def _table(header: list, rows: list) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def zone_method_block(key: str) -> str:
    """Parameters and fit criteria of one zone method, both sexes."""
    cfg, res = fits()
    rows = []
    for sex in ("Female", "Male"):
        table = evaluation(sex).set_index("method")
        row = table.loc[key]
        rows.append([SEX_NAME[sex], _params_text(key, res[sex][key]), int(row["n"]), f"{row['loglik']:.1f}", f"{row['aic']:.1f}",
                     f"{row['delta_aic']:.1f}", "—" if np.isnan(row["runs_p"]) else f"{row['runs_p']:.3f}", f"{100 * row['outside']:.1f} %"])
    return _table(["Sexe", "Paramètres estimés", "n", "Log-vraisemblance", "AIC", "ΔAIC (zone)", "p(runs)", "\\|r\\| > 2"], rows)


def spline_all_block() -> str:
    """The knots of the global spline and its row in each zone, both sexes."""
    cfg, res = fits()
    lines = []
    for sex in ("Female", "Male"):
        knots = res[sex]["spline_all"].params["knots"]
        lines.append(f"- {SEX_NAME[sex]} : {len(knots)} nœuds ({', '.join(f'{k:g}' for k in knots)}).")
    rows = []
    for sex in ("Female", "Male"):
        table = evaluation(sex, with_global=True)
        for row in table[table["method"] == "spline_all"].itertuples():
            rows.append([SEX_NAME[sex], C.ZONE_LABELS[row.zone], row.k, row.n, f"{row.loglik:.1f}", f"{row.aic:.1f}", f"{row.delta_aic:.1f}",
                         f"{row.runs_p:.3f}", "oui" if row.suggested else "non"])
    return "\n".join(lines) + "\n\n" + _table(["Sexe", "Zone", "p (nœuds de la zone)", "n", "Log-vraisemblance", "AIC", "ΔAIC (zone)", "p(runs)", "Suggérée"], rows)


def junction_block() -> str:
    """The weight of the zone on the left of a boundary, around the boundary, for sigma = 3."""
    sigma = C.DEFAULT_SIGMA_JOIN
    rows = []
    for delta in range(-7, 8):
        u = -delta / sigma
        literal = 0.5 * (1.0 + np.tanh(u))
        kept = float(J.blend_weight(np.array([float(delta)]), 0.0, sigma)[0])
        rows.append([delta, f"{u:+.3f}", f"{literal:.4f}", f"{kept:.4f}", f"{1 - kept:.4f}"])
    return _table(["x − ω", "u = (ω − x) / σ", "w, formule littérale", "w retenu (bords continus)", "1 − w retenu"], rows)


def evaluation_block() -> str:
    """The comparison table of the nine zone methods (France 2020), both sexes."""
    rows = []
    for sex in ("Female", "Male"):
        table = evaluation(sex)
        for r in table.itertuples():
            rows.append([SEX_NAME[sex], C.ZONE_LABELS[r.zone], r.label, r.k, f"{r.aic:.1f}", f"{r.delta_aic:.1f}",
                         "—" if np.isnan(r.runs_p) else f"{r.runs_p:.3f}", f"{100 * r.outside:.1f} %", "oui" if r.suggested else ""])
    return _table(["Sexe", "Zone", "Méthode", "p", "AIC", "ΔAIC", "p(runs)", "\\|r\\| > 2", "Suggérée"], rows)


def life_table_block() -> str:
    """The life table of the suggested selection: e0, e65, e85 and q100, both sexes."""
    cfg, res = fits()
    rows = []
    for sex in ("Female", "Male"):
        table = evaluation(sex)
        pick = EV.suggest_selection(table)
        mu = J.join_zones(*(res[sex][pick[z]] for z in C.ZONES), cfg)
        lt = J.build_life_table(mu).set_index("age")
        labels = " · ".join(C.method_spec(pick[z]).label for z in C.ZONES)
        rows.append([SEX_NAME[sex], labels, f"{lt.loc[0, 'ex']:.2f}", f"{lt.loc[65, 'ex']:.2f}", f"{lt.loc[85, 'ex']:.2f}", f"{lt.loc[100, 'qx']:.4f}"])
    return _table(["Sexe", "Méthodes (suggestion de l'évaluateur)", "e₀", "e₆₅", "e₈₅", "q₁₀₀"], rows)


BLOCKS = {
    "01-polynome-degre-3": lambda: zone_method_block("poly3"),
    "02-spline-naturelle": lambda: zone_method_block("spline"),
    "03-gompertz": lambda: zone_method_block("gompertz"),
    "04-makeham": lambda: zone_method_block("makeham"),
    "05-makeham-2e-ordre": lambda: zone_method_block("makeham2"),
    "06-perks": lambda: zone_method_block("perks"),
    "07-kannisto": lambda: zone_method_block("kannisto"),
    "08-beard": lambda: zone_method_block("beard"),
    "09-coale-kisker": lambda: zone_method_block("coale_kisker"),
    "10-spline-globale": spline_all_block,
    "11-raccord": junction_block,
    "12-evaluation": evaluation_block,
    "13-table-de-vie": life_table_block,
}


def block_pattern() -> re.Pattern:
    return re.compile(re.escape(START) + r"\n(.*?)\n" + re.escape(END), re.S)


def current_block(text: str) -> str | None:
    found = block_pattern().search(text)
    return found.group(1) if found else None


def main(check: bool = False) -> int:
    stale = []
    for slug, make in BLOCKS.items():
        path = DOC_DIR / f"{slug}.md"
        text = path.read_text(encoding="utf-8")
        if current_block(text) is None:
            raise SystemExit(f"{path.name}: no results block between {START} and {END}")
        new = make()
        if current_block(text) != new:
            stale.append(slug)
            if not check:
                path.write_text(block_pattern().sub(lambda m: f"{START}\n{new}\n{END}", text), encoding="utf-8", newline="\n")
    print(("out of date: " if check else "rewritten: ") + (", ".join(stale) if stale else "nothing (all tables up to date)"))
    return 1 if (check and stale) else 0


if __name__ == "__main__":
    raise SystemExit(main(check="--check" in sys.argv))
