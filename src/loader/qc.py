"""Quality controls of spec 1.5, run per sex on a standard table."""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .constants import MIN_AGES


_LEVEL_LABEL = {
    "ok": "OK", "info": "Info", "orange": "Warning",
    "red": "Alerte rouge", "blocking": "Erreur bloquante",
}


def age_list(ages) -> list:
    """Sorted plain-int list (numpy ints would leak into JSON / session_state)."""
    return sorted({int(a) for a in ages})


def fmt_ages(ages, limit: int = 12) -> str:
    ages = [int(a) for a in ages]
    head = ", ".join(str(a) for a in ages[:limit])
    return head + (f", ... (+{len(ages) - limit})" if len(ages) > limit else "")


@dataclass
class QCItem:
    check: str                       # stable identifier, e.g. "dx_nonneg"
    level: str                       # ok | info | orange | red | blocking
    message: str                     # French, ready for the UI
    ages: list = field(default_factory=list)


@dataclass
class QCReport:
    items: list
    n_rows: int
    age_min: int | None
    age_max: int | None
    n_dx_zero: int
    total_deaths: float
    n_missing: int = 0               # cells excluded at load time (HMD ".", empty)
    source: str = ""

    def _level(self, level: str) -> list:
        return [i for i in self.items if i.level == level]

    @property
    def blocking(self) -> list:
        return self._level("blocking")

    @property
    def red(self) -> list:
        return self._level("red")

    @property
    def orange(self) -> list:
        return self._level("orange")

    @property
    def can_proceed(self) -> bool:
        """False as soon as a red alert or a blocking error is present."""
        return not (self.blocking or self.red)

    def to_frame(self) -> pd.DataFrame:
        """One row per control: check, level, message, ages."""
        return pd.DataFrame(
            [(i.check, i.level, i.message, i.ages) for i in self.items],
            columns=["check", "level", "message", "ages"],
        )

    def summary(self) -> pd.DataFrame:
        """The 'tableau de contrôle' of spec 1.3 (all values as text: Arrow-safe)."""
        span = ("-" if self.age_min is None
                else f"{self.age_min} - {self.age_max}")
        rows = [
            ("Plage d'âge", span),
            ("Nombre de lignes", str(self.n_rows)),
            ("Nb Dx = 0", str(self.n_dx_zero)),
            ("Valeurs manquantes (exclues)", str(self.n_missing)),
            ("Décès totaux (somme des Dx)", f"{self.total_deaths:,.0f}"),
        ]
        return pd.DataFrame(rows, columns=["Contrôle", "Valeur"])

    def __str__(self) -> str:
        return "\n".join(f"[{_LEVEL_LABEL[i.level]}] {i.message}" for i in self.items)


def check_quality(df: pd.DataFrame, *, n_missing: int = 0,
                  source: str = "") -> QCReport:
    """Run the six mandatory controls of spec 1.5 (plus a gap check) on a
    standard DataFrame (needs `age`, `Dx`, `Ex`; `mx_raw` is recomputed if absent).

    Public so the app can re-run it on an edited table. NaN is never a pass:
    a NaN Dx fails the Dx >= 0 control.
    """
    age = pd.to_numeric(df["age"], errors="coerce")
    Dx = pd.to_numeric(df["Dx"], errors="coerce")
    Ex = pd.to_numeric(df["Ex"], errors="coerce")
    if "mx_raw" in df:
        mx = pd.to_numeric(df["mx_raw"], errors="coerce")
    else:
        mx = (Dx / Ex.where(Ex > 0))

    def ages_of(mask) -> list:
        return age_list(age[mask].dropna())

    items: list[QCItem] = []

    bad = ~(Dx >= 0)
    if bad.any():
        a = ages_of(bad)
        items.append(QCItem("dx_nonneg", "red",
                            f"Dx négatif ou non numérique à {len(a)} âge(s) : {fmt_ages(a)}", a))
    else:
        items.append(QCItem("dx_nonneg", "ok", "Dx >= 0 pour tous les âges"))

    bad = ((Dx > 0) & ~(Ex > 0)) | (Ex < 0)
    if bad.any():
        a = ages_of(bad)
        items.append(QCItem("ex_positive", "blocking",
                            f"Ex <= 0 alors que Dx > 0 (ou Ex < 0) à {len(a)} âge(s) : {fmt_ages(a)}", a))
    else:
        items.append(QCItem("ex_positive", "ok", "Ex > 0 partout où Dx > 0"))

    dup = ages_of(age.duplicated(keep=False) & age.notna())
    if dup:
        items.append(QCItem("duplicate_ages", "red",
                            f"Âges dupliqués ({len(dup)}) : {fmt_ages(dup)}", dup))
    else:
        items.append(QCItem("duplicate_ages", "ok", "Aucun âge dupliqué"))

    distinct = age_list(age.dropna())
    n_ages = len(distinct)
    span = f"{distinct[0]} - {distinct[-1]}" if distinct else "-"
    if n_ages < MIN_AGES:
        items.append(QCItem("age_range", "orange",
                            f"Seulement {n_ages} âge(s) distinct(s) (minimum conseillé : {MIN_AGES}), plage {span}"))
    else:
        items.append(QCItem("age_range", "ok", f"Plage d'âge {span} ({n_ages} valeurs)"))

    gaps = (sorted(set(range(distinct[0], distinct[-1] + 1)) - set(distinct))
            if distinct else [])
    if gaps:
        items.append(QCItem("age_gaps", "orange",
                            f"{len(gaps)} âge(s) absent(s) à l'intérieur de la plage : {fmt_ages(gaps)}", gaps))
    else:
        items.append(QCItem("age_gaps", "ok", "Plage d'âge continue"))

    zero = Dx == 0
    if zero.any():
        a = ages_of(zero)
        items.append(QCItem("dx_zero", "orange",
                            f"Dx = 0 à {len(a)} âge(s), conservé(s) (contribution nulle au MLE) : {fmt_ages(a)}", a))
    else:
        items.append(QCItem("dx_zero", "ok", "Aucun Dx = 0"))

    over = mx > 1
    if over.any():
        a = ages_of(over)
        items.append(QCItem("mx_range", "orange",
                            f"mx_raw > 1 à {len(a)} âge(s) : {fmt_ages(a)}", a))
    else:
        items.append(QCItem("mx_range", "ok", "mx_raw dans ]0, 1] là où Dx > 0"))

    return QCReport(
        items=items, n_rows=len(df),
        age_min=distinct[0] if distinct else None,
        age_max=distinct[-1] if distinct else None,
        n_dx_zero=int(zero.sum()), total_deaths=float(Dx.sum(skipna=True)),
        n_missing=n_missing, source=source,
    )


def can_proceed(df: pd.DataFrame) -> bool:
    """False if any sex carries a red alert or a blocking error."""
    return all(r.can_proceed for r in df.attrs["qc"].values())


def qc_summary(df: pd.DataFrame) -> pd.DataFrame:
    """The 'tableau de contrôle' of spec 1.3, one column per sex."""
    tables = {s: r.summary().set_index("Contrôle")["Valeur"] for s, r in df.attrs["qc"].items()}
    return pd.DataFrame(tables).rename_axis("Contrôle").reset_index()
