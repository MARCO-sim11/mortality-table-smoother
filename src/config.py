"""config.py - Module 2: the settings every smoothing module consumes.

Exposes `SmootherConfig` (spec 2.3, defaults of spec 2.4) and, because every
module of the chain needs the same contract, `FitResult` (spec, "Convention
commune a tous les modules de lissage", reproduced field for field). The specs
do not say where FitResult lives; it sits here so that zone modules, joiner and
evaluator all import it from the one place they already import `SmootherConfig`.

Also here: the method registry (keys and labels of spec 2.2-2.3, the module
each lives in per the architecture table of spec 0), the zone bounds, and
`validate_config`, which turns a config plus the loaded ages into error and
warning messages for tab 2.

CONVENTIONS THE ZONE MODULES MUST FOLLOW (decided with the app skeleton, so
worth checking when each `fit_*` is written):

  * Each method key `k` is served by `fit_<k>(ages, Dx, Ex, cfg) -> FitResult`
    in the module named by `METHODS[k].module` (spec 3: "fit_*(ages, Dx, Ex, cfg)").
  * `ages, Dx, Ex` are the rows of the loaded table that fall inside the zone,
    both bounds included (`SmootherConfig.zone_bounds`); the zone-3 bound is
    omega3, so a table that stops earlier simply yields fewer rows.
    The module reads omega3, sigma, NR settings... from `cfg`.
  * `FitResult.zone` is 'zone1' | 'zone2' | 'zone3', or 'all' for a GLOBAL method (see below).

GLOBAL METHODS (not in the specs, asked for after the seven modules were built): a method of `GLOBAL_METHODS` is fitted ONCE
on every loaded age up to omega3 (`zone_bounds('all')`), whatever the zone boundaries, and is then offered in each of the
three zones. `fit_<key>` follows the same contract, its FitResult has zone='all'. Anything that reads a result for one zone
(the evaluator, the joiner) goes through `zone_view`, or accepts zone='all' directly.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

# --------------------------------------------------------------------------
# Defaults (spec 2.4)
# --------------------------------------------------------------------------

DEFAULT_OMEGA1 = 20          # end of infant and juvenile excess mortality
DEFAULT_OMEGA2 = 85          # start of the extreme-ageing zone
DEFAULT_OMEGA3 = 110         # conventional end of HMD tables
DEFAULT_SIGMA_JOIN = 3.0     # smooth transition over +/- 1 standard deviation
DEFAULT_NR_MAXITER = 200
DEFAULT_NR_TOL = 1e-8
DEFAULT_GRID_N = 100         # 100 x 100 = 10 000 grid points

SIGMA_JOIN_RANGE = (1.0, 10.0)      # spec 2.2: slider 1-10

# --------------------------------------------------------------------------
# Zones and methods (spec 2.2, 2.3 and the architecture table)
# --------------------------------------------------------------------------

ZONES = ("zone1", "zone2", "zone3")
ZONE_ALL = "all"                    # the 'zone' of a global method: every age from the first one to omega3
ZONE_LABELS = {"zone1": "Zone I", "zone2": "Zone II", "zone3": "Zone III", ZONE_ALL: "Tous âges"}


@dataclass(frozen=True)
class MethodSpec:
    key: str            # identifier stored in SmootherConfig and in FitResult.method
    label: str          # checkbox label (spec 2.2)
    zone: str           # 'zone1' | 'zone2' | 'zone3'
    module: str         # file in src/ that serves it (spec 0, architecture table)
    n_params: int       # free parameters, from the methodology doc's summary table

    @property
    def fit_name(self) -> str:
        return f"fit_{self.key}"


METHODS: dict[str, MethodSpec] = {m.key: m for m in (
    MethodSpec("poly3",        "Polynôme deg 3",     "zone1", "zone1_poly",      4),
    MethodSpec("spline",       "Splines naturelles", "zone1", "zone1_poly",      4),   # K + 3, K >= 1
    MethodSpec("gompertz",     "Gompertz",           "zone2", "zone2_gompertz",  2),
    MethodSpec("makeham",      "Makeham",            "zone2", "zone2_makeham",   3),
    MethodSpec("makeham2",     "Makeham 2e ordre",   "zone2", "zone2_makeham2",  5),   # spec: zone2_perks; own file as asked
    MethodSpec("perks",        "Perks",              "zone2", "zone2_perks",     4),
    MethodSpec("kannisto",     "Kannisto",           "zone3", "zone3",           2),
    MethodSpec("beard",        "Beard",              "zone3", "zone3",           3),
    MethodSpec("coale_kisker", "Coale-Kisker",       "zone3", "zone3",           1),
)}


# Fitted once on all the ages and offered in every zone (zone = ZONE_ALL). `n_params` is only the registry's fallback: the
# real count is the number of knots, `params['n_params']`, and `zone_view` gives each zone its own share of it.
GLOBAL_METHODS: dict[str, MethodSpec] = {m.key: m for m in (
    MethodSpec("spline_all", "Spline globale (tous âges)", ZONE_ALL, "spline_all", 4),
)}


def methods_of(zone: str) -> list[MethodSpec]:
    """The registered methods of a zone, in display order (ZONE_ALL: the global methods)."""
    if zone == ZONE_ALL:
        return list(GLOBAL_METHODS.values())
    return [m for m in METHODS.values() if m.zone == zone]


def every_method() -> list[MethodSpec]:
    """The zone methods in registry order, then the global ones."""
    return [*METHODS.values(), *GLOBAL_METHODS.values()]


def method_spec(key: str) -> MethodSpec:
    """The registered method `key`, zone or global. KeyError when unknown."""
    return METHODS[key] if key in METHODS else GLOBAL_METHODS[key]


def zones_served(method: MethodSpec) -> tuple:
    """The zones a method is offered in: its own zone, or the three zones for a global method."""
    return ZONES if method.zone == ZONE_ALL else (method.zone,)


# --------------------------------------------------------------------------
# FitResult: the return type of every fit_* (spec 3, verbatim)
# --------------------------------------------------------------------------


@dataclass
class FitResult:
    method: str          # method identifier
    zone: str            # 'zone1' | 'zone2' | 'zone3'
    ages: np.ndarray     # ages covered
    mu_hat: np.ndarray   # smoothed rates
    params: dict         # estimated parameters
    loglik: float        # final log-likelihood
    converged: bool      # convergence flag
    n_iter: int          # iterations performed
    message: str         # diagnostic message


# --------------------------------------------------------------------------
# SmootherConfig (spec 2.3; defaults 2.4)
# --------------------------------------------------------------------------


def _all_keys(zone: str) -> list:
    return [m.key for m in methods_of(zone)]


@dataclass
class SmootherConfig:
    omega1: int = DEFAULT_OMEGA1             # zone I / zone II boundary
    omega2: int = DEFAULT_OMEGA2             # zone II / zone III boundary
    omega3: int = DEFAULT_OMEGA3             # age limit (end of zone III)
    z1_methods: list = field(default_factory=lambda: _all_keys("zone1"))   # ['poly3', 'spline']
    z2_methods: list = field(default_factory=lambda: _all_keys("zone2"))   # ['gompertz', 'makeham', 'makeham2', 'perks']
    z3_methods: list = field(default_factory=lambda: _all_keys("zone3"))   # ['kannisto', 'beard', 'coale_kisker']
    sigma_join: float = DEFAULT_SIGMA_JOIN   # half-width of the tanh window
    nr_maxiter: int = DEFAULT_NR_MAXITER     # Newton-Raphson iteration cap
    nr_tol: float = DEFAULT_NR_TOL           # NR stopping criterion
    grid_n: int = DEFAULT_GRID_N             # Kannisto 2-D grid size
    all_methods: list = field(default_factory=list)   # global methods ticked (off by default): ['spline_all']

    def methods_in(self, zone: str) -> list:
        return {"zone1": self.z1_methods, "zone2": self.z2_methods,
                "zone3": self.z3_methods, ZONE_ALL: self.all_methods}[zone]

    def selected_methods(self) -> list[MethodSpec]:
        """Every ticked method, zone by zone, in registry order, the global ones last. Unknown keys are
        skipped here and reported by `validate_config`."""
        out = []
        for zone in (*ZONES, ZONE_ALL):
            ticked = set(self.methods_in(zone))
            out.extend(m for m in methods_of(zone) if m.key in ticked)
        return out

    def zone_bounds(self, zone: str) -> tuple:
        """(low, high) ages of a zone, both included. Zone I starts at the first
        loaded age, which the config does not know: its low bound is None. ZONE_ALL (global methods) spans
        everything up to omega3."""
        return {"zone1": (None, self.omega1),
                "zone2": (self.omega1, self.omega2),
                "zone3": (self.omega2, self.omega3),
                ZONE_ALL: (None, self.omega3)}[zone]


def zone_slice(df, cfg: SmootherConfig, zone: str):
    """Rows of a loaded table (columns age, Dx, Ex, ...) that belong to `zone`."""
    lo, hi = cfg.zone_bounds(zone)
    keep = df["age"] <= hi
    if lo is not None:
        keep &= df["age"] >= lo
    return df.loc[keep]


def adapt_config(cfg: SmootherConfig, ages) -> SmootherConfig:
    """A copy of `cfg` for a dataset whose ages are `ages`: used to give one sex the settings of the other.

    Every setting is copied (the method lists too, so the two configs never share a list). The boundaries are brought inside
    the new age range the way tab 2's sliders need them: omega1 in [first age, last age - 1], omega2 in [omega1 + 1, last
    age], omega3 at least first age + 2. Nothing changes when the two sexes have the same ages."""
    ages = np.asarray(ages)
    lo, hi = int(ages.min()), int(ages.max())
    omega1 = min(max(cfg.omega1, lo), hi - 1)
    omega2 = min(max(cfg.omega2, omega1 + 1), hi)
    return replace(cfg, omega1=omega1, omega2=omega2, omega3=max(cfg.omega3, lo + 2),
                   z1_methods=list(cfg.z1_methods), z2_methods=list(cfg.z2_methods),
                   z3_methods=list(cfg.z3_methods), all_methods=list(cfg.all_methods))


def zone_view(result: FitResult, zone: str, cfg: SmootherConfig) -> FitResult:
    """`result` as one zone sees it: a zone result comes back as is, a global one (zone='all') as a copy labelled with `zone`.

    The curve is the same whole curve (ages, mu_hat and the formula's parameters untouched, so the joiner can read it at any
    age); only `params['n_params']` changes: the number of knots inside the zone's bounds (both included, so a knot on a
    boundary counts in the two zones; at least 1). That is what the zone's AIC / BIC are charged for, against the k of the
    methods that live in the zone alone. Raises KeyError for a zone that is not one of ZONES."""
    if result.zone != ZONE_ALL:
        return result
    if zone not in ZONES:
        raise KeyError(zone)
    lo, hi = cfg.zone_bounds(zone)
    knots = np.asarray(result.params.get("knots", []), dtype=float)
    inside = (knots <= hi) & (knots >= (-np.inf if lo is None else lo))
    return replace(result, zone=zone, params={**result.params, "n_params": max(int(inside.sum()), 1)})


def default_config(ages) -> SmootherConfig:
    """Spec 2.4 defaults, made to fit the loaded age range.

    The recommended omega1 = 20 / omega2 = 85 are kept when they fall strictly
    inside the data; otherwise (a table starting at 40, or stopping at 70) the
    boundaries are placed at 20% and 75% of the range so the app starts valid.
    omega3 is 110 unless the data go beyond it.
    """
    ages = np.asarray(ages)
    lo, hi = int(ages.min()), int(ages.max())
    if lo < DEFAULT_OMEGA1 < DEFAULT_OMEGA2 < hi:
        w1, w2 = DEFAULT_OMEGA1, DEFAULT_OMEGA2
    else:
        w1, w2 = lo + round(0.20 * (hi - lo)), lo + round(0.75 * (hi - lo))
    return SmootherConfig(omega1=w1, omega2=w2, omega3=max(DEFAULT_OMEGA3, hi))


# --------------------------------------------------------------------------
# Validation (messages are shown as is in tab 2, hence in French)
# --------------------------------------------------------------------------


@dataclass
class ConfigIssues:
    errors: list = field(default_factory=list)       # block the launch
    warnings: list = field(default_factory=list)     # shown, never block

    @property
    def ok(self) -> bool:
        return not self.errors


def validate_config(cfg: SmootherConfig, ages=None) -> ConfigIssues:
    """Check a config, and against the loaded ages when they are given.

    Errors: boundaries out of order or outside the data, unknown method keys,
    nothing ticked, non-positive numeric settings. Warnings: a zone with no
    method (the final assembly needs one per zone), a zone with no more observed
    ages than the free parameters of a ticked method (nothing left to estimate
    with), omega3 beyond the last observed age (zone III is then extrapolated),
    and overlapping junction windows (spec 4.2: +/- 2 sigma around each boundary).
    """
    out = ConfigIssues()

    if not (0 <= cfg.omega1 < cfg.omega2 < cfg.omega3):
        out.errors.append(f"Les frontières doivent vérifier 0 ≤ ω₁ < ω₂ < ω₃ "
                          f"(reçu : ω₁ = {cfg.omega1}, ω₂ = {cfg.omega2}, ω₃ = {cfg.omega3}).")
    if not cfg.sigma_join > 0:
        out.errors.append(f"σ de jonction doit être > 0 (reçu : {cfg.sigma_join}).")
    if cfg.nr_maxiter < 1:
        out.errors.append("Le nombre maximal d'itérations Newton-Raphson doit être ≥ 1.")
    if not cfg.nr_tol > 0:
        out.errors.append("Le critère d'arrêt Newton-Raphson doit être > 0.")
    if cfg.grid_n < 2:
        out.errors.append("La grille Kannisto doit compter au moins 2 points par axe.")

    for zone in (*ZONES, ZONE_ALL):
        registry = GLOBAL_METHODS if zone == ZONE_ALL else METHODS
        unknown = [k for k in cfg.methods_in(zone)
                   if k not in registry or registry[k].zone != zone]
        if unknown:
            out.errors.append(f"{ZONE_LABELS[zone]} : méthode(s) inconnue(s) {unknown}.")
    chosen = cfg.selected_methods()
    if not chosen:
        out.errors.append("Cochez au moins une méthode.")
    else:
        for zone in ZONES:
            if not any(m.zone in (zone, ZONE_ALL) for m in chosen):     # a global method serves every zone
                out.warnings.append(f"Aucune méthode cochée en {ZONE_LABELS[zone]} : "
                                    "l'assemblage final exigera une méthode par zone.")

    if cfg.omega2 - cfg.omega1 < 4 * cfg.sigma_join and cfg.omega1 < cfg.omega2:
        out.warnings.append(f"Les fenêtres de jonction (±2σ = ±{2 * cfg.sigma_join:g} ans) "
                            f"se chevauchent : ω₂ − ω₁ = {cfg.omega2 - cfg.omega1} < 4σ.")

    if ages is not None and len(ages):
        ages = np.asarray(ages)
        lo, hi = int(ages.min()), int(ages.max())
        for name, w in (("ω₁", cfg.omega1), ("ω₂", cfg.omega2)):
            if not lo <= w <= hi:
                out.errors.append(f"{name} = {w} est hors de la plage des données ({lo}–{hi}).")
        if cfg.omega3 > hi + 1:        # HMD data stop at 109 and omega3 = 110 by convention: not worth a warning
            out.warnings.append(f"ω₃ = {cfg.omega3} dépasse de {cfg.omega3 - hi} ans le dernier âge "
                                f"observé ({hi}) : la fin de la Zone III sera extrapolée.")
        if out.ok:
            for zone in (*ZONES, ZONE_ALL):
                zlo, zhi = cfg.zone_bounds(zone)
                n_obs = int(((ages <= zhi) & (ages >= (lo if zlo is None else zlo))).sum())
                for m in (m for m in chosen if m.zone == zone):
                    if n_obs <= m.n_params:
                        out.warnings.append(
                            f"{ZONE_LABELS[zone]} : {n_obs} âge(s) observé(s) pour {m.label} "
                            f"({m.n_params} paramètre(s)) - estimation non identifiable.")
    return out
