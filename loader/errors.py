"""Errors the loader raises: every message is fit for the UI, and carries the data a selector needs."""
from __future__ import annotations

import pandas as pd


class LoaderError(ValueError):
    """Base class: whatever the loader refuses, with a message fit for the UI."""


class FileFormatError(LoaderError):
    """The file cannot be read as the requested / detected format."""


class ColumnMappingError(LoaderError):
    """A needed column or quantity could not be identified.

    `missing` names what is lacking ("age", "Dx", "Ex", ...) and `available` the
    columns found in the file - the two lists a "map columns" dropdown needs.
    """

    def __init__(self, message: str, *, missing=(), available=()):
        super().__init__(message)
        self.missing = list(missing)
        self.available = list(available)


class AmbiguousSelectionError(LoaderError):
    """The file holds several years / cohorts / sexes and none (or an absent one) was chosen.

    `what` is "year", "cohort", "sex" or "segment" (then `column` names it), `options` the values to
    offer in a selector.
    """

    def __init__(self, message: str, *, what: str, options, column: str | None = None):
        super().__init__(message)
        self.what = what
        self.options = list(options)
        self.column = column                       # for what == "segment": the column to choose a value in


class DataQualityError(LoaderError):
    """A blocking quality failure. `reports` maps each sex to its QCReport and `data` is the table
    as loaded, so the UI can still show which rows are at fault."""

    def __init__(self, reports: dict, data: pd.DataFrame | None = None):
        parts = [f"{s} : " + "; ".join(i.message for i in r.blocking)
                 for s, r in reports.items() if r.blocking]
        where = next((f" ({r.source})" for r in reports.values() if r.source), "")
        super().__init__(f"Erreur bloquante{where} - " + " | ".join(parts))
        self.reports = reports
        self.data = data
