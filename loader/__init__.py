"""loader - Module 1 of the Mortality Table Smoother: import crude mortality data.

The loader IMPORTS; it does not compute crude rates. It reads data that are already processed - crude
rates m(x), or the deaths and exposure that give them - in the formats the sources use, and hands the
rest of the app ONE table

    sex | age | Dx | Ex | mx_raw | source              (mx_raw = Dx / Ex, a plain division)

with both sexes stacked (or one, by choice) and a quality report per sex (`df.attrs["qc"]`).
Nothing here returns a FitResult: that is the `fit_*` protocol of modules 3a-3e.

THREE WAYS IN
    Option C  load_data(file)                one file of elementary data - age, sex, year, deaths, exposure
                                             (recommended: robust, automatable, extends to other segments)
    Option A  load_separate(deaths=, exposure=, rates=)
                                             a file of rates (or deaths) and a file of exposure, sexes in
                                             columns; the exact deaths need the exposure and its convention
    Option B  load_by_sex(female=, male=)    one file per sex, each with deaths, exposures or rates by age

  Each input is a path, an uploaded file, or `Source(...)` carrying its own Excel sheet, header row,
  column mapping, segment filter... Enough inputs: Dx + Ex (a rates file, if given, is only
  cross-checked); Ex + mx (Dx = mx * Ex, so not an integer); mx alone with E_ref (Ex = E_ref).

WHAT IT READS
  HMD output files (tab-delimited with a title line; JMD, CHMD... share the layout), period or cohort:
      Mx_1x1  Exposures_1x1  Deaths_1x1        "Year    Age  Female  Male  Total"   period
      cMx_1x1 cExposures_1x1 (cohort files)    "Cohort  Age  Female  Male  Total"   cohort
  INSEE state-civil tables (T74BIS: CHAMP2;ANNEE_1975_2024;SEXE;AGE2_REVOLU;NBDEC - sexes F/H/E, a
  TOT line, a geography to choose with where={'CHAMP2': 'FM'}, and a last age '100' that means 100 and
  over). Eurostat bulk TSV ('unit,sex,age,geo\\TIME_PERIOD', one column per year, flags, ':' missing).
  Any CSV / TSV / Excel that is wide in sexes, wide in years, or long (one row per cell), with the
  column names found below title lines, French or English, and French number formats.
  Ages: '85', '85 ans', 'Y85', 'D85', '< 1 year', 'Y_LT1'; open groups ('110+', '105_PL', 'Y_OPEN'),
  totals and bands are left out. Population at 1 January (INSEE, Eurostat) -> exposure_basis='jan1_mean'.
  NOT read, with a message saying why: the HMD InputDB (Lexis triangles - raw counts to aggregate),
  `.xls` (needs xlrd), age bands ('0-4': the smoothing needs single ages).

TWO VIEWS ("tables du moment" / "tables générationnelles")
  period   rates by age in one calendar year     -> year=
  cohort   rates by age along one birth cohort   -> cohort=
  The loader slices a file; it never turns period data into a cohort (a cohort is not the diagonal of
  period rates). view='auto' follows the file's own column (Year or Cohort).

SEXES  sex='both' (default: Female and Male, one table), 'Female' or 'Male'. A file that does not say
  which sex it holds is attributed to the single sex asked for. Total - both sexes pooled - is ignored.

QUALITY LEVELS (spec 1.5), per sex
    ok        control passed
    info      worth knowing (excluded open age group, reconstructed Dx, ...)   -> df.attrs["notes"]
    orange    warning, never blocks
    red       "alerte rouge": returned, flagged, not fit to smooth (negative Dx, duplicate ages)
    blocking  "erreur bloquante": raises DataQualityError (Ex <= 0 with Dx > 0)
  Ages with Dx = 0 are KEPT (zero contribution to the Poisson MLE). Only Ex = 0 AND Dx = 0 is
  dropped (no exposure, 0/0).

Streamlit: every file argument accepts the `UploadedFile` of `st.file_uploader` as is.

Layout of the package: errors, constants, qc (controls), parsing (numbers, ages), columns (vocabulary),
layouts (wide-in-years, Eurostat keys), readers (files -> tables, `Source`), select (a slice, a sex),
standardize (Dx / Ex / mx), conventions (population -> exposure), api (the three options).
"""
from .api import FileInfo, inspect_file, load_by_sex, load_data, load_separate, standard_table
from .constants import MIN_AGES, SEXES, STANDARD_COLUMNS, VIEWS
from .errors import (AmbiguousSelectionError, ColumnMappingError, DataQualityError, FileFormatError,
                     LoaderError)
from .qc import QCItem, QCReport, can_proceed, check_quality, qc_summary
from .readers import Source

__all__ = [
    "STANDARD_COLUMNS", "SEXES", "VIEWS", "MIN_AGES",
    "LoaderError", "FileFormatError", "ColumnMappingError",
    "AmbiguousSelectionError", "DataQualityError",
    "QCItem", "QCReport", "check_quality", "can_proceed", "qc_summary",
    "Source", "FileInfo", "inspect_file",
    "load_data", "load_by_sex", "load_separate", "standard_table",
]
