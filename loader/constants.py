"""Constants shared by the loader's modules."""
from __future__ import annotations

STANDARD_COLUMNS = ["sex", "age", "Dx", "Ex", "mx_raw", "source"]
SEXES = ("Female", "Male")
VIEWS = ("period", "cohort")
MIN_AGES = 20                       # spec 1.5: age range of at least 20 values

# Tolerance when a rates file is cross-checked against Dx / Ex. HMD prints mx
# to 6 decimals, so the absolute slack must cover half a unit of that.
MX_RTOL, MX_ATOL = 1e-3, 1e-6

Q_KEY = {"Dx": "dx", "Ex": "ex", "mx": "mx"}       # quantity -> the role name of its column
