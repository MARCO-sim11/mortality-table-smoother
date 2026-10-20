"""Module 1 against the REAL France HMD files of Post 3 (skipped if they are absent).

Run:   python -m pytest tests/test_loader_real_hmd.py -v

The oracle is smoothing_lab/io_hmd.py, which reads the same Exposures and Mx files independently.
Folder override: set MORTALITY_POST3_DIR to the Post3 folder holding HMDB/ and smoothing_lab/.

Which of the HMDB files the loader takes: Mx_1x1 + Exposures_1x1, HMD's own processed outputs
(crude rates and central exposure, period 1x1, both sexes). Not FRATNPdeath.txt - the InputDB, raw
Lexis-triangle counts - which is refused; not the *ltper life tables - their rates are closed at the
oldest ages by HMD's own model, so they are not crude. No Deaths_1x1 is in the folder: the
Deaths + Exposure route is covered by tests/test_loader.py on synthetic files, and by the last test
below if a *Deaths_1x1.txt is dropped into HMDB.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

from src import loader as L

POST3 = Path(os.environ.get(
    "MORTALITY_POST3_DIR",
    r"C:\Users\simar\OneDrive\Bureau\Persos\LinkedIn\Life_Risks_Series\Post3-!Smoothing1"))
HMDB = POST3 / "HMDB"
EXPOSURE, RATES, INPUTDB = HMDB / "FRA_Exposures_1x1.txt", HMDB / "FRATNP_Mx_1x1.txt", HMDB / "FRATNPdeath.txt"

pytestmark = pytest.mark.skipif(not (EXPOSURE.exists() and RATES.exists()),
                                reason=f"France HMD files not found under {HMDB}")


@pytest.fixture(scope="module")
def io_hmd():
    sys.path.insert(0, str(POST3))
    pytest.importorskip("smoothing_lab.io_hmd")
    from smoothing_lab import io_hmd as mod
    return mod


@pytest.fixture(scope="module")
def france_2020():
    return L.load_separate(exposure=EXPOSURE, rates=RATES, year=2020)


def test_inspect_sees_the_period_views_and_every_year_of_the_files():
    info = L.inspect_file(EXPOSURE)
    assert (info.fmt, info.layout, info.quantity, info.views) == ("hmd", "sex_columns", "Ex", ["period"])
    assert info.years[0] == 1816 and info.years[-1] == 2024 and len(info.years) == 209
    assert info.cohorts == [] and info.sexes == ["Female", "Male"]
    assert L.inspect_file(RATES).quantity == "mx"


@pytest.mark.parametrize("sex,code", [("Female", "f"), ("Male", "m")])
def test_france_2020_agrees_with_the_smoothing_lab_reader(io_hmd, france_2020, sex, code):
    """Measured (2020, ages 0-104): Ex identical, mx identical to 2e-16; reconstructed Dx = mx * Ex
    within 0.13% of the InputDB count where D >= 100 and within 0.21 deaths where D < 100 - the
    print rounding of mx."""
    ref = io_hmd.crude_slice(code, 2020)                  # ages 0-104 (config.AGE_MAX_RAW)
    j = ref.merge(france_2020[france_2020["sex"] == sex], on="age")
    assert len(j) == 105
    assert (j["E"] == j["Ex"]).all()
    np.testing.assert_allclose(j["mx_raw"], j["m"], rtol=1e-12)
    rel = (j["Dx"] - j["D"]).abs() / j["D"]
    big = j["D"] >= 100
    assert rel[big].max() < 3e-3
    assert (j["Dx"] - j["D"]).abs()[~big].max() < 0.5


def test_france_2020_is_a_usable_smoothing_input_for_both_sexes(france_2020):
    df = france_2020
    assert list(df.columns) == L.STANDARD_COLUMNS and len(df) == 220
    assert df.attrs["sexes"] == ["Female", "Male"] and (df.attrs["view"], df.attrs["year"]) == ("period", 2020)
    assert L.can_proceed(df)
    for s in ("Female", "Male"):
        part = df[df["sex"] == s]
        assert (part["age"].min(), part["age"].max()) == (0, 109)            # '110+' is the open group
        assert (part["Ex"] > 0).all() and (part["Dx"] >= 0).all()
        assert 3.0e5 < part["Dx"].sum() < 3.6e5                              # ~327k a sex, France 2020
        non_ok = {i.check for i in df.attrs["qc"][s].items if i.level != "ok"}
        assert non_ok == {"mx_range"}                                        # HMD's own closing artefact
    assert [i.ages for i in df.attrs["qc"]["Female"].items if i.check == "mx_range"] == [[105]]


def test_one_sex_by_choice_gives_the_same_rows_as_the_both_sexes_table(france_2020):
    male = L.load_separate(exposure=EXPOSURE, rates=RATES, year=2020, sex="Male")
    both_male = france_2020[france_2020["sex"] == "Male"].reset_index(drop=True)
    cols = ["sex", "age", "Dx", "Ex", "mx_raw"]
    assert male[cols].equals(both_male[cols])


@pytest.mark.skipif(not INPUTDB.exists(), reason="FRATNPdeath.txt not found")
def test_the_inputdb_deaths_file_is_refused_not_misread():
    """Read as a plain long file it gave 'âges dupliqués' (a TL and a TU row per age): now the
    message says what the file is and which HMD files to use."""
    with pytest.raises(L.FileFormatError, match="InputDB") as exc:
        L.load_separate(deaths=INPUTDB, exposure=EXPOSURE, year=2020)
    assert "Mx_1x1" in str(exc.value)


def test_spec_j1_check_deaths_plus_exposure_2020_if_a_deaths_file_is_present():
    """Spec section 8, J1: 'Charger France HMD Deaths + Exposure 2020. Verifier df_raw coherent,
    Dx >= 0, Ex > 0.' Runs only if a Deaths_1x1 file sits next to the others."""
    candidates = sorted(HMDB.glob("*[Dd]eaths_1x1.txt"))
    if not candidates:
        pytest.skip("no *Deaths_1x1.txt in HMDB (only the InputDB triangles are there)")
    df = L.load_separate(deaths=candidates[0], exposure=EXPOSURE, rates=RATES, year=2020)
    assert (df["Dx"] >= 0).all() and (df["Ex"] > 0).all() and L.can_proceed(df)
    # HMD publishes Deaths and Exposure to 2 decimals and Mx to 6: at the oldest ages (an exposure of 0.72 for the men at 109) the rounding of
    # D and E alone moves D/E by about 0.5 %, past the loader's tolerance. The check may speak there (orange, D/E is kept), and only there.
    flagged = [i for r in df.attrs["qc"].values() for i in r.items if i.check == "mx_consistency"]
    assert all(i.level == "orange" and min(i.ages) >= 105 for i in flagged)
