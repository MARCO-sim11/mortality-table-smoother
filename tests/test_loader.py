"""Module 1 - loader.py on synthetic data (no real file needed).

Run from the project root:   python -m pytest tests/test_loader.py -v

The loader imports crude data - rates, or the deaths + exposure that give them - in market formats
(HMD 1x1 files, CSV, Excel), for period tables (a year) or generation tables (a cohort), and
returns ONE table with both sexes (or one). These tests pin that contract, the layouts it reads,
and everything it refuses.
"""
from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest

import synthetic as S
from src import loader as L

YEARS = (2020, 2021)
BOTH = ("Female", "Male")


@pytest.fixture(scope="module")
def truth():
    """{(sex, year): table with age, Dx, Ex, mu_true}"""
    return {(s, y): S.make_sex_table(s, y) for s in BOTH for y in YEARS}


@pytest.fixture
def hmd(tmp_path):
    """Deaths / Exposure / Rates in HMD layout (Female, Male, Total columns), two years."""
    by_year = S.hmd_tables(YEARS)
    return {
        "dir": tmp_path, "by_year": by_year,
        "deaths": S.write_hmd(tmp_path / "Deaths_1x1.txt", "Dx", by_year),
        "exposure": S.write_hmd(tmp_path / "Exposures_1x1.txt", "Ex", by_year),
        "rates": S.write_hmd(tmp_path / "Mx_1x1.txt", "mx", by_year),
    }


class FakeUploadedFile(io.BytesIO):
    """What st.file_uploader returns: a BytesIO with a name."""

    def __init__(self, data: bytes, name: str):
        super().__init__(data)
        self.name = name


def rows(df, sex):
    return df[df["sex"] == sex].reset_index(drop=True)


def check_contract(df, sexes=BOTH):
    """The shape every loader returns."""
    assert list(df.columns) == L.STANDARD_COLUMNS
    assert df["age"].dtype == np.int64
    for c in ("Dx", "Ex", "mx_raw"):
        assert df[c].dtype == np.float64
    assert list(dict.fromkeys(df["sex"])) == list(sexes)
    assert df.attrs["sexes"] == list(sexes) and set(df.attrs["qc"]) == set(sexes)
    for s in sexes:
        r = rows(df, s)
        assert r["age"].is_monotonic_increasing
        pos = r["Ex"] > 0
        np.testing.assert_allclose(r.loc[pos, "mx_raw"], r.loc[pos, "Dx"] / r.loc[pos, "Ex"], rtol=1e-12)
        assert isinstance(df.attrs["qc"][s], L.QCReport)


def item(df, sex, check):
    (it,) = [i for i in df.attrs["qc"][sex].items if i.check == check]
    return it


def levels(df, sex, check):
    return [i.level for i in df.attrs["qc"][sex].items if i.check == check]


def note(df, check):
    return [i for i in df.attrs["notes"] if i.check == check]


def per_sex_csv(path, table, **kw):
    return S.write_per_sex_csv(path, table.drop(columns="mu_true", errors="ignore"), **kw)


def all_in_one(tmp_path, truth, edit=None, name="all.csv"):
    """One long file, both sexes: Sex, Age, Deaths, Exposure (no year column). `edit(f, m)` may
    alter the two sex tables to plant a defect."""
    f = truth[("Female", 2020)].drop(columns="mu_true").copy()
    m = truth[("Male", 2020)].drop(columns="mu_true").copy()
    if edit:
        f, m = edit(f, m)
    long = pd.concat([f.assign(Sex="f"), m.assign(Sex="m")])
    long = long.rename(columns={"age": "Age", "Dx": "Deaths", "Ex": "Exposure"})[["Sex", "Age", "Deaths", "Exposure"]]
    p = tmp_path / name
    long.to_csv(p, index=False)
    return p


# --------------------------------------------------------------------------
# One table, both sexes - or one, by choice
# --------------------------------------------------------------------------


def test_both_sexes_come_back_in_one_table(hmd, truth):
    df = L.load_separate(deaths=hmd["deaths"], exposure=hmd["exposure"], year=2020)
    check_contract(df)
    assert len(df) == 202
    for s in BOTH:
        np.testing.assert_array_equal(rows(df, s)["age"], truth[(s, 2020)]["age"])    # '110+' is not an age
        np.testing.assert_array_equal(rows(df, s)["Dx"], truth[(s, 2020)]["Dx"])
        np.testing.assert_array_equal(rows(df, s)["Ex"], truth[(s, 2020)]["Ex"])
    assert (df.attrs["view"], df.attrs["year"], df.attrs["cohort"]) == ("period", 2020, None)
    assert df["source"].iloc[0] == "Deaths_1x1.txt + Exposures_1x1.txt"
    assert len(note(df, "non_single_ages")) == 1 and "110+" in note(df, "non_single_ages")[0].message


@pytest.mark.parametrize("sex,want", [("Female", ("Female",)), ("Male", ("Male",)), ("f", ("Female",)),
                                      ("Hommes", ("Male",)), ("both", BOTH), ("les deux", BOTH)])
def test_the_sex_choice(hmd, sex, want):
    df = L.load_separate(deaths=hmd["deaths"], exposure=hmd["exposure"], year=2021, sex=sex)
    check_contract(df, want)


def test_total_is_not_handled_and_its_column_never_leaks_in(hmd):
    with pytest.raises(L.LoaderError, match="Sexe inconnu"):
        L.load_separate(deaths=hmd["deaths"], exposure=hmd["exposure"], year=2020, sex="Total")
    df = L.load_separate(deaths=hmd["deaths"], exposure=hmd["exposure"], year=2020)       # files carry Total
    assert set(df["sex"]) == set(BOTH)

    total_only = S.write_hmd(hmd["dir"] / "E_total.txt", "Ex", hmd["by_year"], columns=("Total",))
    with pytest.raises(L.ColumnMappingError, match="deux sexes confondus") as exc:
        L.load_separate(deaths=hmd["deaths"], exposure=total_only, year=2020)
    assert exc.value.missing == ["Female"]


def test_one_sex_missing_from_a_wide_file_is_refused_for_both(hmd):
    only_f = S.write_hmd(hmd["dir"] / "E_f.txt", "Ex", hmd["by_year"], columns=("Female",))
    with pytest.raises(L.ColumnMappingError, match="pas de colonne Male"):
        L.load_separate(deaths=hmd["deaths"], exposure=only_f, year=2020)
    df = L.load_separate(deaths=hmd["deaths"], exposure=only_f, year=2020, sex="Female")
    check_contract(df, ("Female",))


def test_bad_arguments_fail_before_any_file_is_touched():
    with pytest.raises(L.LoaderError, match="Sexe inconnu"):
        L.load_data("missing.csv", sex="Total")
    with pytest.raises(L.LoaderError, match="quantity inconnue"):
        L.load_data("missing.csv", quantity="qx")
    with pytest.raises(L.LoaderError, match="Vision inconnue"):
        L.load_data("missing.csv", view="annual")
    with pytest.raises(L.LoaderError, match="exclusifs"):
        L.load_data("missing.csv", year=2000, cohort=1950)
    with pytest.raises(L.LoaderError, match="générationnelle"):
        L.load_data("missing.csv", view="period", cohort=1950)
    with pytest.raises(FileNotFoundError):
        L.load_data("missing.csv", sex="Female")


# --------------------------------------------------------------------------
# Period tables from HMD-style files: what is dropped, what is rebuilt, what is refused
# --------------------------------------------------------------------------


def test_hmd_text_missing_marker_open_group_and_commented_header(hmd):
    deaths = S.write_hmd(hmd["dir"] / "D_gap.txt", "Dx", hmd["by_year"], comment_header=True,
                         missing={(2020, 30, "Female"), (2020, 31, "Female")})
    df = L.load_separate(deaths=deaths, exposure=hmd["exposure"], year=2020)
    assert len(rows(df, "Female")) == 99 and len(rows(df, "Male")) == 101    # the gap hits one sex only
    rep = df.attrs["qc"]["Female"]
    assert rep.n_missing == 2 and item(df, "Female", "missing_values").ages == [30, 31]
    assert "(Dx absent)" in item(df, "Female", "missing_values").message
    assert item(df, "Female", "age_gaps").ages == [30, 31]
    assert df.attrs["qc"]["Male"].n_missing == 0 and not df.attrs["qc"]["Male"].orange


def test_exposure_and_rates_rebuild_dx_and_say_so(hmd, truth):
    df = L.load_separate(exposure=hmd["exposure"], rates=hmd["rates"], year=2020)
    check_contract(df)
    assert len(note(df, "dx_reconstructed")) == 1
    for s in BOTH:
        t, r = truth[(s, 2020)], rows(df, s)
        # mx is printed to 6 decimals: Dx = mx * Ex is out by at most 0.5e-6 * Ex ...
        assert (np.abs(r["Dx"] - t["Dx"]) <= 0.5e-6 * t["Ex"] + 1e-9).all()
        # ... and mx_raw is exactly the published rate
        np.testing.assert_allclose(r["mx_raw"], np.round(t["Dx"] / t["Ex"], 6), rtol=1e-9, atol=0)


def test_rates_alone_need_a_reference_exposure(hmd):
    with pytest.raises(L.ColumnMappingError, match="E_ref") as exc:
        L.load_separate(rates=hmd["rates"], year=2020)
    assert "Ex" in exc.value.missing
    df = L.load_separate(rates=hmd["rates"], year=2020, E_ref=50_000)
    assert (df["Ex"] == 50_000).all() and len(note(df, "ex_reference")) == 1
    pd.testing.assert_frame_equal(df, L.load_data(hmd["rates"], year=2020, E_ref=50_000))
    for bad in (0, -5, float("nan")):
        with pytest.raises(L.LoaderError, match="E_ref"):
            L.load_separate(rates=hmd["rates"], year=2020, E_ref=bad)


def test_rates_given_beside_deaths_and_exposure_are_cross_checked(hmd, truth):
    df = L.load_separate(deaths=hmd["deaths"], exposure=hmd["exposure"], rates=hmd["rates"], year=2020)
    assert not any(r.orange for r in df.attrs["qc"].values())              # printed rates agree with Dx / Ex

    bad = {y: {s: t.copy() for s, t in d.items()} for y, d in hmd["by_year"].items()}
    bad[2020]["Female"].loc[50, "Dx"] += 400                                # the rates file says 400 more deaths
    rates = S.write_hmd(hmd["dir"] / "Mx_bad.txt", "mx", bad)
    df = L.load_separate(deaths=hmd["deaths"], exposure=hmd["exposure"], rates=rates, year=2020)
    assert item(df, "Female", "mx_consistency").ages == [50]
    assert not df.attrs["qc"]["Male"].orange
    np.testing.assert_array_equal(rows(df, "Female")["Dx"], truth[("Female", 2020)]["Dx"])   # Dx / Ex wins


def test_not_enough_inputs_say_what_is_missing(hmd):
    with pytest.raises(L.ColumnMappingError) as exc:
        L.load_separate(deaths=hmd["deaths"], year=2020)
    assert exc.value.missing == ["Ex"]
    with pytest.raises(L.ColumnMappingError) as exc:
        L.load_separate(exposure=hmd["exposure"], year=2020)
    assert exc.value.missing == ["Dx"]
    with pytest.raises(L.ColumnMappingError, match="Dx = 0"):                # Dx + mx cannot give back Ex
        L.load_separate(deaths=hmd["deaths"], rates=hmd["rates"], year=2020)
    with pytest.raises(L.LoaderError, match="Aucun fichier"):
        L.load_separate()


def test_year_must_be_chosen_when_the_files_hold_several(hmd):
    with pytest.raises(L.AmbiguousSelectionError) as exc:
        L.load_separate(deaths=hmd["deaths"], exposure=hmd["exposure"])
    assert exc.value.what == "year" and exc.value.options == list(YEARS)
    with pytest.raises(L.AmbiguousSelectionError, match="1999"):
        L.load_separate(deaths=hmd["deaths"], exposure=hmd["exposure"], year=1999)


def test_files_must_cover_the_same_year(tmp_path):
    d = S.write_hmd(tmp_path / "D_2020.txt", "Dx", S.hmd_tables((2020,)))
    e = S.write_hmd(tmp_path / "E_2019.txt", "Ex", S.hmd_tables((2019,)))
    with pytest.raises(L.LoaderError, match="années différentes"):
        L.load_separate(deaths=d, exposure=e)


def test_pairing_ages_across_files(hmd, truth):
    short = {y: {s: t.iloc[:90] for s, t in d.items()} for y, d in hmd["by_year"].items()}
    expo = S.write_hmd(hmd["dir"] / "E_short.txt", "Ex", short)
    df = L.load_separate(deaths=hmd["deaths"], exposure=expo, year=2020)
    assert rows(df, "Male")["age"].max() == 89 and df.attrs["qc"]["Male"].n_missing == 11
    assert item(df, "Male", "missing_values").ages == list(range(90, 101))

    t = truth[("Female", 2020)]
    d = hmd["dir"] / "d_dup.csv"
    pd.DataFrame({"Age": list(t["age"]) + [7], "Female": list(t["Dx"]) + [1], "Male": list(t["Dx"]) + [1]}
                 ).to_csv(d, index=False)
    with pytest.raises(L.LoaderError, match="dupliqués"):
        L.load_separate(deaths=d, exposure=hmd["exposure"], year=2020)


def test_a_french_excel_style_csv_without_a_year_column(tmp_path, truth):
    f, m = truth[("Female", 2020)], truth[("Male", 2020)]

    def write(name, a, b):
        p = tmp_path / name
        pd.DataFrame({"Âge": f["age"], "Femmes": a, "Hommes": b}).to_csv(p, sep=";", index=False, decimal=",")
        return p

    df = L.load_separate(deaths=write("d.csv", f["Dx"], m["Dx"]), exposure=write("e.csv", f["Ex"], m["Ex"]))
    check_contract(df)
    np.testing.assert_array_equal(rows(df, "Male")["Dx"], m["Dx"])
    np.testing.assert_allclose(rows(df, "Female")["Ex"], f["Ex"])
    assert df.attrs["year"] is None


def test_one_row_per_sex_layout(tmp_path, truth):
    by_year = S.hmd_tables(YEARS)
    d = S.write_long_csv(tmp_path / "D_long.csv", "Dx", by_year)
    e = S.write_long_csv(tmp_path / "E_long.csv", "Ex", by_year)
    df = L.load_separate(deaths=d, exposure=e, year=2020)
    check_contract(df)
    for s in BOTH:
        np.testing.assert_array_equal(rows(df, s)["Dx"], truth[(s, 2020)]["Dx"])
    male = L.load_separate(deaths=d, exposure=e, year=2020, sex="Male")
    check_contract(male, ("Male",))

    only_m = tmp_path / "D_m.csv"
    pd.read_csv(d).query("Sex == 'm'").to_csv(only_m, index=False)
    with pytest.raises(L.AmbiguousSelectionError) as exc:
        L.load_separate(deaths=only_m, exposure=e, year=2020)
    assert exc.value.what == "sex" and exc.value.options == ["Male"]


def test_the_layouts_can_differ_from_file_to_file(tmp_path, hmd, truth):
    d = S.write_long_csv(tmp_path / "D_long.csv", "Dx", hmd["by_year"])                 # a row per sex
    df = L.load_separate(deaths=d, exposure=hmd["exposure"], year=2021)                  # a column per sex
    for s in BOTH:
        np.testing.assert_array_equal(rows(df, s)["Dx"], truth[(s, 2021)]["Dx"])
        np.testing.assert_array_equal(rows(df, s)["Ex"], truth[(s, 2021)]["Ex"])


def test_a_file_that_does_not_say_its_sex_is_attributed_to_the_single_sex_asked(tmp_path, hmd, truth):
    t = truth[("Female", 2020)]
    d, e = tmp_path / "d.csv", tmp_path / "e.csv"
    pd.DataFrame({"Age": t["age"], "Deaths": t["Dx"]}).to_csv(d, index=False)
    pd.DataFrame({"Age": t["age"], "Valeur": t["Ex"]}).to_csv(e, index=False)         # the only column left
    df = L.load_separate(deaths=d, exposure=e, sex="Female")
    check_contract(df, ("Female",))
    np.testing.assert_array_equal(df["Ex"], t["Ex"])
    with pytest.raises(L.ColumnMappingError, match="n'indique pas le sexe") as exc:
        L.load_separate(deaths=d, exposure=e)                                          # 'both': cannot split
    assert exc.value.missing == ["sex"]


def test_source_gives_each_input_its_own_reading_options(tmp_path, hmd, truth):
    f, m = truth[("Female", 2020)], truth[("Male", 2020)]
    xlsx = tmp_path / "deaths.xlsx"
    with pd.ExcelWriter(xlsx) as w:
        pd.DataFrame({"readme": ["x"]}).to_excel(w, sheet_name="notes", index=False)
        pd.DataFrame({"Age": f["age"], "Female": f["Dx"], "Male": m["Dx"]}).to_excel(
            w, sheet_name="Décès", index=False, startrow=2)                            # names on row 2
    odd = tmp_path / "expo_odd.csv"
    pd.DataFrame({"Alter": f["age"], "Female": f["Ex"], "Male": m["Ex"]}).to_csv(odd, index=False)

    with pytest.raises(L.ColumnMappingError, match="âge"):
        L.load_separate(deaths=L.Source(xlsx, sheet="Décès", header=2), exposure=odd)
    df = L.load_separate(deaths=L.Source(xlsx, sheet="Décès", header=2),
                         exposure=L.Source(odd, col_map={"age": "Alter"}))
    np.testing.assert_array_equal(rows(df, "Male")["Dx"], m["Dx"])
    np.testing.assert_array_equal(rows(df, "Female")["Ex"], f["Ex"])
    with pytest.raises(L.ColumnMappingError, match="introuvable"):
        L.load_separate(deaths=L.Source(xlsx, sheet="Décès", header=2),
                        exposure=L.Source(odd, col_map={"age": "nope"}))
    with pytest.raises(L.LoaderError, match="clé inconnue"):
        L.load_separate(deaths=L.Source(xlsx, sheet="Décès", header=2),
                        exposure=L.Source(odd, col_map={"colour": "Alter"}))


def test_uploaded_file_objects_are_accepted(hmd):
    up = FakeUploadedFile(hmd["exposure"].read_bytes(), "FRA_Exposures_1x1.txt")
    up.read()                                                      # pointer at the end: must not matter
    d = FakeUploadedFile(hmd["deaths"].read_bytes(), "Deaths_1x1.txt")
    df = L.load_separate(deaths=d, exposure=up, year=2020)
    assert len(df) == 202 and df["source"].iloc[0] == "Deaths_1x1.txt + FRA_Exposures_1x1.txt"
    assert L.inspect_file(up).quantity == "Ex"


# --------------------------------------------------------------------------
# What is refused, and how it says so
# --------------------------------------------------------------------------


def test_the_hmd_inputdb_is_refused_with_the_files_to_use_instead(hmd):
    """FRATNPdeath.txt: one row per age and Lexis triangle. Read as is, every age appears twice."""
    inputdb = S.write_inputdb(hmd["dir"] / "FRATNPdeath.txt", hmd["by_year"])
    with pytest.raises(L.FileFormatError, match="InputDB") as exc:
        L.load_separate(deaths=inputdb, exposure=hmd["exposure"], year=2020)
    assert "Mx_1x1" in str(exc.value) and "Exposures_1x1" in str(exc.value)
    with pytest.raises(L.FileFormatError, match="InputDB"):
        L.inspect_file(inputdb)


def test_old_xls_gets_a_clear_message():
    with pytest.raises(L.FileFormatError, match=r"\.xlsx"):
        L.load_data(FakeUploadedFile(b"\xd0\xcf\x11\xe0" + b"\x00" * 200, "old.xls"), sex="Female")


def test_garbage_in_a_numeric_column_is_an_error_not_a_silent_nan(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("Age,Deaths,Exposure\n" + "\n".join(
        f"{a},{'abc' if a == 4 else 10},1000" for a in range(30)) + "\n")
    with pytest.raises(L.LoaderError, match="non numérique"):
        L.load_data(p, sex="Female")


def test_fractional_ages_are_refused(tmp_path):
    p = tmp_path / "half.csv"
    p.write_text("Age,Deaths,Exposure\n0.5,10,1000\n1.5,12,1000\n")
    with pytest.raises(L.LoaderError, match="non entiers"):
        L.load_data(p, sex="Female")


# --------------------------------------------------------------------------
# An all-in-one file per sex (load_by_sex), and one file (load_data)
# --------------------------------------------------------------------------


def test_load_by_sex_stacks_one_file_per_sex(tmp_path, truth):
    out = L.load_by_sex(female=per_sex_csv(tmp_path / "f.csv", truth[("Female", 2020)]),
                        male=per_sex_csv(tmp_path / "m.csv", truth[("Male", 2020)]))
    check_contract(out)
    for s in BOTH:
        np.testing.assert_array_equal(rows(out, s)["Dx"], truth[(s, 2020)]["Dx"])
    assert out["source"].iloc[0] == "f.csv + m.csv"
    one = L.load_by_sex(male=per_sex_csv(tmp_path / "m.csv", truth[("Male", 2020)]))
    check_contract(one, ("Male",))
    with pytest.raises(L.LoaderError):
        L.load_by_sex()


def test_load_by_sex_slot_decides_the_sex_of_a_file_that_does_not_say(tmp_path, truth):
    """Female's file is given in the Male slot: the loader trusts the slot and cannot know better."""
    p = per_sex_csv(tmp_path / "x.csv", truth[("Female", 2020)])
    assert set(L.load_by_sex(male=p)["sex"]) == {"Male"}


def test_load_by_sex_a_file_with_a_sex_column_is_filtered_to_its_slot(tmp_path, truth):
    p = all_in_one(tmp_path, truth)
    out = L.load_by_sex(female=p)
    check_contract(out, ("Female",))
    np.testing.assert_array_equal(out["Dx"], truth[("Female", 2020)]["Dx"])


def test_load_by_sex_per_file_sheet_header_and_mapping(tmp_path, truth):
    f, m = truth[("Female", 2020)], truth[("Male", 2020)]
    pf, pm = tmp_path / "f.xlsx", tmp_path / "m.xlsx"
    with pd.ExcelWriter(pf) as w:
        f.drop(columns="mu_true").rename(columns={"age": "Age", "Dx": "Deaths", "Ex": "Exposure"}).to_excel(
            w, sheet_name="F", index=False)
    with pd.ExcelWriter(pm) as w:
        pd.DataFrame({"x": ["note"]}).to_excel(w, sheet_name="skip", index=False)
        m.drop(columns="mu_true").rename(columns={"age": "a", "Dx": "b", "Ex": "c"}).to_excel(
            w, sheet_name="M", index=False, startrow=3)
    out = L.load_by_sex(female=L.Source(pf, sheet="F"),
                        male=L.Source(pm, sheet="M", header=3, col_map={"age": "a", "Dx": "b", "Ex": "c"}))
    np.testing.assert_array_equal(rows(out, "Male")["Dx"], m["Dx"])
    np.testing.assert_array_equal(rows(out, "Female")["Dx"], f["Dx"])


def test_load_by_sex_files_on_different_years_are_refused(tmp_path, truth):
    def with_year(name, sex, year):
        t = truth[(sex, 2020)].drop(columns="mu_true").rename(columns={"age": "Age", "Dx": "Deaths", "Ex": "Exposure"})
        p = tmp_path / name
        t.assign(Year=year).to_csv(p, index=False)
        return p

    with pytest.raises(L.LoaderError, match="années différentes"):
        L.load_by_sex(female=with_year("f.csv", "Female", 2019), male=with_year("m.csv", "Male", 2020))


def test_load_by_sex_rates_only_files_take_a_reference_exposure(tmp_path, truth):
    paths = {}
    for s in BOTH:
        t = truth[(s, 2020)]
        paths[s] = tmp_path / f"{s}.csv"
        pd.DataFrame({"Age": t["age"], "mx": t["Dx"] / t["Ex"]}).to_csv(paths[s], index=False)
    with pytest.raises(L.ColumnMappingError, match="E_ref"):
        L.load_by_sex(female=paths["Female"], male=paths["Male"])
    out = L.load_by_sex(female=paths["Female"], male=paths["Male"], E_ref=100_000)
    assert (out["Ex"] == 100_000).all()


def test_load_data_reads_an_all_in_one_file_with_both_sexes(tmp_path, truth):
    p = all_in_one(tmp_path, truth)
    df = L.load_data(p)
    check_contract(df)
    for s in BOTH:
        np.testing.assert_array_equal(rows(df, s)["Dx"], truth[(s, 2020)]["Dx"])
    check_contract(L.load_data(p, sex="Male"), ("Male",))


def test_load_data_a_file_without_sex_info_needs_one_sex_chosen(tmp_path, truth):
    p = per_sex_csv(tmp_path / "f.csv", truth[("Female", 2020)])
    with pytest.raises(L.ColumnMappingError, match="n'indique pas le sexe") as exc:
        L.load_data(p)
    assert exc.value.missing == ["sex"]
    df = L.load_data(p, sex="Female")
    check_contract(df, ("Female",))


def test_load_data_unrecognised_columns_get_a_mapping(tmp_path, truth):
    t = truth[("Female", 2020)]
    p = tmp_path / "weird.csv"
    pd.DataFrame({"a": t["age"], "b": t["Dx"], "c": t["Ex"]}).to_csv(p, index=False)
    with pytest.raises(L.ColumnMappingError) as exc:
        L.load_data(p, sex="Female")
    assert exc.value.missing == ["age"] and exc.value.available == ["a", "b", "c"]
    df = L.load_data(p, sex="Female", col_map={"age": "a", "Dx": "b", "Ex": "c"})
    np.testing.assert_array_equal(df["Dx"], t["Dx"])


def test_load_data_hmd_file_of_one_quantity(hmd, tmp_path):
    with pytest.raises(L.ColumnMappingError, match="Ex introuvable"):
        L.load_data(hmd["deaths"], year=2020)
    with pytest.raises(L.ColumnMappingError, match="Dx introuvable"):
        L.load_data(hmd["exposure"], year=2020)
    mystery = S.write_hmd(tmp_path / "table.txt", "mx", hmd["by_year"], title="Synthland, Mystery table")
    with pytest.raises(L.ColumnMappingError) as exc:
        L.load_data(mystery, year=2020, E_ref=1e5)
    assert exc.value.missing == ["quantity"]
    assert len(L.load_data(mystery, year=2020, E_ref=1e5, quantity="mx")) == 202


def test_unsorted_input_comes_back_sorted(tmp_path, truth):
    t = truth[("Female", 2020)].sample(frac=1, random_state=0)
    df = L.load_data(per_sex_csv(tmp_path / "f.csv", t), sex="Female")
    np.testing.assert_array_equal(df["age"], np.arange(101))


# --------------------------------------------------------------------------
# Period or cohort: the same ages, a different slice
# --------------------------------------------------------------------------


@pytest.fixture
def cohort_files(tmp_path):
    """Generation tables: the keys are birth years, the first column is called Cohort."""
    by_cohort = S.hmd_tables((1950, 1951))
    return {
        "by_cohort": by_cohort,
        "deaths": S.write_hmd(tmp_path / "cDeaths_1x1.txt", "Dx", by_cohort, key_name="Cohort"),
        "exposure": S.write_hmd(tmp_path / "cExposures_1x1.txt", "Ex", by_cohort, key_name="Cohort"),
        "rates": S.write_hmd(tmp_path / "cMx_1x1.txt", "mx", by_cohort, key_name="Cohort"),
    }


def test_cohort_files_are_read_as_generation_tables(cohort_files):
    info = L.inspect_file(cohort_files["rates"])
    assert (info.views, info.years, info.cohorts, info.quantity) == (["cohort"], [], [1950, 1951], "mx")

    with pytest.raises(L.AmbiguousSelectionError) as exc:
        L.load_separate(deaths=cohort_files["deaths"], exposure=cohort_files["exposure"])
    assert exc.value.what == "cohort" and exc.value.options == [1950, 1951]

    df = L.load_separate(deaths=cohort_files["deaths"], exposure=cohort_files["exposure"], cohort=1951)
    check_contract(df)
    assert (df.attrs["view"], df.attrs["cohort"], df.attrs["year"]) == ("cohort", 1951, None)
    for s in BOTH:
        np.testing.assert_array_equal(rows(df, s)["Dx"], cohort_files["by_cohort"][1951][s]["Dx"])
    rates = L.load_separate(exposure=cohort_files["exposure"], rates=cohort_files["rates"], cohort=1950)
    assert rates.attrs["cohort"] == 1950


def test_period_and_cohort_never_get_mixed_up(cohort_files, hmd, tmp_path):
    with pytest.raises(L.LoaderError, match="générations.*pas des années"):
        L.load_separate(deaths=cohort_files["deaths"], exposure=cohort_files["exposure"], year=2020)
    with pytest.raises(L.ColumnMappingError, match="aucune année") as exc:
        L.load_separate(deaths=cohort_files["deaths"], exposure=cohort_files["exposure"], view="period")
    assert exc.value.missing == ["year"]
    with pytest.raises(L.ColumnMappingError, match="pas de colonne de génération") as exc:
        L.load_separate(deaths=hmd["deaths"], exposure=hmd["exposure"], view="cohort")
    assert exc.value.missing == ["cohort"]

    # one period file and one cohort file, each with a single key: nothing to choose, yet not the same view
    d = S.write_hmd(tmp_path / "D_2020.txt", "Dx", S.hmd_tables((2020,)))
    e = S.write_hmd(tmp_path / "cE_1950.txt", "Ex", S.hmd_tables((1950,)), key_name="Cohort")
    with pytest.raises(L.LoaderError, match="même vision"):
        L.load_separate(deaths=d, exposure=e)

    # two cohort files on different generations
    d = S.write_hmd(tmp_path / "cD_1950.txt", "Dx", S.hmd_tables((1950,)), key_name="Cohort")
    e = S.write_hmd(tmp_path / "cE_1949.txt", "Ex", S.hmd_tables((1949,)), key_name="Cohort")
    with pytest.raises(L.LoaderError, match="générations différentes"):
        L.load_separate(deaths=d, exposure=e)


def test_a_file_with_both_a_year_and_a_cohort_column_follows_the_view_asked(tmp_path, truth):
    """A Lexis-style long file: each cell has its year and its generation."""
    parts = []
    for yr in YEARS:
        for s, label in (("Female", "f"), ("Male", "m")):
            t = truth[(s, yr)]
            parts.append(pd.DataFrame({"Year": yr, "Cohort": yr - t["age"], "Age": t["age"], "Sex": label,
                                       "Deaths": t["Dx"], "Exposure": t["Ex"]}))
    p = tmp_path / "lexis.csv"
    pd.concat(parts).to_csv(p, index=False)

    info = L.inspect_file(p)
    assert info.views == ["period", "cohort"] and info.years == list(YEARS) and 1920 in info.cohorts

    period = L.load_data(p, year=2020)                                    # auto: the year column comes first
    assert period.attrs["view"] == "period" and len(rows(period, "Female")) == 101
    gen = L.load_data(p, cohort=1950)                                     # a cohort given: generation view
    assert gen.attrs["view"] == "cohort" and gen.attrs["cohort"] == 1950
    assert list(rows(gen, "Female")["age"]) == [70, 71]                   # the cells of cohort 1950: ages 70 (2020), 71 (2021)
    assert L.load_data(p, view="cohort", cohort=1950).attrs["view"] == "cohort"
    with pytest.raises(L.AmbiguousSelectionError) as exc:
        L.load_data(p, view="cohort")
    assert exc.value.what == "cohort"


# --------------------------------------------------------------------------
# Quality controls (spec 1.5), one report per sex
# --------------------------------------------------------------------------


def test_clean_data_passes_every_control_for_both_sexes(tmp_path, truth):
    df = L.load_data(all_in_one(tmp_path, truth))
    for s in BOTH:
        rep = df.attrs["qc"][s]
        assert rep.can_proceed and not rep.orange and {i.level for i in rep.items} == {"ok"}
        assert [i.check for i in rep.items] == ["dx_nonneg", "ex_positive", "duplicate_ages", "age_range",
                                                "age_gaps", "dx_zero", "mx_range"]
    assert L.can_proceed(df)


def test_a_defect_on_one_sex_is_reported_on_that_sex_only(tmp_path, truth):
    def plant(f, m):
        f.loc[40, "Dx"] = -3
        return f, m
    df = L.load_data(all_in_one(tmp_path, truth, plant))
    assert item(df, "Female", "dx_nonneg").level == "red" and item(df, "Female", "dx_nonneg").ages == [40]
    assert df.attrs["qc"]["Male"].can_proceed and not df.attrs["qc"]["Female"].can_proceed
    assert not L.can_proceed(df) and len(df) == 202                    # returned, not raised


def test_deaths_without_exposure_block_and_name_the_sex(tmp_path, truth):
    def plant(f, m):
        m.loc[60, ["Dx", "Ex"]] = [5, 0.0]
        return f, m
    with pytest.raises(L.DataQualityError) as exc:
        L.load_data(all_in_one(tmp_path, truth, plant))
    err = exc.value
    assert err.reports["Male"].blocking[0].ages == [60] and not err.reports["Female"].blocking
    assert "Male" in str(err) and "all.csv" in str(err) and err.data is not None


def test_negative_exposure_blocks_even_without_deaths(tmp_path, truth):
    def plant(f, m):
        f.loc[10, ["Dx", "Ex"]] = [0, -4.0]
        return f, m
    with pytest.raises(L.DataQualityError) as exc:
        L.load_data(all_in_one(tmp_path, truth, plant))
    assert exc.value.reports["Female"].blocking[0].ages == [10]


def test_no_exposure_and_no_deaths_is_dropped_and_reported(tmp_path, truth):
    def plant(f, m):
        f.loc[70, ["Dx", "Ex"]] = [0, 0.0]
        return f, m
    df = L.load_data(all_in_one(tmp_path, truth, plant))
    assert 70 not in set(rows(df, "Female")["age"]) and len(rows(df, "Male")) == 101
    assert item(df, "Female", "no_exposure").level == "orange" and item(df, "Female", "no_exposure").ages == [70]


def test_duplicate_ages_are_red_and_kept(tmp_path, truth):
    def plant(f, m):
        return pd.concat([f, f.iloc[[5, 6]]]), m
    df = L.load_data(all_in_one(tmp_path, truth, plant))
    assert item(df, "Female", "duplicate_ages").level == "red" and item(df, "Female", "duplicate_ages").ages == [5, 6]
    assert len(rows(df, "Female")) == 103


def test_dx_zero_is_kept_flagged_and_gives_a_zero_rate(tmp_path, truth):
    def plant(f, m):
        f.loc[[3, 4], "Dx"] = 0
        return f, m
    df = L.load_data(all_in_one(tmp_path, truth, plant))
    r = rows(df, "Female")
    assert len(r) == 101                                               # spec 1.6: NOT suppressed
    assert item(df, "Female", "dx_zero").level == "orange" and item(df, "Female", "dx_zero").ages == [3, 4]
    assert df.attrs["qc"]["Female"].n_dx_zero == 2 and (r.loc[r["age"].isin([3, 4]), "mx_raw"] == 0).all()
    assert df.attrs["qc"]["Female"].can_proceed                        # orange never stops the app


def test_short_age_range_and_rate_above_one_are_warnings(tmp_path, truth):
    def plant(f, m):
        m.loc[99, "Dx"] = int(m.loc[99, "Ex"]) + 50
        return f.iloc[60:75], m
    df = L.load_data(all_in_one(tmp_path, truth, plant))
    assert item(df, "Female", "age_range").level == "orange" and "15" in item(df, "Female", "age_range").message
    assert item(df, "Male", "mx_range").level == "orange" and item(df, "Male", "mx_range").ages == [99]


def test_check_quality_stands_alone_and_nan_never_passes(tmp_path, truth):
    df = L.load_data(all_in_one(tmp_path, truth))
    one = rows(df, "Female")
    one.loc[2, "Dx"] = np.nan
    rep = L.check_quality(one)
    assert rep.red and rep.red[0].check == "dx_nonneg" and rep.red[0].ages == [2]
    assert list(rep.to_frame().columns) == ["check", "level", "message", "ages"]
    assert all(type(a) is int for i in rep.items for a in i.ages)
    assert "[Alerte rouge]" in str(rep)


def test_qc_summary_has_one_column_per_sex(tmp_path, truth):
    def plant(f, m):
        f.loc[[3, 4], "Dx"] = 0
        return f, m
    df = L.load_data(all_in_one(tmp_path, truth, plant))
    s = L.qc_summary(df).set_index("Contrôle")
    assert list(s.columns) == ["Female", "Male"]
    assert s.loc["Plage d'âge", "Female"] == "0 - 100" and s.loc["Nb Dx = 0", "Female"] == "2"
    assert s.loc["Nb Dx = 0", "Male"] == "0" and s.loc["Valeurs manquantes (exclues)", "Male"] == "0"
    assert s.loc["Décès totaux (somme des Dx)", "Male"] == f"{rows(df, 'Male')['Dx'].sum():,.0f}"
    assert all(isinstance(v, str) for v in s.to_numpy().ravel())      # Arrow-safe for st.dataframe


# --------------------------------------------------------------------------
# The standard file, and what the UI asks before loading
# --------------------------------------------------------------------------


def test_the_standard_table_holds_both_sexes_and_reads_back(hmd, tmp_path):
    df = L.load_separate(deaths=hmd["deaths"], exposure=hmd["exposure"], year=2020)
    st = L.standard_table(df)
    assert list(st.columns) == ["year", "sex", "age", "Dx", "Ex", "mx_raw"] and set(st["year"]) == {2020}
    p = tmp_path / "standard.csv"
    st.to_csv(p, index=False)
    back = L.load_data(p)                                             # the file the app hands out
    cols = ["sex", "age", "Dx", "Ex", "mx_raw"]
    pd.testing.assert_frame_equal(back[cols], df[cols], check_exact=False, rtol=1e-12)
    assert back.attrs["year"] == 2020


def test_the_standard_table_of_a_cohort_says_cohort(cohort_files, tmp_path):
    df = L.load_separate(deaths=cohort_files["deaths"], exposure=cohort_files["exposure"], cohort=1950)
    st = L.standard_table(df)
    assert list(st.columns)[:2] == ["cohort", "sex"] and set(st["cohort"]) == {1950}
    p = tmp_path / "standard_cohort.csv"
    st.to_csv(p, index=False)
    back = L.load_data(p)
    assert back.attrs["view"] == "cohort" and back.attrs["cohort"] == 1950


def test_inspect_file_feeds_the_selectors(hmd, tmp_path, truth):
    info = L.inspect_file(hmd["exposure"])
    assert (info.fmt, info.layout, info.quantity) == ("hmd", "sex_columns", "Ex")
    assert (info.views, info.years, info.cohorts, info.sexes) == (["period"], list(YEARS), [], ["Female", "Male"])
    assert info.auto_map["age"] == "Age" and info.auto_map["female"] == "Female"
    assert L.inspect_file(hmd["rates"]).quantity == "mx" and L.inspect_file(hmd["deaths"]).quantity == "Dx"

    info = L.inspect_file(all_in_one(tmp_path, truth))
    assert (info.fmt, info.layout, info.views, info.sexes) == ("csv", "roles", [], ["Female", "Male"])
    assert set(info.auto_map) == {"age", "sex", "dx", "ex"}

    plain = per_sex_csv(tmp_path / "f.csv", truth[("Female", 2020)])
    assert L.inspect_file(plain).sexes == []                           # the file does not say its sex
    odd = tmp_path / "odd.csv"
    pd.DataFrame({"a": [1], "b": [2], "c": [3]}).to_csv(odd, index=False)
    assert L.inspect_file(odd).auto_map == {}                          # nothing found -> the UI shows dropdowns

    xlsx = tmp_path / "f.xlsx"
    with pd.ExcelWriter(xlsx) as w:
        pd.DataFrame({"Age": [0, 1], "Deaths": [3, 4], "Exposure": [10.0, 20.0]}).to_excel(w, sheet_name="data", index=False)
        pd.DataFrame({"x": [1]}).to_excel(w, sheet_name="notes", index=False)
    assert L.inspect_file(L.Source(xlsx, sheet="data")).sheets == ["data", "notes"]
