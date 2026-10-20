"""Module 1 - the three options (A, B, C) on the same data, and the sources beyond HMD.

Run:   python -m pytest tests/test_loader_sources.py -v

INSEE (T74BIS) and Eurostat (bulk TSV) fixtures follow the real files' layout - read from INSEE's
T74BIS.csv and Eurostat's TSV documentation. The population / exposure and wide-in-years fixtures are
generic stand-ins for national-office tables; no real INSEE population file was available to test.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import synthetic as S
from src import loader as L
from src.loader.parsing import age_from_label, parse_ages, to_numeric

BOTH = ("Female", "Male")


@pytest.fixture(scope="module")
def truth():
    return {(s, y): S.make_sex_table(s, y) for s in BOTH for y in (2020, 2021)}


def rows(df, sex):
    return df[df["sex"] == sex].reset_index(drop=True)


def note(df, check):
    return [i for i in df.attrs["notes"] if i.check == check]


def item(df, sex, check):
    (it,) = [i for i in df.attrs["qc"][sex].items if i.check == check]
    return it


COLS = ["sex", "age", "Dx", "Ex", "mx_raw"]


# --------------------------------------------------------------------------
# Option A, B, C: one dataset, three dispositions, one table
# --------------------------------------------------------------------------


def test_options_a_b_and_c_give_the_same_table(tmp_path, truth):
    by_year = S.hmd_tables((2020,))
    a = L.load_separate(deaths=S.write_hmd(tmp_path / "D.txt", "Dx", by_year),
                        exposure=S.write_hmd(tmp_path / "E.txt", "Ex", by_year))          # A: a file per quantity
    b = L.load_by_sex(female=S.write_per_sex_csv(tmp_path / "f.csv", truth[("Female", 2020)].drop(columns="mu_true")),
                      male=S.write_per_sex_csv(tmp_path / "m.csv", truth[("Male", 2020)].drop(columns="mu_true")))
    c = L.load_data(S.write_elementary_csv(tmp_path / "c.csv", by_year))                    # C: one elementary file
    pd.testing.assert_frame_equal(a[COLS], b[COLS], check_exact=False, rtol=1e-12)
    pd.testing.assert_frame_equal(a[COLS], c[COLS], check_exact=False, rtol=1e-12)
    assert a.attrs["year"] == c.attrs["year"] == 2020 and L.can_proceed(c)


def test_option_a_with_rates_warns_that_deaths_are_rebuilt(tmp_path, truth):
    """The text of option A: rates alone do not give the exact deaths without the exposure."""
    by_year = S.hmd_tables((2020,))
    df = L.load_separate(rates=S.write_hmd(tmp_path / "M.txt", "mx", by_year),
                         exposure=S.write_hmd(tmp_path / "E.txt", "Ex", by_year))
    assert "Dx reconstruit" in note(df, "dx_reconstructed")[0].message
    with pytest.raises(L.ColumnMappingError, match="E_ref"):                              # rates with no exposure at all
        L.load_separate(rates=S.write_hmd(tmp_path / "M.txt", "mx", by_year))


def test_option_c_many_years_and_a_segment_in_one_file(tmp_path, truth):
    p = S.write_elementary_csv(tmp_path / "c.csv", S.hmd_tables((2020, 2021)),
                               segment={"Region": {"North": 1.0, "South": 0.5}})
    info = L.inspect_file(p)
    assert info.segments == {"Region": ["North", "South"]} and info.years == [2020, 2021]

    with pytest.raises(L.AmbiguousSelectionError) as exc:                                  # the year comes first
        L.load_data(p)
    assert exc.value.what == "year"
    with pytest.raises(L.AmbiguousSelectionError) as exc:
        L.load_data(p, year=2020)
    assert (exc.value.what, exc.value.column, exc.value.options) == ("segment", "Region", ["North", "South"])

    north = L.load_data(p, year=2020, where={"Region": "North"})
    south = L.load_data(p, year=2020, where={"region": "South"})                           # column name: any case
    for s in BOTH:
        np.testing.assert_array_equal(rows(north, s)["Dx"], truth[(s, 2020)]["Dx"])
        np.testing.assert_allclose(rows(south, s)["Ex"], truth[(s, 2020)]["Ex"] * 0.5, atol=0.01)
    with pytest.raises(L.AmbiguousSelectionError, match="West") as exc:
        L.load_data(p, year=2020, where={"Region": "West"})
    assert exc.value.options == ["North", "South"]
    with pytest.raises(L.ColumnMappingError, match="Colour"):
        L.load_data(p, year=2020, where={"Colour": "red"})


# --------------------------------------------------------------------------
# INSEE state-civil tables (T74BIS layout)
# --------------------------------------------------------------------------


@pytest.fixture
def insee(tmp_path):
    by_year = S.hmd_tables((2020, 2021))
    return {"dir": tmp_path, "by_year": by_year,
            "deaths": S.write_insee_deaths(tmp_path / "T74BIS.csv", by_year, fr_extra=5),
            "exposure": S.write_hmd(tmp_path / "E.txt", "Ex", by_year)}


def test_insee_t74bis_is_recognised(insee):
    info = L.inspect_file(insee["deaths"])
    assert info.origin == "INSEE" and info.layout == "roles"
    assert {"age", "year", "sex", "dx"} <= set(info.auto_map)           # AGE2_REVOLU, ANNEE_1975_2024, SEXE, NBDEC
    assert info.years == [2020, 2021] and info.sexes == ["Female", "Male"]   # F / H; the E (ensemble) rows are ignored
    assert info.segments == {"CHAMP2": ["FM", "FR"]}
    assert L.inspect_file(insee["exposure"]).origin == "HMD"


def test_insee_the_field_must_be_chosen_and_selects_its_rows(insee, truth):
    with pytest.raises(L.AmbiguousSelectionError) as exc:
        L.load_separate(deaths=insee["deaths"], exposure=insee["exposure"], year=2020)
    assert (exc.value.what, exc.value.column, exc.value.options) == ("segment", "CHAMP2", ["FM", "FR"])

    fm = L.load_separate(deaths=L.Source(insee["deaths"], where={"CHAMP2": "FM"}),
                         exposure=insee["exposure"], year=2020)
    fr = L.load_separate(deaths=L.Source(insee["deaths"], where={"CHAMP2": "FR"}),
                         exposure=insee["exposure"], year=2020)
    for s in BOTH:
        np.testing.assert_array_equal(rows(fm, s)["Dx"], truth[(s, 2020)]["Dx"].iloc[:100])
        np.testing.assert_array_equal(rows(fr, s)["Dx"], truth[(s, 2020)]["Dx"].iloc[:100] + 5)


def test_insee_the_last_age_code_means_100_and_over_and_is_left_out(insee, truth):
    df = L.load_separate(deaths=L.Source(insee["deaths"], where={"CHAMP2": "FM"}),
                         exposure=insee["exposure"], year=2020)
    assert rows(df, "Female")["age"].max() == 99                          # AGE2_REVOLU: '100' = 100 and over
    msg = note(df, "non_single_ages")[0].message
    assert "100+" in msg and "TOT" in msg
    assert item(df, "Female", "missing_values").ages == [100]            # the exposure has it, the deaths cannot

    # any other file declares its open code with open_from
    plain = S.write_elementary_csv(insee["dir"] / "c.csv", insee["by_year"])
    closed = L.load_data(L.Source(plain, open_from=100), year=2020)
    assert rows(closed, "Male")["age"].max() == 99 and "100+" in note(closed, "non_single_ages")[0].message
    assert rows(L.load_data(plain, year=2020), "Male")["age"].max() == 100


def test_the_real_t74bis_header_and_first_rows_are_read(tmp_path):
    """The header and the first 12 data rows of INSEE's T74BIS.csv, as the file starts (statistics
    published under the Licence Ouverte): sex 'E' = both sexes, so the excerpt names neither F nor H."""
    p = tmp_path / "T74BIS.csv"
    p.write_text("CHAMP2;ANNEE_1975_2024;SEXE;AGE2_REVOLU;NBDEC\n" + "\n".join([
        "FM;1975;E;TOT;560353", "FM;1975;E;0;10277", "FM;1975;E;1;819", "FM;1975;E;2;604", "FM;1975;E;3;458",
        "FM;1975;E;4;383", "FM;1975;E;5;301", "FM;1975;E;6;328", "FM;1975;E;7;312", "FM;1975;E;8;276",
        "FM;1975;E;9;270", "FM;1975;E;10;251"]) + "\n", encoding="utf-8")
    info = L.inspect_file(p)
    assert info.origin == "INSEE" and info.layout == "roles" and info.years == [1975]
    assert set(info.auto_map) == {"age", "year", "sex", "dx"} and info.sexes == [] and info.segments == {}
    expo = S.write_hmd(tmp_path / "E.txt", "Ex", S.hmd_tables((1975,)))
    with pytest.raises(L.AmbiguousSelectionError, match="Female") as exc:        # only 'E' rows: no F, no H
        L.load_separate(deaths=p, exposure=expo)
    assert exc.value.what == "sex" and exc.value.options == []


# --------------------------------------------------------------------------
# Eurostat bulk TSV, and the formats with years in columns
# --------------------------------------------------------------------------


def test_eurostat_tsv_keys_flags_missing_and_age_codes(tmp_path, truth):
    by_year = S.hmd_tables((2020, 2021))
    tsv = S.write_eurostat_tsv(tmp_path / "demo_magec.tsv", "Dx", by_year,
                               missing={("F", "Y5", 2020)}, provisional_year=2021)
    expo = S.write_hmd(tmp_path / "E.txt", "Ex", by_year)
    info = L.inspect_file(tsv)
    assert info.origin == "Eurostat" and info.years == [2020, 2021]
    assert info.segments == {"geo": ["FR", "DE"]} and info.sexes == ["Female", "Male"]    # F / M; T ignored

    with pytest.raises(L.AmbiguousSelectionError) as exc:
        L.load_separate(deaths=tsv, exposure=expo, year=2020)
    assert (exc.value.column, exc.value.options) == ("geo", ["FR", "DE"])

    df = L.load_separate(deaths=L.Source(tsv, where={"geo": "FR"}), exposure=expo, year=2020)
    male = rows(df, "Male")
    np.testing.assert_allclose(male["Dx"], truth[("Male", 2020)]["Dx"].iloc[:100])        # Y_LT1 = age 0, Y_OPEN left out
    assert male["age"].max() == 99
    assert item(df, "Female", "missing_values").ages == [5, 100]    # ':' is missing, not zero; 100 = the open group
    assert item(df, "Male", "missing_values").ages == [100]

    prov = L.load_separate(deaths=L.Source(tsv, where={"geo": "DE"}), exposure=expo, year=2021)   # '2772.00 p'
    np.testing.assert_allclose(rows(prov, "Female")["Dx"], truth[("Female", 2021)]["Dx"].iloc[:100] + 7)


def test_years_as_columns_with_title_lines_and_french_age_labels(tmp_path, truth):
    by_year = S.hmd_tables((2020, 2021))
    d = S.write_wide_years_xlsx(tmp_path / "deces_femmes.xlsx", "Dx", by_year, "Female")
    e = S.write_wide_years_xlsx(tmp_path / "expo_femmes.xlsx", "Ex", by_year, "Female")
    info = L.inspect_file(d)
    assert info.years == [2020, 2021] and info.sexes == []                  # years found; the sheet names no sex

    df = L.load_separate(deaths=d, exposure=e, sex="Female", year=2021)
    t = truth[("Female", 2021)]
    np.testing.assert_array_equal(df["age"], np.arange(100))                 # '100 ans ou plus' is an open group
    np.testing.assert_allclose(df["Dx"], t["Dx"].iloc[:100])
    np.testing.assert_allclose(df["Ex"], t["Ex"].iloc[:100], atol=0.01)
    with pytest.raises(L.ColumnMappingError, match="n'indique pas le sexe"):
        L.load_separate(deaths=d, exposure=e, year=2021)


def test_the_value_column_of_a_long_file_takes_the_quantity_of_its_input(tmp_path, truth):
    """Eurostat's CSV: TIME_PERIOD and OBS_VALUE, whose quantity is the file's role, not its name."""
    by_year = S.hmd_tables((2020,))
    lines = ["DATAFLOW,freq,unit,sex,age,geo,TIME_PERIOD,OBS_VALUE,OBS_FLAG"]
    for code, label in (("F", "Female"), ("M", "Male")):
        t = by_year[2020][label]
        for i in range(101):
            age = "Y_LT1" if i == 0 else ("Y_OPEN" if i == 100 else f"Y{i}")
            lines.append(f"ESTAT:DEMO_MAGEC(1.0),A,NR,{code},{age},FR,2020,{int(t['Dx'].iloc[i])},")
    p = tmp_path / "sdmx.csv"
    p.write_text("\n".join(lines) + "\n")
    expo = S.write_hmd(tmp_path / "E.txt", "Ex", by_year)
    df = L.load_separate(deaths=p, exposure=expo)
    np.testing.assert_array_equal(rows(df, "Male")["Dx"], truth[("Male", 2020)]["Dx"].iloc[:100])
    with pytest.raises(L.ColumnMappingError, match="sans grandeur") as exc:          # as a lone file: which quantity?
        L.load_data(p, sex="Female")
    assert exc.value.missing == ["quantity"]


def test_a_recognised_value_column_wins_over_a_lone_unknown_column(tmp_path, truth):
    """Year, Sex, Age, Valeur, Remarque: 'Remarque' is the only column left over, yet it is the value
    column that holds the deaths."""
    by_year = S.hmd_tables((2020,))
    lines = ["Year;Sex;Age;Valeur;Remarque"]
    for label in BOTH:
        t = by_year[2020][label]
        lines += [f"2020;{label};{a};{d};ok" for a, d in zip(t["age"], t["Dx"])]
    p = tmp_path / "valeur.csv"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    df = L.load_separate(deaths=p, exposure=S.write_hmd(tmp_path / "E.txt", "Ex", by_year))
    np.testing.assert_array_equal(rows(df, "Female")["Dx"], truth[("Female", 2020)]["Dx"])


# --------------------------------------------------------------------------
# Files with titles, notes and odd cells
# --------------------------------------------------------------------------


def test_the_header_row_is_found_below_title_lines(tmp_path, truth):
    t = truth[("Female", 2020)]
    body = "\n".join(f"{a};{d};{e}" for a, d, e in zip(t["age"], t["Dx"], t["Ex"]))
    p = tmp_path / "titled.csv"
    p.write_text("Statistiques de mortalité 2020\nSource : test\n\nÂge;Décès;Exposition\n" + body + "\n", encoding="utf-8")
    df = L.load_data(p, sex="Female")                                                 # header=None: found
    np.testing.assert_array_equal(df["Dx"], t["Dx"])
    pd.testing.assert_frame_equal(df, L.load_data(L.Source(p, header=2), sex="Female"))   # or told (non-blank line no. 2)


def test_footnotes_under_the_data_labels_and_suppressed_cells(tmp_path, truth):
    p = S.write_tab_with_footnotes(tmp_path / "export.tsv", S.hmd_tables((2020,)))
    df = L.load_data(p, year=2020)                       # 'Gender', 'Deaths', 'Population': read as sex, Dx, exposure
    np.testing.assert_array_equal(rows(df, "Female")["age"], np.arange(100))           # '< 1 year' = 0, '100+ years' left out
    np.testing.assert_array_equal(rows(df, "Male")["age"], np.delete(np.arange(100), 3))   # age 3 is 'Suppressed': missing
    np.testing.assert_array_equal(rows(df, "Female")["Dx"], truth[("Female", 2020)]["Dx"].iloc[:100])
    np.testing.assert_allclose(rows(df, "Female")["Ex"], truth[("Female", 2020)]["Ex"].iloc[:100], atol=0.5)
    assert item(df, "Male", "missing_values").ages == [3]                              # 'Suppressed' is missing, not zero
    assert any("mal formée" in i.message for i in df.attrs["notes"])                   # the notes were set aside, counted
    assert note(df, "population_as_exposure")                                          # and the reading of Population said


def test_age_bands_are_refused_with_a_clear_message(tmp_path):
    p = tmp_path / "bands.csv"
    p.write_text("Age,Deaths,Exposure\n0-4,10,1000\n5-9,12,1000\n10-14,9,1000\n")
    with pytest.raises(L.LoaderError, match="tranches"):
        L.load_data(p, sex="Female")


# --------------------------------------------------------------------------
# Population at 1 January -> exposure (the one computed quantity)
# --------------------------------------------------------------------------


def _january_population(by_year, growth=0.1):
    """P(2021) = (1 + growth) * P(2020): population at 1 January, same layout as an exposure file."""
    return {y: {s: t.assign(Ex=t["Ex"] * (1 + growth * (y - 2020))) for s, t in d.items()}
            for y, d in by_year.items()}


def test_january_population_becomes_exposure_by_the_trapezoid_rule(tmp_path, truth):
    by_year = S.hmd_tables((2020, 2021))
    deaths = S.write_hmd(tmp_path / "D.txt", "Dx", by_year)
    pop = S.write_hmd(tmp_path / "P.txt", "Ex", _january_population(by_year))

    df = L.load_separate(deaths=deaths, exposure=L.Source(pop, exposure_basis="jan1_mean"), year=2020)
    for s in BOTH:
        np.testing.assert_allclose(rows(df, s)["Ex"], truth[(s, 2020)]["Ex"] * 1.05, rtol=0, atol=0.01)
        np.testing.assert_array_equal(rows(df, s)["Dx"], truth[(s, 2020)]["Dx"])
    assert "1er janvier 2020 et 2021" in note(df, "exposure_convention")[0].message

    plain = L.load_separate(deaths=deaths, exposure=pop, year=2020)                   # default: taken as it is
    np.testing.assert_allclose(rows(plain, "Male")["Ex"], truth[("Male", 2020)]["Ex"])
    assert not note(plain, "exposure_convention")


def test_january_population_needs_the_next_year_and_a_dated_period_table(tmp_path):
    by_year = S.hmd_tables((2020, 2021))
    deaths = S.write_hmd(tmp_path / "D.txt", "Dx", by_year)
    pop = S.write_hmd(tmp_path / "P.txt", "Ex", _january_population(by_year))
    with pytest.raises(L.LoaderError, match="2022"):                                  # no 1 January 2022 in the file
        L.load_separate(deaths=deaths, exposure=L.Source(pop, exposure_basis="jan1_mean"), year=2021)
    with pytest.raises(L.LoaderError, match="exposure_basis inconnu"):
        L.load_separate(deaths=deaths, exposure=L.Source(pop, exposure_basis="mid"), year=2020)
    cohorts = S.hmd_tables((1950, 1951))
    cd = S.write_hmd(tmp_path / "cD.txt", "Dx", cohorts, key_name="Cohort")
    cp = S.write_hmd(tmp_path / "cP.txt", "Ex", cohorts, key_name="Cohort")
    with pytest.raises(L.LoaderError, match="table du moment"):
        L.load_separate(deaths=cd, exposure=L.Source(cp, exposure_basis="jan1_mean"), cohort=1950)


def test_option_c_file_with_deaths_and_a_january_population_column(tmp_path, truth):
    parts = []
    for y in (2020, 2021):
        for s in BOTH:
            t = truth[(s, y)]
            parts.append(pd.DataFrame({"Year": y, "Sex": s, "Age": t["age"], "Deaths": t["Dx"],
                                       "Population": truth[(s, 2020)]["Ex"] * (1 + 0.1 * (y - 2020))}))
    p = tmp_path / "pop.csv"
    pd.concat(parts).to_csv(p, index=False)
    df = L.load_data(L.Source(p, exposure_basis="jan1_mean"), year=2020)
    for s in BOTH:
        np.testing.assert_allclose(rows(df, s)["Ex"], truth[(s, 2020)]["Ex"] * 1.05)
        np.testing.assert_array_equal(rows(df, s)["Dx"], truth[(s, 2020)]["Dx"])
    central = L.load_data(p, year=2020)
    assert note(central, "population_as_exposure") and not note(central, "exposure_convention")


# --------------------------------------------------------------------------
# The vocabulary: ages and numbers
# --------------------------------------------------------------------------


@pytest.mark.parametrize("label,want", [
    ("85", ("age", 85)), ("85.0", ("age", 85)), ("85 ans", ("age", 85)), ("0 an", ("age", 0)),
    ("1 year", ("age", 1)), ("Y85", ("age", 85)), ("Y_LT1", ("age", 0)), ("< 1 year", ("age", 0)),
    ("Moins de 1 an", ("age", 0)), ("D12", ("age", 12)), ("Âge 12", ("age", 12)),
    ("100+", ("open", None)), ("110+", ("open", None)), ("100 ans ou plus", ("open", None)),
    ("105_PL", ("open", None)), ("Y_OPEN", ("open", None)), ("Y_GE100", ("open", None)),
    ("100+ years", ("open", None)),
    ("TOT", ("total", None)), ("Ensemble", ("total", None)), ("Total", ("total", None)),
    ("0-4", ("group", None)), ("Y5-9", ("group", None)), ("1-4", ("group", None)),
    ("2.5", ("fraction", None)), ("", ("blank", None)), ("???", ("other", None)), ("UNK", ("other", None)),
])
def test_age_labels(label, want):
    assert age_from_label(label) == want


def test_numbers_flags_and_missing_markers():
    out = to_numeric(pd.Series(["1 234,5", "98.5 p", "12 (p)", ":", "x", "Suppressed", "1.234,5", "7 ", "", "-",
                                "Not Applicable", "5e3"]), "t")
    assert list(out.iloc[:3]) == [1234.5, 98.5, 12.0]
    assert out.iloc[3:6].isna().all() and out.iloc[6] == 1234.5 and out.iloc[7] == 7.0
    assert out.iloc[8:11].isna().all() and out.iloc[11] == 5000.0
    with pytest.raises(L.LoaderError, match="non numérique"):
        to_numeric(pd.Series(["abc", "12"]), "t")


def test_parse_ages_open_from_and_bands():
    parsed = parse_ages(pd.Series(["0", "1", "99", "100", "TOT"]), "t", open_from=100)
    assert list(parsed.ages.dropna()) == [0.0, 1.0, 99.0] and parsed.excluded == ["TOT", "100+"]
    assert parse_ages(pd.Series(["0-4", "5-9"]), "t").n_group == 2
