"""Page 1 - loading and checking the data (spec 1)."""
from __future__ import annotations

import hashlib

import streamlit as st

from src import loader as L
from ui import theme
from ui.common import ACCEPTED_TYPES, MODE_A, MODE_B, MODE_C, MODES, SEX_LABEL
from ui.loading import file_block, key_choices, segment_choices, sex_choices, tab1_result, view_choices
from ui.state import remember, remembered, reset_downstream

theme.page_header("Chargement et contrôle des données", 1,
                  "Déposez vos fichiers, contrôlez leur qualité, puis chargez la table.")
wanted = st.session_state.view_choice
kind_col, change_col = st.columns([2, 1], vertical_alignment="center")
kind_col.markdown(f"Type de table : **{theme.view_phrase(wanted)}**")
change_col.page_link("views/accueil.py", label="Changer de type", icon=":material/swap_horiz:")

mode = st.segmented_control("Disposition des fichiers", MODES, default=remembered("load_mode", MODE_C),
                            required=True, key="load_mode")
remember("load_mode", mode)
setups: dict = {}                                   # input name -> FileSetup

if mode == MODE_C:
    st.caption("**Recommandé.** Un seul fichier, une ligne par cellule : âge, sexe, année, décès, exposition. "
               "Le chargeur sélectionne l'année (ou la génération) et le sexe, contrôle les données et "
               "calcule mₓ = Dₓ / Eₓ. Formats : export de base de données, INSEE (T74BIS...), Excel...")
    up_c = st.file_uploader("Fichier de données élémentaires", type=ACCEPTED_TYPES, key="up_elementary")
    setups["elementary"] = file_block("elementary", up_c, "sexfile")
elif mode == MODE_A:
    st.caption("Un fichier de taux (ou de décès) et un fichier d'exposition, âges en lignes et sexes en colonnes "
               "(sorties HMD, tableaux actuariels, Excel). Les taux seuls ne redonnent pas les décès exacts : "
               "il faut l'exposition et sa convention (Dx = mx × Ex est alors reconstruit). "
               "Fichiers HMD : Deaths_1x1, Exposures_1x1, Mx_1x1 (et c… pour les générations).")
    col_d, col_e, col_r = st.columns(3)
    up_d = col_d.file_uploader("Deaths (Dx)", type=ACCEPTED_TYPES, key="up_deaths")
    up_e = col_e.file_uploader("Exposure (Ex)", type=ACCEPTED_TYPES, key="up_exposure")
    up_r = col_r.file_uploader("Rates (mx)", type=ACCEPTED_TYPES, key="up_rates")
    with col_d:
        setups["deaths"] = file_block("deaths", up_d, "Dx")
    with col_e:
        setups["exposure"] = file_block("exposure", up_e, "Ex")
    with col_r:
        setups["rates"] = file_block("rates", up_r, "mx")
else:
    st.caption("Un fichier par sexe, chacun avec ses décès, ses expositions ou ses taux par âge. Simple, "
               "mais il faut deux fichiers de même année (ou génération) et de même structure.")
    col_f, col_m = st.columns(2)
    up_f = col_f.file_uploader("Fichier Féminin", type=ACCEPTED_TYPES, key="up_female")
    up_m = col_m.file_uploader("Fichier Masculin", type=ACCEPTED_TYPES, key="up_male")
    with col_f:
        setups["female"] = file_block("female", up_f, "sexfile")
    with col_m:
        setups["male"] = file_block("male", up_m, "sexfile")

given = [s for s in setups.values() if s]
infos = [s.info for s in given if s.info]
sig = hashlib.md5("|".join(f"{n}:{getattr(s.uploaded, 'file_id', s.name)}"
                           for n, s in setups.items() if s).encode()).hexdigest()[:8]
ready = bool(given) and all(s.error is None for s in given)
mode_key = {MODE_C: "C", MODE_A: "A", MODE_B: "B"}[mode]

if given:
    with st.container(border=True):
        sex_cols = st.columns(3)
        with sex_cols[0]:
            options, sex_note = sex_choices(mode, setups)
            if not options:
                sexes_choice = None
                ready = False
            elif len(options) > 1:
                sexes_choice = st.segmented_control(
                    "Sexe(s) à traiter", options, default=options[0], required=True,
                    format_func=SEX_LABEL.get, key=f"sexes_{mode_key}_{sig}")
                if sex_note:
                    st.caption(sex_note)
            else:
                sexes_choice = options[0]
                st.caption(f"Sexe : {SEX_LABEL[sexes_choice].lower()}")
        with sex_cols[1]:
            views = view_choices(infos)                   # the kind of table was chosen on the home page
            if not views:
                st.error("Les fichiers ne sont pas dans la même vision (année / génération).")
                view, ready = "auto", False
            elif views == ["auto"]:
                view = "auto"
                st.caption(f"Vision : {theme.view_phrase(wanted)} (les fichiers ne donnent ni année ni génération)")
            elif wanted in views:
                view = wanted
                st.caption(f"Vision : {theme.view_phrase(wanted)}")
            else:
                view, ready = "auto", False
                st.error(f"Vous avez choisi les {theme.view_phrase(wanted)}, mais ces fichiers ne contiennent que des "
                         f"{theme.view_phrase(views[0])}. Déposez d'autres fichiers ou changez de type de table.")
        with sex_cols[2]:
            keys = key_choices(infos, view)
            year = cohort = None
            if len(keys) > 1:
                chosen = st.selectbox("Année" if view == "period" else "Génération", keys,
                                      index=len(keys) - 1, key=f"key_{view}_{sig}")
                year, cohort = (chosen, None) if view == "period" else (None, chosen)
            elif len(keys) == 1:
                st.caption(f"{'Année' if view == 'period' else 'Génération'} : {keys[0]} (seule présente)")
            elif view != "auto" and any(i.years if view == "period" else i.cohorts for i in infos):
                st.error("Les fichiers n'ont aucune valeur commune.")
                ready = False

        extra = st.columns(3)
        with extra[0]:                                          # segments: INSEE CHAMP2, Eurostat geo...
            where_map = {}
            for col, values in segment_choices(infos).items():
                if not values:
                    st.error(f"Aucune valeur commune de [{col}] entre les fichiers.")
                    ready = False
                    continue
                where_map[col] = st.selectbox(f"Segment [{col}]", values, key=f"seg_{col}_{sig}",
                                              help="Ce fichier contient plusieurs sous-populations "
                                                   "(champ géographique, pays...) : choisissez-en une.")
            for setup in given:
                if setup.info:
                    setup.where = {c: v for c, v in where_map.items() if c in setup.info.columns}
        with extra[1]:                                          # population at 1 January -> exposure
            uses_pop = mode != MODE_A or setups.get("exposure")
            if uses_pop and view != "cohort":
                jan1 = st.checkbox("Exposition = population au 1er janvier", key=f"jan1_{sig}",
                                   help="Coché : l'exposition de l'année t est la moyenne des populations au "
                                        "1er janvier de t et t+1 (règle du trapèze, approximation de "
                                        "l'exposition centrale). Décoché : la colonne est l'exposition.")
                for name, setup in setups.items():
                    if setup and (mode != MODE_A or name == "exposure"):
                        setup.exposure_basis = "jan1_mean" if jan1 else "central"
        with extra[2]:
            rates_only = (mode == MODE_A and setups["rates"] and not setups["deaths"]
                          and not setups["exposure"]) or (
                mode != MODE_A and any(
                    s.info and ((s.info.layout == "roles" and "mx" in s.roles
                                 and "dx" not in s.roles and "ex" not in s.roles)
                                or (s.info.layout == "sex_columns" and s.info.quantity == "mx"))
                    for s in given))
            if rates_only:
                E_ref = st.number_input("Exposition de référence E_ref", min_value=1.0,
                                        value=100_000.0, step=10_000.0, format="%.0f",
                                        key=f"eref_{sig}",
                                        help="Seul mx est disponible : Ex = E_ref pour tous les âges, "
                                             "Dx = mx × E_ref. Les tests statistiques en dépendent.")
            else:
                E_ref = None
        clicked = st.button("Charger", type="primary", disabled=not ready, key="btn_load",
                            icon=":material/upload_file:")
else:
    clicked = False
    st.info("Déposez un ou plusieurs fichiers (.txt, .csv, .tsv, .xlsx, .xls) pour commencer.",
            icon=":material/upload_file:")

if clicked:
    reset_downstream()
    common = dict(view=view, year=year, cohort=cohort, E_ref=E_ref)
    try:
        if mode == MODE_C:
            df = L.load_data(setups["elementary"].source, sex=sexes_choice, **common)
        elif mode == MODE_A:
            src = lambda name: setups[name].source if setups[name] else None
            df = L.load_separate(deaths=src("deaths"), exposure=src("exposure"), rates=src("rates"),
                                 sex=sexes_choice, **common)
        else:
            take = lambda s: setups[s.lower()].source if (
                setups[s.lower()] and sexes_choice in ("both", s)) else None
            df = L.load_by_sex(female=take("Female"), male=take("Male"), **common)
        st.session_state.df_raw = df
        st.session_state.load_error = None
    except L.LoaderError as exc:
        st.session_state.df_raw = None
        st.session_state.load_error = exc

error = st.session_state.load_error
if error is not None:
    st.error(str(error), icon=":material/error:")
    if isinstance(error, L.ColumnMappingError) and error.available:
        st.caption(f"Colonnes trouvées : {', '.join(error.available)}. "
                   "Ouvrez « Paramètres » du fichier pour les associer.")
    if isinstance(error, L.DataQualityError) and error.data is not None:
        tab1_result(error.data, downloadable=False)
elif st.session_state.df_raw is not None:
    tab1_result(st.session_state.df_raw)

