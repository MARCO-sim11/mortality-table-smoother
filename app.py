"""app.py - Module 7: the Streamlit entry point.

    python -m streamlit run app.py        # from the project root, mortality_smoother/

A home page asks which tables to smooth, then four pages follow; a marine sidebar, on all five, links them:

    accueil      choice of the tables: 'period' (tables du moment) or 'cohort' (generational)      -> session_state.view_choice
    chargement   upload + 'Charger'                  -> loader       -> session_state.df_raw                    (spec 1)
    parametrage  sliders / checkboxes, 'Lancer'      -> config, zone modules -> config, fit_results, fit_config (spec 2, 3)
    evaluation   table of criteria, menus, 'Valider' -> evaluator, joiner -> selected_z1..3, mu_joint, life_table (spec 5, 4)
    export       chart, preview, downloads           -> exporter     -> Excel, CSV, HMD-style text               (spec 6)

The pages are the files of `views/`, the pieces they share are in `ui/` (constants, session state, helpers, the Risk Data Lab look).
The file name of a page is its URL path, which is also how the tests reach it (`AppTest.switch_page("views/chargement.py")`).

Execution order is that of spec 7.3. Both sexes mean two datasets to smooth, each with its own boundaries and methods: against
the single-dataset shapes of spec 7.2, `config` is {sex: SmootherConfig} and `fit_results` is {sex: {method_key: FitResult}}; the
settings page switches between the sexes. More keys: `fit_status` (one row per sex and method of the last run), `load_error`,
`fit_config` {sex: SmootherConfig} (the settings each sex was ESTIMATED with: the sliders may move afterwards, but the evaluation and
the join read the zones with the bounds the fits were made with), `validated_selection` {sex: {zone: method_key}} (what `mu_joint` and
`life_table` were built from), `export_params` {sex: {cols, sep, decimal, multi}}, `view_choice`, and `ui` (what a page remembers
between visits: a widget that is not drawn in a run loses its state, and a page is not drawn while another is shown).

Three additions to the specs, asked for afterwards:
  * a GLOBAL spline over every age (the settings page ticks it, `config.GLOBAL_METHODS`): fitted once on 0..omega3, offered in the
    menu of each of the three zones on the evaluation page, evaluated zone by zone (`config.zone_view`);
  * omega2 can never be set below omega1 + 1;
  * a "lier les deux sexes" checkbox on the settings, evaluation and export pages, independent in each: while ticked, what is set for
    the sex on screen (the configuration, the method picked in each zone, the export options) is copied to the other sex at every run.

Zone modules are called as `fit_<key>(ages, Dx, Ex, cfg)` from `src/<module>.py` (config.METHODS and config.GLOBAL_METHODS), with the
rows of one sex and one zone (all the ages up to omega3 for a global method). A module or function that does not exist yet is
reported as "non implémenté", never as a crash.
"""
from __future__ import annotations

import streamlit as st

from ui import theme
from ui.state import init_state

st.set_page_config(page_title="Mortality Table Smoother", page_icon=":material/show_chart:", layout="wide")
init_state()
theme.inject_css()

home = st.Page("views/accueil.py", title="Accueil", icon=":material/home:", default=True)
chargement = st.Page("views/chargement.py", title="Chargement", icon=":material/upload_file:")
parametrage = st.Page("views/parametrage.py", title="Paramétrage", icon=":material/tune:")
evaluation = st.Page("views/evaluation.py", title="Évaluation", icon=":material/query_stats:")
export = st.Page("views/export.py", title="Export", icon=":material/download:")

current = st.navigation([home, chargement, parametrage, evaluation, export], position="hidden")
current.run()

ss = st.session_state                                # the sidebar is drawn after the page, so that its ticks show what the page just did
theme.render_sidebar([
    (chargement, "1 · Chargement", ss.df_raw is not None),
    (parametrage, "2 · Paramétrage", bool(ss.fit_results)),
    (evaluation, "3 · Évaluation", bool(ss.validated_selection)),
    (export, "4 · Export", False),
], home, show_choice=current.title != home.title or ss.view_chosen)   # on the home page, a table type only once one was chosen
