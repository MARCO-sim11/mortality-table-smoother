"""Home page: which tables to smooth. The choice is kept in `view_choice` and the loading page opens."""
from __future__ import annotations

import streamlit as st

from ui import theme
from ui.state import reset_downstream


def choose(view: str) -> None:
    """Remember the choice and go to the loading page. A different choice than before drops what was loaded: tables of
    another kind would not match the estimations made on the old ones."""
    if view != st.session_state.view_choice:
        reset_downstream()
        st.session_state.update(df_raw=None, load_error=None)
    st.session_state.update(view_choice=view, view_chosen=True)
    st.switch_page("views/chargement.py")


theme.hero()

st.subheader("Quelles tables voulez-vous lisser ?")
left, right = st.columns(2)
with left, st.container(border=True):
    theme.choice_card("period", "Tables du moment",
                      "La mortalité d'une année civile, âge par âge : par exemple la France en 2020. "
                      "C'est le cas courant, celui de tout ce que l'outil sait faire aujourd'hui.")
    if st.button("Lisser des tables du moment", key="home_period", type="primary", width="stretch",
                 icon=":material/calendar_month:"):
        choose("period")
with right, st.container(border=True):
    theme.choice_card("cohort", "Tables générationnelles",
                      "La mortalité d'une même génération suivie d'âge en âge : par exemple les personnes nées en 1950. "
                      "Même chaîne de lissage, appliquée aux taux d'une génération.", badge="extension")
    if st.button("Lisser des tables générationnelles", key="home_cohort", width="stretch", icon=":material/timeline:"):
        choose("cohort")

st.subheader("Quatre étapes")
theme.steps_strip()
