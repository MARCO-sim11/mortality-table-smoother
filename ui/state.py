"""Session state: the keys of spec 7.2 (per-sex shapes, see app.py) and what the pages remember between visits."""
from __future__ import annotations

import copy

import streamlit as st

DEFAULTS = {
    "df_raw": None,            # pd.DataFrame      loaded data, both sexes stacked (loader)
    "config": {},              # {sex: SmootherConfig}  smoothing settings (config)
    "fit_results": {},         # {sex: {method_key: FitResult}}
    "mu_joint": None,          # joined smoothed rates (joiner)
    "life_table": None,        # final life table
    "selected_z1": None,       # method chosen for zone I
    "selected_z2": None,
    "selected_z3": None,
    "fit_status": [],          # additions: see module docstring
    "load_error": None,
    "fit_config": {},          # {sex: SmootherConfig} the settings of the last estimation
    "validated_selection": None,   # {sex: {zone: method_key}} behind mu_joint / life_table
    "view_choice": "period",   # 'period' | 'cohort': what the home page asked for, read by the loading page
    "view_chosen": False,      # True once the home page was used (until then the sidebar of the home page has no table type to show)
    "ui": {},                  # what the pages remember between visits (a widget not drawn in a run loses its state)
    "export_params": {},       # {sex: {cols, sep, decimal, multi}} the export options of tab 4, one set per sex
}


def init_state() -> None:
    """Give every session key its default (called at each run by the entry script, a no-op once set).
    Each session gets its own copy of the mutable defaults: DEFAULTS lives at module level, shared by every session."""
    for key, default in DEFAULTS.items():
        st.session_state.setdefault(key, copy.deepcopy(default))


def reset_downstream() -> None:
    """A new load invalidates everything computed from the previous one."""
    st.session_state.update(config={}, fit_results={}, fit_status=[], fit_config={}, mu_joint=None,
                            life_table=None, selected_z1=None, selected_z2=None, selected_z3=None,
                            validated_selection=None)


def remembered(name: str, default):
    """What a page stored under `name` on an earlier visit, or `default`."""
    return st.session_state.ui.get(name, default)


def remember(name: str, value):
    """Store `value` for the next visit and hand it back (`x = remember("k", widget(...))`)."""
    st.session_state.ui[name] = value
    return value
