"""Page 2 - settings and launching the estimations (spec 2, 3)."""
from __future__ import annotations

import copy

import pandas as pd
import streamlit as st

from src import config as C
from ui import theme
from ui.common import FIT_STATUS_COLUMNS, SEX_BY_LABEL, SEX_LABEL
from ui.fitting import run_estimations
from ui.state import remember, remembered

header = st.empty()                                    # the banner is drawn last: its curve follows the sliders
banner = (20, 85, 110)
df = st.session_state.df_raw
if df is None:
    st.info("Chargez d'abord des données (page « Chargement »).", icon=":material/upload_file:")
    st.page_link("views/chargement.py", label="Aller au chargement", icon=":material/upload_file:")
else:
    sexes = df.attrs["sexes"]
    data = {s: df[df["sex"] == s] for s in sexes}
    if len(sexes) > 1:
        last = remembered("sex_cfg", sexes[0])
        picked = st.segmented_control("Jeu de données à paramétrer", [SEX_LABEL[s] for s in sexes],
                                      default=SEX_LABEL[last if last in sexes else sexes[0]], required=True, key="active_sex")
        active = remember("sex_cfg", SEX_BY_LABEL[picked])
        link_cfg = st.checkbox(
            "Lier le féminin et le masculin", value=remembered("link_cfg", False), key="link_cfg",
            help="Coché, la configuration affichée est copiée tout de suite sur l'autre sexe, et chaque réglage fait "
                 "ensuite sur l'un est reporté sur l'autre. Pour copier le féminin vers le masculin, cochez la case "
                 "alors que le féminin est affiché.")
        remember("link_cfg", link_cfg)
        st.caption("Les deux sexes sont liés : tout réglage fait ici est reporté tel quel sur l'autre sexe."
                   if link_cfg else
                   "Chaque sexe a sa propre configuration (frontières, méthodes, jonction) ; "
                   "« Lancer les estimations » traite les deux.")
    else:
        active, link_cfg = sexes[0], False
    stored = st.session_state.config
    ages_of = {s: data[s]["age"].to_numpy() for s in sexes}
    d, ages = data[active], ages_of[active]

    if len(set(ages)) < 3:
        st.error(f"Moins de 3 âges chargés ({SEX_LABEL[active].lower()}) : impossible de définir des zones.")
    else:
        lo, hi = int(ages.min()), int(ages.max())
        cfg0 = stored.get(active) or C.default_config(ages)     # what this sex had at its last visit
        dsig = f"{active}_{lo}_{hi}"                            # new sex / age range -> fresh widgets

        with st.container(border=True):
            st.subheader("Frontières de zones")
            c1, c2 = st.columns(2)
            omega1 = c1.slider("ω₁ — frontière Zone I / Zone II", lo, hi - 1, min(max(cfg0.omega1, lo), hi - 1),
                               key=f"omega1_{dsig}")
            floor2 = omega1 + 1                                     # omega2 starts at omega1 + 1 and follows omega1
            omega2 = c2.slider("ω₂ — frontière Zone II / Zone III", floor2, hi, min(max(cfg0.omega2, floor2), hi),
                               key=f"omega2_{dsig}", help="Commence à ω₁ + 1 : déplacer ω₁ au-delà de ω₂ pousse ω₂.")

        with st.container(border=True):
            st.subheader("Méthodes à tester")
            ticked = {}
            cols = st.columns(3)
            bounds = {"zone1": f"{lo} – {omega1}", "zone2": f"{omega1} – {omega2}",
                      "zone3": f"{omega2} – ω₃"}
            for col, zone in zip(cols, C.ZONES):
                with col:
                    st.markdown(f"**{C.ZONE_LABELS[zone]}** ({bounds[zone]})")
                    ticked[zone] = [m.key for m in C.methods_of(zone)
                                    if st.checkbox(m.label, value=m.key in cfg0.methods_in(zone),
                                                   key=f"use_{active}_{m.key}")]
            st.markdown(f"**{C.ZONE_LABELS[C.ZONE_ALL]}** ({lo} – ω₃)")
            ticked_all = [m.key for m in C.methods_of(C.ZONE_ALL)
                          if st.checkbox(m.label, value=m.key in cfg0.all_methods, key=f"use_{active}_{m.key}",
                                         help="Une seule spline naturelle (un nœud tous les 5 âges) ajustée sur tous les "
                                              "âges, du premier à ω₃. Elle est proposée dans le menu de chacune des trois "
                                              "zones (page Évaluation), évaluée zone par zone ; choisie dans les trois, la table "
                                              "finale est cette seule courbe.")]

        with st.container(border=True):
            st.subheader("Jonction")
            sigma = st.slider("σ de jonction (tanh)", C.SIGMA_JOIN_RANGE[0], C.SIGMA_JOIN_RANGE[1],
                              cfg0.sigma_join, 0.5, format="%.1f", key=f"sigma_join_{active}",
                              help="Demi-largeur de la fenêtre de raccord entre zones : "
                                   "transition appliquée sur [ω − 2σ, ω + 2σ].")
            with st.expander("Options avancées"):
                a1, a2, a3, a4 = st.columns(4)
                omega3 = int(a1.number_input("ω₃ — âge limite", min_value=lo + 2, max_value=150,
                                             value=cfg0.omega3, step=1, key=f"omega3_{dsig}"))
                nr_maxiter = int(a2.number_input("Itérations max Newton-Raphson", min_value=1,
                                                 max_value=10_000, value=cfg0.nr_maxiter, step=10,
                                                 key=f"nr_maxiter_{active}"))
                nr_tol = a3.select_slider("Critère d'arrêt NR", [1e-4, 1e-6, 1e-8, 1e-10, 1e-12],
                                          value=cfg0.nr_tol, format_func=lambda x: f"{x:.0e}",
                                          key=f"nr_tol_{active}")
                grid_n = int(a4.number_input("Grille Kannisto (points par axe)", min_value=2,
                                             max_value=500, value=cfg0.grid_n, step=10,
                                             key=f"grid_n_{active}"))

        banner = (omega1, omega2, omega3)
        stored[active] = C.SmootherConfig(
            omega1=omega1, omega2=omega2, omega3=omega3, z1_methods=ticked["zone1"],
            z2_methods=ticked["zone2"], z3_methods=ticked["zone3"], sigma_join=sigma,
            nr_maxiter=nr_maxiter, nr_tol=nr_tol, grid_n=grid_n, all_methods=ticked_all)   # spec 7.3, step 2
        if link_cfg:                                                     # the sex on screen sets the other one
            for other in sexes:
                if other != active and len(set(ages_of[other])) >= 3:
                    stored[other] = C.adapt_config(stored[active], ages_of[other])
        issues = C.validate_config(stored[active], ages)
        for message in issues.errors:
            st.error(message, icon=":material/error:")
        for message in issues.warnings:
            st.warning(message, icon=":material/warning:")
        if issues.ok:
            sizes = [f"{C.ZONE_LABELS[z]} : {len(C.zone_slice(d, stored[active], z))} âges"
                     for z in C.ZONES]
            st.caption("Âges observés par zone (bornes incluses) — " + " · ".join(sizes))

    # every loaded sex needs a valid configuration to launch (one never opened gets the defaults)
    cfgs = {s: stored.get(s) or C.default_config(ages_of[s]) for s in sexes}
    problems = {s: C.validate_config(cfgs[s], ages_of[s]) for s in sexes}
    for s in sexes:
        if s != active and not problems[s].ok:
            st.error(f"{SEX_LABEL[s]} : configuration invalide ({problems[s].errors[0]}) — "
                     "passez sur ce jeu de données pour la corriger.", icon=":material/error:")
    launchable = all(p.ok for p in problems.values()) and len(set(ages)) >= 3

    launch = st.button("Lancer les estimations", type="primary", disabled=not launchable,
                       key="btn_run", icon=":material/play_arrow:")
    if launch:
        bar = st.progress(0.0, text="Préparation…")
        results, status = {}, []
        for i, s in enumerate(sexes):
            results[s], rows = run_estimations(
                data[s], cfgs[s],
                progress=lambda frac, text, i=i, s=s: bar.progress(
                    (i + frac) / len(sexes), text=f"{SEX_LABEL[s]} · {text}"))
            status += [{"sex": SEX_LABEL[s], **r} for r in rows]
        bar.empty()
        st.session_state.update(fit_results=results, fit_status=status,
                                fit_config={s: copy.deepcopy(cfgs[s]) for s in sexes},
                                mu_joint=None, life_table=None, selected_z1=None, selected_z2=None,
                                selected_z3=None, validated_selection=None)

    status = st.session_state.fit_status
    if status:
        n_ok = sum(r["status"] == "ok" for r in status)
        st.markdown(f"**Estimations** : {n_ok} méthode(s) estimée(s) sur {len(status)}.")
        table = pd.DataFrame(status)
        table["converged"] = table["converged"].map({True: "oui", False: "non"}).fillna("—")
        table["loglik"] = table["loglik"].map(lambda v: "—" if pd.isna(v) else f"{v:,.2f}")
        table["n_iter"] = table["n_iter"].map(lambda v: "—" if pd.isna(v) else str(int(v)))
        st.dataframe(table[list(FIT_STATUS_COLUMNS)].rename(columns=FIT_STATUS_COLUMNS),
                     hide_index=True)
        if n_ok < len(status):
            st.caption("« non implémenté » : le module de lissage correspondant n'est pas encore "
                       "écrit (ordre de construction : specs, section 8).")

with header:
    theme.page_header("Paramétrage et lancement", 2,
                      "Fixez les frontières de zones, cochez les méthodes à tester, puis lancez les estimations.", banner)
