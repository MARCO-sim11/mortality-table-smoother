"""Page 3 - evaluation and selection of one method per zone (spec 5, 4)."""
from __future__ import annotations

import hashlib

import streamlit as st

from src import config as C
from src import evaluator as EV
from ui import theme
from ui.common import SEX_BY_LABEL, SEX_LABEL
from ui.evaluation import (RESIDUAL_CLIP, assemble_tables, evaluation_display, evaluation_tables, fit_figure,
                           offered_in_zone, residual_figure)
from ui.state import remember, remembered

fits = st.session_state.fit_results
df3 = st.session_state.df_raw
sexes3 = [s for s in (df3.attrs["sexes"] if df3 is not None else []) if fits.get(s)]
theme.page_header("Évaluation et sélection", 3,
                  "Comparez les méthodes zone par zone, choisissez-en une par zone, puis validez.",
                  tuple(st.session_state.fit_config[sexes3[0]].__dict__[k] for k in ("omega1", "omega2", "omega3"))
                  if sexes3 else (20, 85, 110))
if not sexes3:
    st.info("Lancez d'abord les estimations (page « Paramétrage ») : cette page compare les méthodes estimées, "
            "puis assemble les zones.", icon=":material/play_arrow:")
    st.page_link("views/parametrage.py", label="Aller au paramétrage", icon=":material/tune:")
else:
    if len(sexes3) > 1:
        last3 = remembered("sex_eval", sexes3[0])
        picked3 = st.segmented_control("Jeu de données", [SEX_LABEL[s] for s in sexes3],
                                       default=SEX_LABEL[last3 if last3 in sexes3 else sexes3[0]], required=True, key="eval_sex")
        active3 = remember("sex_eval", SEX_BY_LABEL[picked3])
        link3 = st.checkbox(
            "Lier le féminin et le masculin", value=remembered("link_pick", False), key="link_pick",
            help="Coché, la méthode choisie pour chaque zone est reportée sur l'autre sexe quand celui-ci l'a estimée "
                 "(sinon son propre choix est conservé). Pour copier le féminin vers le masculin, cochez la case alors "
                 "que le féminin est affiché.")
        remember("link_pick", link3)
    else:
        active3, link3 = sexes3[0], False
    with st.expander("Options d'évaluation"):
        h3 = remember("wh_h", st.number_input(
            "h — lissage de Whittaker-Henderson", min_value=0.0, value=remembered("wh_h", EV.WH_H), format="%.3g",
            key="wh_h", help="Poids de la rugosité dans WH = Σ w (q − z)² + h Σ (Δ²z)². "
                             "Les deux termes sont aussi donnés séparément."))
    table3, residuals3 = evaluation_tables(active3, h3)
    cfg3 = st.session_state.fit_config[active3]
    df3_sex = df3[df3["sex"] == active3]

    if table3.empty:
        st.warning("Aucune méthode estimée n'a de données dans sa zone : rien à évaluer.", icon=":material/warning:")
    else:
        st.subheader("Tableau récapitulatif")
        show3, config3 = evaluation_display(table3)
        st.dataframe(show3, hide_index=True, column_config=config3)
        st.caption("L'AIC et le BIC ne se comparent qu'entre méthodes d'une même zone (ΔAIC = écart au meilleur de "
                   "la zone). Règle suggérée : AIC le plus bas de la zone parmi les méthodes dont le runs test "
                   "n'est pas significatif (p > 0,05) ; à défaut, l'AIC le plus bas.")

        st.plotly_chart(residual_figure(residuals3, cfg3), key=f"residuals_{active3}")           # one under the other: the legend
        st.plotly_chart(fit_figure(fits[active3], df3_sex, cfg3, active3), key=f"fit_{active3}")   # takes a column on the right
        st.caption(f"Résidus affichés dans [−{RESIDUAL_CLIP:g}, +{RESIDUAL_CLIP:g}] (valeur réelle au survol). "
                   "Bandes pointillées : ±2.")

        st.subheader("Méthode finale par zone")
        suggested3 = {r.zone: r.method for r in table3.itertuples() if r.suggested}
        sig3 = hashlib.md5("|".join(f"{k}:{r.loglik:.6f}" for k, r in fits[active3].items()).encode()).hexdigest()[:8]
        picks3 = {}
        for col, (i, zone) in zip(st.columns(3), enumerate(C.ZONES, start=1)):
            keys = [m for m, z in zip(table3["method"], table3["zone"]) if z == zone]
            with col:
                if keys:
                    kept = st.session_state[f"selected_z{i}"]        # streamlit drops the state of a menu that is not
                    stored = kept.get(active3) if isinstance(kept, dict) else None   # shown, so the choice is read back here
                    start = stored if stored in keys else suggested3.get(zone, keys[0])
                    picks3[zone] = st.selectbox(
                        f"Méthode finale — {C.ZONE_LABELS[zone]}", keys, index=keys.index(start),
                        format_func=lambda k: C.method_spec(k).label, key=f"final_{active3}_{zone}_{sig3}")
                else:
                    picks3[zone] = None
                    st.warning(f"Aucune méthode estimée pour la {C.ZONE_LABELS[zone]}.", icon=":material/warning:")
        not_copied = []
        for i, zone in enumerate(C.ZONES, start=1):                  # spec 7.2 selected_z1..3, per sex
            kept = st.session_state[f"selected_z{i}"]
            updated = {**(kept if isinstance(kept, dict) else {}), active3: picks3[zone]}
            if link3 and picks3[zone]:                               # the sex on screen sets the other one
                for other in sexes3:
                    if other == active3:
                        continue
                    if picks3[zone] in offered_in_zone(fits[other], zone):
                        updated[other] = picks3[zone]
                    else:
                        not_copied.append(f"{C.ZONE_LABELS[zone]} ({SEX_LABEL[other].lower()})")
            st.session_state[f"selected_z{i}"] = updated
        st.caption("Menus préremplis avec la suggestion de la règle ci-dessus."
                   + (" Les deux sexes sont liés : le choix est reporté sur l'autre sexe." if link3 else ""))
        if not_copied:
            st.caption("Non reporté (méthode non estimée pour l'autre sexe) : " + ", ".join(not_copied) + ".")

        if st.button("Valider la sélection", type="primary", key="btn_validate", icon=":material/check_circle:"):
            chosen = {}
            for s in sexes3:                                        # a sex never opened here gets the suggestion
                own = {z: (st.session_state[f"selected_z{i}"] or {}).get(s) for i, z in enumerate(C.ZONES, start=1)}
                if not all(own.values()):
                    own = {**EV.suggest_selection(evaluation_tables(s, h3)[0]), **{z: k for z, k in own.items() if k}}
                chosen[s] = own
            mu_joint, life_table, validated, problems = assemble_tables(fits, st.session_state.fit_config, chosen)
            st.session_state.update(mu_joint=mu_joint or None, life_table=life_table or None,
                                    validated_selection=validated or None)
            for message in problems:
                st.error(message, icon=":material/error:")

        done = st.session_state.validated_selection
        if done and active3 in done:
            lt3 = st.session_state.life_table[active3]
            ex = lambda a: lt3.loc[lt3["age"] == a, "ex"]
            facts = " · ".join(f"e{a} = {ex(a).iloc[0]:.2f} ans" for a in (0, 65) if len(ex(a)))
            if done[active3] != {z: picks3[z] for z in C.ZONES}:
                st.warning("Les menus ne correspondent plus à la sélection validée : validez de nouveau pour "
                           "reconstruire la table.", icon=":material/warning:")
            else:
                st.success(f"Sélection validée — table de vie assemblée ({SEX_LABEL[active3].lower()}) : {facts}. "
                           "Visualisation et export : page « Export ».", icon=":material/check_circle:")

