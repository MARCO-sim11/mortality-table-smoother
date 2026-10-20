"""Page 4 - final chart and downloads (spec 6)."""
from __future__ import annotations

import functools

import streamlit as st

from src import config as C
from src import evaluator as EV
from src import exporter as X
from ui import theme
from ui.common import SEX_BY_LABEL, SEX_LABEL
from ui.exporting import (EXPORT_DEFAULTS, PREVIEW_FORMATS, PREVIEW_ROWS, SEX_FILE, csv_payload, excel_payload,
                          final_figure, hmd_payload)
from ui.state import remember, remembered

mu_all, tables_all = st.session_state.mu_joint, st.session_state.life_table
df4 = st.session_state.df_raw
sexes4 = [s for s in (df4.attrs["sexes"] if df4 is not None else []) if mu_all and s in mu_all]
theme.page_header("Visualisation et export", 4, "Contrôlez la table lissée, puis téléchargez-la.",
                  tuple(st.session_state.fit_config[sexes4[0]].__dict__[k] for k in ("omega1", "omega2", "omega3"))
                  if sexes4 else (20, 85, 110))
if not sexes4:
    st.info("Validez d'abord la sélection des méthodes (page « Évaluation ») : cette page affiche la table assemblée et "
            "l'exporte.", icon=":material/check_circle:")
    st.page_link("views/evaluation.py", label="Aller à l'évaluation", icon=":material/query_stats:")
else:
    if len(sexes4) > 1:
        last4 = remembered("sex_export", sexes4[0])
        picked4 = st.segmented_control("Jeu de données", [SEX_LABEL[s] for s in sexes4],
                                       default=SEX_LABEL[last4 if last4 in sexes4 else sexes4[0]], required=True, key="export_sex")
        active4 = remember("sex_export", SEX_BY_LABEL[picked4])
        link4 = st.checkbox(
            "Lier le féminin et le masculin", value=remembered("link_export", False), key="link_export",
            help="Coché, les options d'export (colonnes, séparateur, marque décimale, multi-méthodes) sont copiées tout "
                 "de suite sur l'autre sexe et restent identiques. Pour copier le féminin vers le masculin, cochez la "
                 "case alors que le féminin est affiché.")
        remember("link_export", link4)
    else:
        active4, link4 = sexes4[0], False
    df4_sex = df4[df4["sex"] == active4]
    cfg4, table4 = st.session_state.fit_config[active4], tables_all[active4]
    chosen4 = st.session_state.validated_selection[active4]

    st.caption("Méthodes retenues — " + " · ".join(f"{C.ZONE_LABELS[z]} : {C.method_spec(chosen4[z]).label}" for z in C.ZONES))
    current4 = {z: (st.session_state[f"selected_z{i}"] or {}).get(active4) for i, z in enumerate(C.ZONES, start=1)}
    if all(current4.values()) and current4 != chosen4:
        st.warning("Les menus de la page Évaluation ont changé depuis la validation : la table ci-dessous est celle de la "
                   "sélection validée.", icon=":material/warning:")

    st.plotly_chart(final_figure(df4_sex, mu_all[active4], cfg4, active4), key=f"final_chart_{active4}")

    st.subheader("Table qₓ")
    formats4 = {c: st.column_config.NumberColumn(c, format=f) for c, f in PREVIEW_FORMATS.items()}
    if len(table4) > 2 * PREVIEW_ROWS:
        st.caption(f"{PREVIEW_ROWS} premières lignes")
        st.dataframe(table4.head(PREVIEW_ROWS), hide_index=True, column_config=formats4)
        st.caption(f"{PREVIEW_ROWS} dernières lignes")
        st.dataframe(table4.tail(PREVIEW_ROWS), hide_index=True, column_config=formats4)
    else:
        st.dataframe(table4, hide_index=True, column_config=formats4)

    st.subheader("Export")
    if link4:
        st.caption("Les deux sexes sont liés : les options d'export sont les mêmes pour les deux.")
    own4 = {**EXPORT_DEFAULTS, **st.session_state.export_params.get(active4, {})}     # this sex's last options
    c1, c2, c3 = st.columns(3)
    cols4 = c1.multiselect("Colonnes à exporter (l'âge est toujours inclus)", X.TABLE_COLUMNS[1:],
                           default=[c for c in own4["cols"] if c in X.TABLE_COLUMNS[1:]], key=f"export_cols_{active4}")
    sep4 = c2.segmented_control("Séparateur du CSV", list(X.SEPARATORS), default=own4["sep"], required=True,
                                key=f"export_sep_{active4}")
    dec4 = c3.segmented_control("Marque décimale du CSV", list(X.DECIMALS), default=own4["decimal"], required=True,
                                key=f"export_decimal_{active4}")
    multi4 = st.checkbox("Export multi-méthodes : une feuille par méthode testée dans l'Excel", value=own4["multi"],
                         key=f"export_multi_{active4}")
    now4 = {"cols": list(cols4), "sep": sep4, "decimal": dec4, "multi": bool(multi4)}
    st.session_state.export_params = {
        **st.session_state.export_params, active4: now4,
        **({s: dict(now4) for s in sexes4 if s != active4} if link4 else {})}              # the sex on screen sets the other
    if not cols4:
        st.error("Choisissez au moins une colonne à exporter.", icon=":material/error:")
    if sep4 == dec4:
        st.error("Le séparateur et la marque décimale ne peuvent pas être identiques.", icon=":material/error:")

    year4 = df4.attrs.get("year") if df4.attrs.get("view") != "cohort" else None
    cohort4 = df4.attrs.get("cohort") if df4.attrs.get("view") == "cohort" else None
    base4 = "_".join(str(x) for x in ("table_lissee", SEX_FILE[active4], cohort4 or year4) if x)
    title4 = f"{df4.attrs.get('source') or 'Mortality Table Smoother'} ({SEX_LABEL[active4].lower()})"
    d1, d2, d3 = st.columns(3)
    d1.download_button(
        "Excel (.xlsx)", icon=":material/table:", key="dl_excel", disabled=not cols4,
        data=functools.partial(excel_payload, df4_sex, cfg4, st.session_state.fit_results[active4], chosen4, table4,
                               mu_all[active4], active4, cols4, multi4, title4, remembered("wh_h", EV.WH_H)),
        file_name=f"{base4}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        help="Feuilles : Table_lissée, Taux_bruts, Paramètres, Critères, Graphique (+ une par méthode si l'option est cochée).")
    d2.download_button(
        "CSV", icon=":material/csv:", key="dl_csv", disabled=not cols4 or sep4 == dec4,
        data=functools.partial(csv_payload, table4, cols4, sep4, dec4), file_name=f"{base4}.csv", mime="text/csv")
    d3.download_button(
        "Texte (format HMD)", icon=":material/description:", key="dl_txt", disabled=not cols4,
        data=functools.partial(hmd_payload, table4, cols4, title4, year4, cohort4), file_name=f"{base4}.txt",
        mime="text/plain", help="Entête standardisée, colonnes alignées séparées par des espaces.")
