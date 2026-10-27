"""Methodology page: one document per method (Markdown, formulas included) and a live fit on the France 2020 sample."""
from __future__ import annotations

import streamlit as st

from src import config as C
from ui import methodology as M
from ui import theme
from ui.common import SEX_BY_LABEL, SEX_LABEL


def illustration(doc: M.Doc) -> None:
    """The toggle that fits the method to the France 2020 sample, and what it shows."""
    st.subheader("Illustration sur la France 2020")
    if not M.SAMPLE.exists():
        st.info(f"Le fichier d'exemple {M.SAMPLE.name} est introuvable : l'illustration n'est pas disponible.", icon=":material/info:")
        return
    left, right = st.columns([2, 3], vertical_alignment="center")
    sex_label = left.segmented_control("Jeu de données", [SEX_LABEL[s] for s in M.SEXES], default=SEX_LABEL["Female"],
                                       required=True, key=f"meth_sex_{doc.key}")
    show = right.toggle("Ajuster la méthode à ces données", key=f"meth_show_{doc.key}",
                        help="Réglages par défaut de l'application (ω₁ = 20, ω₂ = 85, ω₃ = 110). Les méthodes à optimisation prennent quelques secondes.")
    if not show:
        st.caption("Activez l'interrupteur : la méthode est ajustée sur les taux bruts de la zone, avec le code de l'application.")
        return
    sex = SEX_BY_LABEL[sex_label]
    result, cfg = M.sample_fit(doc.key, sex)
    df = M.sample_table()
    st.plotly_chart(M.illustration_figure(doc.key, sex, result, cfg, df[df["sex"] == sex]), key=f"meth_fig_{doc.key}_{sex}")
    cols = st.columns(4)
    cols[0].metric("Log-vraisemblance", f"{result.loglik:,.1f}")
    cols[1].metric("Paramètres", f"{int(result.params['n_params'])}")
    cols[2].metric("Convergence", "oui" if result.converged else "non")
    cols[3].metric("Itérations", f"{result.n_iter}")
    st.dataframe(M.parameter_table(result), hide_index=True, width=420)
    st.caption(result.message)


def show(doc: M.Doc) -> None:
    st.markdown(M.page_text(doc.slug))
    if doc.key is not None:
        st.divider()
        illustration(doc)


theme.page_header("Méthodologie", None,
                  "Chaque méthode du lisseur : le modèle, l'estimation pas à pas, ses limites, et ce qu'elle donne sur la France 2020.")
st.caption("Les fiches sont les fichiers de `docs/methodes/`. Les tableaux de résultats sont produits par le code et vérifiés par les tests.")

tabs = st.tabs([name for name, _ in M.GROUPS])
for tab, (_, docs) in zip(tabs, M.GROUPS):
    with tab:
        if len(docs) == 1:
            show(docs[0])
        else:
            for sub, doc in zip(st.tabs([d.label for d in docs]), docs):
                with sub:
                    show(doc)
