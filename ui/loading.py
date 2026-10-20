"""Loading page helpers: file inspection, per-file settings, what the files allow, the result of a load."""
from __future__ import annotations

import io
from dataclasses import dataclass, field

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import loader as L
from ui.common import LEVEL_TEXT, MODE_B, QUANTITY_ROLE, ROLE_TEXT, SEX_COLOR, SEX_LABEL, VIEW_LABEL

# --------------------------------------------------------------------------
# Tab 1 helpers: file inspection, per-file settings, what the files allow
# --------------------------------------------------------------------------


class _Upload(io.BytesIO):
    """Bytes with a .name, which is all the loader asks of an uploaded file."""

    def __init__(self, data: bytes, name: str):
        super().__init__(data)
        self.name = name


@st.cache_data(show_spinner=False, max_entries=32)
def inspect_upload(name: str, data: bytes, sheet, header):
    """(FileInfo, None) or (None, message) - cached, the uploader reruns constantly."""
    try:
        return L.inspect_file(L.Source(_Upload(data, name), sheet=sheet, header=header)), None
    except L.LoaderError as exc:
        return None, str(exc)


@st.cache_data(show_spinner=False, max_entries=32)
def excel_sheet_names(data: bytes) -> list:
    try:
        return list(pd.ExcelFile(io.BytesIO(data), engine="openpyxl").sheet_names)
    except Exception:                                  # .xls, corrupt file: inspect_upload explains
        return []


def plain(df: pd.DataFrame) -> pd.DataFrame:
    """A view without `attrs`: the loader hangs its QC reports there, and st.dataframe
    tries to serialise attrs to JSON (and warns, once per display, when it cannot)."""
    out = df.copy()
    out.attrs = {}
    return out


@dataclass
class FileSetup:
    """What a file block hands back: the file, what the loader saw in it, and how to read it
    (sheet, header row, hand-made column mapping, segment, exposure basis)."""
    name: str
    uploaded: object
    info: object = None                                # loader.FileInfo
    error: str | None = None
    opts: dict = field(default_factory=dict)
    roles: set = field(default_factory=set)            # roles found, automatically or by hand
    where: dict = field(default_factory=dict)          # segment column -> chosen value
    exposure_basis: str = "central"

    @property
    def source(self) -> L.Source:
        return L.Source(self.uploaded, sheet=self.opts.get("sheet", 0), header=self.opts.get("header"),
                        col_map=self.opts.get("col_map"), where=self.where or None,
                        exposure_basis=self.exposure_basis)


def needed_roles(info, wanted: str) -> list:
    """Roles the loader could not find by itself - each gets a dropdown (spec 1.3).

    wanted = 'sexfile' (options C and B: age, Dx, Ex, or mx alone) or 'Dx' | 'Ex' | 'mx' (option A)."""
    found = info.auto_map
    need = [] if "age" in found else ["age"]
    if wanted == "sexfile":
        if info.layout == "sex_columns":               # HMD-style: the sex columns are read as is
            return need
        if "mx" in found and "dx" not in found and "ex" not in found:
            return need                                # rates only: E_ref is asked instead
        if "value" in found and not {"dx", "ex", "mx"} & set(found):
            return need                                # a lone value column: its quantity is asked
        need += [r for r in ("dx", "ex") if r not in found]
    else:
        role = QUANTITY_ROLE[wanted]
        leftover = [c for c in info.columns if c not in found.values()]
        if info.layout == "roles" and role not in found and "value" not in found and len(leftover) != 1:
            need.append(role)                          # one column left over = the loader takes it
    return need


def file_block(kind: str, uploaded, wanted: str) -> FileSetup | None:
    """Settings of one uploaded file: Excel sheet, header row, column mapping."""
    if uploaded is None:
        return None
    uid = getattr(uploaded, "file_id", None) or uploaded.name       # new upload -> fresh widgets
    k = f"{kind}_{uid}"
    data = uploaded.getvalue()
    setup = FileSetup(name=uploaded.name, uploaded=uploaded)
    sheet, header = 0, None
    with st.expander(f"Paramètres — {uploaded.name}", expanded=False):
        suffix = uploaded.name.lower().rsplit(".", 1)[-1]
        if suffix in ("xlsx", "xlsm", "xls"):
            sheets = excel_sheet_names(data)
            if sheets:
                sheet = st.selectbox("Feuille", sheets, key=f"{k}_sheet")
        if suffix in ("xlsx", "xlsm", "xls", "csv", "tsv"):
            row = int(st.number_input("Ligne d'en-tête (0 = détection automatique)", min_value=0, value=0,
                                      step=1, key=f"{k}_header"))
            header = row - 1 if row else None
        info, err = inspect_upload(uploaded.name, data, sheet, header)
        if err:
            st.error(err)
            setup.error = err
            return setup
        setup.info = info
        col_map = {}
        need = needed_roles(info, wanted)
        if need:
            st.caption("Colonnes non reconnues automatiquement : associez-les.")
        for role in need:
            choice = st.selectbox(ROLE_TEXT[role], ["—", *info.columns], key=f"{k}_map_{role}")
            if choice != "—":
                col_map[role] = choice
        origin = f"format reconnu : {info.origin} · " if info.origin else ""
        st.caption(f"{origin}{info.n_rows} lignes · colonnes : {', '.join(info.columns)}")
    setup.opts = {"sheet": sheet, "header": header, "col_map": col_map or None}
    setup.roles = set(info.auto_map) | set(col_map)
    return setup


def sex_choices(mode: str, setups: dict) -> tuple:
    """-> (options, note). Option B: the sexes whose file was given. Options C and A: what every file can
    serve; a file that does not say its sex can only be attributed to ONE sex, so 'both' is not offered."""
    if mode == MODE_B:
        have = [s for s, name in (("Female", "female"), ("Male", "male")) if setups.get(name)]
        note = ""
    else:
        infos = [s.info for s in setups.values() if s and s.info]
        if not infos:
            return [], ""
        if any(not i.sexes for i in infos):
            return list(L.SEXES), ("un fichier au moins ne précise pas le sexe (colonne unique) : "
                                   "choisissez celui auquel il se rapporte")
        have, note = [s for s in L.SEXES if all(s in i.sexes for i in infos)], ""
    return (["both"] if len(have) == 2 else []) + have, note


def segment_choices(infos: list) -> dict:
    """{segment column: values every file that has the column can serve} - INSEE CHAMP2 (FM / FR),
    Eurostat geo. One value must be chosen in each, or the same age would appear twice."""
    out: dict = {}
    for info in infos:
        for col, values in info.segments.items():
            out[col] = [v for v in out[col] if v in values] if col in out else list(values)
    return out


def view_choices(infos: list) -> list:
    """Views every file can serve. ['auto'] = no file carries a Year or Cohort column: nothing to choose."""
    lists = [set(i.views) for i in infos if i.views]
    if not lists:
        return ["auto"]
    return [v for v in L.VIEWS if v in set.intersection(*lists)]


def key_choices(infos: list, view: str) -> list:
    """Years (period view) or cohorts (cohort view) every file that has some can serve."""
    if view == "auto":
        return []
    lists = [set(i.years if view == "period" else i.cohorts) for i in infos
             if (i.years if view == "period" else i.cohorts)]
    return sorted(set.intersection(*lists)) if lists else []


# --------------------------------------------------------------------------
# Tab 1: showing what was loaded
# --------------------------------------------------------------------------


def tab1_result(df: pd.DataFrame, *, downloadable: bool = True) -> None:
    """Control table, notes, alerts per sex, crude-rate chart, the standard file, the table itself."""
    qc, sexes = df.attrs["qc"], df.attrs["sexes"]
    view = df.attrs["view"]
    key = df.attrs["cohort"] if view == "cohort" else df.attrs["year"]
    st.divider()
    st.subheader("Contrôle des données")
    st.caption(" · ".join(x for x in (
        df.attrs["source"], VIEW_LABEL[view].split(" (")[0].lower(),
        (f"génération {key}" if view == "cohort" else f"année {key}") if key is not None else "",
        " + ".join(SEX_LABEL[s].lower() for s in sexes)) if x))
    st.dataframe(L.qc_summary(df).rename(columns=SEX_LABEL), hide_index=True)
    for n in df.attrs["notes"]:
        st.info(n.message, icon=":material/info:")

    for s in sexes:
        rep, part = qc[s], df[df["sex"] == s]
        with st.container(border=True):
            st.markdown(f"**{SEX_LABEL[s]}**")
            for item in rep.items:
                if item.level in ("blocking", "red"):
                    st.error(f"**{LEVEL_TEXT[item.level]}** · {item.message}", icon=":material/error:")
                    if item.ages:                                       # spec 1.5: rows listed
                        st.dataframe(plain(part[part["age"].isin(item.ages)]), hide_index=True)
                elif item.level == "orange":
                    st.warning(f"**{LEVEL_TEXT['orange']}** · {item.message}", icon=":material/warning:")
            passed = [i for i in rep.items if i.level == "ok"]
            if passed:
                with st.expander(f"Contrôles réussis ({len(passed)})"):
                    for item in passed:
                        st.markdown(f":green[:material/check:] {item.message}")

    st.plotly_chart(crude_rates_figure(df), key="crude_rates")
    zeros = {SEX_LABEL[s]: int((df.loc[df["sex"] == s, "mx_raw"] <= 0).sum()) for s in sexes}
    if any(zeros.values()):
        shown = ", ".join(f"{n} ({s.lower()})" for s, n in zeros.items() if n)
        st.caption(f"Âge(s) avec mx = 0 (Dx = 0), non représentables en échelle logarithmique : {shown}.")
    if downloadable:
        name = f"taux_bruts_{'generation' if view == 'cohort' else 'moment'}_{key}_{'_'.join(s.lower() for s in sexes)}.csv"
        st.download_button("Télécharger la table standardisée (CSV)", icon=":material/download:",
                           data=L.standard_table(df).to_csv(index=False).encode("utf-8"),
                           file_name=name, mime="text/csv", key="dl_standard",
                           help="Les sexes chargés dans un seul fichier : année ou génération, sexe, âge, "
                                "Dx, Ex, mx_raw. Le loader sait le relire.")
    with st.expander("Table chargée"):
        st.dataframe(plain(df), hide_index=True)


def crude_rates_figure(df: pd.DataFrame) -> go.Figure:
    """Spec 1.3: crude rates, log(mx) against age (log axis, so zeros are not drawn). One series
    per sex: colour AND marker shape tell them apart (the brand green is light on white)."""
    fig = go.Figure()
    for s in df.attrs["sexes"]:
        pos = df[(df["sex"] == s) & (df["mx_raw"] > 0)]
        color, symbol = SEX_COLOR[s]
        fig.add_trace(go.Scatter(
            x=pos["age"], y=pos["mx_raw"], mode="markers", name=SEX_LABEL[s],
            marker=dict(size=7, color=color, symbol=symbol, line=dict(width=1, color="white")),
            customdata=pos[["Dx", "Ex"]].to_numpy(),
            hovertemplate=f"{SEX_LABEL[s]} · âge %{{x}}<br>mx = %{{y:.3e}}<br>Dx = %{{customdata[0]:,.1f}}"
                          "<br>Ex = %{customdata[1]:,.1f}<extra></extra>"))
    view = df.attrs["view"]
    key = df.attrs["cohort"] if view == "cohort" else df.attrs["year"]
    label = (f"génération {key}" if view == "cohort" else f"{key}") if key is not None else ""
    fig.update_layout(
        title=dict(text="Taux bruts mₓ = Dₓ / Eₓ" + (f" — {label}" if label else ""),
                   x=0, xanchor="left", font=dict(size=15)),
        xaxis_title="Âge", yaxis_title="mₓ (échelle logarithmique)", showlegend=True,
        legend=dict(orientation="h", x=0, y=1.0, yanchor="bottom", xanchor="left"),
        height=420, margin=dict(l=10, r=10, t=70, b=10))
    fig.update_yaxes(type="log", gridcolor="rgba(128,128,128,0.2)")
    fig.update_xaxes(gridcolor="rgba(128,128,128,0.0)")
    return fig
