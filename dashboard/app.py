# -*- coding: utf-8 -*-
import plotly.graph_objects as go
import streamlit as st

from utils.data import load_scored_data, compute_live_metrics, AGE_GROUP_LABELS
from utils.filters import render_sidebar_filters
from utils.style import inject_css, page_header, footer, NAVY, ACCENT, GREEN, RED, ICE, TEXT_MUTED

st.set_page_config(
    page_title="Airline Satisfaction Intelligence",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()

df = load_scored_data()
filtered, model_choice = render_sidebar_filters(df)

page_header(
    "Vue d'ensemble — Satisfaction Passagers",
    "Outil d'aide à la décision · 129 880 passagers analysés · Modèles Régression Logistique &amp; KNN",
    badges=["RUSH 3", "MSc MSI", "DATA / BI"],
)

if filtered.empty:
    st.warning("Aucun passager ne correspond aux filtres sélectionnés.")
    st.stop()

# ---------------------------------------------------------------- KPI row
n_total = len(filtered)
pct_satisfied = (filtered["satisfaction"] == "satisfied").mean() * 100
live = compute_live_metrics(filtered, model_choice)

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    with st.container(border=True):
        st.metric("Passagers (segment filtré)", f"{n_total:,}".replace(",", " "))
with c2:
    with st.container(border=True):
        st.metric("Taux de satisfaction", f"{pct_satisfied:.1f} %")
with c3:
    with st.container(border=True):
        st.metric(
            f"Recall — {model_choice}",
            f"{live['recall_class0']*100:.1f} %" if live else "n/a",
            help="Capacité du modèle à détecter les passagers insatisfaits, sur le jeu de test du segment filtré.",
        )
with c4:
    with st.container(border=True):
        st.metric(
            f"ROC AUC — {model_choice}",
            f"{live['roc_auc']:.3f}" if live else "n/a",
        )
with c5:
    with st.container(border=True):
        st.metric("Modèle retenu (prod.)", "KNN", help="Meilleures performances sur toutes les métriques (voir page Performance).")

st.caption(
    f"Métriques de performance calculées sur les {live['n']:,} passagers du **jeu de test** compris dans le segment filtré."
    if live else "Pas assez de données de test dans ce segment pour calculer les métriques du modèle."
)

st.write("")

# ---------------------------------------------------------------- Row 2 : donut + findings
col_left, col_right = st.columns([1, 1.4])

with col_left:
    with st.container(border=True):
        st.markdown("#### Répartition de la satisfaction")
        counts = filtered["satisfaction"].value_counts()
        fig = go.Figure(
            data=[go.Pie(
                labels=["Satisfait" if l == "satisfied" else "Neutre / Insatisfait" for l in counts.index],
                values=counts.values,
                hole=0.62,
                marker=dict(colors=[GREEN, RED] if counts.index[0] == "satisfied" else [RED, GREEN]),
                textinfo="percent",
                textfont=dict(size=14, color="white"),
            )]
        )
        fig.update_layout(
            showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=-0.15),
            margin=dict(t=10, b=10, l=10, r=10), height=300,
            annotations=[dict(text=f"{n_total:,}".replace(",", " "), x=0.5, y=0.5, font_size=22, showarrow=False, font_color=NAVY)],
        )
        st.plotly_chart(fig, use_container_width=True)

with col_right:
    with st.container(border=True):
        st.markdown("#### 3 constats structurants de l'analyse")
        st.markdown(
            """
            **1. Le profil « business + fidèle » domine**
            Un voyageur d'affaires fidèle est satisfait dans 58 à 69 % des cas, contre 10 % pour un
            voyageur personnel — l'écart le plus fort du dataset.

            **2. Le service perçu prime sur la ponctualité**
            Corrélation quasi nulle entre retard et satisfaction (-0.05), contre 0.30 à 0.50 pour les
            notes de service (embarquement en ligne, divertissement...).

            **3. Effet non-linéaire de l'âge**
            La satisfaction culmine à 45-60 ans (57 %) puis rechute chez les 60 ans et plus (21 %) —
            un profil autant à risque que les plus jeunes.
            """
        )

st.write("")

# ---------------------------------------------------------------- Row 3 : satisfaction by segment
with st.container(border=True):
    st.markdown("#### Taux de satisfaction par segment")
    dim = st.radio(
        "Dimension d'analyse", options=["Type de voyage", "Classe", "Type de client", "Tranche d'âge"],
        horizontal=True, label_visibility="collapsed",
    )
    dim_col = {
        "Type de voyage": "Type of Travel", "Classe": "Class",
        "Type de client": "Customer Type", "Tranche d'âge": "age_group",
    }[dim]

    grp = filtered.groupby(dim_col, observed=True)["satisfaction"].apply(lambda s: (s == "satisfied").mean() * 100)
    labels = grp.index.map(lambda k: AGE_GROUP_LABELS.get(k, k)) if dim_col == "age_group" else grp.index

    fig2 = go.Figure(go.Bar(
        x=list(labels), y=grp.values, marker_color=NAVY,
        text=[f"{v:.1f}%" for v in grp.values], textposition="outside",
    ))
    fig2.update_layout(
        margin=dict(t=10, b=10, l=10, r=10), height=340,
        yaxis_title="% satisfaits", yaxis_range=[0, max(grp.values) * 1.2 if len(grp) else 100],
        plot_bgcolor="white",
    )
    st.plotly_chart(fig2, use_container_width=True)

footer()
