# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.data import load_scored_data, load_service_cols
from utils.filters import render_sidebar_filters
from utils.style import inject_css, page_header, footer, NAVY, ACCENT, GREEN, RED

st.set_page_config(page_title="Facteurs de Satisfaction", page_icon="📊", layout="wide")
inject_css()

df = load_scored_data()
service_cols = load_service_cols()
filtered, model_choice = render_sidebar_filters(df)

page_header(
    "Facteurs de Satisfaction",
    "Quelles variables expliquent réellement la satisfaction, sur le segment actuellement filtré ?",
    badges=["EDA", "ANALYSE BIVARIÉE"],
)

if filtered.empty:
    st.warning("Aucun passager ne correspond aux filtres sélectionnés.")
    st.stop()

y_bin = (filtered["satisfaction"] == "satisfied").astype(int)

col1, col2 = st.columns(2)

with col1:
    with st.container(border=True):
        st.markdown("#### Écart de perception par service")
        st.caption("Note moyenne (satisfait − insatisfait), sur une échelle 0-5")
        means = filtered.groupby("satisfaction")[service_cols].mean().T
        if "satisfied" in means.columns and "neutral or dissatisfied" in means.columns:
            means["diff"] = means["satisfied"] - means["neutral or dissatisfied"]
            means = means.sort_values("diff")
            colors = [GREEN if v > 0 else RED for v in means["diff"]]
            span = means["diff"].max() - means["diff"].min()
            fig = go.Figure(go.Bar(
                x=means["diff"], y=means.index, orientation="h", marker_color=colors,
                text=[f"{v:+.2f}" for v in means["diff"]], textposition="outside",
                cliponaxis=False,
            ))
            fig.update_layout(
                margin=dict(t=10, b=10, l=10, r=40), height=460, plot_bgcolor="white",
                xaxis_title="Écart de note moyenne",
                xaxis_range=[means["diff"].min() - span * 0.25, means["diff"].max() + span * 0.2],
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Les deux classes de satisfaction doivent être présentes dans le segment pour ce calcul.")

with col2:
    with st.container(border=True):
        st.markdown("#### Classement des facteurs par corrélation")
        st.caption("Corrélation de chaque variable numérique avec la satisfaction (binaire)")
        num_cols = service_cols + ["Age", "Flight Distance", "Departure Delay in Minutes", "Arrival Delay in Minutes", "mean_service_score"]
        corrs = filtered[num_cols].corrwith(y_bin).sort_values()
        colors2 = [ACCENT if v < 0 else NAVY for v in corrs]
        fig2 = go.Figure(go.Bar(
            x=corrs.values, y=corrs.index, orientation="h", marker_color=colors2,
        ))
        fig2.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=460, plot_bgcolor="white",
                            xaxis_title="Corrélation avec la satisfaction")
        st.plotly_chart(fig2, use_container_width=True)

st.write("")

with st.container(border=True):
    st.markdown("#### Satisfaction croisée : Classe × Type de voyage")
    st.caption("Identifie les combinaisons de profil les plus / moins satisfaites")
    pivot = filtered.pivot_table(index="Class", columns="Type of Travel", values="satisfaction",
                                  aggfunc=lambda s: (s == "satisfied").mean() * 100)
    fig3 = go.Figure(go.Heatmap(
        z=pivot.values, x=pivot.columns, y=pivot.index,
        colorscale=[[0, "#FDECEA"], [0.5, "#CADCFC"], [1, NAVY]],
        text=[[f"{v:.0f}%" for v in row] for row in pivot.values],
        texttemplate="%{text}", textfont=dict(size=14),
        colorbar=dict(title="% satisfaits"),
    ))
    fig3.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=280)
    st.plotly_chart(fig3, use_container_width=True)

st.write("")

with st.container(border=True):
    st.markdown("#### Détail des notes de service — moyennes par groupe")
    show_cols = ["satisfaction"] + service_cols
    summary = filtered[show_cols].groupby("satisfaction").mean().round(2).T
    summary.columns = ["Neutre / Insatisfait" if c == "neutral or dissatisfied" else "Satisfait" for c in summary.columns]
    st.dataframe(
        summary.style.background_gradient(cmap="Blues", axis=1).format("{:.2f}"),
        use_container_width=True,
    )

footer()
