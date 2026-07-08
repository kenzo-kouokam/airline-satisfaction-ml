# -*- coding: utf-8 -*-
import plotly.graph_objects as go
import streamlit as st

from utils.data import load_scored_data, load_service_cols
from utils.filters import render_sidebar_filters
from utils.style import inject_css, page_header, footer, NAVY, ACCENT, GREEN

st.set_page_config(page_title="Recommandations", page_icon="📋", layout="wide")
inject_css()

df = load_scored_data()
service_cols = load_service_cols()
filtered, model_choice = render_sidebar_filters(df)

SERVICE_LABELS = {
    "Inflight wifi service": "Wifi à bord", "Departure/Arrival time convenient": "Horaires pratiques",
    "Ease of Online booking": "Facilité de réservation en ligne", "Gate location": "Emplacement de la porte",
    "Food and drink": "Restauration", "Online boarding": "Embarquement en ligne",
    "Seat comfort": "Confort du siège", "Inflight entertainment": "Divertissement à bord",
    "On-board service": "Service à bord", "Leg room service": "Espace pour les jambes",
    "Baggage handling": "Gestion des bagages", "Checkin service": "Service d'enregistrement",
    "Inflight service": "Service en vol", "Cleanliness": "Propreté",
}

page_header(
    "Synthèse & Recommandations",
    "Ce qu'il faut retenir pour la décision — priorités d'investissement et profils à risque",
    badges=["EXECUTIVE SUMMARY"],
)

if filtered.empty:
    st.warning("Aucun passager ne correspond aux filtres sélectionnés.")
    st.stop()

# ---------------------------------------------------------------- 4 objectifs
st.markdown("#### Réponse aux 4 objectifs de la mission")
o1, o2, o3, o4 = st.columns(4)
objectives = [
    ("1", "Analyser", "Profil satisfait : voyageur d'affaires, classe Business, client fidèle, 45-60 ans."),
    ("2", "Identifier des motifs", "Synergie « business + fidèle », service > ponctualité, effet non-linéaire de l'âge."),
    ("3", "Corrélation & causalité", "Distinction entre variables de profil (non actionnables) et leviers de service (actionnables)."),
    ("4", "Prédire", "KNN retenu (recall 96.7 %, ROC AUC 0.980) ; régression logistique pour l'interprétation."),
]
for col, (num, title, text) in zip([o1, o2, o3, o4], objectives):
    with col:
        with st.container(border=True):
            st.markdown(f"**{num}. {title}**")
            st.caption(text)

st.write("")

col_left, col_right = st.columns([1.3, 1])

with col_left:
    with st.container(border=True):
        st.markdown("#### Priorités d'investissement — services à améliorer en premier")
        st.caption("Classement basé sur l'écart de perception (satisfait − insatisfait), segment filtré")
        means = filtered.groupby("satisfaction")[service_cols].mean().T
        if "satisfied" in means.columns and "neutral or dissatisfied" in means.columns:
            means["diff"] = means["satisfied"] - means["neutral or dissatisfied"]
            top = means.sort_values("diff", ascending=False).head(5)
            fig = go.Figure(go.Bar(
                x=top["diff"], y=[SERVICE_LABELS.get(i, i) for i in top.index], orientation="h",
                marker_color=NAVY, text=[f"+{v:.2f}" for v in top["diff"]], textposition="outside",
            ))
            fig.update_layout(margin=dict(t=10, b=10, l=10, r=40), height=320, plot_bgcolor="white",
                               yaxis=dict(autorange="reversed"), xaxis_title="Écart de note moyenne")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Les deux classes de satisfaction doivent être présentes pour ce calcul.")

    with st.container(border=True):
        st.markdown("#### Recommandations opérationnelles")
        st.markdown(
            """
            - **Prioriser l'expérience digitale d'embarquement** et le divertissement à bord — les deux leviers
              les plus discriminants du dataset — plutôt que la seule réduction des retards.
            - **Utiliser le score de risque prédictif** (page Simulateur) pour cibler en priorité les voyageurs
              personnels en classe Eco, jeunes ou seniors — les profils les plus à risque identifiés.
            - **Ajuster le seuil de décision** du modèle en production (page Performance) selon la capacité
              opérationnelle de l'équipe relation-client à traiter les alertes générées.
            - **Ne pas confondre corrélation et causalité** : la fidélité et la classe sont des marqueurs de
              profil, pas des leviers d'action directs — agir sur la qualité de service réellement perçue.
            """
        )

with col_right:
    with st.container(border=True):
        st.markdown("#### Profils les plus à risque")
        segment_risk = filtered.groupby(["Type of Travel", "Class"], observed=True)["satisfaction"].apply(
            lambda s: (s == "satisfied").mean() * 100
        ).sort_values().head(5)
        for (travel, cls), rate in segment_risk.items():
            with st.container(border=True):
                st.markdown(f"**{travel} · {cls}**")
                st.progress(min(rate / 100, 1.0), text=f"{rate:.1f} % satisfaits")

    with st.container(border=True):
        st.markdown("#### Limites à garder en tête")
        st.caption(
            """
            - Notes de service auto-déclarées : effet de halo possible.
            - Corrélation ≠ causalité sur les variables de profil.
            - KNN coûteux à grande échelle en production (algorithme paresseux).
            - Compagnie, marché et période de collecte non précisés dans le dataset source.
            """
        )

footer()
