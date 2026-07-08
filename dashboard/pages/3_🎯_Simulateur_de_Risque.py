# -*- coding: utf-8 -*-
import plotly.graph_objects as go
import streamlit as st

from utils.data import load_scored_data, load_service_cols, load_models
from utils.filters import render_sidebar_filters
from utils.model import build_input_row, predict, lr_local_contributions
from utils.style import inject_css, page_header, footer, NAVY, ACCENT, GREEN, RED, AMBER

st.set_page_config(page_title="Simulateur de Risque", page_icon="🎯", layout="wide")
inject_css()

df = load_scored_data()
service_cols = load_service_cols()
filtered, model_choice = render_sidebar_filters(df)
models = load_models()

page_header(
    "Simulateur de Risque d'Insatisfaction",
    "Estimez la probabilité de satisfaction d'un passager à partir de son profil et de son expérience de vol",
    badges=["SCORING TEMPS RÉEL", model_choice.upper()],
)

SERVICE_LABELS = {
    "Inflight wifi service": "Wifi à bord", "Departure/Arrival time convenient": "Horaires pratiques",
    "Ease of Online booking": "Facilité de réservation en ligne", "Gate location": "Emplacement de la porte",
    "Food and drink": "Restauration", "Online boarding": "Embarquement en ligne",
    "Seat comfort": "Confort du siège", "Inflight entertainment": "Divertissement à bord",
    "On-board service": "Service à bord", "Leg room service": "Espace pour les jambes",
    "Baggage handling": "Gestion des bagages", "Checkin service": "Service d'enregistrement",
    "Inflight service": "Service en vol", "Cleanliness": "Propreté",
}

with st.form("passenger_form"):
    st.markdown("#### Profil du passager")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        gender = st.selectbox("Genre", ["Male", "Female"])
        customer_type = st.selectbox("Type de client", ["Loyal Customer", "disloyal Customer"])
    with c2:
        age = st.slider("Âge", 7, 85, 38)
        travel_type = st.selectbox("Type de voyage", ["Business travel", "Personal Travel"])
    with c3:
        travel_class = st.selectbox("Classe", ["Business", "Eco", "Eco Plus"])
        flight_distance = st.number_input("Distance de vol (km)", 31, 5000, 1200)
    with c4:
        dep_delay = st.number_input("Retard au départ (min)", 0, 1600, 0)
        arr_delay = st.number_input("Retard à l'arrivée (min)", 0, 1600, 0)

    st.markdown("#### Notes de service (0 = très mauvais / non applicable · 5 = excellent)")
    service_values = {}
    cols = st.columns(2)
    for i, sc in enumerate(service_cols):
        with cols[i % 2]:
            service_values[sc] = st.slider(SERVICE_LABELS.get(sc, sc), 0, 5, 3, key=f"svc_{sc}")

    submitted = st.form_submit_button("🔍 Calculer le score de risque", use_container_width=True)

if submitted:
    inputs = {
        "Gender": gender, "Customer Type": customer_type, "Age": age,
        "Type of Travel": travel_type, "Class": travel_class, "Flight Distance": flight_distance,
        "Departure Delay in Minutes": dep_delay, "Arrival Delay in Minutes": arr_delay,
        **service_values,
    }
    input_row = build_input_row(inputs, service_cols)

    proba_satisfied = predict(models[model_choice], input_row)
    proba_risk = 1 - proba_satisfied

    if proba_risk >= 0.6:
        risk_label, risk_color = "RISQUE ÉLEVÉ", RED
    elif proba_risk >= 0.35:
        risk_label, risk_color = "RISQUE MODÉRÉ", AMBER
    else:
        risk_label, risk_color = "RISQUE FAIBLE", GREEN

    st.write("")
    col_gauge, col_info = st.columns([1, 1.2])

    with col_gauge:
        with st.container(border=True):
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=proba_satisfied * 100,
                number={"suffix": " %", "font": {"size": 40, "color": NAVY}},
                title={"text": f"Probabilité de satisfaction — {model_choice}", "font": {"size": 14}},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": NAVY},
                    "steps": [
                        {"range": [0, 35], "color": "#FDECEA"},
                        {"range": [35, 65], "color": "#FEF3E2"},
                        {"range": [65, 100], "color": "#E9F7EF"},
                    ],
                    "threshold": {"line": {"color": ACCENT, "width": 4}, "thickness": 0.8, "value": 50},
                },
            ))
            fig.update_layout(height=280, margin=dict(t=50, b=10, l=30, r=30))
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(
                f'<div style="text-align:center;"><span class="risk-pill" style="background-color:{risk_color}22; color:{risk_color};">● {risk_label}</span></div>',
                unsafe_allow_html=True,
            )

    with col_info:
        with st.container(border=True):
            st.markdown("#### Recommandation")
            if risk_label == "RISQUE ÉLEVÉ":
                st.error(
                    "Ce passager présente un profil à forte probabilité d'insatisfaction. "
                    "Action recommandée : suivi proactif par l'équipe relation-client "
                    "(geste commercial, enquête de satisfaction ciblée) avant qu'il ne laisse un avis négatif."
                )
            elif risk_label == "RISQUE MODÉRÉ":
                st.warning(
                    "Profil incertain. Pas d'action immédiate nécessaire, mais à surveiller si "
                    "ce profil se répète sur un segment de clientèle plus large."
                )
            else:
                st.success("Profil globalement satisfait. Aucune action corrective nécessaire.")

    st.write("")
    with st.container(border=True):
        st.markdown("#### Pourquoi ce score ? — Facteurs contributifs (modèle interprétable)")
        st.caption(
            "Le KNN ne fournissant pas de coefficients, cette décomposition utilise la régression logistique "
            "pour expliquer la direction et le poids de chaque variable sur CE passager précis."
        )
        contrib = lr_local_contributions(models["Régression Logistique"], input_row, top_n=10)
        colors = [GREEN if v > 0 else RED for v in contrib["contribution"]]
        fig_c = go.Figure(go.Bar(
            x=contrib["contribution"], y=contrib["feature"], orientation="h", marker_color=colors,
        ))
        fig_c.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=380, plot_bgcolor="white",
                             xaxis_title="Contribution au score (positif = pousse vers 'satisfait')")
        st.plotly_chart(fig_c, use_container_width=True)
else:
    st.info("Renseignez le profil du passager ci-dessus puis cliquez sur **Calculer le score de risque**.")

footer()
