# -*- coding: utf-8 -*-
"""Shared sidebar filters — persisted across pages via st.session_state keys."""
import streamlit as st

from utils.data import AGE_GROUP_LABELS


def render_sidebar_filters(df):
    with st.sidebar:
        st.markdown("### ✈️ Airline Satisfaction")
        st.caption("Intelligence Dashboard")
        st.divider()
        st.markdown("**Filtres**")

        travel_types = st.multiselect(
            "Type de voyage", options=sorted(df["Type of Travel"].unique()),
            default=sorted(df["Type of Travel"].unique()), key="f_travel_type",
        )
        classes = st.multiselect(
            "Classe", options=sorted(df["Class"].unique()),
            default=sorted(df["Class"].unique()), key="f_class",
        )
        customer_types = st.multiselect(
            "Type de client", options=sorted(df["Customer Type"].unique()),
            default=sorted(df["Customer Type"].unique()), key="f_customer_type",
        )
        age_groups = st.multiselect(
            "Tranche d'âge",
            options=list(AGE_GROUP_LABELS.keys()),
            default=list(AGE_GROUP_LABELS.keys()),
            format_func=lambda k: AGE_GROUP_LABELS[k],
            key="f_age_group",
        )
        dataset_choice = st.radio(
            "Jeu de données", options=["Tous", "Entraînement", "Test"], index=0,
            key="f_dataset", horizontal=False,
        )

        st.divider()
        model_choice = st.selectbox(
            "Modèle de scoring", options=["KNN", "Régression Logistique"], index=0, key="f_model",
        )
        st.caption("KNN : meilleure performance (recall 96.7%). Régression logistique : lecture interprétable.")

    filtered = df[
        df["Type of Travel"].isin(travel_types)
        & df["Class"].isin(classes)
        & df["Customer Type"].isin(customer_types)
        & df["age_group"].isin(age_groups)
    ]
    if dataset_choice == "Entraînement":
        filtered = filtered[filtered["dataset"] == "train"]
    elif dataset_choice == "Test":
        filtered = filtered[filtered["dataset"] == "test"]

    return filtered, model_choice
