# -*- coding: utf-8 -*-
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score, accuracy_score

from utils.data import load_scored_data, load_metrics
from utils.filters import render_sidebar_filters
from utils.style import inject_css, page_header, footer, NAVY, ACCENT, GREEN, RED, TEXT_MUTED

st.set_page_config(page_title="Performance des Modèles", page_icon="🤖", layout="wide")
inject_css()

df = load_scored_data()
metrics = load_metrics()
filtered, model_choice = render_sidebar_filters(df)

page_header(
    "Performance des Modèles",
    "Régression Logistique vs KNN — comparaison, courbes ROC et seuil de décision interactif",
    badges=["ÉVALUATION", "JEU DE TEST EXTERNE"],
)

# ---------------------------------------------------------------- Comparison table
with st.container(border=True):
    st.markdown("#### Comparaison sur le jeu de test complet (25 976 passagers, seuil 0.5)")
    rows = []
    for name in ["Régression Logistique", "KNN"]:
        m = metrics[name]
        rows.append({
            "Modèle": name,
            "Accuracy": f"{m['accuracy']:.3f}",
            "Recall (insatisfait)": f"{m['recall_class0']:.3f}",
            "Précision (insatisfait)": f"{m['precision_class0']:.3f}",
            "F1 macro": f"{m['f1_macro']:.3f}",
            "ROC AUC": f"{m['roc_auc']:.3f}",
        })
    st.dataframe(rows, use_container_width=True, hide_index=True)
    st.success("**Modèle retenu pour la production : KNN** — recall de 96.7 % sur la classe prioritaire, "
               "ROC AUC de 0.980. La régression logistique reste utilisée comme modèle d'interprétation (page Recommandations).")

st.write("")

col1, col2 = st.columns(2)
with col1:
    with st.container(border=True):
        st.markdown("#### Courbes ROC")
        fig = go.Figure()
        colors = {"Régression Logistique": ACCENT, "KNN": NAVY}
        for name in ["Régression Logistique", "KNN"]:
            m = metrics[name]
            fig.add_trace(go.Scatter(
                x=m["fpr"], y=m["tpr"], mode="lines", name=f"{name} (AUC={m['roc_auc']:.3f})",
                line=dict(color=colors[name], width=3),
            ))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Aléatoire",
                                  line=dict(color="lightgray", dash="dash")))
        fig.update_layout(
            margin=dict(t=10, b=10, l=10, r=10), height=380, plot_bgcolor="white",
            xaxis_title="Taux de faux positifs", yaxis_title="Taux de vrais positifs",
            legend=dict(orientation="h", yanchor="bottom", y=-0.35),
        )
        st.plotly_chart(fig, use_container_width=True)

with col2:
    with st.container(border=True):
        st.markdown("#### Matrice de confusion (référence, seuil 0.5)")
        cm = np.array(metrics[model_choice]["confusion_matrix"])
        fig_cm = go.Figure(go.Heatmap(
            z=cm, x=["Prédit : Insatisfait", "Prédit : Satisfait"], y=["Réel : Insatisfait", "Réel : Satisfait"],
            colorscale=[[0, "#F3F6FC"], [1, NAVY]], text=cm, texttemplate="%{text}", textfont=dict(size=18),
            showscale=False,
        ))
        fig_cm.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=380,
                              yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_cm, use_container_width=True)

st.write("")

# ---------------------------------------------------------------- Interactive threshold tuner
with st.container(border=True):
    st.markdown(f"#### 🎚️ Simulateur de seuil de décision — {model_choice}")
    st.caption(
        "Le seuil par défaut (0.5) n'est pas forcément optimal pour la priorité métier. "
        "Déplacez le curseur pour voir l'arbitrage précision / recall en temps réel, sur le segment filtré."
    )

    test_rows = filtered[filtered["dataset"] == "test"]
    if len(test_rows) < 10 or test_rows["satisfaction"].nunique() < 2:
        st.info("Pas assez de données de test dans ce segment pour simuler un seuil.")
    else:
        proba_col = "knn_proba" if model_choice == "KNN" else "lr_proba"
        y_true = (test_rows["satisfaction"] == "satisfied").astype(int)
        proba = test_rows[proba_col].values

        threshold = st.slider("Seuil de décision (probabilité d'être satisfait)", 0.05, 0.95, 0.50, 0.01)
        y_pred = (proba >= threshold).astype(int)

        cA, cB, cC, cD = st.columns(4)
        with cA:
            with st.container(border=True):
                st.metric("Recall (insatisfait)", f"{recall_score(y_true, y_pred, pos_label=0)*100:.1f} %")
        with cB:
            with st.container(border=True):
                st.metric("Précision (insatisfait)", f"{precision_score(y_true, y_pred, pos_label=0)*100:.1f} %")
        with cC:
            with st.container(border=True):
                st.metric("F1 macro", f"{f1_score(y_true, y_pred, average='macro'):.3f}")
        with cD:
            with st.container(border=True):
                st.metric("Accuracy", f"{accuracy_score(y_true, y_pred)*100:.1f} %")

        cm_live = confusion_matrix(y_true, y_pred)
        fig_live = go.Figure(go.Heatmap(
            z=cm_live, x=["Prédit : Insatisfait", "Prédit : Satisfait"], y=["Réel : Insatisfait", "Réel : Satisfait"],
            colorscale=[[0, "#F3F6FC"], [1, ACCENT]], text=cm_live, texttemplate="%{text}", textfont=dict(size=16),
            showscale=False,
        ))
        fig_live.update_layout(margin=dict(t=30, b=10, l=10, r=10), height=320,
                                yaxis=dict(autorange="reversed"),
                                title=dict(text=f"Matrice de confusion au seuil {threshold:.2f}", font=dict(size=13, color=TEXT_MUTED)))
        st.plotly_chart(fig_live, use_container_width=True)

        st.info(
            "**Lecture métier** : baisser le seuil augmente le recall (moins de passagers insatisfaits ratés) "
            "mais fait chuter la précision (plus de fausses alertes). Le bon compromis dépend de la capacité "
            "opérationnelle de l'équipe relation-client à traiter les alertes générées."
        )

footer()
