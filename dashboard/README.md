# Airline Satisfaction Intelligence — Dashboard

**🔴 En ligne : [satisfaction-client-airline.streamlit.app](https://satisfaction-client-airline.streamlit.app/)**

Outil d'aide à la décision interactif construit sur les modèles du projet (régression logistique & KNN), pensé pour un public non-technique (décideurs, équipes relation-client).

## Pages

| Page | Contenu |
|---|---|
| **Vue d'ensemble** | KPIs, répartition de la satisfaction, 3 constats clés, taux de satisfaction par segment |
| **Facteurs de Satisfaction** | Écarts de perception par service, classement des corrélations, croisement Classe × Type de voyage |
| **Performance des Modèles** | Comparaison LR vs KNN, courbes ROC, matrices de confusion, **simulateur de seuil de décision interactif** |
| **Simulateur de Risque** | Score de risque d'insatisfaction en temps réel pour un passager donné, avec explication des facteurs contributifs |
| **Recommandations** | Synthèse exécutive : objectifs, priorités d'investissement, profils à risque, limites |

Tous les filtres de la barre latérale (type de voyage, classe, fidélité, âge, jeu de données, modèle de scoring) s'appliquent en temps réel à l'ensemble des pages.

## Lancer le dashboard en local

```bash
cd dashboard
pip install -r requirements.txt
streamlit run app.py
```

L'application s'ouvre sur `http://localhost:8501`.

Les modèles (`models/*.joblib`) et le jeu de données scoré (`data/passengers_scored.csv.gz`) sont déjà générés et versionnés — aucune étape supplémentaire n'est nécessaire pour lancer l'app.

## Régénérer les modèles et les données (optionnel)

Si vous voulez ré-entraîner les modèles ou reconstruire le jeu de données scoré à partir des fichiers sources :

```bash
cd dashboard/scripts
# placer train.csv et test.csv dans ce dossier (voir ../../data/readme.md)
python train_and_export.py
```

Le script régénère `models/lr_pipeline.joblib`, `models/knn_pipeline.joblib`, `data/passengers_scored.csv.gz`, `data/metrics.json` et `data/lr_coefficients.json`.

## Déploiement (Streamlit Community Cloud)

L'app est déployée gratuitement à cette adresse : **[satisfaction-client-airline.streamlit.app](https://satisfaction-client-airline.streamlit.app/)**, connectée directement à ce dépôt (branche `main`, fichier `dashboard/app.py`) — chaque push met à jour l'app automatiquement.

Pour déployer sa propre copie :

1. Se rendre sur [share.streamlit.io](https://share.streamlit.io) et se connecter avec son compte GitHub.
2. Cliquer sur **New app**, sélectionner le dépôt (forké), la branche `main`, et le fichier principal `dashboard/app.py`.
3. **Important : dans "Advanced settings", fixer la version Python sur 3.11 ou 3.12.** Laisser la version par défaut (souvent la plus récente, ex. 3.14) peut faire échouer l'installation — certains paquets scientifiques (notamment les dépendances internes de Streamlit) n'ont pas encore de version précompilée pour les toutes dernières versions de Python, et la compilation depuis les sources échoue dans l'environnement de build.
4. Déployer — Streamlit installe automatiquement `dashboard/requirements.txt`.

Aucune donnée sensible n'est requise : les modèles et le jeu de données scoré sont déjà inclus dans le dépôt.

## Structure

```bash
dashboard/
├── app.py                  ← Page d'accueil (Vue d'ensemble)
├── pages/                  ← 4 pages additionnelles (numérotées pour l'ordre du menu)
├── utils/                  ← Style, chargement des données/modèles, filtres partagés
├── models/                 ← Pipelines scikit-learn entraînés (joblib)
├── data/                   ← Dataset scoré (csv.gz) + métriques précalculées (json)
├── scripts/                ← Script de ré-entraînement / export
├── .streamlit/config.toml  ← Thème corporate (navy / glace)
└── requirements.txt
```
