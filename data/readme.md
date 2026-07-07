# Dataset

Ce projet utilise le dataset public **Airline Passenger Satisfaction**, mobilisé dans le cadre d'une mission fictive de consultant Data/BI pour une compagnie aérienne (Rush 3 — MSc MSI, Epitech).

## Source

- Dataset : [Airline Passenger Satisfaction — Kaggle](https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction)

## Description

Le jeu de données recense **129 880 passagers** (103 904 en entraînement, 25 976 en test, déjà séparés à la source) et décrit, pour chacun :

- un profil socio-démographique (`Gender`, `Age`, `Customer Type`) ;
- un contexte de voyage (`Type of Travel`, `Class`, `Flight Distance`) ;
- **14 notes de satisfaction par service** sur une échelle 0–5 (wifi, embarquement en ligne, confort du siège, divertissement, propreté, etc.) ;
- les retards au départ et à l'arrivée (`Departure Delay in Minutes`, `Arrival Delay in Minutes`) ;
- la **variable cible** `satisfaction` : `satisfied` / `neutral or dissatisfied`.

## Remarque

Les fichiers `train.csv` et `test.csv` ne sont pas versionnés dans ce dépôt afin de garder le repository léger.
Pour reproduire l'analyse :

1. Télécharger les deux fichiers depuis Kaggle (lien ci-dessus).
2. Les placer dans un dossier local `data/` (ou directement à la racine, à côté du notebook).
3. Adapter `DATA_DIR` en tête du notebook si nécessaire.
