# Project-Python

Ce projet est un moteur de recherche basé sur Python, conçu pour analyser et rechercher des informations dans des corpus textuels provenant de Reddit et Arxiv.

## Fonctionnalités

- **Collecte de données** :
  - Collecte des données textuelles à partir de Reddit via son API.
  - Récupération des résumés de recherches scientifiques depuis Arxiv.

- **Analyse et traitement** :
  - Construction de matrices TF (Term Frequency) et TF-IDF.
  - Calcul des statistiques textuelles (nombre moyen de mots, de phrases, etc.).
  - Recherche par mots-clés avec la similarité cosinus.

- **Interface utilisateur** :
  - Utilisation de Streamlit pour une interface simple et interactive.
  - Visualisation des résultats et exportation en CSV.

## Installation

1. **Cloner le projet :**
   ```bash
   git clone https://github.com/<votre_nom_utilisateur>/<nom_du_projet>.git
   cd <nom_du_projet>
