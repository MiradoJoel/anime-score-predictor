# Anime Score Predictor

Projet réalisé dans le cadre du cours **Algorithme et Programmation** (M2).

Ce projet prédit le score MyAnimeList d'un animé à partir de ses caractéristiques, en comparant plusieurs modèles de Machine Learning et en exposant le meilleur via une interface Streamlit.

---

## Données

Le dataset provient de [Kaggle - MyAnimeList Dataset](https://www.kaggle.com/datasets/nikhil1e9/myanimelist-anime-and-manga).

Télécharger le fichier `MAL-anime.csv` et le placer dans le dossier `data/` avant de lancer le notebook.

Colonnes utilisées :
- `Type` : catégorie de l'animé (TV, Movie, OVA, etc.)
- `Episodes` : nombre d'épisodes
- `Members` : nombre de membres MAL ayant regardé l'animé
- `Score` : score moyen (variable cible)

---

## Structure du projet
anime-score-predictor/
│
├── data/                        # Dataset (non versionné)
│   └── MAL-anime.csv
├── .streamlit/
│   └── config.toml              # Configuration Streamlit
├── notebook_modelisation.ipynb  # Notebook complet de modélisation
├── app.py                       # Interface Streamlit
├── best_model.pkl               # Meilleur modèle sauvegardé
├── label_encoder.pkl            # Encodeur des types d'animés
├── resultats_modeles.csv        # Tableau comparatif des modèles
├── pyproject.toml               # Dépendances du projet (uv)
└── README.md
---

## Démarche de modélisation

Le notebook `notebook_modelisation.ipynb` contient toutes les étapes :

1. Chargement et exploration des données
2. Nettoyage et encodage des variables
3. Séparation train/test (80/20)
4. Optimisation des hyperparamètres avec **GridSearchCV** (5-fold cross-validation, métrique R²) pour chaque modèle :
   - Decision Tree
   - Random Forest
   - AdaBoost
   - XGBoost
   - LightGBM
   - CatBoost
5. Comparaison des modèles sur le jeu de test (RMSE et R²)
6. Sauvegarde du meilleur modèle

Le meilleur modèle obtenu est **LightGBM**.

---

## Installation et lancement

### Prérequis

- Python >= 3.12
- [uv](https://docs.astral.sh/uv/) installé

### Installation

```bash
git clone https://github.com/MiradoJoel/anime-score-predictor.git
cd anime-score-predictor
uv sync
```

Placer le fichier `MAL-anime.csv` dans le dossier `data/`.

### Lancer le notebook de modélisation

```bash
uv run jupyter notebook notebook_modelisation.ipynb
```

### Lancer l'interface Streamlit

```bash
uv run streamlit run app.py
```

L'application s'ouvre sur `http://localhost:8501`.

---

## Interface Streamlit

L'interface comporte trois pages :

- **Prediction** : saisir le type, le nombre d'épisodes et le nombre de membres pour obtenir un score prédit
- **Comparaison des modèles** : tableau et graphiques comparant les 6 modèles testés
- **À propos** : description du projet

---

## Dépendances principales

- pandas, numpy
- scikit-learn
- xgboost, lightgbm, catboost
- streamlit
- plotly
- jupyter

Toutes les dépendances sont gérées via `uv` et listées dans `pyproject.toml`.