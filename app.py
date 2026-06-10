import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Anime Score Predictor",
    page_icon="🎌",
    layout="wide"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Bangers&family=Nunito:wght@400;700&display=swap');

    .stApp {
        background: linear-gradient(135deg, #f5f0ff 0%, #ede8ff 50%, #f0f5ff 100%);
        color: #1a1a2e;
    }

    h1, h2, h3 {
        font-family: 'Bangers', cursive !important;
        letter-spacing: 2px;
        color: #3a0078;
    }

    .main-title {
        font-family: 'Bangers', cursive;
        font-size: 4em;
        text-align: center;
        background: linear-gradient(90deg, #7B2FBE, #3a0078);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }

    .subtitle {
        text-align: center;
        color: #555577;
        font-size: 1.1em;
        margin-bottom: 2em;
    }

    .score-card {
        background: linear-gradient(135deg, #ffffff, #f0e8ff);
        border: 2px solid #7B2FBE;
        border-radius: 15px;
        padding: 30px;
        text-align: center;
        box-shadow: 0 0 25px rgba(123, 47, 190, 0.2);
    }

    .score-number {
        font-family: 'Bangers', cursive;
        font-size: 5em;
        background: linear-gradient(90deg, #7B2FBE, #3a0078);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .stButton > button {
        background: linear-gradient(90deg, #7B2FBE, #3a0078);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 12px 40px;
        font-size: 1.1em;
        font-weight: bold;
        width: 100%;
        cursor: pointer;
    }

    .stButton > button:hover {
        box-shadow: 0 0 15px rgba(123, 47, 190, 0.5);
    }

    .info-box {
        background: #ffffff;
        border-left: 4px solid #7B2FBE;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }

    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ede8ff, #f5f0ff);
        border-right: 1px solid #7B2FBE;
    }

    div[data-testid="stSidebar"] * {
        color: #1a1a2e !important;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    with open("best_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
    return model, le

@st.cache_data
def load_results():
    return pd.read_csv("resultats_modeles.csv")

model, le = load_model()
results_df = load_results()

with st.sidebar:
    st.markdown("## 🎌 Navigation")
    page = st.radio("", ["Prediction", "Comparaison des modeles", "A propos"])

    st.markdown("---")
    st.markdown("### Meilleur modele")
    best = results_df.iloc[0]
    st.markdown(f"**{best['Modèle']}**")
    st.metric("R² (test)", f"{best['R² (test)']:.4f}")
    st.metric("RMSE (test)", f"{best['RMSE (test)']:.4f}")

if page == "Prediction":

    st.markdown('<p class="main-title">🎌 ANIME SCORE PREDICTOR 🎌</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Predit le score MyAnimeList de ton anime grace au Machine Learning</p>', unsafe_allow_html=True)
    st.markdown("---")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### Caracteristiques de l'anime")

        anime_type = st.selectbox(
            "Type d'anime",
            options=list(le.classes_),
            help="Selectionne la categorie de l'anime"
        )

        episodes = st.number_input(
            "Nombre d'episodes",
            min_value=1,
            max_value=5000,
            value=12,
            step=1
        )

        members = st.number_input(
            "Nombre de membres",
            min_value=100,
            max_value=10000000,
            value=50000,
            step=1000
        )

        predict_btn = st.button("Predire le score !")

    with col2:
        st.markdown("### Resultat de la prediction")

        if predict_btn:
            type_encoded = le.transform([anime_type])[0]
            X_input = np.array([[type_encoded, episodes, members]])
            score = model.predict(X_input)[0]
            score = np.clip(score, 1.0, 10.0)

            if score >= 8.0:
                niveau = "CHEF D'OEUVRE"
                color = "#2E7D32"
            elif score >= 7.0:
                niveau = "TRES BON"
                color = "#388E3C"
            elif score >= 6.0:
                niveau = "BON"
                color = "#1565C0"
            elif score >= 5.0:
                niveau = "MOYEN"
                color = "#E65100"
            else:
                niveau = "A EVITER"
                color = "#B71C1C"

            st.markdown(f"""
            <div class="score-card">
                <div class="score-number">{score:.2f} / 10</div>
                <div style="color:{color}; font-size:1.5em; font-weight:bold; margin-top:10px;">{niveau}</div>
                <hr style="border-color:#ddd; margin:15px 0;">
                <div style="color:#555; margin-top:10px;">
                    {episodes} episodes &nbsp;|&nbsp;
                    {anime_type} &nbsp;|&nbsp;
                    {members:,} membres
                </div>
            </div>
            """, unsafe_allow_html=True)

            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=score,
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [0, 10], 'tickcolor': "#1a1a2e"},
                    'bar': {'color': "#7B2FBE"},
                    'bgcolor': "#f5f0ff",
                    'steps': [
                        {'range': [0, 5], 'color': '#ffe0e0'},
                        {'range': [5, 7], 'color': '#fff8e0'},
                        {'range': [7, 10], 'color': '#e0ffe0'}
                    ],
                    'threshold': {
                        'line': {'color': "#3a0078", 'width': 4},
                        'thickness': 0.75,
                        'value': score
                    }
                }
            ))
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#1a1a2e',
                height=250,
                margin=dict(t=20, b=20)
            )
            st.plotly_chart(fig, use_container_width=True)

        else:
            st.markdown("""
            <div style="text-align:center; padding:60px; color:#999;">
                <div style="font-size:4em">🎌</div>
                <p>Remplis les caracteristiques et clique sur<br><strong>Predire le score !</strong></p>
            </div>
            """, unsafe_allow_html=True)

elif page == "Comparaison des modeles":

    st.markdown("## Comparaison des modeles ML")
    st.markdown("Performances de tous les modeles testes avec **GridSearchCV** (5-fold cross-validation, metrique R²) :")

    st.dataframe(
        results_df[["Modèle", "R² CV (train)", "RMSE (test)", "R² (test)"]].style.highlight_max(
            subset=["R² (test)", "R² CV (train)"], color="#d4edda"
        ).highlight_min(
            subset=["RMSE (test)"], color="#d4edda"
        ),
        use_container_width=True
    )

    col1, col2 = st.columns(2)

    with col1:
        fig1 = px.bar(
            results_df.sort_values("RMSE (test)"),
            x="RMSE (test)", y="Modèle",
            orientation='h',
            title="RMSE par modele (plus bas = meilleur)",
            color="RMSE (test)",
            color_continuous_scale="RdYlGn_r"
        )
        fig1.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#1a1a2e')
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        fig2 = px.bar(
            results_df.sort_values("R² (test)"),
            x="R² (test)", y="Modèle",
            orientation='h',
            title="R² par modele (plus haut = meilleur)",
            color="R² (test)",
            color_continuous_scale="RdYlGn"
        )
        fig2.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#1a1a2e')
        st.plotly_chart(fig2, use_container_width=True)

elif page == "A propos":

    st.markdown("## 🎌 A propos du projet")

    st.markdown("""
    <div class="info-box">
        <h3>Objectif</h3>
        <p>Ce projet predit le score MyAnimeList d'un anime a partir de ses caracteristiques
        en utilisant des algorithmes de Machine Learning.</p>
    </div>

    <div class="info-box">
        <h3>Donnees</h3>
        <p>Dataset issu de <strong>MyAnimeList</strong> via Kaggle, contenant des informations
        sur des milliers d'animes : type, episodes, membres, scores.</p>
    </div>

    <div class="info-box">
        <h3>Modeles testes</h3>
        <p>Decision Tree, Random Forest, AdaBoost, XGBoost, LightGBM, CatBoost —
        chacun optimise avec <strong>GridSearchCV</strong> (5-fold cross-validation, metrique R²).</p>
    </div>

    <div class="info-box">
        <h3>Meilleur modele</h3>
        <p><strong>LightGBM</strong> — selectionne pour son meilleur R² sur le jeu de test.</p>
    </div>
    """, unsafe_allow_html=True)