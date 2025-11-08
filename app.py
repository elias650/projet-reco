# app.py — UI stylée + métriques + tableau + bar chart + 10 décimales
import streamlit as st
import pandas as pd
import altair as alt

# ---------- Page config & thème de base ----------
st.set_page_config(
    page_title="Reco Renforcement (6 semaines)",
    page_icon="💪",
    layout="centered",
)

# ---------- Masquer éléments Streamlit + styles légers ----------
st.markdown("""
<style>
/* Masquer le menu/hint/footer Streamlit */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
/* Boutons arrondis et un peu plus visibles */
.stButton>button {
    border-radius: 12px;
    padding: 0.6rem 1rem;
    font-weight: 600;
}
/* Conteneur plus respirant */
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 900px;
}
/* Titres */
h1, h2, h3 { font-weight: 700; }
/* Cartes (containers border=True) */
div[data-testid="stVerticalBlock"] > div[style*="border"] {
    border-radius: 12px !important;
}
</style>
""", unsafe_allow_html=True)

# ---------- Chargement du module modèle (mis en cache) ----------
@st.cache_resource(show_spinner=True)
def get_model_module():
    import modele  # ton fichier modele.py
    return modele

modele = get_model_module()
reco_depuis_inputs = modele.reco_depuis_inputs
MUSCLES = modele.MUSCLES

# ---------- Sidebar "À propos" ----------
with st.sidebar:
    st.header("À propos")
    st.write("Application de **recommandation d’intervention** sur 6 semaines.")
    st.write("Sélection de modèle par muscle (Ridge / RandomForest, LOOCV).")
    st.caption("Données d'exemple anonymisées. Nov. 2025.")

# ---------- En-tête ----------
st.title("Recommandation d’intervention — 6 semaines")
st.caption(
    "Renseigne les informations de base et les niveaux de force (0–100, jusqu’à 10 décimales). "
    "L’application calcule un **%RM** et un **volume (séries/semaine)** équilibrés, avec les **gains estimés par groupe**."
)

# ---------- Formulaire ----------
with st.container(border=True):
    st.subheader("Informations de base")
    c1, c2, c3 = st.columns(3)
    age    = c1.number_input("Âge", min_value=10, max_value=100, value=28)
    poids  = c2.number_input("Poids (kg)", min_value=30.0, max_value=200.0, value=70.0, step=0.5)
    sexe   = c3.selectbox("Sexe", ["H", "F", "Autre"])

    c4, c5 = st.columns(2)
    lat    = c4.selectbox("Latéralité", ["droitier", "gaucher"])  # pas d'ambidextre
    niv    = c5.selectbox("Niveau de pratique", ["débutant", "intermédiaire", "avancé", "très avancé"])
    one_rm = st.number_input("1RM de référence (kg)", min_value=10.0, max_value=300.0, value=100.0, step=0.5)

with st.container(border=True):
    st.subheader("Niveaux de force normalisés (0–100, précision 0,001)")
    cL, cR = st.columns(2)

    hip_ext_G   = cL.number_input("Extenseurs de hanche — Gauche",  min_value=0.0, max_value=100.0, value=55.0, step=0.001, format="%.3f")
    hip_ext_D   = cR.number_input("Extenseurs de hanche — Droite",  min_value=0.0, max_value=100.0, value=60.0, step=0.001, format="%.3f")
    knee_flex_G = cL.number_input("Fléchisseurs de genou — Gauche", min_value=0.0, max_value=100.0, value=60.0, step=0.001, format="%.3f")
    knee_flex_D = cR.number_input("Fléchisseurs de genou — Droite", min_value=0.0, max_value=100.0, value=62.0, step=0.001, format="%.3f")
    knee_ext_G  = cL.number_input("Extenseurs de genou — Gauche",   min_value=0.0, max_value=100.0, value=52.0, step=0.001, format="%.3f")
    knee_ext_D  = cR.number_input("Extenseurs de genou — Droite",   min_value=0.0, max_value=100.0, value=58.0, step=0.001, format="%.3f")
0.0000000001, format="%.10f")

st.write("")  # petit espace

# ---------- Action ----------
if st.button("Proposer l’intervention optimale"):
    with st.spinner("Calcul en cours..."):
        # normalisation robuste
        lat_norm = lat.strip().lower()
        niv_norm = niv.strip().lower()

        pct_rm, series_sem, gains, score = reco_depuis_inputs(
            age, poids, sexe, lat_norm, niv_norm, one_rm,
            hip_ext_G, hip_ext_D, knee_flex_G, knee_flex_D, knee_ext_G, knee_ext_D
        )

    # --- Cartes métriques ---
    m1, m2, m3 = st.columns(3)
    m1.metric(label="%RM recommandé", value=f"{pct_rm:.1f} %")
    m2.metric(label="Séries / semaine", value=f"{series_sem:.1f}")
    m3.metric(label="Score équilibre", value=f"{score:.2f}")

    st.divider()
    st.subheader("Gains estimés par groupe (%)")

    # --- Tableau + Bar chart ---
    df_gains = pd.DataFrame({
        "Groupe musculaire": MUSCLES,
        "Gain (%)": [float(g) for g in gains],
    })

    st.dataframe(
        df_gains.style.format({"Gain (%)": "{:.2f}"}),
        use_container_width=True,
        hide_index=True
    )

    chart = (
        alt.Chart(df_gains)
        .mark_bar()
        .encode(
            x=alt.X("Gain (%)", title="Gain estimé (%)"),
            y=alt.Y("Groupe musculaire", sort="-x", title=""),
            tooltip=["Groupe musculaire", alt.Tooltip("Gain (%)", format=".2f")]
        )
        .properties(height=280)
    )
    st.altair_chart(chart, use_container_width=True)

    st.caption("Note : %RM et séries/semaine sont globaux (équilibrés) ; les gains varient par groupe.")
