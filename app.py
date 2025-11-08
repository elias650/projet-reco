# app.py — version finale (forces à 10 décimales)
import streamlit as st

# Cache le chargement du module modèle (entraîne une seule fois par session)
@st.cache_resource(show_spinner=True)
def get_model_module():
    import modele  # importe ton fichier modele.py
    return modele

modele = get_model_module()
reco_depuis_inputs = modele.reco_depuis_inputs
MUSCLES = modele.MUSCLES

st.set_page_config(page_title="Reco Renforcement (6 semaines)", layout="centered")
st.title("Recommandation d’intervention — 6 semaines")

st.caption(
    "Renseigne les informations de base et les niveaux de force (0–100, jusqu’à 10 décimales). "
    "L’application calcule un %RM et un volume (séries/semaine) **équilibrés**, avec les gains estimés par groupe."
)

# ====== Informations de base ======
c1, c2, c3 = st.columns(3)
age    = c1.number_input("Âge", min_value=10, max_value=100, value=28)
poids  = c2.number_input("Poids (kg)", min_value=30.0, max_value=200.0, value=70.0, step=0.5)
sexe   = c3.selectbox("Sexe", ["H", "F", "Autre"])

# Latéralité : SANS ambidextre
lat    = st.selectbox("Latéralité", ["droitier", "gaucher"])

# Niveau de pratique : ajoute "très avancé"
niv    = st.selectbox("Niveau de pratique", ["débutant", "intermédiaire", "avancé", "très avancé"])

one_rm = st.number_input("1RM de référence (kg)", min_value=10.0, max_value=300.0, value=100.0, step=0.5)

# ====== Niveaux de force (jusqu’à 10 décimales) ======
st.markdown("### Niveaux de force normalisés (0–100, jusqu’à 10 décimales)")
cL, cR = st.columns(2)

hip_ext_G   = cL.number_input("Extenseurs de hanche — Gauche",  min_value=0.0, max_value=100.0, value=55.0, step=0.0000000001, format="%.10f")
hip_ext_D   = cR.number_input("Extenseurs de hanche — Droite",  min_value=0.0, max_value=100.0, value=60.0, step=0.0000000001, format="%.10f")
knee_flex_G = cL.number_input("Fléchisseurs de genou — Gauche", min_value=0.0, max_value=100.0, value=60.0, step=0.0000000001, format="%.10f")
knee_flex_D = cR.number_input("Fléchisseurs de genou — Droite", min_value=0.0, max_value=100.0, value=62.0, step=0.0000000001, format="%.10f")
knee_ext_G  = cL.number_input("Extenseurs de genou — Gauche",   min_value=0.0, max_value=100.0, value=52.0, step=0.0000000001, format="%.10f")
knee_ext_D  = cR.number_input("Extenseurs de genou — Droite",   min_value=0.0, max_value=100.0, value=58.0, step=0.0000000001, format="%.10f")

# ====== Bouton d'action ======
if st.button("Proposer l’intervention optimale"):
    with st.spinner("Calcul en cours..."):
        # normalisation robuste (évite espaces/variantes vus dans les CSV)
        lat_norm = lat.strip().lower()
        niv_norm = niv.strip().lower()

        pct_rm, series_sem, gains, score = reco_depuis_inputs(
            age, poids, sexe, lat_norm, niv_norm, one_rm,
            hip_ext_G, hip_ext_D, knee_flex_G, knee_flex_D, knee_ext_G, knee_ext_D
        )

    st.success(f"%RM recommandé : **{pct_rm:.1f}%**  |  Séries/semaine : **{series_sem:.1f}**  |  Score équilibre : {score:.2f}")
    st.markdown("#### Gains estimés sur 6 semaines (%)")
    for m, g in zip(MUSCLES, gains):
        st.write(f"- **{m}** : {g:.2f} %")

    st.caption("Note : %RM et séries/semaine sont globaux (équilibrés) ; les gains varient par groupe.")
