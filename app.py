# app.py — version 2-plans (équilibré + performance) — champs force à 3 décimales

import streamlit as st

# ---------------------------------------------------------
# Cache module + reload pour éviter le cache de Streamlit Cloud
# Incrémente 'version' si tu modifies fortement modele.py pour casser le cache.
# ---------------------------------------------------------
@st.cache_resource(show_spinner=True)
def get_model_module(version: int = 3):
    import importlib
    import modele  # ton fichier modele.py
    importlib.reload(modele)
    return modele

modele = get_model_module()
MUSCLES = modele.MUSCLES

# On essaie la nouvelle API (2 plans). Si absente, fallback sur l'historique (1 plan).
reco2_depuis_inputs = getattr(modele, "reco2_depuis_inputs", None)
reco_depuis_inputs  = getattr(modele, "reco_depuis_inputs",  None)

# ---------------------------------------------------------
# UI de base
# ---------------------------------------------------------
st.set_page_config(page_title="Reco Renforcement (6 semaines)", layout="centered")
st.title("Recommandation d’intervention — 6 semaines")

st.caption(
    "Renseigne les informations de base et les niveaux de force (0–100). "
    "L’application propose jusqu’à **deux plans** : un **Équilibré** (mean − λ·std) et un **Performance** "
    "(maximisation du gain moyen sous contrainte d’équilibre)."
)

# ---------------------------------------------------------
# Informations de base
# ---------------------------------------------------------
c1, c2, c3 = st.columns(3)
age    = c1.number_input("Âge", min_value=10, max_value=100, value=28)
poids  = c2.number_input("Poids (kg)", min_value=30.0, max_value=200.0, value=70.0, step=0.5)
sexe   = c3.selectbox("Sexe", ["H", "F", "Autre"])

# Latéralité : sans "ambidextre"
lat    = st.selectbox("Latéralité", ["droitier", "gaucher"])

# Niveau de pratique : inclut "très avancé"
niv    = st.selectbox("Niveau de pratique", ["débutant", "intermédiaire", "avancé", "très avancé"])

one_rm = st.number_input("1RM de référence (kg)", min_value=10.0, max_value=300.0, value=100.0, step=0.5)

# ---------------------------------------------------------
# Niveaux de force (3 décimales)
# ---------------------------------------------------------
with st.container(border=True):
    st.subheader("Niveaux de force normalisés (0–100, précision 0,001)")
    cL, cR = st.columns(2)

    hip_ext_G   = cL.number_input("Extenseurs de hanche — Gauche",  min_value=0.0, max_value=100.0, value=55.0, step=0.001, format="%.3f")
    hip_ext_D   = cR.number_input("Extenseurs de hanche — Droite",  min_value=0.0, max_value=100.0, value=60.0, step=0.001, format="%.3f")
    knee_flex_G = cL.number_input("Fléchisseurs de genou — Gauche", min_value=0.0, max_value=100.0, value=60.0, step=0.001, format="%.3f")
    knee_flex_D = cR.number_input("Fléchisseurs de genou — Droite", min_value=0.0, max_value=100.0, value=62.0, step=0.001, format="%.3f")
    knee_ext_G  = cL.number_input("Extenseurs de genou — Gauche",   min_value=0.0, max_value=100.0, value=52.0, step=0.001, format="%.3f")
    knee_ext_D  = cR.number_input("Extenseurs de genou — Droite",   min_value=0.0, max_value=100.0, value=58.0, step=0.001, format="%.3f")

st.divider()

# ---------------------------------------------------------
# Bouton d'action : propose 2 plans si dispo, sinon fallback à 1 plan
# ---------------------------------------------------------
if st.button("Proposer l’intervention optimale"):
    with st.spinner("Calcul en cours..."):
        # normalisation simple (évite espaces/variantes)
        lat_norm = lat.strip().lower()
        niv_norm = niv.strip().lower()

        if callable(reco2_depuis_inputs):
            # --- NOUVELLE API : deux plans ---
            res = reco2_depuis_inputs(
                age, poids, sexe, lat_norm, niv_norm, one_rm,
                hip_ext_G, hip_ext_D, knee_flex_G, knee_flex_D, knee_ext_G, knee_ext_D
            )
            plan_eq  = res.get("equilibre", {})
            plan_pf  = res.get("performance", {})

            c1, c2 = st.columns(2)

            # ------ Option 1 : Équilibré ------
            with c1:
                st.subheader("Option 1 — Équilibré")
                if plan_eq.get("rm") is None:
                    st.info("Aucun plan équilibré disponible pour ces entrées.")
                else:
                    st.success(f"%RM : **{plan_eq['rm']:.1f}%**  •  Séries/sem : **{plan_eq['series']:.1f}**")
                    st.caption(
                        f"Gain moyen estimé : **{plan_eq['mean']:.2f}%**  •  "
                        f"Déséquilibre (écart-type) : **{plan_eq['std']:.2f}**  •  "
                        f"Score : **{plan_eq['score']:.2f}**"
                    )
                    st.markdown("**Gains estimés par groupe (%)**")
                    gains_eq = plan_eq.get("gains", [])
                    for m, g in zip(MUSCLES, gains_eq):
                        st.write(f"- **{m}** : {g:.2f} %")

            # ------ Option 2 : Performance ------
            with c2:
                st.subheader("Option 2 — Performance (contrainte)")
                if plan_pf.get("rm") is None:
                    st.info("Aucun plan performance disponible répondant aux contraintes.")
                else:
                    st.success(f"%RM : **{plan_pf['rm']:.1f}%**  •  Séries/sem : **{plan_pf['series']:.1f}**")
                    st.caption(
                        f"Gain moyen estimé : **{plan_pf['mean']:.2f}%**  •  "
                        f"Déséquilibre (écart-type) : **{plan_pf['std']:.2f}**  •  "
                        f"Score : **{plan_pf['score']:.2f}**"
                    )
                    st.markdown("**Gains estimés par groupe (%)**")
                    gains_pf = plan_pf.get("gains", [])
                    for m, g in zip(MUSCLES, gains_pf):
                        st.write(f"- **{m}** : {g:.2f} %")

            st.caption(
                "Note : le plan **Équilibré** maximise *mean − λ·std* ; "
                "le plan **Performance** maximise le gain moyen sous contrainte d'équilibre (τ) et reste distinct de l'option 1."
            )

        elif callable(reco_depuis_inputs):
            # --- FALLBACK : ancienne API 1 plan ---
            pct_rm, series_sem, gains, score = reco_depuis_inputs(
                age, poids, sexe, lat_norm, niv_norm, one_rm,
                hip_ext_G, hip_ext_D, knee_flex_G, knee_flex_D, knee_ext_G, knee_ext_D
            )
            st.subheader("Option unique (compatibilité)")
            st.success(f"%RM recommandé : **{pct_rm:.1f}%**  |  Séries/semaine : **{series_sem:.1f}**  |  Score équilibre : {score:.2f}")
            st.markdown("**Gains estimés sur 6 semaines (%)**")
            for m, g in zip(MUSCLES, gains):
                st.write(f"- **{m}** : {g:.2f} %")
            st.caption("Mode compatibilité : une seule proposition (ancienne API).")

        else:
            st.error("Aucune fonction de recommandation détectée dans `modele.py` (ni 2 plans, ni 1 plan). Vérifie les noms des fonctions.")

# ---------------------------------------------------------
# Fin
# ---------------------------------------------------------
