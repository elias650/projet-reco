# app.py
import streamlit as st
import pandas as pd

# === Imports du modèle ===
try:
    from modele import MUSCLES, reco_depuis_inputs, reco3_depuis_inputs
    _HAS_RECO3 = True
except Exception as e:
    # Fallback si la fonction 3 plans n'existe pas encore dans modele.py
    try:
        from modele import MUSCLES, reco_depuis_inputs
        _HAS_RECO3 = False
    except Exception as ee:
        st.error("Impossible d'importer les fonctions depuis modele.py. Vérifie que le fichier est bien présent et à jour.")
        st.exception(ee)
        st.stop()

# === Config UI ===
st.set_page_config(page_title="Reco d'intervention — Cohorte 1", layout="wide")
st.title("Recommandation d’intervention — Cohorte 1")

with st.expander("⚙️ Diagnostic rapide", expanded=False):
    st.write("`reco_depuis_inputs` :", "✅" if 'reco_depuis_inputs' in dir() else "❌")
    st.write("`reco3_depuis_inputs` :", "✅" if _HAS_RECO3 else "❌ (mode 3 plans indisponible)")

st.markdown(
    "Saisis les **informations initiales** et les **forces à J0**. "
    "Choisis le mode de recommandation : **plan unique (Équilibre)** ou **trois plans**."
)

# === Saisie patient ===
with st.expander("Informations initiales", expanded=True):
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        age = st.number_input("Âge (années)", min_value=14, max_value=100, value=22)
        sexe = st.selectbox("Sexe", ["H", "F"], index=0)
        poids = st.number_input("Poids (kg)", min_value=30.0, max_value=200.0, value=70.0, step=0.1)
    with c2:
        lateralite = st.selectbox("Latéralité", ["droitier", "gaucher", "ambidextre"], index=0)
        niveau = st.selectbox("Niveau", ["débutant", "intermédiaire", "avancé", "très avancé"], index=1)
        one_rm = st.number_input("1RM (kg)", min_value=10.0, max_value=300.0, value=100.0, step=0.1)

with st.expander("Forces à J0 (moyennes par groupe musculaire)", expanded=True):
    cA, cB, cC = st.columns(3)
    with cA:
        hip_ext_G = st.number_input("Ext hanche G — J0", min_value=0.0, value=5.000000, step=0.001, format="%.6f", key="hg")
        hip_ext_D = st.number_input("Ext hanche D — J0", min_value=0.0, value=5.000000, step=0.001, format="%.6f", key="hd")
    with cB:
        knee_flex_G = st.number_input("Flx genou G — J0", min_value=0.0, value=3.000000, step=0.001, format="%.6f", key="kfg")
        knee_flex_D = st.number_input("Flx genou D — J0", min_value=0.0, value=3.000000, step=0.001, format="%.6f", key="kfd")
    with cC:
        knee_ext_G = st.number_input("Ext genou G — J0", min_value=0.0, value=5.000000, step=0.001, format="%.6f", key="keg")
        knee_ext_D = st.number_input("Ext genou D — J0", min_value=0.0, value=5.000000, step=0.001, format="%.6f", key="ked")

# === Mode ===
choices = ["Plan unique (Équilibre)"]
if _HAS_RECO3:
    choices.append("Trois plans (Équilibre / Performance / Stabilité)")
mode = st.radio("Mode de recommandation", choices, horizontal=True)

st.divider()

# === Helpers UI ===
def _plan_card(titre: str, plan: dict, aide: str = ""):
    st.markdown(f"### {titre}")
    if aide:
        st.caption(aide)
    top = st.columns(3)
    top[0].metric("%RM", f"{plan['rm']:.1f}")
    top[1].metric("Séries/sem", f"{plan['series']:.1f}")
    top[2].metric("Score (mean−λ·std)", f"{plan['score']:.2f}")
    mid = st.columns(2)
    mid[0].metric("Gain moyen (%)", f"{plan['mean']:.2f}")
    mid[1].metric("Écart-type (pts)", f"{plan['std']:.2f}")
    dfp = pd.DataFrame({"Muscle": MUSCLES, "Gain prédit (%)": [round(g, 2) for g in plan["gains"]]})
    st.dataframe(dfp, hide_index=True, use_container_width=True)

# === Bouton calcul ===
if st.button("Calculer la/les recommandations", type="primary"):
    with st.spinner("Calcul en cours…"):
        try:
            if mode.startswith("Plan unique"):
                pct_rm, series, gains, score = reco_depuis_inputs(
                    float(age), float(poids), str(sexe), str(lateralite), str(niveau), float(one_rm),
                    float(hip_ext_G), float(hip_ext_D), float(knee_flex_G), float(knee_flex_D),
                    float(knee_ext_G), float(knee_ext_D)
                )
                st.subheader("Plan Équilibre (mean − λ·std)")
                c1, c2, c3 = st.columns(3)
                c1.metric("%RM recommandé", f"{pct_rm:.1f}")
                c2.metric("Séries/semaine", f"{series:.1f}")
                c3.metric("Score (mean−λ·std)", f"{score:.2f}")
                df = pd.DataFrame({"Muscle": MUSCLES, "Gain prédit (%)": [round(g, 2) for g in gains]})
                st.dataframe(df, hide_index=True, use_container_width=True)

            else:
                # 3 plans
                plans = reco3_depuis_inputs(
                    float(age), float(poids), str(sexe), str(lateralite), str(niveau), float(one_rm),
                    float(hip_ext_G), float(hip_ext_D), float(knee_flex_G), float(knee_flex_D),
                    float(knee_ext_G), float(knee_ext_D)
                )

                col1, col2, col3 = st.columns(3)
                with col1:
                    _plan_card("Option 1 — Équilibre", plans["equilibre"], "Optimise mean − λ·std (λ calibré).")
                with col2:
                    _plan_card("Option 2 — Performance", plans["performance"], "Maximise la moyenne des gains.")
                with col3:
                    _plan_card("Option 3 — Stabilité", plans["stabilite"], "Minimise l’écart-type (départage par la moyenne).")

                st.markdown("#### Comparatif synthétique")
                comp = pd.DataFrame([
                    {"Option": "Équilibre", "%RM": round(plans["equilibre"]["rm"], 1),
                     "Séries/sem": round(plans["equilibre"]["series"], 1),
                     "Gain moyen (%)": round(plans["equilibre"]["mean"], 2),
                     "Écart-type (pts)": round(plans["equilibre"]["std"], 2),
                     "Score (mean−λ·std)": round(plans["equilibre"]["score"], 2)},
                    {"Option": "Performance", "%RM": round(plans["performance"]["rm"], 1),
                     "Séries/sem": round(plans["performance"]["series"], 1),
                     "Gain moyen (%)": round(plans["performance"]["mean"], 2),
                     "Écart-type (pts)": round(plans["performance"]["std"], 2),
                     "Score (mean−λ·std)": round(plans["performance"]["score"], 2)},
                    {"Option": "Stabilité", "%RM": round(plans["stabilite"]["rm"], 1),
                     "Séries/sem": round(plans["stabilite"]["series"], 1),
                     "Gain moyen (%)": round(plans["stabilite"]["mean"], 2),
                     "Écart-type (pts)": round(plans["stabilite"]["std"], 2),
                     "Score (mean−λ·std)": round(plans["stabilite"]["score"], 2)},
                ])
                st.dataframe(comp, hide_index=True, use_container_width=True)

        except Exception as e:
            st.error("Une erreur est survenue pendant le calcul. Vérifie que `modele.py` est bien à jour et rechargé.")
            st.exception(e)

st.markdown("---")
st.caption("Les recommandations sont une aide à la décision issue de la cohorte 1 et doivent être interprétées par le praticien.")
