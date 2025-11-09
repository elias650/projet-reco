# app.py
import streamlit as st
import pandas as pd
from modele import (
    MUSCLES,
    reco_depuis_inputs,   # plan unique (Équilibre)
    reco3_depuis_inputs,  # trois plans (Équilibre / Performance / Stabilité)
)

st.set_page_config(page_title="Reco d'intervention", layout="wide")

st.title("Recommandation d'intervention — Cohorte 1")

st.markdown(
    "Choisis le mode de recommandation puis saisis les caractéristiques du patient et ses forces à **J0**."
)

# -------------------- Saisie patient --------------------
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
        hip_ext_G = st.number_input("Ext hanche G — J0", min_value=0.0, value=5.0, step=0.001, format="%.6f")
        hip_ext_D = st.number_input("Ext hanche D — J0", min_value=0.0, value=5.0, step=0.001, format="%.6f")
    with cB:
        knee_flex_G = st.number_input("Flx genou G — J0", min_value=0.0, value=3.0, step=0.001, format="%.6f")
        knee_flex_D = st.number_input("Flx genou D — J0", min_value=0.0, value=3.0, step=0.001, format="%.6f")
    with cC:
        knee_ext_G = st.number_input("Ext genou G — J0", min_value=0.0, value=5.0, step=0.001, format="%.6f")
        knee_ext_D = st.number_input("Ext genou D — J0", min_value=0.0, value=5.0, step=0.001, format="%.6f")

# -------------------- Mode --------------------
mode = st.radio(
    "Mode de recommandation",
    ["Plan unique (Équilibre)", "Trois plans (Équilibre / Performance / Stabilité)"],
    horizontal=True,
)

# -------------------- Calcul --------------------
if st.button("Calculer la/les recommandations"):
    if mode.startswith("Plan unique"):
        pct_rm, series, gains, score = reco_depuis_inputs(
            age, poids, sexe, lateralite, niveau, one_rm,
            hip_ext_G, hip_ext_D, knee_flex_G, knee_flex_D, knee_ext_G, knee_ext_D
        )
        st.subheader("Plan Équilibre (mean − λ·std)")
        mcol1, mcol2 = st.columns([1,1])
        with mcol1:
            st.metric("%RM recommandé", f"{pct_rm:.1f}")
            st.metric("Séries/semaine", f"{series:.1f}")
        with mcol2:
            st.metric("Score (mean−λ·std)", f"{score:.2f}")
        df = pd.DataFrame({"Muscle": MUSCLES, "Gain prédit (%)": [round(g,2) for g in gains]})
        st.dataframe(df, hide_index=True, use_container_width=True)

    else:
        plans = reco3_depuis_inputs(
            age, poids, sexe, lateralite, niveau, one_rm,
            hip_ext_G, hip_ext_D, knee_flex_G, knee_flex_D, knee_ext_G, knee_ext_D
        )

        # Mise en forme des trois plans
        def plan_card(titre, plan, help_text=""):
            st.markdown(f"### {titre}")
            if help_text:
                st.caption(help_text)
            top = st.columns(3)
            top[0].metric("%RM", f"{plan['rm']:.1f}")
            top[1].metric("Séries/sem", f"{plan['series']:.1f}")
            top[2].metric("Score (mean−λ·std)", f"{plan['score']:.2f}")
            mid = st.columns(2)
            mid[0].metric("Gain moyen (%)", f"{plan['mean']:.2f}")
            mid[1].metric("Écart-type (pts)", f"{plan['std']:.2f}")
            dfp = pd.DataFrame({"Muscle": MUSCLES, "Gain prédit (%)": [round(g,2) for g in plan["gains"]]})
            st.dataframe(dfp, hide_index=True, use_container_width=True)

        c1, c2, c3 = st.columns(3)
        with c1:
            plan_card("Option 1 — Équilibre", plans["equilibre"], "Optimise mean − λ·std (λ calibré).")
        with c2:
            plan_card("Option 2 — Performance", plans["performance"], "Maximise la moyenne des gains.")
        with c3:
            plan_card("Option 3 — Stabilité", plans["stabilite"], "Minimise l’écart-type (départage par la moyenne).")

        # Tableau comparatif
        comp = pd.DataFrame([
            {
                "Option": "Équilibre",
                "%RM": round(plans["equilibre"]["rm"],1),
                "Séries/sem": round(plans["equilibre"]["series"],1),
                "Gain moyen (%)": round(plans["equilibre"]["mean"],2),
                "Écart-type (pts)": round(plans["equilibre"]["std"],2),
                "Score (mean−λ·std)": round(plans["equilibre"]["score"],2),
            },
            {
                "Option": "Performance",
                "%RM": round(plans["performance"]["rm"],1),
                "Séries/sem": round(plans["performance"]["series"],1),
                "Gain moyen (%)": round(plans["performance"]["mean"],2),
                "Écart-type (pts)": round(plans["performance"]["std"],2),
                "Score (mean−λ·std)": round(plans["performance"]["score"],2),
            },
            {
                "Option": "Stabilité",
                "%RM": round(plans["stabilite"]["rm"],1),
                "Séries/sem": round(plans["stabilite"]["series"],1),
                "Gain moyen (%)": round(plans["stabilite"]["mean"],2),
                "Écart-type (pts)": round(plans["stabilite"]["std"],2),
                "Score (mean−λ·std)": round(plans["stabilite"]["score"],2),
            },
        ])
        st.markdown("#### Comparatif synthétique")
        st.dataframe(comp, hide_index=True, use_container_width=True)

st.markdown("---")
st.caption("Les données affichées proviennent des modèles entraînés sur la cohorte 1. Les recommandations sont des aides à la décision et doivent être interprétées par le praticien.")
