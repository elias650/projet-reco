# ==========================================================
# MODELE D'APPRENTISSAGE AUTOMATISÉ — RECO D'INTERVENTION
# Cohorte 1 — Elias Simon — Novembre 2025
# ==========================================================

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor

# ==========================================================
# === PARAMETRES (avec anonymes auto si dispo) =============
# ==========================================================
import os

DATA_DIR = "data"
ANON_DIR = os.path.join(DATA_DIR, "anonymes")

def csv_path(name: str) -> str:
    """Utilise data/anonymes/<name> si le fichier existe, sinon data/<name>."""
    anon = os.path.join(ANON_DIR, name)
    return anon if os.path.exists(anon) else os.path.join(DATA_DIR, name)

CSV_INFOS = csv_path("information initiales.csv")
CSV_J0    = csv_path("Mesures J0.csv")
CSV_J6    = csv_path("Mesures J+6 semaines.csv")
CSV_INT   = csv_path("Interventions.csv")

MUSCLES = ["Ext hanche D","Ext hanche G","Flx genou D","Flx genou G","Ext genou D","Ext genou G"]
TEST_SUBJECTS = ['Alexia','Romain','Elise']   # sous-ensemble test
CSV_SORTIE_RECO = "recommandations.csv"       # fichier de sauvegarde des recos

# ==========================================================
# === LECTURE CSV (encodage Windows) =======================
# ==========================================================
info  = pd.read_csv(CSV_INFOS, sep=';', decimal=',', encoding='latin1')
j0    = pd.read_csv(CSV_J0,    sep=';', decimal=',', encoding='latin1')
j6    = pd.read_csv(CSV_J6,    sep=';', decimal=',', encoding='latin1')
interv= pd.read_csv(CSV_INT,   sep=';', decimal=',', encoding='latin1')

print("=== Fichiers chargés ===")
print(f"Information initiales : {info.shape}")
print(f"Mesures J0 : {j0.shape}")
print(f"Mesures J+6 semaines : {j6.shape}")
print(f"Interventions : {interv.shape}")

# ==========================================================
# === NORMALISATION DES TABLES =============================
# ==========================================================
def normalize_df(df):
    # strip des noms de colonnes
    df.columns = [str(c).strip() for c in df.columns]
    # si 'Sujets' absent, prendre la 1re colonne comme 'Sujets'
    if 'Sujets' not in df.columns:
        df.rename(columns={df.columns[0]: 'Sujets'}, inplace=True)
    # normalise la colonne Sujets
    df['Sujets'] = df['Sujets'].astype(str).str.strip()
    # supprime lignes vides/NaN et doublons
    df = df[df['Sujets'].notna() & (df['Sujets'] != "")]
    df = df.drop_duplicates(subset=['Sujets'], keep='first')
    return df

info   = normalize_df(info)
j0     = normalize_df(j0)
j6     = normalize_df(j6)
interv = normalize_df(interv)
# harmonise nom colonne d'intervention
interv.rename(columns={'Séries/sem':'Séries/semaine','Séries/semai':'Séries/semaine'}, inplace=True)

print("\n--- Après normalisation ---")
print("Colonnes Interventions :", list(interv.columns))
print("Effectifs (info, j0, j6, interv) :", len(info), len(j0), len(j6), len(interv))

# diagnostics de correspondance des sujets
set_info = set(info['Sujets']); set_j0 = set(j0['Sujets']); set_j6 = set(j6['Sujets']); set_int = set(interv['Sujets'])
print("Sujets manquants dans Interventions :", sorted(set_info - set_int))
print("Sujets en trop dans Interventions   :", sorted(set_int - set_info))

# ==========================================================
# === TEST_SUBJECTS robuste (IDs anonymes) =================
# ==========================================================
# On reconstruit automatiquement une liste de test compatible (Sujet1, Sujet2, Sujet3, ...)
# si les anciens prénoms n'existent plus.
all_ids = sorted(set(info['Sujets'].astype(str).str.strip()))
DEFAULT_TEST = [f"Sujet{i}" for i in range(1, 50)]  # large au cas où
TEST_SUBJECTS = [x for x in DEFAULT_TEST if x in all_ids][:3] or all_ids[:min(3, len(all_ids))]
print("TEST utilisé :", TEST_SUBJECTS)

# ==========================================================
# === FUSIONS & CONVERSION NUMERIQUE =======================
# ==========================================================
# conversion numérique des mesures J0/J6 si besoin
for m in MUSCLES:
    if m in j0.columns: j0[m] = pd.to_numeric(j0[m], errors='coerce')
    if m in j6.columns: j6[m] = pd.to_numeric(j6[m], errors='coerce')

base = info.merge(j0, on="Sujets").merge(j6, on="Sujets", suffixes=("_J0","_J6"))
# 'left' pour ne pas perdre de sujets si Interventions a une ligne surnuméraire
data = base.merge(interv, on="Sujets", how="left")

# gains %
for m in MUSCLES:
    data[f"Gain_{m}_%"] = 100 * (data[f"{m}_J6"] - data[f"{m}_J0"]) / data[f"{m}_J0"]

# %RM -> numérique moyen (pour aider l'apprentissage)
def rm_to_numeric(v):
    if isinstance(v, str):
        v = v.strip()
        if v == '40-60': return 50.0
        if v == '60-75': return 67.5
        if v == '75-85': return 80.0
        if v == '80-90': return 85.0
    try:
        return float(v)
    except:
        return np.nan

data['%RM_num'] = data['%RM'].apply(rm_to_numeric)

# ==========================================================
# === FEATURES (BASELINE UNIQUEMENT) & PIPELINE ============
# ==========================================================
# Cibles = gains (%) calculés à partir de J0 et J6
Y_cols = [f"Gain_{m}_%" for m in MUSCLES]

# 1) Colonnes d'infos initiales (depuis 'info')
info_cols = [c for c in info.columns if c != 'Sujets']  # ex: âge, sexe, poids, latéralité, niveau, 1RM
# 2) Colonnes de force à J0
j0_cols = [f"{m}_J0" for m in MUSCLES if f"{m}_J0" in data.columns]
# 3) Variables d’intervention (explorées par l’optimiseur)
interv_cols = [c for c in ['%RM', '%RM_num', 'Séries/semaine'] if c in data.columns]

# >>>> FEATURES DÉFINITIVES = BASELINE UNIQUEMENT (PAS DE J6) <<<<
feat_cols = info_cols + j0_cols + interv_cols
assert not any(col.endswith('_J6') for col in feat_cols), "Les colonnes J6 ne doivent pas être dans feat_cols."

# Préprocesseur (numériques vs catégorielles)
num_cols = [c for c in feat_cols if data[c].dtype != 'object']
cat_cols = [c for c in feat_cols if data[c].dtype == 'object']

prep = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
])

# ==========================================================
# === ENTRAINEMENT : Ridge vs RandomForest =================
# ==========================================================
loo = LeaveOneOut()
MODELS = {}
rows = []

def make_pipe(base_estimator):
    return Pipeline([('prep', prep), ('model', base_estimator)])

for m in MUSCLES:
    ycol = f'Gain_{m}_%'
    if ycol not in data.columns: 
        continue
    X = data[feat_cols]
    y = pd.to_numeric(data[ycol], errors='coerce')

    # Modèle 1 : Ridge (linéaire robuste petits n)
    ridge = RidgeCV(alphas=np.logspace(-3, 3, 13))
    pipe_ridge = make_pipe(ridge)
    y_pred_ridge = cross_val_predict(pipe_ridge, X, y, cv=loo)
    r2_ridge = r2_score(y, y_pred_ridge)
    mae_ridge = mean_absolute_error(y, y_pred_ridge)
    rmse_ridge = mean_squared_error(y, y_pred_ridge) ** 0.5

    # Modèle 2 : RandomForest (non linéaire)
    rf_est = RandomForestRegressor(n_estimators=600, random_state=42)
    pipe_rf = make_pipe(rf_est)
    y_pred_rf = cross_val_predict(pipe_rf, X, y, cv=loo)
    r2_rf = r2_score(y, y_pred_rf)
    mae_rf = mean_absolute_error(y, y_pred_rf)
    rmse_rf = mean_squared_error(y, y_pred_rf) ** 0.5

    # Choix du meilleur
    if r2_ridge >= r2_rf:
        best_name, best_pipe, best_r2, best_mae, best_rmse = 'Ridge', pipe_ridge, r2_ridge, mae_ridge, rmse_ridge
    else:
        best_name, best_pipe, best_r2, best_mae, best_rmse = 'RF', pipe_rf, r2_rf, mae_rf, rmse_rf

    MODELS[m] = best_pipe.fit(X, y)
    rows.append({'muscle': m, 'n': len(y), 'modele': best_name,
                 'MAE': best_mae, 'RMSE': best_rmse, 'R2': best_r2})

perf = pd.DataFrame(rows)
print("\n=== Performances LOOCV (meilleur modèle par muscle) ===")
print(perf.to_string(index=False))

# ==========================================================
# === SPLIT TRAIN/TEST EXPLICITE ===========================
# ==========================================================
print("\n=== Split train/test explicite ===")
train = data[~data['Sujets'].isin(TEST_SUBJECTS)].copy()
test  = data[ data['Sujets'].isin(TEST_SUBJECTS)].copy()
print("TEST demandé :", TEST_SUBJECTS)
print("TEST présent :", test['Sujets'].tolist())
if len(TEST_SUBJECTS) != len(test):
    print("⚠️ Tous les sujets de TEST_SUBJECTS ne sont pas présents après fusion/dropna().")
    if len(test) > 0:
        print("Lignes test avec valeurs manquantes par cible :")
        print(test[['Sujets'] + Y_cols].isna().groupby(test['Sujets']).sum())

for m in MUSCLES:
    ycol = f'Gain_{m}_%'
    if ycol not in data.columns: 
        continue
    Xtr, Xte = train[feat_cols], test[feat_cols]
    ytr, yte = train[ycol],  test[ycol]
    model = MODELS[m]
    if len(yte) > 0:
        yhat = model.predict(Xte)
        rmse = mean_squared_error(yte, yhat) ** 0.5
        print(f"[TEST] {m}: R2={r2_score(yte,yhat):.2f} | MAE={mean_absolute_error(yte,yhat):.2f} | RMSE={rmse:.2f} | n={len(yte)}")
    else:
        print(f"[TEST] {m}: n=0 (aucun sujet test disponible)")

# ==========================================================
# === RECOMMANDATION OPTIMALE & ÉQUILIBRÉE (DEMO) ==========
# ==========================================================
print("\n>>> Patient utilisé pour la recommandation : (échantillon aléatoire)")
patient_row = data.sample(1).iloc[0]
print({k: patient_row[k] for k in list(data.columns)[:8]}, "...")

def recommander(patient_row):
    # grille raisonnable (tu peux affiner)
    rm_values = np.linspace(40, 90, 11)       # 40,45,...,90
    series_values = np.linspace(4, 12, 9)     # 4,5,...,12
    best_score, best_rm, best_series = -1e9, None, None
    best_preds = None

    for rm in rm_values:
        for s in series_values:
            row = patient_row.copy()
            # on écrase uniquement les features utiles à l'intervention
            row['%RM_num'] = rm
            # si le modèle a encodé %RM (catégoriel), on peut garder la valeur en place (%RM) — peu importe ici
            row['Séries/semaine'] = s

            x = pd.DataFrame([row[feat_cols]])
            preds = []
            for m in MUSCLES:
                preds.append(MODELS[m].predict(x)[0])
            preds = np.array(preds)

            # Score "équilibré" = moyenne - écart-type (lambda=1)
            score = preds.mean() - preds.std()
            if score > best_score:
                best_score, best_rm, best_series, best_preds = score, rm, s, preds

    return best_rm, best_series, best_preds, best_score

print("\n>>> Début optimisation de l'intervention")
opt_rm, opt_ser, opt_preds, opt_score = recommander(patient_row)
print(">>> Optimisation terminée\n")

print("=== Recommandation OPTIMALE & ÉQUILIBRÉE ===")
print(f"%RM recommandé : {opt_rm:.1f}")
print(f"Séries/semaine recommandé : {opt_ser:.1f}")
print("Gains prédits par muscle (%):")
for m, g in zip(MUSCLES, opt_preds):
    print(f"  - {m}: {g:.2f}")
print(f"Score global (moy - écart-type) : {opt_score:.2f}")

# ==========================================================
# === INTERFACE CONSOLE POUR SAISIR UN NOUVEAU PATIENT =====
# ==========================================================
def _safe_input(prompt, default=None, dtype=str):
    """Lecture robuste : accepte décimales avec virgule, vide => default."""
    s = input(prompt).strip()
    if s == "":
        return default
    if dtype in (float, int):
        s = s.replace(",", ".")
        try:
            x = float(s)
            return int(x) if dtype is int else x
        except:
            print("  ⚠️ Entrée non valide, on garde la valeur par défaut.")
            return default
    return s

def _mode_or_first(series):
    vc = series.dropna().astype(str).str.strip().value_counts()
    return vc.index[0] if len(vc)>0 else None

def _default_patient_from_data(df):
    """Construit un 'patient-type' (médian pour numériques, modalité la plus fréquente pour catégorielles)."""
    d = {}
    # infos initiales
    d['Sujets'] = 'Nouveau_patient'
    d['âge'] = float(df['âge'].median()) if 'âge' in df.columns else 25.0
    d['sexe'] = _mode_or_first(df['sexe']) if 'sexe' in df.columns else 'H'
    d['poids'] = float(df['poids'].median()) if 'poids' in df.columns else 70.0
    d['latéralité'] = _mode_or_first(df['latéralité']) if 'latéralité' in df.columns else 'droitier'
    d['niveau'] = _mode_or_first(df['niveau']) if 'niveau' in df.columns else 'intermédiaire'
    d['1RM'] = float(df['1RM'].median()) if '1RM' in df.columns else 100.0
    # forces J0
    for m in ["Ext hanche D","Ext hanche G","Flx genou D","Flx genou G","Ext genou D","Ext genou G"]:
        col = f"{m}_J0"
        if col in df.columns:
            d[col] = float(pd.to_numeric(df[col], errors='coerce').median())
        else:
            d[col] = 4.0
    # placeholders d'intervention (seront écrasés pendant l'optimisation)
    d['%RM'] = '60-75'
    d['%RM_num'] = 67.5
    d['Séries/semaine'] = 6.0
    return d

def saisir_patient_depuis_console(df):
    """Affiche les valeurs par défaut (issues du dataset) et propose de les modifier."""
    defaults = _default_patient_from_data(df)
    print("\n--- Saisie d'un nouveau patient (laisser vide pour garder la valeur par défaut) ---")
    p = {}

    p['Sujets'] = _safe_input(f"Nom du sujet [{defaults['Sujets']}]: ", defaults['Sujets'], str)
    p['âge'] = _safe_input(f"Âge [{defaults['âge']}]: ", defaults['âge'], float)
    p['sexe'] = _safe_input(f"Sexe (H/F) [{defaults['sexe']}]: ", defaults['sexe'], str)
    p['poids'] = _safe_input(f"Poids (kg) [{defaults['poids']}]: ", defaults['poids'], float)
    p['latéralité'] = _safe_input(f"Latéralité (droitier/gaucher) [{defaults['latéralité']}]: ", defaults['latéralité'], str)
    p['niveau'] = _safe_input(f"Niveau (débutant/intermédiaire/avancé) [{defaults['niveau']}]: ", defaults['niveau'], str)
    p['1RM'] = _safe_input(f"1RM (kg) [{defaults['1RM']}]: ", defaults['1RM'], float)

    # Mesures J0 (accepte virgules)
    for m in ["Ext hanche D","Ext hanche G","Flx genou D","Flx genou G","Ext genou D","Ext genou G"]:
        col = f"{m}_J0"
        p[col] = _safe_input(f"{col} [{defaults[col]}]: ", defaults[col], float)

    # Placeholders (seront écrasés pour la reco)
    p['%RM'] = '60-75'
    p['%RM_num'] = 67.5
    p['Séries/semaine'] = 6.0

    # On convertit en Series pour compat avec recommander(patient_row)
    return pd.Series(p)

def sauvegarder_recommandation(path_csv, sujet, opt_rm, opt_ser, opt_preds, opt_score):
    """Append (ou crée) un CSV de recommandations ; séparateur ';', décimale ','."""
    out_row = {
        'Sujet': sujet,
        '%RM_reco': round(float(opt_rm), 1),
        'Series_semaine_reco': round(float(opt_ser), 1),
        'Score_equilibre': round(float(opt_score), 2),
    }
    for m, g in zip(MUSCLES, opt_preds):
        out_row[f'Gain_pred_{m}_%'] = round(float(g), 2)

    df_out = pd.DataFrame([out_row])
    if os.path.exists(path_csv):
        # append sans doubler l’en-tête
        df_out.to_csv(path_csv, sep=';', decimal=',', mode='a', header=False, index=False, encoding='latin1')
    else:
        df_out.to_csv(path_csv, sep=';', decimal=',', index=False, encoding='latin1')

# ==========================================================
# === MODE INTERACTIF CONSOLE (optionnel) ==================
# ==========================================================
if __name__ == "__main__":
    while True:
        choice = input("\nSouhaites-tu saisir un NOUVEAU patient maintenant ? (o/n) : ").strip().lower()
        if choice not in ('o', 'oui', 'y'):
            print("\nFin du mode interactif. À bientôt !")
            break

        patient_new = saisir_patient_depuis_console(data)
        print("\n>>> Début optimisation de l'intervention (nouveau patient)")
        opt_rm2, opt_ser2, opt_preds2, opt_score2 = recommander(patient_new)
        print(">>> Optimisation terminée\n")

        print("=== RECOMMANDATION POUR NOUVEAU PATIENT ===")
        print(f"Sujet : {patient_new['Sujets']}")
        print(f"%RM recommandé : {opt_rm2:.1f}")
        print(f"Séries/semaine recommandé : {opt_ser2:.1f}")
        print("Gains prédits par muscle (%):")
        for m, g in zip(MUSCLES, opt_preds2):
            print(f"  - {m}: {g:.2f}")
        print(f"Score global (moy - écart-type) : {opt_score2:.2f}")

        sauvegarder_recommandation(CSV_SORTIE_RECO, patient_new['Sujets'],
                                   opt_rm2, opt_ser2, opt_preds2, opt_score2)
        print(f"✔ Recommandation enregistrée automatiquement dans '{CSV_SORTIE_RECO}'")

# === API UTILITAIRE POUR L'APP STREAMLIT ======================================
# Doit être au niveau racine du module (pas dans if __name__ == "__main__")

import pandas as pd  # au cas où

def reco_depuis_inputs(age, poids, sexe, lateralite, niveau, one_rm,
                       hip_ext_G, hip_ext_D, knee_flex_G, knee_flex_D, knee_ext_G, knee_ext_D):
    """
    Construit une ligne 'patient' au format attendu par ton code, puis appelle `recommander`.
    Retourne: (pct_rm: float, series_sem: float, gains: list[float in MUSCLES order], score: float)
    """
    row = {
        'Sujets': 'API_patient',
        'âge': age,
        'sexe': sexe,
        'poids': poids,
        'latéralité': lateralite,           # 'droitier' / 'gaucher' / 'ambidextre'
        'niveau': niveau,                   # 'débutant' / 'intermédiaire' / 'avancé'
        '1RM': one_rm,
        'Ext hanche G_J0': hip_ext_G,
        'Ext hanche D_J0': hip_ext_D,
        'Flx genou G_J0': knee_flex_G,
        'Flx genou D_J0': knee_flex_D,
        'Ext genou G_J0': knee_ext_G,
        'Ext genou D_J0': knee_ext_D,
        # placeholders intervention – seront écrasés dans la boucle d’optim
        '%RM': '60-75',
        '%RM_num': 67.5,
        'Séries/semaine': 6.0
    }
    s = pd.Series(row)

    # Appel de TA fonction d'optimisation déjà définie dans ce fichier
    opt_rm, opt_ser, opt_preds, opt_score = recommander(s)

    # opt_preds est aligné avec l’ordre MUSCLES de ton script
    return float(opt_rm), float(opt_ser), [float(x) for x in opt_preds], float(opt_score)
