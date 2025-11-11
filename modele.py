# ==========================================================
# MODELE D'APPRENTISSAGE AUTOMATISÉ — RECO D'INTERVENTION
# Cohorte 1 — Version: lambda auto avec borne minimale
# ==========================================================

import os
import numpy as np
import pandas as pd

from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor

# -------------------- Localisation des données --------------------
DATA_DIRS = ["data", "Data"]
ANON_SUB = "anonymes"

def csv_path(name: str) -> str:
    for base in DATA_DIRS:
        p = os.path.join(base, ANON_SUB, name)
        if os.path.exists(p):
            return p
    for base in DATA_DIRS:
        p = os.path.join(base, name)
        if os.path.exists(p):
            return p
    return os.path.join(DATA_DIRS[0], name)

CSV_INFOS = csv_path("information initiales.csv")
CSV_J0    = csv_path("Mesures J0.csv")
CSV_J6    = csv_path("Mesures J+6 semaines.csv")
CSV_INT   = csv_path("Interventions.csv")

MUSCLES = ["Ext hanche D","Ext hanche G","Flx genou D","Flx genou G","Ext genou D","Ext genou G"]
CSV_SORTIE_RECO = "recommandations.csv"

# -------------------- Lecture CSV --------------------
info   = pd.read_csv(CSV_INFOS, sep=';', decimal=',', encoding='latin1')
j0     = pd.read_csv(CSV_J0,    sep=';', decimal=',', encoding='latin1')
j6     = pd.read_csv(CSV_J6,    sep=';', decimal=',', encoding='latin1')
interv = pd.read_csv(CSV_INT,   sep=';', decimal=',', encoding='latin1')

print("=== Fichiers chargés ===")
print(f"Information initiales : {info.shape}")
print(f"Mesures J0 : {j0.shape}")
print(f"Mesures J+6 semaines : {j6.shape}")
print(f"Interventions : {interv.shape}")

# -------------------- Normalisation tables --------------------
def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    if 'Sujets' not in df.columns:
        df.rename(columns={df.columns[0]: 'Sujets'}, inplace=True)
    df['Sujets'] = df['Sujets'].astype(str).str.strip()
    df = df[df['Sujets'].notna() & (df['Sujets'] != "")]
    df = df.drop_duplicates(subset=['Sujets'], keep='first')
    if 'niveau' in df.columns:
        df['niveau'] = (df['niveau'].astype(str)
                        .str.lower()
                        .str.replace(r"\s+", " ", regex=True)
                        .str.replace("tres avance", "très avancé")
                        .str.replace("très avance", "très avancé")
                        .str.strip())
    return df

info   = normalize_df(info)
j0     = normalize_df(j0)
j6     = normalize_df(j6)
interv = normalize_df(interv)
interv.rename(columns={'Séries/sem':'Séries/semaine','Séries/semai':'Séries/semaine'}, inplace=True)

print("\n--- Après normalisation ---")
print("Colonnes Interventions :", list(interv.columns))
if 'niveau' in info.columns:
    print("Valeurs uniques de 'niveau' :", sorted(info['niveau'].dropna().unique()))
print("Sujets manquants dans Interventions :", sorted(set(info['Sujets']) - set(interv['Sujets'])))
print("Sujets en trop dans Interventions   :", sorted(set(interv['Sujets']) - set(info['Sujets'])))

# -------------------- Fusion & features --------------------
for m in MUSCLES:
    if m in j0.columns: j0[m] = pd.to_numeric(j0[m], errors='coerce')
    if m in j6.columns: j6[m] = pd.to_numeric(j6[m], errors='coerce')

base = info.merge(j0, on="Sujets").merge(j6, on="Sujets", suffixes=("_J0","_J6"))
data = base.merge(interv, on="Sujets", how="left")

for m in MUSCLES:
    data[f"Gain_{m}_%"] = 100 * (data[f"{m}_J6"] - data[f"{m}_J0"]) / data[f"{m}_J0"]

def rm_to_numeric(v):
    if isinstance(v, str):
        s = v.strip()
        if s == '40-60': return 50.0
        if s == '60-75': return 67.5
        if s == '75-85': return 80.0
        if s == '80-90': return 85.0
    try:
        return float(v)
    except:
        return np.nan

if '%RM' in data.columns:
    data['%RM_num'] = data['%RM'].apply(rm_to_numeric)

info_cols   = [c for c in info.columns if c != 'Sujets']  # âge, sexe, poids, latéralité, niveau, 1RM
j0_cols     = [f"{m}_J0" for m in MUSCLES if f"{m}_J0" in data.columns]
interv_cols = [c for c in ['%RM','%RM_num','Séries/semaine'] if c in data.columns]
feat_cols   = info_cols + j0_cols + interv_cols
Y_cols      = [f"Gain_{m}_%" for m in MUSCLES]

num_cols = [c for c in feat_cols if data[c].dtype != 'object']
cat_cols = [c for c in feat_cols if data[c].dtype == 'object']

prep = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
])

# -------------------- Entraînement (LOOCV) --------------------
loo = LeaveOneOut()
MODELS = {}
rows = []

def make_pipe(est):
    return Pipeline([('prep', prep), ('model', est)])

for m in MUSCLES:
    ycol = f"Gain_{m}_%"
    X = data[feat_cols]
    y = pd.to_numeric(data[ycol], errors='coerce')

    ridge = RidgeCV(alphas=np.logspace(-3, 3, 13))
    rf    = RandomForestRegressor(n_estimators=600, random_state=42)

    pipe_ridge = make_pipe(ridge)
    pipe_rf    = make_pipe(rf)

    ypr = cross_val_predict(pipe_ridge, X, y, cv=loo)
    ypf = cross_val_predict(pipe_rf,    X, y, cv=loo)

    r2r, maer, rmser = r2_score(y, ypr), mean_absolute_error(y, ypr), mean_squared_error(y, ypr)**0.5
    r2f, maef, rmsef = r2_score(y, ypf), mean_absolute_error(y, ypf), mean_squared_error(y, ypf)**0.5

    if r2r >= r2f:
        best_name, best_pipe, best_r2, best_mae, best_rmse = 'Ridge', pipe_ridge, r2r, maer, rmser
    else:
        best_name, best_pipe, best_r2, best_mae, best_rmse = 'RF', pipe_rf, r2f, maef, rmsef

    MODELS[m] = best_pipe.fit(X, y)
    rows.append({'muscle': m, 'n': len(y), 'modele': best_name,
                 'MAE': best_mae, 'RMSE': best_rmse, 'R2': best_r2})

perf = pd.DataFrame(rows)
print("\n=== Performances LOOCV (meilleur modèle par muscle) ===")
print(perf.to_string(index=False))

# -------------------- Calibration TAU_STD (seuil de déséquilibre) --------------------
def calibrate_tau_std(df: pd.DataFrame) -> float:
    stds = []
    for i in range(len(df)):
        vals = [df[f"Gain_{m}_%"].iloc[i] for m in MUSCLES]
        if all(np.isfinite(vals)):
            stds.append(np.nanstd(vals))
    stds = np.array(stds, dtype=float)
    stds = stds[np.isfinite(stds)]
    if len(stds) == 0:
        return 12.0
    q = np.quantile(stds, 0.40)  # 40e percentile
    return float(np.clip(q, 8.0, 14.0))

TAU_STD = calibrate_tau_std(data)
print(f"\n[INFO] TAU_STD calibré automatiquement = {TAU_STD:.1f} (bornes 8–14)")

# -------------------- Calibration LAMBDA_EQ (avec borne minimale) --------------------
LAMBDA_MIN = 0.35  # 👈 borne basse cohérente, évite Équilibre ≈ Performance
LAMBDA_MAX = 0.70  # borne haute raisonnable

def calibrate_lambda_eq(df: pd.DataFrame, lambda_grid=None, coverage_target=0.70):
    if lambda_grid is None:
        lambda_grid = np.round(np.linspace(0.05, 0.50, 10), 2)

    def score_eq(mean, std, lam): return mean - lam*std

    best = lambda_grid[0]
    best_cov = 0.0
    for lam in lambda_grid:
        ok, tot = 0, 0
        for i in range(len(df)):
            vals = [df[f"Gain_{m}_%"].iloc[i] for m in MUSCLES]
            if not all(np.isfinite(vals)):
                continue
            mean_i = float(np.nanmean(vals))
            std_i  = float(np.nanstd(vals))
            sc = score_eq(mean_i, std_i, lam)
            if sc >= 0:
                ok += 1
            tot += 1
        if tot == 0:
            continue
        cov = ok/tot
        if cov >= coverage_target:
            best = lam
            best_cov = cov
            break
    return best, best_cov

lam_cal, cov = calibrate_lambda_eq(data, coverage_target=0.70)
LAMBDA_EQ = float(np.clip(lam_cal, LAMBDA_MIN, LAMBDA_MAX))

print(f"[INFO] LAMBDA_EQ calibré = {lam_cal:.2f} | couverture={cov*100:.1f}% ; "
      f"après borne min→ LAMBDA_EQ={LAMBDA_EQ:.2f} (min={LAMBDA_MIN}, max={LAMBDA_MAX})")

# -------------------- Prédicteur d’un plan (%RM, séries) --------------------
def _predict_gains_for(row: pd.Series, rm: float, series: float):
    r = row.copy()
    r['%RM_num'] = float(rm)
    r['Séries/semaine'] = float(series)
    x = pd.DataFrame([r[feat_cols]])
    preds = np.array([MODELS[m].predict(x)[0] for m in MUSCLES], dtype=float)
    mean = float(np.mean(preds))
    std  = float(np.std(preds))
    return preds, mean, std

# -------------------- Optimisation (grilles et règles) --------------------
RM_VALUES     = np.linspace(40, 90, 11)   # 40,45,...,90
SERIES_VALUES = np.linspace(6, 12, 7)     # 6,7,...,12 (borne basse clinique)

def _best_equilibre(row: pd.Series):
    best = None
    for rm in RM_VALUES:
        for s in SERIES_VALUES:
            preds, mean, std = _predict_gains_for(row, rm, s)
            score = mean - LAMBDA_EQ*std
            cand = (score, mean, std, rm, s, preds)
            if (best is None) or (cand[0] > best[0]):
                best = cand
    score, mean, std, rm, s, preds = best
    return {"rm": rm, "series": s, "gains": preds, "mean": mean, "std": std, "score": score}

def _best_performance(row: pd.Series):
    best = None
    for rm in RM_VALUES:
        for s in SERIES_VALUES:
            preds, mean, std = _predict_gains_for(row, rm, s)
            cand = (mean, std, rm, s, preds)
            if (best is None) or (cand[0] > best[0]):
                best = cand
    mean, std, rm, s, preds = best
    return {"rm": rm, "series": s, "gains": preds, "mean": mean, "std": std, "score": mean}

def _best_stabilite(row: pd.Series):
    best = None
    for rm in RM_VALUES:
        for s in SERIES_VALUES:
            preds, mean, std = _predict_gains_for(row, rm, s)
            cand = (std, mean, rm, s, preds)
            if (best is None) or (cand[0] < best[0]) or (cand[0] == best[0] and cand[1] > best[1]):
                best = cand
    std, mean, rm, s, preds = best
    if s < 6.0:
        s = 6.0
        preds, mean, std = _predict_gains_for(row, rm, s)
    return {"rm": rm, "series": s, "gains": preds, "mean": mean, "std": std, "score": -std}

# -------------------- API Streamlit — 1 plan --------------------
def reco_depuis_inputs(age, poids, sexe, lateralite, niveau, one_rm,
                       hip_ext_G, hip_ext_D, knee_flex_G, knee_flex_D, knee_ext_G, knee_ext_D):
    row = {
        'Sujets': 'API_patient',
        'âge': age, 'sexe': sexe, 'poids': poids, 'latéralité': lateralite, 'niveau': niveau, '1RM': one_rm,
        'Ext hanche G_J0': hip_ext_G, 'Ext hanche D_J0': hip_ext_D,
        'Flx genou G_J0': knee_flex_G, 'Flx genou D_J0': knee_flex_D,
        'Ext genou G_J0': knee_ext_G, 'Ext genou D_J0': knee_ext_D,
        '%RM': '60-75', '%RM_num': 67.5, 'Séries/semaine': 6.0
    }
    s = pd.Series(row)
    p = _best_equilibre(s)
    return float(p["rm"]), float(p["series"]), [float(x) for x in p["gains"]], float(p["score"])

# -------------------- API Streamlit — 3 plans --------------------
def reco3_depuis_inputs(age, poids, sexe, lateralite, niveau, one_rm,
                        hip_ext_G, hip_ext_D, knee_flex_G, knee_flex_D, knee_ext_G, knee_ext_D):
    row = {
        'Sujets': 'API_patient',
        'âge': age, 'sexe': sexe, 'poids': poids, 'latéralité': lateralite, 'niveau': niveau, '1RM': one_rm,
        'Ext hanche G_J0': hip_ext_G, 'Ext hanche D_J0': hip_ext_D,
        'Flx genou G_J0': knee_flex_G, 'Flx genou D_J0': knee_flex_D,
        'Ext genou G_J0': knee_ext_G, 'Ext genou D_J0': knee_ext_D,
        '%RM': '60-75', '%RM_num': 67.5, 'Séries/semaine': 6.0
    }
    s = pd.Series(row)
    eq = _best_equilibre(s)
    pf = _best_performance(s)
    st = _best_stabilite(s)
    if st["series"] < 6.0:
        st["series"] = 6.0
        preds, mean, std = _predict_gains_for(s, st["rm"], st["series"])
        st["gains"], st["mean"], st["std"], st["score"] = preds, mean, std, -std
    return {
        "equilibre":  {k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in eq.items()},
        "performance":{k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in pf.items()},
        "stabilite":  {k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in st.items()},
    }

# -------------------- Mode console (optionnel) --------------------
def _mode_or_first(series):
    vc = series.dropna().astype(str).str.strip().value_counts()
    return vc.index[0] if len(vc) > 0 else None

def _default_patient_from_data(df):
    d = {}
    d['Sujets'] = 'Nouveau_patient'
    d['âge'] = float(df['âge'].median()) if 'âge' in df.columns else 25.0
    d['sexe'] = _mode_or_first(df['sexe']) if 'sexe' in df.columns else 'H'
    d['poids'] = float(df['poids'].median()) if 'poids' in df.columns else 70.0
    d['latéralité'] = _mode_or_first(df['latéralité']) if 'latéralité' in df.columns else 'droitier'
    d['niveau'] = _mode_or_first(df['niveau']) if 'niveau' in df.columns else 'intermédiaire'
    d['1RM'] = float(df['1RM'].median()) if '1RM' in df.columns else 100.0
    for m in MUSCLES:
        col = f"{m}_J0"
        d[col] = float(pd.to_numeric(data[col], errors='coerce').median()) if col in data.columns else 4.0
    d['%RM'] = '60-75'; d['%RM_num'] = 67.5; d['Séries/semaine'] = 6.0
    return d

def _safe_input(prompt, default=None, dtype=str):
    s = input(prompt).strip()
    if s == "": return default
    if dtype in (float, int):
        s = s.replace(",", ".")
        try:
            x = float(s)
            return int(x) if dtype is int else x
        except:
            print("  Entrée non valide, valeur par défaut conservée.")
            return default
    return s

def saisir_patient_depuis_console(df):
    defaults = _default_patient_from_data(df)
    print("\n--- Saisie d'un nouveau patient (laisser vide pour défaut) ---")
    p = {}
    p['Sujets'] = _safe_input(f"Nom [{defaults['Sujets']}]: ", defaults['Sujets'], str)
    p['âge'] = _safe_input(f"Âge [{defaults['âge']}]: ", defaults['âge'], float)
    p['sexe'] = _safe_input(f"Sexe (H/F) [{defaults['sexe']}]: ", defaults['sexe'], str)
    p['poids'] = _safe_input(f"Poids (kg) [{defaults['poids']}]: ", defaults['poids'], float)
    p['latéralité'] = _safe_input(f"Latéralité [{defaults['latéralité']}]: ", defaults['latéralité'], str)
    p['niveau'] = _safe_input(f"Niveau [{defaults['niveau']}]: ", defaults['niveau'], str)
    p['1RM'] = _safe_input(f"1RM (kg) [{defaults['1RM']}]: ", defaults['1RM'], float)
    for m in MUSCLES:
        col = f"{m}_J0"
        p[col] = _safe_input(f"{col} [{defaults[col]}]: ", defaults[col], float)
    p['%RM'] = '60-75'; p['%RM_num'] = 67.5; p['Séries/semaine'] = 6.0
    return pd.Series(p)

def sauvegarder_recommandation(path_csv, sujet, plan_name, plan):
    out_row = {
        'Sujet': sujet,
        'Plan': plan_name,
        '%RM_reco': round(float(plan['rm']), 1),
        'Series_semaine_reco': round(float(plan['series']), 1),
        'Gain_moyen_%': round(float(plan['mean']), 2),
        'Ecart_type_pts': round(float(plan['std']), 2),
        'Score_mean_minus_lambda_std': round(float(plan['score']), 2),
    }
    for m, g in zip(MUSCLES, plan['gains']):
        out_row[f'Gain_pred_{m}_%'] = round(float(g), 2)
    df_out = pd.DataFrame([out_row])
    if os.path.exists(path_csv):
        df_out.to_csv(path_csv, sep=';', decimal=',', mode='a', header=False, index=False, encoding='latin1')
    else:
        df_out.to_csv(path_csv, sep=';', decimal=',', index=False, encoding='latin1')

if __name__ == "__main__":
    print("\n>>> Patient utilisé pour la recommandation (démo plan unique)")
    patient_row = data.sample(1).iloc[0]
    print({k: patient_row[k] for k in ['Sujets','âge','sexe','poids','latéralité','niveau','1RM']}, "...")

    eq_demo = _best_equilibre(patient_row)
    print("\n=== Recommandation OPTIMALE & ÉQUILIBRÉE (démo) ===")
    print(f"%RM: {eq_demo['rm']:.1f} | Séries/sem: {eq_demo['series']:.1f}")
    print(f"Gain moyen: {eq_demo['mean']:.2f}% | Écart-type: {eq_demo['std']:.2f} | Score: {eq_demo['score']:.2f}")

    while True:
        choice = input("\nSaisir un NOUVEAU patient ? (o=1 plan / 2=trois plans / n=non) : ").strip().lower()
        if choice not in ('o','oui','y','2'):
            print("Fin du mode interactif.")
            break
        p = saisir_patient_depuis_console(data)
        if choice == '2':
            eq = _best_equilibre(p)
            pf = _best_performance(p)
            st = _best_stabilite(p)
            print("\n=== OPTION 1 — ÉQUILIBRE ===")
            print(f"%RM: {eq['rm']:.1f} | Séries/sem: {eq['series']:.1f} | mean: {eq['mean']:.2f} | std: {eq['std']:.2f} | score: {eq['score']:.2f}")
            print("=== OPTION 2 — PERFORMANCE ===")
            print(f"%RM: {pf['rm']:.1f} | Séries/sem: {pf['series']:.1f} | mean: {pf['mean']:.2f} | std: {pf['std']:.2f} | score: {pf['score']:.2f}")
            print("=== OPTION 3 — STABILITÉ ===")
            print(f"%RM: {st['rm']:.1f} | Séries/sem: {st['series']:.1f} | mean: {st['mean']:.2f} | std: {st['std']:.2f} | score: {st['score']:.2f}")
            sauvegarder_recommandation(CSV_SORTIE_RECO, p['Sujets'], "Equilibre", eq)
            sauvegarder_recommandation(CSV_SORTIE_RECO, p['Sujets'], "Performance", pf)
            sauvegarder_recommandation(CSV_SORTIE_RECO, p['Sujets'], "Stabilite", st)
            print(f"✔ Propositions enregistrées dans '{CSV_SORTIE_RECO}'")
        else:
            eq = _best_equilibre(p)
            print("\n=== RECOMMANDATION (ÉQUILIBRE) ===")
            print(f"%RM: {eq['rm']:.1f} | Séries/sem: {eq['series']:.1f} | mean: {eq['mean']:.2f} | std: {eq['std']:.2f} | score: {eq['score']:.2f}")
            sauvegarder_recommandation(CSV_SORTIE_RECO, p['Sujets'], "Equilibre", eq)
            print(f"✔ Recommandation enregistrée dans '{CSV_SORTIE_RECO}'")
