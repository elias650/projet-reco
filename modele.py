# ==========================================================
# MODELE D'APPRENTISSAGE AUTOMATISÉ — RECO D'INTERVENTION
# Cohorte 1 — Elias Simon — Novembre 2025
# ==========================================================

import os
import re
import unicodedata
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
DATA_DIR = "data"
ANON_DIR = os.path.join(DATA_DIR, "anonymes")

def csv_path(name: str) -> str:
    """Cherche le fichier CSV dans data/anonymes/ ou Data/anonymes/."""
    DATA_DIRS = ["data", "Data"]  # accepte les deux
    ANON_SUB = "anonymes"
    # 1) Cherche d'abord dans data*/anonymes/
    for base in DATA_DIRS:
        anon = os.path.join(base, ANON_SUB, name)
        if os.path.exists(anon):
            return anon
    # 2) Sinon cherche dans data*/ directement
    for base in DATA_DIRS:
        p = os.path.join(base, name)
        if os.path.exists(p):
            return p
    # 3) Sinon retourne un chemin par défaut (pour message clair)
    return os.path.join(DATA_DIRS[0], name)

CSV_INFOS = csv_path("information initiales.csv")
CSV_J0    = csv_path("Mesures J0.csv")
CSV_J6    = csv_path("Mesures J+6 semaines.csv")
CSV_INT   = csv_path("Interventions.csv")

MUSCLES = ["Ext hanche D","Ext hanche G","Flx genou D","Flx genou G","Ext genou D","Ext genou G"]

# ⚠️ TEST_SUBJECTS : reconstruit si anonymisés (conserve compatibilité)
TEST_SUBJECTS_DEFAULT = ['Alexia','Romain','Elise']
CSV_SORTIE_RECO = "recommandations.csv"

# ------------------ Hyperparamètres de décision ------------------
# (1) équilibre : score = mean - LAMBDA_EQ * std
LAMBDA_EQ        = 0.35     # défaut (sera recalibré si LAMBDA_EQ_AUTO=True)
LAMBDA_EQ_AUTO   = True     # calibration data-driven ci-dessous
LAMBDA_GRID      = np.linspace(0.20, 0.60, 9)
ALPHA_PERF       = 0.90     # mean >= ALPHA_PERF * mean_max
COVERAGE_TARGET  = 0.70
# (2) contrainte d'équilibre pour le plan "Performance"
TAU_STD   = 10.0
TAU_STD_DYNAMIC = True
TAU_STD_Q   = 0.40
TAU_STD_MAX = 14.0
TAU_STD_MIN = 8.0
# Diversité entre plans (non apprise, à ta demande)
DIVERGENCE_MIN = 1.0  # |Δ%RM| + |Δseries|
LAMBDA_EQ_EFFECTIVE = LAMBDA_EQ  # sera mis à jour si calibration auto

# ==========================================================
# === UTILITAIRES D'HARMONISATION (niveau) =================
# ==========================================================
def _strip_accents(s: str) -> str:
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

def canon_niveau_keep_tres_avance(niv: str) -> str:
    """
    Harmonise en 4 catégories : 'débutant', 'intermédiaire', 'avancé', 'très avancé'
    """
    if not isinstance(niv, str):
        return 'intermédiaire'
    s = _strip_accents(niv).lower()
    s = re.sub(r'\s+', ' ', s).strip()
    if s.startswith('deb'):
        return 'débutant'
    if s.startswith('inter'):
        return 'intermédiaire'
    if s in {'tres avance', 'tres avances', 'tres-avance', 'tres_avance'} or ('tres' in s and 'avance' in s):
        return 'très avancé'
    if s in {'avance', 'avancee', 'avancees', 'avancer', 'avancé'}:
        return 'avancé'
    return 'intermédiaire'

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
    df.columns = [str(c).strip() for c in df.columns]
    if 'Sujets' not in df.columns:
        df.rename(columns={df.columns[0]: 'Sujets'}, inplace=True)
    df['Sujets'] = df['Sujets'].astype(str).str.strip()
    df = df[df['Sujets'].notna() & (df['Sujets'] != "")]
    df = df.drop_duplicates(subset=['Sujets'], keep='first')
    return df

info   = normalize_df(info)
j0     = normalize_df(j0)
j6     = normalize_df(j6)
interv = normalize_df(interv)
interv.rename(columns={'Séries/sem':'Séries/semaine','Séries/semai':'Séries/semaine'}, inplace=True)

# Harmonisation explicite du 'niveau'
if 'niveau' in info.columns:
    info['niveau'] = info['niveau'].astype(str).apply(canon_niveau_keep_tres_avance)

print("\n--- Après normalisation ---")
print("Colonnes Interventions :", list(interv.columns))
print("Effectifs (info, j0, j6, interv) :", len(info), len(j0), len(j6), len(interv))
if 'niveau' in info.columns:
    print("Valeurs uniques de 'niveau' :", sorted(info['niveau'].unique()))

# diagnostics
set_info = set(info['Sujets']); set_j0 = set(j0['Sujets']); set_j6 = set(j6['Sujets']); set_int = set(interv['Sujets'])
print("Sujets manquants dans Interventions :", sorted(set_info - set_int))
print("Sujets en trop dans Interventions   :", sorted(set_int - set_info))

# ==========================================================
# === TEST_SUBJECTS robuste (IDs anonymes) =================
# ==========================================================
all_ids = sorted(set(info['Sujets'].astype(str).str.strip()))
DEFAULT_TEST = [f"Sujet{i}" for i in range(1, 50)]
TEST_SUBJECTS = [x for x in TEST_SUBJECTS_DEFAULT if x in all_ids]
if len(TEST_SUBJECTS) < 3:
    fallback = [x for x in DEFAULT_TEST if x in all_ids][:3-len(TEST_SUBJECTS)]
    TEST_SUBJECTS = TEST_SUBJECTS + fallback
if not TEST_SUBJECTS:
    TEST_SUBJECTS = all_ids[:min(3, len(all_ids))]
print("TEST utilisé :", TEST_SUBJECTS)

# ==========================================================
# === FUSIONS & CONVERSION NUMERIQUE =======================
# ==========================================================
for m in MUSCLES:
    if m in j0.columns: j0[m] = pd.to_numeric(j0[m], errors='coerce')
    if m in j6.columns: j6[m] = pd.to_numeric(j6[m], errors='coerce')

base = info.merge(j0, on="Sujets").merge(j6, on="Sujets", suffixes=("_J0","_J6"))
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

# --- PATCH CRITIQUE : cast robuste en numérique pour éviter le crash Streamlit/NumPy ---
data['%RM_num'] = pd.to_numeric(data['%RM_num'], errors='coerce')
if 'Séries/semaine' in data.columns:
    data['Séries/semaine'] = pd.to_numeric(data['Séries/semaine'], errors='coerce')

# ==========================================================
# === CALIBRAGE DATA-DRIVEN DE TAU_STD =====================
# ==========================================================
if TAU_STD_DYNAMIC:
    gains = data[[f"Gain_{m}_%" for m in MUSCLES]].copy().dropna()
    if len(gains) > 0:
        gains['mean'] = gains.mean(axis=1)
        gains['std']  = gains[[f"Gain_{m}_%" for m in MUSCLES]].std(axis=1)
        med_gain = gains['mean'].median()
        pool = gains.loc[gains['mean'] >= med_gain, 'std']
        if len(pool) > 0:
            tau_candidate = float(pool.quantile(TAU_STD_Q))
            tau_candidate = max(TAU_STD_MIN, min(TAU_STD_MAX, tau_candidate))
            TAU_STD = round(tau_candidate, 1)
            print(f"\n[INFO] TAU_STD calibré automatiquement = {TAU_STD:.1f} "
                  f"(q={int(TAU_STD_Q*100)}e percentile parmi sujets performants, borné [{TAU_STD_MIN},{TAU_STD_MAX}])")
        else:
            print("\n[INFO] TAU_STD dynamique: aucun sujet performant identifiable — valeur par défaut conservée.")
    else:
        print("\n[INFO] TAU_STD dynamique: aucune ligne complète de gains — valeur par défaut conservée.")
else:
    print(f"\n[INFO] TAU_STD fixé manuellement = {TAU_STD:.1f}")

# ==========================================================
# === FEATURES (BASELINE UNIQUEMENT) & PIPELINE ============
# ==========================================================
Y_cols = [f"Gain_{m}_%" for m in MUSCLES]
info_cols = [c for c in info.columns if c != 'Sujets']
j0_cols = [f"{m}_J0" for m in MUSCLES if f"{m}_J0" in data.columns]
interv_cols = [c for c in ['%RM', '%RM_num', 'Séries/semaine'] if c in data.columns]
feat_cols = info_cols + j0_cols + interv_cols
assert not any(col.endswith('_J6') for col in feat_cols), "Les colonnes J6 ne doivent pas être dans feat_cols."

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

    ridge = RidgeCV(alphas=np.logspace(-3, 3, 13))
    pipe_ridge = make_pipe(ridge)
    y_pred_ridge = cross_val_predict(pipe_ridge, X, y, cv=loo)
    r2_ridge = r2_score(y, y_pred_ridge)
    mae_ridge = mean_absolute_error(y, y_pred_ridge)
    rmse_ridge = mean_squared_error(y, y_pred_ridge) ** 0.5

    rf_est = RandomForestRegressor(n_estimators=600, random_state=42)
    pipe_rf = make_pipe(rf_est)
    y_pred_rf = cross_val_predict(pipe_rf, X, y, cv=loo)
    r2_rf = r2_score(y, y_pred_rf)
    mae_rf = mean_absolute_error(y, y_pred_rf)
    rmse_rf = mean_squared_error(y, y_pred_rf) ** 0.5

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
# === OUTILS D'EVALUATION DE PLANS =========================
# ==========================================================
def _predict_gains_from_row(row_dict):
    """Retourne (preds array, mean, std) pour une ligne (dict/Series) déjà équipée des features."""
    x = pd.DataFrame([row_dict[feat_cols]])
    preds = np.array([MODELS[m].predict(x)[0] for m in MUSCLES])
    return preds, preds.mean(), preds.std()

def _evaluate_grid_for_row(row, rm_values, series_values):
    """Balaye la grille pour un sujet: retourne liste [(rm, s, mean, std, preds)]."""
    out = []
    for rm in rm_values:
        for s in series_values:
            tmp = row.copy()
            tmp['%RM_num'] = rm
            tmp['Séries/semaine'] = s
            preds, mean_gain, std_gain = _predict_gains_from_row(tmp)
            out.append((rm, s, mean_gain, std_gain, preds))
    return out

# ==========================================================
# === CALIBRATION DE LAMBDA_EQ (objectif clinique) =========
# ==========================================================
LAMBDA_EQ_EFFECTIVE = LAMBDA_EQ  # par défaut
if LAMBDA_EQ_AUTO:
    print("\n[INFO] Calibration automatique de LAMBDA_EQ (objectif clinique)")
    rm_values = np.linspace(40, 90, 11)
    series_values = np.linspace(4, 12, 9)

    subjects = data[feat_cols + ['Sujets']].dropna().copy()
    subject_grids = []
    for _, row in subjects.iterrows():
        cands = _evaluate_grid_for_row(row, rm_values, series_values)
        mean_max = max(c[2] for c in cands) if cands else np.nan
        subject_grids.append({'Sujets': row['Sujets'], 'cands': cands, 'mean_max': mean_max})

    def _coverage_for_lambda(lmbd):
        ok = 0; total = 0
        viol_std = []; viol_perf = []
        for sg in subject_grids:
            cands = sg['cands']
            if not cands or not np.isfinite(sg['mean_max']):
                continue
            total += 1
            best = max(cands, key=lambda t: (t[2] - lmbd * t[3]))
            mean_ok = best[2] >= ALPHA_PERF * sg['mean_max']
            std_ok  = best[3] <= TAU_STD
            if mean_ok and std_ok:
                ok += 1
            else:
                viol_std.append(max(0.0, best[3] - TAU_STD))
                viol_perf.append(max(0.0, (ALPHA_PERF * sg['mean_max']) - best[2]))
        cov = ok / total if total > 0 else 0.0
        v_std  = np.mean(viol_std)  if viol_std  else 0.0
        v_perf = np.mean(viol_perf) if viol_perf else 0.0
        return cov, v_std, v_perf

    best_lambda = None
    best_viol = (1e9, 1e9)
    best_lambda_if_fail = LAMBDA_GRID[0]
    for lmbd in LAMBDA_GRID:
        cov, vstd, vperf = _coverage_for_lambda(lmbd)
        print(f"  - lambda={lmbd:.2f} -> couverture {cov*100:.1f}% | viol(std)={vstd:.2f} | viol(perf)={vperf:.2f}")
        if cov >= COVERAGE_TARGET:
            best_lambda = lmbd
            break
        if (vstd, vperf) < best_viol:
            best_viol = (vstd, vperf)
            best_lambda_if_fail = lmbd

    LAMBDA_EQ_EFFECTIVE = float(best_lambda if best_lambda is not None else best_lambda_if_fail)
    mode_msg = "atteint la couverture cible" if best_lambda is not None else "fallback (min violations)"
    print(f"[INFO] LAMBDA_EQ fixé à {LAMBDA_EQ_EFFECTIVE:.2f} ({mode_msg}: "
          f"alpha={ALPHA_PERF:.2f}, couverture cible={int(COVERAGE_TARGET*100)}%, tau={TAU_STD:.1f})")
else:
    LAMBDA_EQ_EFFECTIVE = LAMBDA_EQ
    print(f"\n[INFO] LAMBDA_EQ fixé manuellement = {LAMBDA_EQ_EFFECTIVE:.2f}")

# ==========================================================
# === CALIBRATION DUALE (ACCORD AVEC INTERVENTIONS) ========
# ==========================================================
def _plan_eq_perf_with_params(row, lambda_eq, tau_std, alpha_perf, div_min=1.0):
    """Renvoie (best_eq, best_pf) pour un patient donné et un triple d'hyperparams."""
    rm_values = np.linspace(40, 90, 11)
    series_values = np.linspace(4, 12, 9)

    # 1) plan équilibré
    best_eq = {'score': -1e9}
    for rm in rm_values:
        for s in series_values:
            tmp = row.copy()
            tmp['%RM_num'] = rm
            tmp['Séries/semaine'] = s
            preds, mean_gain, std_gain = _predict_gains_from_row(tmp)
            score_eq = mean_gain - lambda_eq * std_gain
            if score_eq > best_eq['score']:
                best_eq = {'score':score_eq, 'rm':rm, 'ser':s,
                           'preds':preds, 'mean':mean_gain, 'std':std_gain}

    # 2) plan performance (contrainte + diversité fixe)
    def _dist(p, rm, s):
        return abs(rm - p['rm']) + abs(s - p['ser'])

    best_pf = {'score': -1e9}
    candidates = []
    for rm in rm_values:
        for s in series_values:
            tmp = row.copy()
            tmp['%RM_num'] = rm
            tmp['Séries/semaine'] = s
            preds, mean_gain, std_gain = _predict_gains_from_row(tmp)
            candidates.append((mean_gain, std_gain, rm, s, preds))
            if (std_gain <= tau_std) and (_dist(best_eq, rm, s) >= div_min):
                if mean_gain > best_pf['score']:
                    best_pf = {'score':mean_gain, 'rm':rm, 'ser':s,
                               'preds':preds, 'mean':mean_gain, 'std':std_gain}

    # 3) fallback fondé sur alpha_perf
    if best_pf['score'] == -1e9:
        if candidates:
            means = np.array([c[0] for c in candidates])
            pth = alpha_perf * float(np.max(means))
            pool = [c for c in candidates if c[0] >= pth] or candidates
            pool.sort(key=lambda z: (z[1], -z[0]))  # std ↑ puis mean ↓
            for mean_gain, std_gain, rm, s, preds in pool:
                if _dist(best_eq, rm, s) >= div_min:
                    best_pf = {'score':mean_gain, 'rm':rm, 'ser':s,
                               'preds':preds, 'mean':mean_gain, 'std':std_gain}
                    break
            if best_pf['score'] == -1e9:
                mean_gain, std_gain, rm, s, preds = pool[0]
                best_pf = {'score':mean_gain, 'rm':rm, 'ser':s,
                           'preds':preds, 'mean':mean_gain, 'std':std_gain}
        else:
            best_pf = {'score': -1e9, 'rm': np.nan, 'ser': np.nan,
                       'preds': np.array([np.nan]*len(MUSCLES)), 'mean': np.nan, 'std': np.nan}

    return best_eq, best_pf

def _dist_plan_interv(plan, rm_real, s_real):
    if plan is None or (isinstance(plan.get('rm', np.nan), float) and np.isnan(plan['rm'])):
        return np.inf
    return abs(float(plan['rm']) - float(rm_real)) + abs(float(plan['ser']) - float(s_real))

def calibrate_dual_params(data, subjects_df, lambda_grid, tau_grid, alpha_grid, div_min=1.0):
    """
    data: table fusionnée complète
    subjects_df: features + 'Sujets' + '%RM_num' + 'Séries/semaine' (peuvent être strings -> on cast)
    Retourne dict(best) et DataFrame(stats)
    """
    best = None
    stats = []
    for lam in lambda_grid:
        for tau in tau_grid:
            for alp in alpha_grid:
                agree = 0; total = 0
                std_viol = []
                for _, row in subjects_df.iterrows():
                    # --- PATCH CRITIQUE : caster en numérique avant isfinite ---
                    val_rm = pd.to_numeric(row.get('%RM_num', np.nan), errors='coerce')
                    val_ser = pd.to_numeric(row.get('Séries/semaine', np.nan), errors='coerce')
                    if not (np.isfinite(val_rm) and np.isfinite(val_ser)):
                        continue

                    eq, pf = _plan_eq_perf_with_params(row, lam, tau, alp, div_min=div_min)
                    d_eq = _dist_plan_interv(eq, val_rm, val_ser)
                    d_pf = _dist_plan_interv(pf, val_rm, val_ser)

                    if np.isfinite(d_eq) or np.isfinite(d_pf):
                        total += 1
                        if d_eq < d_pf:
                            agree += 1
                        elif d_pf < d_eq:
                            agree += 1
                        else:
                            agree += 0.5

                        if pf and np.isfinite(pf.get('std', np.nan)):
                            std_viol.append(max(0.0, float(pf['std']) - tau))

                acc = agree / total if total > 0 else 0.0
                v_std = float(np.mean(std_viol)) if std_viol else 0.0
                stats.append({'lambda': lam, 'tau': tau, 'alpha': alp, 'accord': acc, 'viol_std': v_std, 'n': total})
                key = (acc, -v_std)  # max accord, puis min violation std
                if (best is None) or (key > best[0]):
                    best = (key, (lam, tau, alp, acc, v_std, total))
    if best is None:
        return None, pd.DataFrame(stats)
    (lam, tau, alp, acc, vstd, n) = best[1]
    return {'lambda': lam, 'tau': tau, 'alpha': alp, 'accord': acc, 'n': n, 'viol_std': vstd}, pd.DataFrame(stats)

# --- Calibration duale globale ---
print("\n[INFO] Calibration duale (accord avec interventions réelles)")
subjects_for_dual = data[feat_cols + ['Sujets', '%RM_num', 'Séries/semaine']].copy()
dual_mask = subjects_for_dual[['%RM_num','Séries/semaine']].apply(pd.to_numeric, errors='coerce').notna().all(axis=1)
subjects_for_dual = subjects_for_dual.loc[dual_mask]

TAU_GRID = np.linspace(TAU_STD_MIN, TAU_STD_MAX, 7)
ALPHA_GRID = np.linspace(0.80, 0.95, 4)
dual_best, dual_stats = calibrate_dual_params(
    data, subjects_for_dual,
    lambda_grid=LAMBDA_GRID,
    tau_grid=TAU_GRID,
    alpha_grid=ALPHA_GRID,
    div_min=DIVERGENCE_MIN
)
if dual_best:
    LAMBDA_EQ_EFFECTIVE = float(dual_best['lambda'])
    TAU_STD = float(dual_best['tau'])
    ALPHA_PERF = float(dual_best['alpha'])
    print(f"[OK] Hyperparams appris (globaux): lambda={LAMBDA_EQ_EFFECTIVE:.2f} | tau={TAU_STD:.1f} | alpha={ALPHA_PERF:.2f} "
          f"(accord={dual_best['accord']*100:.1f}% sur n={dual_best['n']})")
else:
    print("[INFO] Calibration duale: insuffisante pour améliorer les réglages actuels.")

# ==========================================================
# === HYPERPARAMS PAR NIVEAU (data-driven) =================
# ==========================================================
PARAMS_PAR_NIVEAU = {}

if 'niveau' in data.columns:
    niveaux_uniques = sorted(data['niveau'].dropna().unique().tolist())
    for niv in niveaux_uniques:
        df_niv = data.loc[data['niveau'] == niv, :]
        subj_niv = df_niv[feat_cols + ['Sujets', '%RM_num', 'Séries/semaine']].copy()
        mask = subj_niv[['%RM_num','Séries/semaine']].apply(pd.to_numeric, errors='coerce').notna().all(axis=1)
        subj_niv = subj_niv.loc[mask]
        if len(subj_niv) < 5:
            continue
        dual_best_niv, _ = calibrate_dual_params(
            data, subj_niv,
            lambda_grid=LAMBDA_GRID,
            tau_grid=np.linspace(TAU_STD_MIN, TAU_STD_MAX, 7),
            alpha_grid=np.linspace(0.80, 0.95, 4),
            div_min=DIVERGENCE_MIN
        )
        if dual_best_niv:
            PARAMS_PAR_NIVEAU[niv] = {
                'lambda': float(dual_best_niv['lambda']),
                'tau': float(dual_best_niv['tau']),
                'alpha': float(dual_best_niv['alpha'])
            }

def _params_for_patient(niveau_norm):
    if niveau_norm in PARAMS_PAR_NIVEAU:
        p = PARAMS_PAR_NIVEAU[niveau_norm]
        return p['lambda'], p['tau'], p['alpha']
    return LAMBDA_EQ_EFFECTIVE, TAU_STD, ALPHA_PERF

# ==========================================================
# === RECOMMANDATIONS : 1 plan + 2 plans ===================
# ==========================================================
print("\n>>> Patient utilisé pour la recommandation : (échantillon aléatoire)")
patient_row = data.sample(1).iloc[0]
print({k: patient_row[k] for k in list(data.columns)[:8]}, "...")

def recommander(patient_row):
    """Plan unique (historique) : score = moyenne - std (lambda=1)."""
    rm_values = np.linspace(40, 90, 11)
    series_values = np.linspace(4, 12, 9)
    best_score, best_rm, best_series = -1e9, None, None
    best_preds = None
    for rm in rm_values:
        for s in series_values:
            row = patient_row.copy()
            row['%RM_num'] = rm
            row['Séries/semaine'] = s
            preds, mean_gain, std_gain = _predict_gains_from_row(row)
            score = mean_gain - std_gain
            if score > best_score:
                best_score, best_rm, best_series, best_preds = score, rm, s, preds
    return best_rm, best_series, best_preds, best_score

def recommander_deux_plans(patient_row):
    """Deux propositions avec hyperparams appris (par niveau si dispo, sinon globaux)."""
    niv_norm = patient_row.get('niveau', 'intermédiaire')
    niv_norm = canon_niveau_keep_tres_avance(niv_norm)
    lam_niv, tau_niv, alp_niv = _params_for_patient(niv_norm)
    best_eq, best_perf = _plan_eq_perf_with_params(patient_row, lam_niv, tau_niv, alp_niv, div_min=DIVERGENCE_MIN)
    return best_eq, best_perf

def _afficher_plan(titre, plan):
    print(titre)
    if plan is None or (isinstance(plan.get('rm', np.nan), float) and np.isnan(plan['rm'])):
        print("  Aucun plan disponible.")
        return
    print(f"  %RM: {plan['rm']:.1f} | Séries/sem: {plan['ser']:.1f}")
    print(f"  Gain moyen: {plan['mean']:.2f}% | Déséquilibre (écart-type): {plan['std']:.2f}")
    for m, g in zip(MUSCLES, plan['preds']):
        print(f"    - {m}: {g:.2f}%")
    print(f"  Score: {plan['score']:.2f}")

print("\n>>> Patient utilisé pour la recommandation (démo plan unique)")
opt_rm, opt_ser, opt_preds, opt_score = recommander(patient_row)
print("=== Recommandation OPTIMALE & ÉQUILIBRÉE (plan unique historique) ===")
print(f"%RM recommandé : {opt_rm:.1f}")
print(f"Séries/semaine recommandé : {opt_ser:.1f}")
print("Gains prédits par muscle (%):")
for m, g in zip(MUSCLES, opt_preds):
    print(f"  - {m}: {g:.2f}")
print(f"Score global (moy - écart-type) : {opt_score:.2f}")

# ==========================================================
# === I/O UTILITAIRES (console & CSV) ======================
# ==========================================================
def _safe_input(prompt, default=None, dtype=str):
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
    d = {'Sujets':'Nouveau_patient'}
    d['âge'] = float(df['âge'].median()) if 'âge' in df.columns else 25.0
    d['sexe'] = _mode_or_first(df['sexe']) if 'sexe' in df.columns else 'H'
    d['poids'] = float(df['poids'].median()) if 'poids' in df.columns else 70.0
    d['latéralité'] = _mode_or_first(df['latéralité']) if 'latéralité' in df.columns else 'droitier'
    d['niveau'] = _mode_or_first(df['niveau']) if 'niveau' in df.columns else 'intermédiaire'
    d['1RM'] = float(df['1RM'].median()) if '1RM' in df.columns else 100.0
    for m in ["Ext hanche D","Ext hanche G","Flx genou D","Flx genou G","Ext genou D","Ext genou G"]:
        col = f"{m}_J0"
        d[col] = float(pd.to_numeric(df[col], errors='coerce').median()) if col in df.columns else 4.0
    d['%RM'] = '60-75'; d['%RM_num'] = 67.5; d['Séries/semaine'] = 6.0
    return d

def saisir_patient_depuis_console(df):
    defaults = _default_patient_from_data(df)
    print("\n--- Saisie d'un nouveau patient (laisser vide pour garder la valeur par défaut) ---")
    p = {}
    p['Sujets'] = _safe_input(f"Nom du sujet [{defaults['Sujets']}]: ", defaults['Sujets'], str)
    p['âge'] = _safe_input(f"Âge [{defaults['âge']}]: ", defaults['âge'], float)
    p['sexe'] = _safe_input(f"Sexe (H/F) [{defaults['sexe']}]: ", defaults['sexe'], str)
    p['poids'] = _safe_input(f"Poids (kg) [{defaults['poids']}]: ", defaults['poids'], float)
    p['latéralité'] = _safe_input(f"Latéralité (droitier/gaucher) [{defaults['latéralité']}]: ", defaults['latéralité'], str)
    p['niveau'] = canon_niveau_keep_tres_avance(_safe_input(
        f"Niveau (débutant/intermédiaire/avancé/très avancé) [{defaults['niveau']}]: ",
        defaults['niveau'], str))
    p['1RM'] = _safe_input(f"1RM (kg) [{defaults['1RM']}]: ", defaults['1RM'], float)
    for m in ["Ext hanche D","Ext hanche G","Flx genou D","Flx genou G","Ext genou D","Ext genou G"]:
        col = f"{m}_J0"
        p[col] = _safe_input(f"{col} [{defaults[col]}]: ", defaults[col], float)
    p['%RM'] = '60-75'; p['%RM_num'] = 67.5; p['Séries/semaine'] = 6.0
    return pd.Series(p)

def sauvegarder_recommandation(path_csv, sujet, opt_rm, opt_ser, opt_preds, opt_score):
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
        df_out.to_csv(path_csv, sep=';', decimal=',', mode='a', header=False, index=False, encoding='latin1')
    else:
        df_out.to_csv(path_csv, sep=';', decimal=',', index=False, encoding='latin1')

def sauvegarder_plan(path_csv, sujet, plan, etiquette):
    if plan is None or (isinstance(plan.get('rm', np.nan), float) and np.isnan(plan['rm'])):
        return
    out_row = {
        'Sujet': sujet,
        'Type_plan': etiquette,
        'Intensite_%RM': int(round(float(plan['rm']))),
        'Series_semaine': round(float(plan['ser']), 1),
        'Gain_moyen_%': round(float(plan['mean']), 2),
        'Desequilibre_std_pts': round(float(plan['std']), 2),
        'Score': round(float(plan['score']), 2)
    }
    for m, g in zip(MUSCLES, plan['preds']):
        out_row[f'Gain_pred_{m}_%'] = round(float(g), 2)
    df_out = pd.DataFrame([out_row])
    if os.path.exists(path_csv):
        df_out.to_csv(path_csv, sep=';', decimal=',', mode='a', header=False, index=False, encoding='latin1')
    else:
        df_out.to_csv(path_csv, sep=';', decimal=',', index=False, encoding='latin1')

# ==========================================================
# === MODE INTERACTIF CONSOLE (optionnel) ==================
# ==========================================================
if __name__ == "__main__":
    while True:
        choice = input("\nSaisir un NOUVEAU patient ? (o=1 plan / 2=deux plans / n=non) : ").strip().lower()
        if choice not in ('o', 'oui', 'y', '2'):
            print("\nFin du mode interactif. À bientôt !")
            break

        patient_new = saisir_patient_depuis_console(data)

        if choice == '2':
            print("\n>>> Début optimisation (DEUX PLANS)")
            best_eq, best_perf = recommander_deux_plans(patient_new)
            print(">>> Optimisation terminée\n")
            _afficher_plan("=== OPTION 1 — PLAN ÉQUILIBRÉ ===", best_eq)
            _afficher_plan(f"=== OPTION 2 — PLAN PERFORMANCE (std ≤ τ={TAU_STD:.1f}) ===", best_perf)
            sauvegarder_plan(CSV_SORTIE_RECO, patient_new['Sujets'], best_eq, "Equilibre")
            sauvegarder_plan(CSV_SORTIE_RECO, patient_new['Sujets'], best_perf, "Performance_contrainte")
            print(f"✔ Deux propositions enregistrées (si disponibles) dans '{CSV_SORTIE_RECO}'")
        else:
            print("\n>>> Début optimisation (PLAN UNIQUE historique)")
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

# ==========================================================
# === API POUR STREAMLIT (compat + 2 plans) ================
# ==========================================================
def reco_depuis_inputs(age, poids, sexe, lateralite, niveau, one_rm,
                       hip_ext_G, hip_ext_D, knee_flex_G, knee_flex_D, knee_ext_G, knee_ext_D):
    """API historique (1 plan)."""
    row = {
        'Sujets': 'API_patient',
        'âge': age, 'sexe': sexe, 'poids': poids,
        'latéralité': lateralite, 'niveau': canon_niveau_keep_tres_avance(niveau), '1RM': one_rm,
        'Ext hanche G_J0': hip_ext_G, 'Ext hanche D_J0': hip_ext_D,
        'Flx genou G_J0': knee_flex_G, 'Flx genou D_J0': knee_flex_D,
        'Ext genou G_J0': knee_ext_G, 'Ext genou D_J0': knee_ext_D,
        '%RM': '60-75', '%RM_num': 67.5, 'Séries/semaine': 6.0
    }
    s = pd.Series(row)
    opt_rm, opt_ser, opt_preds, opt_score = recommander(s)
    return float(opt_rm), float(opt_ser), [float(x) for x in opt_preds], float(opt_score)

def reco2_depuis_inputs(age, poids, sexe, lateralite, niveau, one_rm,
                        hip_ext_G, hip_ext_D, knee_flex_G, knee_flex_D, knee_ext_G, knee_ext_D):
    """Nouvelle API (2 plans) — retourne un dict avec deux propositions."""
    row = {
        'Sujets': 'API_patient',
        'âge': age, 'sexe': sexe, 'poids': poids,
        'latéralité': lateralite, 'niveau': canon_niveau_keep_tres_avance(niveau), '1RM': one_rm,
        'Ext hanche G_J0': hip_ext_G, 'Ext hanche D_J0': hip_ext_D,
        'Flx genou G_J0': knee_flex_G, 'Flx genou D_J0': knee_flex_D,
        'Ext genou G_J0': knee_ext_G, 'Ext genou D_J0': knee_ext_D,
        '%RM': '60-75', '%RM_num': 67.5, 'Séries/semaine': 6.0
    }
    s = pd.Series(row)
    niv_norm = canon_niveau_keep_tres_avance(niveau)
    lam_niv, tau_niv, alp_niv = _params_for_patient(niv_norm)
    best_eq, best_perf = _plan_eq_perf_with_params(s, lam_niv, tau_niv, alp_niv, div_min=DIVERGENCE_MIN)

    def _pack(plan):
        if plan is None:
            return {"rm": None, "series": None, "mean": None, "std": None, "score": None, "gains": []}
        return {
            "rm": (None if isinstance(plan.get('rm', np.nan), float) and np.isnan(plan.get('rm', np.nan)) else float(plan['rm'])),
            "series": (None if isinstance(plan.get('ser', np.nan), float) and np.isnan(plan.get('ser', np.nan)) else float(plan['ser'])),
            "mean": (None if isinstance(plan.get('mean', np.nan), float) and np.isnan(plan.get('mean', np.nan)) else float(plan['mean'])),
            "std": (None if isinstance(plan.get('std', np.nan), float) and np.isnan(plan.get('std', np.nan)) else float(plan['std'])),
            "score": (None if isinstance(plan.get('score', np.nan), float) and np.isnan(plan.get('score', np.nan)) else float(plan['score'])),
            "gains": ([] if (isinstance(plan.get('rm', np.nan), float) and np.isnan(plan.get('rm', np.nan))) else [float(x) for x in plan['preds']])
        }
    return {"equilibre": _pack(best_eq), "performance": _pack(best_perf)}
