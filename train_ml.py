import pandas as pd
import numpy as np
import joblib
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import OrdinalEncoder
from sklearn.linear_model import Ridge
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. โหลดข้อมูล
# ============================================================
print("⏳ กำลังโหลดข้อมูล...")
df = pd.read_csv('dataset/salary.csv', skipinitialspace=True)
print(f"✅ โหลดข้อมูลสำเร็จ: {len(df):,} แถว")

# ============================================================
# 2. Currency Conversion → USD จริงๆ
# ============================================================
print("💱 กำลัง normalize salary เป็น USD...")
FX = {
    'USD': 1.000, 'EUR': 1.080, 'GBP': 1.270, 'CHF': 1.120,
    'CAD': 0.740, 'AUD': 0.650, 'SGD': 0.740,
    'JPY': 0.0067, 'INR': 0.012, 'KRW': 0.00075,
}
df = df.dropna(subset=['salary_usd', 'salary_currency'])
df['salary_usd'] = df.apply(
    lambda r: r['salary_usd'] * FX.get(str(r['salary_currency']).strip(), 1.0), axis=1
)
print(f"✅ แปลงสกุลเงินเสร็จ: {df['salary_currency'].value_counts().to_dict()}")

# ============================================================
# 3. Feature Engineering
# ============================================================
print("⚙️ กำลังทำ Feature Engineering...")
df['skill_count']     = df['required_skills'].str.split(',').str.len().fillna(0).astype(int)
edu_map               = {'Associate': 0, 'Bachelor': 1, 'Master': 2, 'PhD': 3}
df['edu_score']       = df['education_required'].map(edu_map).fillna(1).astype(int)
df['posting_date']        = pd.to_datetime(df['posting_date'])
df['application_deadline'] = pd.to_datetime(df['application_deadline'])
df['days_to_deadline']    = (df['application_deadline'] - df['posting_date']).dt.days
df['exp_edu_interact'] = df['years_experience'] * df['edu_score']
df['exp_per_skill']    = df['years_experience'] / (df['skill_count'] + 1)
size_map               = {'S': 0, 'M': 1, 'L': 2}
df['size_num']         = df['company_size'].map(size_map).fillna(1)
df['remote_x_size']   = df['remote_ratio'] * df['size_num']
df['is_urgent']        = (df['days_to_deadline'] <= 14).astype(int)

# ============================================================
# 4. Features + Split + Encode
# ============================================================
CAT_FEATURES = ['experience_level', 'employment_type', 'industry', 'company_location', 'company_size']
NUM_FEATURES = [
    'years_experience', 'skill_count', 'edu_score',
    'remote_ratio', 'days_to_deadline', 'benefits_score',
    'job_description_length',
    'exp_edu_interact', 'exp_per_skill', 'remote_x_size', 'is_urgent',
]
FEATURES = CAT_FEATURES + NUM_FEATURES

X = df[FEATURES].copy()
y = df['salary_usd'].copy()
for c in CAT_FEATURES:
    X[c] = X[c].fillna('Unknown')

Q1, Q3 = y.quantile(0.01), y.quantile(0.99)
mask   = y.between(Q1, Q3)
X, y   = X[mask], y[mask]
print(f"✅ หลังกรอง Outlier เหลือ: {len(X):,} แถว  (${y.min():,.0f} – ${y.max():,.0f})")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
X_train = X_train.copy(); X_test = X_test.copy()
X_train[CAT_FEATURES] = enc.fit_transform(X_train[CAT_FEATURES])
X_test[CAT_FEATURES]  = enc.transform(X_test[CAT_FEATURES])

feature_names = X_train.columns.tolist()
print(f"✅ Features: {len(feature_names)} | Train: {len(X_train):,} | Test: {len(X_test):,}")

KF = KFold(n_splits=5, shuffle=True, random_state=42)

# ใช้ fold สุดท้ายเป็น quick eval สำหรับ tuning (เร็วกว่า full CV 5x)
TR_IDX, VAL_IDX = list(KF.split(X_train))[-1]
X_tr  = X_train.iloc[TR_IDX];  y_tr  = y_train.iloc[TR_IDX]
X_val = X_train.iloc[VAL_IDX]; y_val = y_train.iloc[VAL_IDX]

N_TRIALS = 100

# ============================================================
# 5. Optuna Tuning
# ============================================================

# ── XGBoost ──────────────────────────────────────────────────
print(f"\n🔍 [1/3] Tuning XGBoost ({N_TRIALS} trials)...")

def xgb_objective(trial):
    m = XGBRegressor(
        n_estimators      = trial.suggest_int('n_estimators', 300, 3000, step=100),
        learning_rate     = trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
        max_depth         = trial.suggest_int('max_depth', 4, 12),
        min_child_weight  = trial.suggest_int('min_child_weight', 1, 10),
        subsample         = trial.suggest_float('subsample', 0.6, 1.0),
        colsample_bytree  = trial.suggest_float('colsample_bytree', 0.6, 1.0),
        colsample_bylevel = trial.suggest_float('colsample_bylevel', 0.6, 1.0),
        gamma             = trial.suggest_float('gamma', 0.0, 0.5),
        reg_alpha         = trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
        reg_lambda        = trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
        tree_method='hist', device='cuda',
        objective='reg:squarederror', random_state=42,
        early_stopping_rounds=30,   # ← ลดจาก 50 เป็น 30
    )
    m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    return r2_score(y_val, m.predict(X_val))

xgb_study = optuna.create_study(direction='maximize',
    sampler=optuna.samplers.TPESampler(seed=42),
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=15))
xgb_study.optimize(xgb_objective, n_trials=N_TRIALS, show_progress_bar=True)
print(f"   ✅ XGB Best R2: {xgb_study.best_value:.4f}")

# ── LightGBM ─────────────────────────────────────────────────
# ★ แก้ปัญหาช้า: ใช้ early_stopping callback แทน n_estimators ใหญ่
#   และ fix n_estimators=3000 ให้ early stopping หยุดเอง
print(f"\n🔍 [2/3] Tuning LightGBM ({N_TRIALS} trials)...")

def lgb_objective(trial):
    m = LGBMRegressor(
        n_estimators      = 3000,    # ← ให้ early stopping หยุดเอง (ไม่ต้อง tune)
        learning_rate     = trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        max_depth         = trial.suggest_int('max_depth', 4, 10),
        num_leaves        = trial.suggest_int('num_leaves', 31, 200),
        min_child_samples = trial.suggest_int('min_child_samples', 10, 100),
        subsample         = trial.suggest_float('subsample', 0.6, 1.0),
        colsample_bytree  = trial.suggest_float('colsample_bytree', 0.6, 1.0),
        reg_alpha         = trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
        reg_lambda        = trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
        device='gpu', random_state=42, verbose=-1,
        subsample_freq=1,  # ต้องตั้งเพื่อให้ subsample ทำงาน
    )
    # ★ early_stopping callback — หยุดเมื่อ val ไม่ดีขึ้น 30 รอบ
    m.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[early_stopping(30, verbose=False), log_evaluation(-1)],
    )
    return r2_score(y_val, m.predict(X_val))

lgb_study = optuna.create_study(direction='maximize',
    sampler=optuna.samplers.TPESampler(seed=42),
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=15))
lgb_study.optimize(lgb_objective, n_trials=N_TRIALS, show_progress_bar=True)
print(f"   ✅ LGB Best R2: {lgb_study.best_value:.4f}")

# ── CatBoost ─────────────────────────────────────────────────
# ★ เพิ่ม od_type early stopping + ลด iterations ceiling
print(f"\n🔍 [3/3] Tuning CatBoost ({N_TRIALS} trials)...")

def cat_objective(trial):
    m = CatBoostRegressor(
        iterations          = 3000,    # ← ให้ early stopping หยุดเอง
        learning_rate       = trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        depth               = trial.suggest_int('depth', 4, 8),  # ← ลด ceiling จาก 10→8
        l2_leaf_reg         = trial.suggest_float('l2_leaf_reg', 1e-4, 10.0, log=True),
        bagging_temperature = trial.suggest_float('bagging_temperature', 0.0, 1.0),
        random_strength     = trial.suggest_float('random_strength', 0.0, 1.0),
        early_stopping_rounds=30,   # ★ เพิ่ม early stopping
        task_type='GPU', random_seed=42, verbose=0,
    )
    m.fit(X_tr, y_tr, eval_set=(X_val, y_val), use_best_model=True)
    return r2_score(y_val, m.predict(X_val))

cat_study = optuna.create_study(direction='maximize',
    sampler=optuna.samplers.TPESampler(seed=42),
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=15))
cat_study.optimize(cat_objective, n_trials=N_TRIALS, show_progress_bar=True)
print(f"   ✅ CAT Best R2: {cat_study.best_value:.4f}")

# ============================================================
# 6. เทรน Final Base Models
# ============================================================
print("\n🚀 เทรน Final Base Models...")

xgb_params = {**xgb_study.best_params,
              'tree_method': 'hist', 'device': 'cuda',
              'objective': 'reg:squarederror', 'random_state': 42,
              'early_stopping_rounds': 30}
xgb_model = XGBRegressor(**xgb_params)
xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

lgb_params = {**lgb_study.best_params,
              'n_estimators': 3000, 'device': 'gpu',
              'random_state': 42, 'verbose': -1, 'subsample_freq': 1}
lgb_model = LGBMRegressor(**lgb_params)
lgb_model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
              callbacks=[early_stopping(30, verbose=False), log_evaluation(-1)])

cat_params = {**cat_study.best_params,
              'iterations': 3000, 'early_stopping_rounds': 30,
              'task_type': 'GPU', 'random_seed': 42, 'verbose': 0}
cat_model = CatBoostRegressor(**cat_params)
cat_model.fit(X_train, y_train, eval_set=(X_test, y_test), use_best_model=True)

print("   ✅ Base models เทรนเสร็จ")

# ============================================================
# 7. Stacking — Out-of-Fold
# ============================================================
print("\n🔗 กำลัง Stack โมเดล (Out-of-Fold 5 folds)...")

def get_oof(model, X_tr, y_tr, X_te, kf, fit_kw={}):
    oof  = np.zeros(len(X_tr))
    test = np.zeros((len(X_te), kf.n_splits))
    for i, (tr, val) in enumerate(kf.split(X_tr)):
        model.fit(X_tr.iloc[tr], y_tr.iloc[tr], **fit_kw)
        oof[val]   = model.predict(X_tr.iloc[val])
        test[:, i] = model.predict(X_te)
    return oof, test.mean(axis=1)

xgb_oof, xgb_tp = get_oof(XGBRegressor(**xgb_params), X_train, y_train, X_test, KF,
    fit_kw={'eval_set': [(X_test, y_test)], 'verbose': False})

lgb_oof, lgb_tp = get_oof(
    LGBMRegressor(**lgb_params), X_train, y_train, X_test, KF,
    fit_kw={'eval_set': [(X_test, y_test)],
            'callbacks': [early_stopping(30, verbose=False), log_evaluation(-1)]})

cat_oof, cat_tp = get_oof(CatBoostRegressor(**cat_params), X_train, y_train, X_test, KF,
    fit_kw={'eval_set': (X_test, y_test), 'use_best_model': True})

meta_train   = np.column_stack([xgb_oof, lgb_oof, cat_oof])
meta_test    = np.column_stack([xgb_tp,  lgb_tp,  cat_tp])
meta_learner = Ridge(alpha=1.0)
meta_learner.fit(meta_train, y_train)
print("   ✅ Meta-learner (Ridge) เทรนเสร็จ")

# ============================================================
# 8. ประเมินผล
# ============================================================
def evaluate(name, y_true, y_pred):
    r2   = r2_score(y_true, y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    print(f"  {name:<22} R2: {r2*100:.2f}%   MAE: ${mae:>10,.0f}   MAPE: {mape:.2f}%")
    return r2, mae, mape

print(f"\n{'='*68}")
print(f"📊  Model Comparison (Test Set — USD normalized)")
print(f"{'='*68}")
evaluate("XGBoost",          y_test, xgb_model.predict(X_test))
evaluate("LightGBM",         y_test, lgb_model.predict(X_test))
evaluate("CatBoost",         y_test, cat_model.predict(X_test))
print(f"  {'─'*64}")
stack_pred    = meta_learner.predict(meta_test)
r2, mae, mape = evaluate("⭐ Stack Ensemble", y_test, stack_pred)
print(f"{'='*68}")

# ============================================================
# 9. บันทึก Artifacts
# ============================================================
joblib.dump(xgb_model,    'model_xgb.pkl')
joblib.dump(lgb_model,    'model_lgb.pkl')
joblib.dump(cat_model,    'model_cat.pkl')
joblib.dump(meta_learner, 'model_meta.pkl')
joblib.dump(enc,          'encoder.pkl')
joblib.dump(feature_names,'feature_names.pkl')
joblib.dump(CAT_FEATURES, 'cat_features.pkl')
joblib.dump(FEATURES,     'all_features.pkl')
joblib.dump(FX,           'fx_rates.pkl')
joblib.dump({'r2': r2, 'mae': mae, 'mape': mape,
             'n_rows_trained': len(X_train),
             'salary_range_usd': {'min': float(y.min()), 'max': float(y.max())}
             }, 'metrics.pkl')
print("\n💾 บันทึกทุก model + encoder + fx_rates + metrics เรียบร้อย!")

# ============================================================
# 10. Inference Helper
# ============================================================
def predict_salary(raw_df: pd.DataFrame) -> np.ndarray:
    _fx    = joblib.load('fx_rates.pkl')
    _enc   = joblib.load('encoder.pkl')
    _feats = joblib.load('feature_names.pkl')
    _cats  = joblib.load('cat_features.pkl')
    _xgb   = joblib.load('model_xgb.pkl')
    _lgb   = joblib.load('model_lgb.pkl')
    _cat   = joblib.load('model_cat.pkl')
    _meta  = joblib.load('model_meta.pkl')

    X = raw_df.copy()
    edu_map_ = {'Associate': 0, 'Bachelor': 1, 'Master': 2, 'PhD': 3}
    if 'edu_score'       not in X: X['edu_score']       = X['education_required'].map(edu_map_).fillna(1)
    if 'skill_count'     not in X: X['skill_count']     = X['required_skills'].str.split(',').str.len().fillna(0)
    if 'exp_edu_interact'not in X: X['exp_edu_interact'] = X['years_experience'] * X['edu_score']
    if 'exp_per_skill'   not in X: X['exp_per_skill']   = X['years_experience'] / (X['skill_count'] + 1)
    if 'size_num'        not in X: X['size_num']        = X['company_size'].map({'S':0,'M':1,'L':2}).fillna(1)
    if 'remote_x_size'   not in X: X['remote_x_size']   = X['remote_ratio'] * X['size_num']
    if 'is_urgent'       not in X: X['is_urgent']       = (X.get('days_to_deadline', 30) <= 14).astype(int)

    X[_cats] = X[_cats].fillna('Unknown')
    X[_cats] = _enc.transform(X[_cats])
    X = X[_feats]

    meta_in = np.column_stack([_xgb.predict(X), _lgb.predict(X), _cat.predict(X)])
    return _meta.predict(meta_in)

print("\n🔎 ทดสอบ Inference Pipeline...")
sample = df[FEATURES].iloc[:5].copy()
preds  = predict_salary(sample)
print(f"✅ Inference ผ่าน: {np.round(preds, 2)}")
print("💡 ใช้ฟังก์ชัน predict_salary(df) ใน production ได้เลย")