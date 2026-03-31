"""
fix_models.py — รันครั้งเดียวก่อน app.py
แปลง XGBoost / LightGBM จาก GPU → CPU แล้ว overwrite .pkl เดิม
"""
import joblib, os

print("⏳ กำลัง fix GPU → CPU...")

# ══════════════════════════════════════════════
# XGBoost  — save booster JSON แล้ว load ใหม่
# ══════════════════════════════════════════════
from xgboost import XGBRegressor

xgb_old = joblib.load("model_xgb.pkl")
xgb_old.get_booster().save_model("_xgb_tmp.ubj")

xgb_cpu = XGBRegressor(tree_method="hist", device="cpu", objective="reg:squarederror")
xgb_cpu.load_model("_xgb_tmp.ubj")
joblib.dump(xgb_cpu, "model_xgb.pkl")
os.remove("_xgb_tmp.ubj")
print("  ✅ XGBoost → CPU")

# ══════════════════════════════════════════════
# LightGBM — save booster .txt แล้ว load ใหม่
# ══════════════════════════════════════════════
import lightgbm as lgb

lgb_old = joblib.load("model_lgb.pkl")
lgb_old.booster_.save_model("_lgb_tmp.txt")

lgb_cpu = lgb.Booster(model_file="_lgb_tmp.txt")

# wrap กลับเป็น LGBMRegressor เพื่อให้ .predict() ใช้งานได้เหมือนเดิม
lgb_wrapper = lgb.LGBMRegressor()
lgb_wrapper._Booster       = lgb_cpu
lgb_wrapper._n_features    = lgb_cpu.num_feature()
lgb_wrapper._n_classes     = 1
lgb_wrapper.fitted_        = True
lgb_wrapper._n_features_in = lgb_cpu.num_feature()

joblib.dump(lgb_wrapper, "model_lgb.pkl")
os.remove("_lgb_tmp.txt")
print("  ✅ LightGBM → CPU")

# ══════════════════════════════════════════════
# CatBoost — รองรับ CPU inference อัตโนมัติ
# ══════════════════════════════════════════════
print("  ✅ CatBoost (ไม่ต้องแก้ — CPU inference อัตโนมัติ)")

print("\n🎉 เสร็จแล้ว! รันต่อได้เลย:")
print("   streamlit run app.py")