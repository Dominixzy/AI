import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="ML Studio",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE = os.path.dirname(os.path.abspath(__file__))

# ─────────────────────────────────────────────
# Global CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.block-container { padding: 2rem 2.5rem 4rem; max-width: 900px; }
h1 { font-size: 1.8rem !important; font-weight: 700 !important; letter-spacing: -0.5px; }
h2 { font-size: 1.2rem !important; font-weight: 600 !important; margin-top: 1.8rem !important; }

.sec-label {
    font-size: 0.62rem; font-weight: 700; letter-spacing: 2.5px;
    text-transform: uppercase; color: #aaa; margin: 1.8rem 0 0.5rem;
}
.hr { border: none; border-top: 1px solid #f0f0f0; margin: 1.4rem 0; }

.info-card {
    background: #f8f9fa; border-left: 3px solid #1D9E75;
    border-radius: 0 8px 8px 0; padding: 1rem 1.2rem; margin: 1rem 0;
    font-size: 0.87rem; line-height: 1.75; color: #333;
}
.info-card-purple { border-left-color: #7c4dbd; }
.info-card-amber  { border-left-color: #BA7517; }
.info-card-blue   { border-left-color: #1a5fa8; }

.step-wrap { display: flex; flex-direction: column; gap: 10px; margin: 1rem 0; }
.step-row  { display: flex; align-items: flex-start; gap: 14px; }
.step-num  {
    min-width: 28px; height: 28px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.75rem; font-weight: 700; color: #fff;
    background: #1D9E75; flex-shrink: 0; margin-top: 2px;
}
.step-num-p { background: #7c4dbd; }
.step-body  { font-size: 0.87rem; line-height: 1.7; color: #333; padding-top: 4px; }

.metric-strip { display: flex; gap: 10px; flex-wrap: wrap; margin: 1rem 0; }
.metric-box {
    flex: 1; min-width: 110px; background: #f8f9fa;
    border: 0.5px solid #e5e5e5; border-radius: 10px; padding: 12px 14px;
}
.mb-label { font-size: 0.62rem; text-transform: uppercase; letter-spacing: 1px; color: #aaa; }
.mb-val   { font-size: 1.3rem; font-weight: 700; color: #111; margin-top: 2px; }
.mb-sub   { font-size: 0.7rem; color: #888; margin-top: 1px; }

.badge-row { display: flex; gap: 8px; flex-wrap: wrap; margin: 0.6rem 0 1.2rem; }
.badge { font-size: 0.67rem; font-weight: 700; padding: 3px 10px; border-radius: 999px; font-family: monospace; }
.badge-green  { background: #e1f5ee; color: #0F6E56; }
.badge-purple { background: #eeedfe; color: #534AB7; }
.badge-amber  { background: #faeeda; color: #854F0B; }
.badge-blue   { background: #e0f0ff; color: #1a5fa8; }
.badge-gray   { background: #f5f5f5; color: #666; }

.ref-card {
    background: #fff; border: 0.5px solid #e5e5e5; border-radius: 8px;
    padding: 10px 14px; margin: 6px 0; font-size: 0.82rem; color: #444; line-height: 1.6;
}
.ref-card a { color: #1a5fa8; text-decoration: none; }
.ref-num { font-weight: 700; color: #aaa; margin-right: 6px; }

.result-card {
    background: #f7fdf9; border: 1.5px solid #1D9E75;
    border-radius: 14px; padding: 1.4rem 1.6rem; margin-top: 1.2rem;
}
.result-label  { font-size: 0.68rem; letter-spacing: 1.5px; text-transform: uppercase; color: #1D9E75; font-weight: 700; }
.result-salary { font-size: 2.4rem; font-weight: 800; color: #0F6E56; letter-spacing: -1.5px; margin: 6px 0 4px; }
.result-range  { font-size: 0.76rem; color: #4caf8a; }
.model-row  { display: flex; gap: 10px; margin-top: 1rem; }
.model-pill { flex: 1; background: #fff; border: 1px solid #e5f5ee; border-radius: 10px; padding: 10px 12px; }
.mp-label   { font-size: 0.62rem; text-transform: uppercase; letter-spacing: 1px; color: #bbb; margin: 0; }
.mp-val     { font-size: 1rem; font-weight: 700; color: #0F6E56; margin: 0; }

.fruit-card {
    background: #f7f4fd; border: 1.5px solid #7c4dbd;
    border-radius: 14px; padding: 1.4rem 1.6rem; margin-top: 1.2rem;
}
.fruit-label { font-size: 0.68rem; letter-spacing: 1.5px; text-transform: uppercase; color: #7c4dbd; font-weight: 700; }
.fruit-name  { font-size: 2.2rem; font-weight: 800; color: #4a2080; letter-spacing: -1px; margin: 6px 0 4px; }
.fruit-conf  { font-size: 0.76rem; color: #9b6dd6; }
.rank-row {
    display: flex; align-items: center; gap: 10px; padding: 8px 12px;
    border-radius: 8px; margin-top: 6px; background: #fff; border: 1px solid #ede8f8;
}
.rank-medal { font-size: 1.1rem; width: 24px; }
.rank-name  { flex: 1; font-size: 0.85rem; font-weight: 600; color: #333; }
.rank-pct   { font-size: 0.82rem; font-weight: 700; color: #7c4dbd; }
.rank-bar-wrap { width: 80px; height: 6px; background: #ede8f8; border-radius: 999px; overflow: hidden; }
.rank-bar      { height: 100%; background: #7c4dbd; border-radius: 999px; }

.conf-note {
    background: #fafafa; border: 1px solid #eee; border-radius: 8px;
    padding: 10px 14px; font-size: 0.76rem; color: #888; margin-top: 0.8rem;
}

table.styled { width: 100%; border-collapse: collapse; font-size: 0.83rem; margin: 0.8rem 0; }
table.styled th { background: #f5f5f5; padding: 8px 12px; text-align: left; font-weight: 600; border-bottom: 1px solid #e5e5e5; }
table.styled td { padding: 7px 12px; border-bottom: 0.5px solid #f0f0f0; }
table.styled tr:last-child td { border-bottom: none; }

div.stButton > button {
    width: 100%; height: 46px; font-weight: 700; font-size: 0.9rem;
    border: none; border-radius: 10px; color: #fff; transition: opacity 0.15s;
}
div.stButton > button:hover { opacity: 0.88; }
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Lookup maps (ตรงกับ train_ml.py)
# ─────────────────────────────────────────────
EXP_LEVEL_MAP = {"EN — Entry-level":"EN","MI — Mid-level":"MI","SE — Senior":"SE","EX — Executive":"EX"}
EMP_TYPE_MAP  = {"FT — Full-time":"FT","PT — Part-time":"PT","CT — Contract":"CT","FL — Freelance":"FL"}
EDU_MAP       = {"Associate":0,"Bachelor":1,"Master":2,"PhD":3}
SIZE_MAP      = {"S":0,"M":1,"L":2}
INDUSTRIES    = sorted(["Automotive","Consulting","Education","Energy","Finance","Gaming",
                         "Government","Healthcare","Manufacturing","Media","Real Estate",
                         "Retail","Technology","Telecommunications"])
LOCATIONS     = sorted(["AT","AU","CA","CH","CN","DE","FR","GB","IL","IN","JP","KR","NO","SE","SG","US"])
SKILL_OPTIONS = sorted(["AWS","Azure","Computer Vision","Data Visualization","Deep Learning",
                         "Docker","GCP","Git","Go","Hadoop","Java","Kubernetes","Linux",
                         "Machine Learning","Mathematics","MLOps","NLP","PyTorch","Python",
                         "R","Scala","Spark","SQL","Statistics","Tableau","TensorFlow"])

# ─────────────────────────────────────────────
# Load assets
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_salary_assets():
    try:
        xgb = joblib.load(os.path.join(BASE,"model_xgb.pkl"))
        lgb = joblib.load(os.path.join(BASE,"model_lgb.pkl"))
        cat = joblib.load(os.path.join(BASE,"model_cat.pkl"))
        # fix GPU→CPU device mismatch
        try: xgb.set_params(device="cpu")
        except: pass
        try: lgb.set_params(device="cpu")
        except: pass
        return (
            xgb, lgb, cat,
            joblib.load(os.path.join(BASE,"model_meta.pkl")),
            joblib.load(os.path.join(BASE,"encoder.pkl")),
            joblib.load(os.path.join(BASE,"feature_names.pkl")),
            joblib.load(os.path.join(BASE,"cat_features.pkl")),
            joblib.load(os.path.join(BASE,"metrics.pkl")),
            True,
        )
    except Exception as e:
        print(f"[Salary] Load error: {e}")
        return None,None,None,None,None,None,None,{},False

@st.cache_resource(show_spinner=False)
def load_fruit_assets():
    try:
        import tensorflow as tf
        # รองรับทั้ง fruit_cnn_model.keras และ fruit_cnn_model_fixed.keras
        for fname in ["fruit_cnn_model_fixed.keras", "fruit_cnn_model.keras"]:
            mp = os.path.join(BASE, fname)
            if os.path.exists(mp):
                model = tf.keras.models.load_model(mp)
                jp = os.path.join(BASE,"class_names.json")
                with open(jp, encoding="utf-8") as f:
                    names = json.load(f)
                return model, names, True
        return None, [], False
    except Exception as e:
        print(f"[Fruit] Load error: {e}")
        return None, [], False

xgb_m,lgb_m,cat_m,meta_m,enc,feature_names,cat_cols,metrics,sal_ok = load_salary_assets()
fruit_model, class_names, fruit_ok = load_fruit_assets()

# ─────────────────────────────────────────────
# Predict helpers
# ─────────────────────────────────────────────
def build_features(row):
    edu = EDU_MAP.get(row["education_required"], 1)
    sk  = len([s.strip() for s in row["required_skills"].split(",") if s.strip()])
    sz  = SIZE_MAP.get(row["company_size"], 1)
    rem = row["remote_ratio"]
    yrs = row["years_experience"]
    dtd = row.get("days_to_deadline", 30)
    df = pd.DataFrame([{
        "experience_level":row["experience_level"], "employment_type":row["employment_type"],
        "industry":row["industry"], "company_location":row["company_location"],
        "company_size":row["company_size"], "years_experience":yrs,
        "skill_count":sk, "edu_score":edu, "remote_ratio":rem,
        "days_to_deadline":dtd, "benefits_score":row.get("benefits_score",5.0),
        "job_description_length":row.get("job_description_length",1200),
        "exp_edu_interact":yrs*edu, "exp_per_skill":yrs/(sk+1),
        "remote_x_size":rem*sz, "is_urgent":int(dtd<=14),
    }])
    df[cat_cols] = df[cat_cols].fillna("Unknown")
    df[cat_cols] = enc.transform(df[cat_cols])
    return df[feature_names]

def run_salary(row):
    X  = build_features(row)
    px = float(xgb_m.predict(X)[0])
    pl = float(lgb_m.predict(X)[0])
    pc = float(cat_m.predict(X)[0])
    ps = float(meta_m.predict(np.array([[px,pl,pc]]))[0])
    return px, pl, pc, ps

def run_fruit(img, top_k=5):
    # ตรงกับ train_nn.py: resize 128×128, rescale /255
    arr   = np.array(img.convert("RGB").resize((128,128)), dtype=np.float32) / 255.0
    probs = fruit_model.predict(np.expand_dims(arr,0), verbose=0)[0]
    idx   = probs.argsort()[::-1][:top_k]
    return [(class_names[i], float(probs[i])) for i in idx]

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 ML Studio")
    st.markdown("---")
    page = st.radio("เมนู", [
        "📖 อธิบายโมเดล ML",
        "📖 อธิบายโมเดล NN",
        "💼 ทดสอบ Salary Predictor",
        "🍎 ทดสอบ Fruit Classifier",
    ], label_visibility="collapsed")
    st.markdown("---")
    st.caption(f"Salary Model : {'✅ พร้อมใช้งาน' if sal_ok else '❌ ไม่พบไฟล์'}")
    st.caption(f"Fruit Model  : {'✅ พร้อมใช้งาน' if fruit_ok else '❌ ไม่พบไฟล์'}")
    if fruit_ok:
        st.caption(f"Classes      : {len(class_names)}")


# ══════════════════════════════════════════════════════════════
# PAGE 1 — อธิบายโมเดล ML
# ══════════════════════════════════════════════════════════════
if page == "📖 อธิบายโมเดล ML":
    st.markdown("# 📖 Machine Learning — Salary Prediction")
    st.markdown("""<div class="badge-row">
      <span class="badge badge-green">XGBoost</span>
      <span class="badge badge-green">LightGBM</span>
      <span class="badge badge-green">CatBoost</span>
      <span class="badge badge-amber">Ridge Meta-learner</span>
      <span class="badge badge-gray">Stack Ensemble · Optuna 100 trials</span>
    </div>""", unsafe_allow_html=True)

    # 1. การเตรียมข้อมูล
    st.markdown("## 1. การเตรียมข้อมูล")
    st.markdown("""<div class="info-card">
    ใช้ <b>AI/ML Job Salary Dataset</b> ในรูปแบบ CSV (<code>dataset/salary.csv</code>)
    รวบรวมตำแหน่งงานด้าน AI ทั่วโลก มีฟีเจอร์ด้านประสบการณ์ การศึกษา ทักษะ อุตสาหกรรม
    และสภาพการทำงาน เป้าหมายคือการทำนาย <b>เงินเดือนต่อปีในหน่วย USD</b>
    </div>""", unsafe_allow_html=True)

    st.markdown('<p class="sec-label">ขั้นตอนการเตรียมข้อมูล</p>', unsafe_allow_html=True)
    st.markdown("""<div class="step-wrap">
      <div class="step-row"><div class="step-num">1</div>
        <div class="step-body"><b>Currency Normalization</b> — แปลงเงินเดือนทุกสกุล (EUR×1.08, GBP×1.27, INR×0.012 ฯลฯ) ให้เป็น USD ด้วย FX rate คงที่ เพื่อให้โมเดลเปรียบเทียบบนหน่วยเดียวกัน</div>
      </div>
      <div class="step-row"><div class="step-num">2</div>
        <div class="step-body"><b>Feature Engineering</b> — สร้างฟีเจอร์ใหม่:
          <code>skill_count</code> (นับจาก required_skills),
          <code>edu_score</code> (Associate=0…PhD=3),
          <code>days_to_deadline</code> (deadline − posting_date),
          <code>exp_edu_interact</code> (years × edu_score),
          <code>exp_per_skill</code> (years ÷ (skill_count+1)),
          <code>remote_x_size</code> (remote_ratio × size_num),
          <code>is_urgent</code> (1 ถ้า deadline ≤ 14 วัน)
        </div>
      </div>
      <div class="step-row"><div class="step-num">3</div>
        <div class="step-body"><b>Outlier Removal</b> — ตัดข้อมูลต่ำกว่า Percentile 1% และสูงกว่า 99% ออกด้วย <code>y.quantile(0.01/0.99)</code></div>
      </div>
      <div class="step-row"><div class="step-num">4</div>
        <div class="step-body"><b>Encoding</b> — ใช้ <code>OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)</code> แปลง 5 categorical features: experience_level, employment_type, industry, company_location, company_size</div>
      </div>
      <div class="step-row"><div class="step-num">5</div>
        <div class="step-body"><b>Train/Test Split</b> — แบ่ง 85% Train / 15% Test (<code>test_size=0.15, random_state=42</code>) จากนั้นใช้ KFold 5-fold สำหรับ OOF Stacking</div>
      </div>
    </div>""", unsafe_allow_html=True)

    st.markdown("""<table class="styled">
      <tr><th>Feature</th><th>ประเภท</th><th>คำอธิบาย</th></tr>
      <tr><td>experience_level</td><td>Categorical</td><td>EN / MI / SE / EX</td></tr>
      <tr><td>employment_type</td><td>Categorical</td><td>FT / PT / CT / FL</td></tr>
      <tr><td>industry</td><td>Categorical</td><td>Technology, Finance, Healthcare ฯลฯ</td></tr>
      <tr><td>years_experience</td><td>Numeric</td><td>จำนวนปีประสบการณ์</td></tr>
      <tr><td>edu_score</td><td>Ordinal (derived)</td><td>Associate=0, Bachelor=1, Master=2, PhD=3</td></tr>
      <tr><td>skill_count</td><td>Numeric (derived)</td><td>จำนวนทักษะที่ต้องการ</td></tr>
      <tr><td>remote_ratio</td><td>Numeric</td><td>0 / 50 / 100</td></tr>
      <tr><td>exp_edu_interact</td><td>Numeric (derived)</td><td>years_experience × edu_score</td></tr>
      <tr><td>is_urgent</td><td>Binary (derived)</td><td>1 ถ้า deadline ≤ 14 วัน</td></tr>
      <tr><td>benefits_score</td><td>Numeric</td><td>คะแนน benefits (1–10)</td></tr>
    </table>""", unsafe_allow_html=True)

    # 2. ทฤษฎี
    st.markdown("## 2. ทฤษฎีของอัลกอริทึม")
    st.markdown("""<div class="info-card">
    <b>Gradient Boosting</b> สร้าง Decision Tree ทีละต้น แต่ละต้นเรียนรู้จาก residual error ของต้นก่อนหน้า:
    <br><code>F_m(x) = F_{m-1}(x) + η · h_m(x)</code><br>
    โดย η คือ learning rate, h_m คือ tree ที่ fit กับ negative gradient ของ loss function
    </div>""", unsafe_allow_html=True)

    c1,c2,c3 = st.columns(3)
    with c1:
        st.markdown("""<div class="info-card">
        <b>XGBoost</b><br>เพิ่ม L1/L2 regularization ใน objective และใช้ second-order gradient
        รองรับ GPU ด้วย <code>tree_method='hist', device='cuda'</code>
        Tune: n_estimators, lr, max_depth, subsample, colsample, gamma, reg_alpha, reg_lambda
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class="info-card info-card-purple">
        <b>LightGBM</b><br>ใช้ Histogram-based + Leaf-wise tree growth และ GOSS sampling
        รองรับ GPU ด้วย <code>device='gpu'</code>
        Tune: lr, max_depth, num_leaves, min_child_samples, subsample, colsample
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""<div class="info-card info-card-blue">
        <b>CatBoost</b><br>จัดการ categorical features ด้วย Ordered Target Statistics
        ใช้ Symmetric Trees รองรับ GPU ด้วย <code>task_type='GPU'</code>
        Tune: lr, depth, l2_leaf_reg, bagging_temperature, random_strength
        </div>""", unsafe_allow_html=True)

    st.markdown("""<div class="info-card info-card-amber" style="margin-top:1rem">
    <b>Stacking Ensemble (Out-of-Fold)</b><br>
    ใช้ KFold 5 folds เทรน base models แต่ละ fold สร้าง OOF predictions ที่ไม่ leak ข้อมูล
    จากนั้น <b>Ridge Regression (α=1.0)</b> เป็น Meta-learner รวม prediction จากทั้ง 3 โมเดล
    เป็น final output
    </div>""", unsafe_allow_html=True)

    # 3. ขั้นตอนพัฒนา
    st.markdown("## 3. ขั้นตอนการพัฒนาโมเดล")
    st.markdown("""<div class="step-wrap">
      <div class="step-row"><div class="step-num">1</div>
        <div class="step-body"><b>Hyperparameter Tuning ด้วย Optuna</b> — TPE Sampler 100 trials/โมเดล พร้อม MedianPruner (n_warmup_steps=15) ใช้ fold สุดท้ายของ KFold เป็น validation เพื่อความเร็ว</div>
      </div>
      <div class="step-row"><div class="step-num">2</div>
        <div class="step-body"><b>Early Stopping (patience=30)</b> — ทุกโมเดลใช้ early stopping โดยดูจาก validation ป้องกัน overfitting และลดเวลาเทรน, XGBoost/LightGBM fix n_estimators=3000 ให้ early stopping หยุดเอง</div>
      </div>
      <div class="step-row"><div class="step-num">3</div>
        <div class="step-body"><b>Final Base Models</b> — นำ best_params จาก Optuna มาเทรนบน X_train ทั้งหมด บันทึกเป็น model_xgb.pkl, model_lgb.pkl, model_cat.pkl</div>
      </div>
      <div class="step-row"><div class="step-num">4</div>
        <div class="step-body"><b>OOF Stacking</b> — เทรน base models อีกรอบด้วย 5-fold CV สร้าง out-of-fold predictions แล้วเทรน Ridge meta-learner บน OOF predictions บันทึกเป็น model_meta.pkl</div>
      </div>
      <div class="step-row"><div class="step-num">5</div>
        <div class="step-body"><b>Evaluation</b> — วัดผลด้วย R², MAE, MAPE บน Test Set ที่แยกไว้ตั้งแต่แรก บันทึกผลใน metrics.pkl</div>
      </div>
    </div>""", unsafe_allow_html=True)

    if sal_ok:
        r2=metrics.get("r2",0); mae=metrics.get("mae",0); mape=metrics.get("mape",0); n=metrics.get("n_rows_trained",0)
        st.markdown(f"""<div class="metric-strip">
          <div class="metric-box"><div class="mb-label">R² Score</div><div class="mb-val">{r2:.1%}</div><div class="mb-sub">Test Set</div></div>
          <div class="metric-box"><div class="mb-label">MAE</div><div class="mb-val">${mae:,.0f}</div><div class="mb-sub">Mean Abs Error</div></div>
          <div class="metric-box"><div class="mb-label">MAPE</div><div class="mb-val">{mape:.1f}%</div><div class="mb-sub">Mean Abs % Error</div></div>
          <div class="metric-box"><div class="mb-label">Training Data</div><div class="mb-val">{n:,}</div><div class="mb-sub">แถว</div></div>
        </div>""", unsafe_allow_html=True)

    # 4. แหล่งอ้างอิง
    st.markdown("## 4. แหล่งอ้างอิง")
    st.markdown("""
    <div class="ref-card"><span class="ref-num">[1]</span> Chen, T., & Guestrin, C. (2016). <i>XGBoost: A Scalable Tree Boosting System</i>. KDD 2016. <a href="https://arxiv.org/abs/1603.02754">arxiv.org/abs/1603.02754</a></div>
    <div class="ref-card"><span class="ref-num">[2]</span> Ke, G. et al. (2017). <i>LightGBM: A Highly Efficient Gradient Boosting Decision Tree</i>. NeurIPS 2017. <a href="https://papers.nips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html">NeurIPS 2017</a></div>
    <div class="ref-card"><span class="ref-num">[3]</span> Prokhorenkova, L. et al. (2018). <i>CatBoost: unbiased boosting with categorical features</i>. NeurIPS 2018. <a href="https://arxiv.org/abs/1706.09516">arxiv.org/abs/1706.09516</a></div>
    <div class="ref-card"><span class="ref-num">[4]</span> Akiba, T. et al. (2019). <i>Optuna: A Next-generation Hyperparameter Optimization Framework</i>. KDD 2019. <a href="https://arxiv.org/abs/1907.10902">arxiv.org/abs/1907.10902</a></div>
    <div class="ref-card"><span class="ref-num">[5]</span> AI/ML Job Salary Dataset — Kaggle. <a href="https://www.kaggle.com/datasets/chopper53/machine-learning-engineer-salary-in-2024">kaggle.com/datasets/chopper53</a></div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# PAGE 2 — อธิบายโมเดล NN
# ══════════════════════════════════════════════════════════════
elif page == "📖 อธิบายโมเดล NN":
    st.markdown("# 📖 Neural Network — Fruit Classification")
    st.markdown("""<div class="badge-row">
      <span class="badge badge-purple">CNN 2-Block</span>
      <span class="badge badge-blue">TensorFlow / Keras</span>
      <span class="badge badge-amber">Mixed Precision float16</span>
      <span class="badge badge-gray">128×128 px · 139 Classes</span>
    </div>""", unsafe_allow_html=True)

    # 1. การเตรียมข้อมูล
    st.markdown("## 1. การเตรียมข้อมูล")
    st.markdown("""<div class="info-card info-card-purple">
    ใช้ <b>Fruit-360 Dataset</b> ประกอบด้วยรูปภาพผลไม้และผัก 139 ประเภท รวมกว่า 24,000 รูป
    โครงสร้างโฟลเดอร์: <code>dataset/Fruit/[label]/image.jpg</code><br>
    โหลดด้วย <code>ImageDataGenerator.flow_from_directory()</code> ซึ่ง map ชื่อโฟลเดอร์เป็น class label อัตโนมัติ
    และแบ่ง Train 80% / Validation 20% ผ่าน <code>validation_split=0.2</code>
    </div>""", unsafe_allow_html=True)

    st.markdown('<p class="sec-label">Data Augmentation (ใช้กับ Training เท่านั้น)</p>', unsafe_allow_html=True)
    st.markdown("""<div class="step-wrap">
      <div class="step-row"><div class="step-num step-num-p">1</div>
        <div class="step-body"><b>rescale=1/255</b> — normalize pixel values จาก [0,255] เป็น [0,1]</div>
      </div>
      <div class="step-row"><div class="step-num step-num-p">2</div>
        <div class="step-body"><b>rotation_range=20</b> — หมุนรูปแบบสุ่มในช่วง ±20 องศา</div>
      </div>
      <div class="step-row"><div class="step-num step-num-p">3</div>
        <div class="step-body"><b>horizontal_flip=True</b> — พลิกรูปในแนวนอนแบบสุ่ม</div>
      </div>
      <div class="step-row"><div class="step-num step-num-p">4</div>
        <div class="step-body"><b>fill_mode='nearest'</b> — เติมพื้นที่ว่างหลัง rotation ด้วย pixel ที่ใกล้ที่สุด</div>
      </div>
    </div>""", unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""<div class="metric-strip" style="flex-direction:column;gap:6px;">
          <div class="metric-box"><div class="mb-label">Dataset Size</div><div class="mb-val">~24,000+</div><div class="mb-sub">รูปภาพ</div></div>
          <div class="metric-box"><div class="mb-label">Classes</div><div class="mb-val">139</div><div class="mb-sub">ประเภท</div></div>
        </div>""", unsafe_allow_html=True)
    with col_b:
        st.markdown("""<div class="metric-strip" style="flex-direction:column;gap:6px;">
          <div class="metric-box"><div class="mb-label">Input Size</div><div class="mb-val">128×128</div><div class="mb-sub">pixels RGB</div></div>
          <div class="metric-box"><div class="mb-label">Batch Size</div><div class="mb-val">64</div><div class="mb-sub">สำหรับ RTX GPU</div></div>
        </div>""", unsafe_allow_html=True)

    # 2. ทฤษฎี
    st.markdown("## 2. ทฤษฎีของอัลกอริทึม")
    st.markdown("""<div class="info-card info-card-purple">
    <b>Convolutional Neural Network (CNN)</b> ใช้ <b>Convolutional Layer</b> เพื่อ extract spatial features
    แต่ละ filter เรียนรู้ pattern ที่แตกต่างกัน เช่น edge, texture, shape
    ตามด้วย <b>MaxPooling</b> เพื่อลด spatial dimension และเพิ่ม translation invariance
    <b>BatchNormalization</b> ช่วย stabilize training โดย normalize activation ใน mini-batch
    </div>""", unsafe_allow_html=True)

    st.markdown('<p class="sec-label">โครงสร้างโมเดล (ตรงตาม train_nn.py)</p>', unsafe_allow_html=True)
    st.markdown("""<table class="styled">
      <tr><th>Layer</th><th>Output Shape</th><th>รายละเอียด</th></tr>
      <tr><td>Input</td><td>(128, 128, 3)</td><td>รูปภาพ RGB</td></tr>
      <tr><td>Conv2D(32, 3×3)</td><td>(128, 128, 32)</td><td>padding='same', extract low-level features</td></tr>
      <tr><td>BatchNormalization</td><td>(128, 128, 32)</td><td>normalize activations</td></tr>
      <tr><td>Activation('relu')</td><td>(128, 128, 32)</td><td>non-linearity</td></tr>
      <tr><td>MaxPooling2D</td><td>(64, 64, 32)</td><td>downsample 2×</td></tr>
      <tr><td>Conv2D(64, 3×3)</td><td>(64, 64, 64)</td><td>padding='same', extract mid-level features</td></tr>
      <tr><td>BatchNormalization</td><td>(64, 64, 64)</td><td>normalize activations</td></tr>
      <tr><td>Activation('relu')</td><td>(64, 64, 64)</td><td>non-linearity</td></tr>
      <tr><td>MaxPooling2D</td><td>(32, 32, 64)</td><td>downsample 2×</td></tr>
      <tr><td>Flatten</td><td>(65,536)</td><td>แปลง 3D → 1D</td></tr>
      <tr><td>Dense(256, relu)</td><td>(256)</td><td>fully connected layer</td></tr>
      <tr><td>Dropout(0.5)</td><td>(256)</td><td>ป้องกัน overfitting</td></tr>
      <tr><td><b>Dense(139, softmax)</b></td><td><b>(139)</b></td><td><b>output layer, dtype=float32</b></td></tr>
    </table>""", unsafe_allow_html=True)

    st.markdown("""<div class="info-card info-card-amber">
    <b>Mixed Precision Training (float16)</b><br>
    ใช้ <code>mixed_precision.set_global_policy('mixed_float16')</code> — คำนวณ forward/backward pass
    ด้วย float16 แต่ output layer สุดท้ายต้องเป็น <code>dtype='float32'</code> เสมอ เพื่อป้องกัน
    numerical instability ในการคำนวณ loss ทำให้เทรนเร็วขึ้นบน RTX GPU ที่มี Tensor Core
    </div>""", unsafe_allow_html=True)

    # 3. ขั้นตอนพัฒนา
    st.markdown("## 3. ขั้นตอนการพัฒนาโมเดล")
    st.markdown("""<div class="step-wrap">
      <div class="step-row"><div class="step-num step-num-p">1</div>
        <div class="step-body"><b>ตั้งค่า GPU</b> — <code>set_memory_growth(True)</code> ป้องกัน TF จอง VRAM ทั้งหมด เปิด Mixed Precision float16 สำหรับ RTX GPU และตั้ง environment variables แก้ปัญหา compatibility RTX 50 Series</div>
      </div>
      <div class="step-row"><div class="step-num step-num-p">2</div>
        <div class="step-body"><b>Data Pipeline</b> — ใช้ <code>ImageDataGenerator</code> พร้อม augmentation โหลดจาก directory โดยตรง แบ่ง 80/20 อัตโนมัติ บันทึก class names ลง <code>class_names.json</code></div>
      </div>
      <div class="step-row"><div class="step-num step-num-p">3</div>
        <div class="step-body"><b>สร้างและ Compile โมเดล</b> — CNN 2-block ด้วย Keras Functional API, optimizer=Adam, loss=categorical_crossentropy, metric=accuracy</div>
      </div>
      <div class="step-row"><div class="step-num step-num-p">4</div>
        <div class="step-body"><b>Custom Callback: StopAt90</b> — หยุดเทรนอัตโนมัติเมื่อ val_accuracy ≥ 90% ประหยัดเวลาเทรน</div>
      </div>
      <div class="step-row"><div class="step-num step-num-p">5</div>
        <div class="step-body"><b>ModelCheckpoint</b> — บันทึก best model ตาม val_accuracy ลง <code>fruit_cnn_model.keras</code> โดยอัตโนมัติ</div>
      </div>
    </div>""", unsafe_allow_html=True)

    # 4. แหล่งอ้างอิง
    st.markdown("## 4. แหล่งอ้างอิง")
    st.markdown("""
    <div class="ref-card"><span class="ref-num">[1]</span> LeCun, Y. et al. (1998). <i>Gradient-Based Learning Applied to Document Recognition</i>. Proceedings of the IEEE. <a href="http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf">yann.lecun.com</a></div>
    <div class="ref-card"><span class="ref-num">[2]</span> Ioffe, S., & Szegedy, C. (2015). <i>Batch Normalization: Accelerating Deep Network Training</i>. ICML 2015. <a href="https://arxiv.org/abs/1502.03167">arxiv.org/abs/1502.03167</a></div>
    <div class="ref-card"><span class="ref-num">[3]</span> Srivastava, N. et al. (2014). <i>Dropout: A Simple Way to Prevent Neural Networks from Overfitting</i>. JMLR 2014. <a href="https://jmlr.org/papers/v15/srivastava14a.html">jmlr.org</a></div>
    <div class="ref-card"><span class="ref-num">[4]</span> Micikevicius, P. et al. (2018). <i>Mixed Precision Training</i>. ICLR 2018. <a href="https://arxiv.org/abs/1710.03740">arxiv.org/abs/1710.03740</a></div>
    <div class="ref-card"><span class="ref-num">[5]</span> Fruit-360 Dataset — Kaggle. <a href="https://www.kaggle.com/datasets/moltean/fruits">kaggle.com/datasets/moltean/fruits</a></div>
    <div class="ref-card"><span class="ref-num">[6]</span> TensorFlow Documentation — tf.keras.preprocessing.image.ImageDataGenerator. <a href="https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator">tensorflow.org</a></div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# PAGE 3 — ทดสอบ Salary Predictor
# ══════════════════════════════════════════════════════════════
elif page == "💼 ทดสอบ Salary Predictor":
    st.markdown("# 💼 ทดสอบ Salary Predictor")
    st.markdown("""<div class="badge-row">
      <span class="badge badge-green">Stack Ensemble</span>
      <span class="badge badge-green">XGBoost + LightGBM + CatBoost</span>
      <span class="badge badge-amber">Ridge Meta-learner</span>
    </div>""", unsafe_allow_html=True)

    if not sal_ok:
        st.error("❌ ไม่พบไฟล์โมเดล — วาง .pkl ทุกไฟล์ในโฟลเดอร์เดียวกับ app.py"); st.stop()

    r2=metrics.get("r2",0); mae=metrics.get("mae",0); mape=metrics.get("mape",0); n=metrics.get("n_rows_trained",0)
    st.markdown(f"""<div class="metric-strip">
      <div class="metric-box"><div class="mb-label">R² Score</div><div class="mb-val">{r2:.1%}</div><div class="mb-sub">Test Set</div></div>
      <div class="metric-box"><div class="mb-label">MAE</div><div class="mb-val">${mae:,.0f}</div><div class="mb-sub">Mean Abs Error</div></div>
      <div class="metric-box"><div class="mb-label">MAPE</div><div class="mb-val">{mape:.1f}%</div><div class="mb-sub">Mean Abs % Error</div></div>
      <div class="metric-box"><div class="mb-label">Training Rows</div><div class="mb-val">{n:,}</div><div class="mb-sub">แถว</div></div>
    </div><hr class="hr">""", unsafe_allow_html=True)

    st.markdown('<p class="sec-label">ประสบการณ์และการศึกษา</p>', unsafe_allow_html=True)
    c1,c2=st.columns(2)
    with c1:
        exp_label=st.selectbox("ระดับประสบการณ์",list(EXP_LEVEL_MAP.keys()),index=2)
        experience_level=EXP_LEVEL_MAP[exp_label]
    with c2:
        years_experience=st.number_input("ประสบการณ์ (ปี)",0,40,5,1)
    c3,c4=st.columns(2)
    with c3:
        education=st.selectbox("ระดับการศึกษา",["Associate","Bachelor","Master","PhD"],index=1)
    with c4:
        emp_label=st.selectbox("ประเภทการจ้างงาน",list(EMP_TYPE_MAP.keys()),index=0)
        employment_type=EMP_TYPE_MAP[emp_label]

    st.markdown('<hr class="hr"><p class="sec-label">บริษัทและอุตสาหกรรม</p>', unsafe_allow_html=True)
    c5,c6=st.columns(2)
    with c5:
        industry=st.selectbox("อุตสาหกรรม",INDUSTRIES,index=INDUSTRIES.index("Technology"))
    with c6:
        company_location=st.selectbox("ที่ตั้งบริษัท (ISO code)",LOCATIONS,index=LOCATIONS.index("US"))
    c7,c8=st.columns([1,2])
    with c7:
        company_size=st.radio("ขนาดบริษัท",["S","M","L"],index=1,horizontal=True)
        st.caption("S<50 · M 50–500 · L 500+")
    with c8:
        remote_ratio=st.select_slider("Remote Work",options=[0,50,100],value=50,
            format_func=lambda x:{0:"0% — Onsite",50:"50% — Hybrid",100:"100% — Remote"}[x])

    st.markdown('<hr class="hr"><p class="sec-label">ทักษะที่มี</p>', unsafe_allow_html=True)
    selected_skills=st.multiselect("เลือกทักษะ",SKILL_OPTIONS,default=["Python","SQL"],placeholder="พิมพ์เพื่อค้นหา...")
    required_skills=", ".join(selected_skills) if selected_skills else "Python"

    with st.expander("ตัวเลือกเพิ่มเติม"):
        ca,cb=st.columns(2)
        with ca:
            benefits_score=st.slider("คะแนน Benefits (1–10)",1.0,10.0,5.0,0.5)
            days_to_deadline=st.number_input("วันถึง Deadline",1,365,30)
        with cb:
            jd_length=st.number_input("ความยาว Job Description (ตัวอักษร)",100,5000,1200,50)

    st.markdown('<hr class="hr">', unsafe_allow_html=True)
    st.markdown("""<style>div.stButton > button{background:#1D9E75;}
    div.stButton > button:hover{background:#0F6E56;}</style>""", unsafe_allow_html=True)

    if st.button("คำนวณเงินเดือน →"):
        if not selected_skills:
            st.warning("กรุณาเลือกทักษะอย่างน้อย 1 อย่าง"); st.stop()
        form={
            "experience_level":experience_level,"employment_type":employment_type,
            "industry":industry,"company_location":company_location,"company_size":company_size,
            "years_experience":years_experience,"education_required":education,
            "required_skills":required_skills,"remote_ratio":remote_ratio,
            "benefits_score":benefits_score,"days_to_deadline":days_to_deadline,
            "job_description_length":jd_length,
        }
        with st.spinner("กำลังประมวลผล ensemble..."):
            try:
                px,pl,pc,ps=run_salary(form)
                lo=max(0,ps-mae*2); hi=ps+mae*2
                st.markdown(f"""
                <div class="result-card">
                  <p class="result-label">Predicted Annual Salary (USD)</p>
                  <p class="result-salary">${ps:,.0f}</p>
                  <p class="result-range">ช่วงความเชื่อมั่น 95%: ${lo:,.0f} – ${hi:,.0f}</p>
                  <div class="model-row">
                    <div class="model-pill"><p class="mp-label">XGBoost</p><p class="mp-val">${px:,.0f}</p></div>
                    <div class="model-pill"><p class="mp-label">LightGBM</p><p class="mp-val">${pl:,.0f}</p></div>
                    <div class="model-pill"><p class="mp-label">CatBoost</p><p class="mp-val">${pc:,.0f}</p></div>
                  </div>
                </div>
                <div class="conf-note">ⓘ Stack Ensemble (Ridge meta-learner) · MAE ±${mae:,.0f} · MAPE {mape:.1f}%</div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"เกิดข้อผิดพลาด: {e}")


# ══════════════════════════════════════════════════════════════
# PAGE 4 — ทดสอบ Fruit Classifier
# ══════════════════════════════════════════════════════════════
elif page == "🍎 ทดสอบ Fruit Classifier":
    st.markdown("# 🍎 ทดสอบ Fruit Classifier")
    st.markdown("""<div class="badge-row">
      <span class="badge badge-purple">CNN 2-Block</span>
      <span class="badge badge-blue">TensorFlow / Keras</span>
      <span class="badge badge-amber">139 Classes</span>
      <span class="badge badge-gray">128×128 px</span>
    </div>""", unsafe_allow_html=True)

    if not fruit_ok:
        st.error("❌ ไม่พบไฟล์โมเดล — วาง fruit_cnn_model.keras และ class_names.json ในโฟลเดอร์เดียวกับ app.py")

    if fruit_ok:
        st.markdown(f"""<div class="metric-strip">
          <div class="metric-box"><div class="mb-label">Classes</div><div class="mb-val">{len(class_names)}</div><div class="mb-sub">ประเภทผลไม้/ผัก</div></div>
          <div class="metric-box"><div class="mb-label">Architecture</div><div class="mb-val">CNN 2-Block</div><div class="mb-sub">Conv→BN→ReLU→Pool</div></div>
          <div class="metric-box"><div class="mb-label">Input Size</div><div class="mb-val">128×128</div><div class="mb-sub">pixels RGB</div></div>
          <div class="metric-box"><div class="mb-label">Framework</div><div class="mb-val">TensorFlow</div><div class="mb-sub">Keras + Mixed Precision</div></div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<p class="sec-label">อัปโหลดรูปภาพผลไม้</p>', unsafe_allow_html=True)
    uploaded=st.file_uploader("เลือกรูปภาพ (JPG, PNG, WEBP)",type=["jpg","jpeg","png","webp"],label_visibility="collapsed")

    if uploaded:
        img=Image.open(uploaded)
        ci,cinfo=st.columns([1,1])
        with ci:
            st.image(img,caption="รูปที่อัปโหลด",use_container_width=True)
        with cinfo:
            w,h=img.size
            st.markdown(f"""<div style="margin-top:8px">
              <p class="sec-label" style="margin-top:0">ข้อมูลไฟล์</p>
              <p style="font-size:0.83rem;color:#444;margin:4px 0">📄 <b>{uploaded.name}</b></p>
              <p style="font-size:0.83rem;color:#444;margin:4px 0">📐 {w} × {h} px</p>
              <p style="font-size:0.83rem;color:#444;margin:4px 0">🎨 Mode: {img.mode}</p>
            </div>""", unsafe_allow_html=True)

        st.markdown('<hr class="hr">', unsafe_allow_html=True)
        st.markdown("""<style>div.stButton > button{background:#7c4dbd;}
        div.stButton > button:hover{background:#4a2080;}</style>""", unsafe_allow_html=True)

        if not fruit_ok:
            st.warning("โมเดลยังไม่ถูกโหลด")
        elif st.button("จำแนกผลไม้ →"):
            with st.spinner("กำลังวิเคราะห์..."):
                try:
                    results=run_fruit(img,top_k=5)
                    top_name,top_conf=results[0]
                    medals=["🥇","🥈","🥉","4️⃣","5️⃣"]
                    conf_color="#0F6E56" if top_conf>=0.7 else "#854F0B" if top_conf>=0.4 else "#a81a1a"
                    conf_label="สูง" if top_conf>=0.7 else "ปานกลาง" if top_conf>=0.4 else "ต่ำ"

                    st.markdown(f"""
                    <div class="fruit-card">
                      <p class="fruit-label">ผลการจำแนก</p>
                      <p class="fruit-name">{top_name}</p>
                      <p class="fruit-conf">ความมั่นใจ: <b style="color:{conf_color}">{top_conf:.1%}</b> ({conf_label})</p>
                    </div>""", unsafe_allow_html=True)

                    st.markdown('<p class="sec-label" style="margin-top:1.4rem">Top-5 Predictions</p>', unsafe_allow_html=True)
                    for i,(name,prob) in enumerate(results):
                        st.markdown(f"""<div class="rank-row">
                          <span class="rank-medal">{medals[i]}</span>
                          <span class="rank-name">{name}</span>
                          <div class="rank-bar-wrap"><div class="rank-bar" style="width:{int(prob*100)}%"></div></div>
                          <span class="rank-pct">{prob:.1%}</span>
                        </div>""", unsafe_allow_html=True)

                    st.markdown(f"""<div class="conf-note">
                      ⓘ CNN 2-Block (Conv32→Conv64→Dense256) · TensorFlow/Keras · {len(class_names)} classes
                      · รูปจะถูก resize เป็น 128×128 และ normalize ÷255 ก่อนทำนาย
                    </div>""", unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"เกิดข้อผิดพลาด: {e}")
    else:
        st.markdown("""<div style="border:2px dashed #e0d5f5;border-radius:12px;padding:3rem;
             text-align:center;color:#bbb;margin-top:0.5rem;">
          <p style="font-size:3rem;margin:0">🍓</p>
          <p style="font-size:0.85rem;margin:10px 0 0">อัปโหลดรูปผลไม้เพื่อเริ่มการจำแนก</p>
        </div>""", unsafe_allow_html=True)