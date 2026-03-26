import streamlit as st
import tensorflow as tf
import numpy as np
import joblib
from PIL import Image

st.set_page_config(page_title="🍎 Fruit Classifier", page_icon="🍎", layout="wide")

IMG_SIZE_NN = (100, 100)
IMG_SIZE_ML = (32, 32)

# ── Load Models ──────────────────────────────────────────
@st.cache_resource
def load_models():
    nn_model    = tf.keras.models.load_model("model/fruit_classifier.h5")
    class_names = np.load("model/class_names.npy", allow_pickle=True).tolist()
    rf_model    = joblib.load("model/rf_model.pkl")
    le          = joblib.load("model/label_encoder.pkl")
    return nn_model, class_names, rf_model, le

# ── Predict Functions ─────────────────────────────────────
def predict_nn(image, model, class_names):
    img = image.convert("RGB").resize(IMG_SIZE_NN)
    arr = np.expand_dims(np.array(img) / 255.0, axis=0)
    probs   = model.predict(arr)[0]
    top3    = probs.argsort()[-3:][::-1]
    return [{"name": class_names[i], "confidence": float(probs[i])} for i in top3]

def predict_ml(image, rf, le):
    img = image.convert("RGB").resize(IMG_SIZE_ML)
    arr = np.array(img).flatten() / 255.0
    probs   = rf.predict_proba([arr])[0]
    top3    = probs.argsort()[-3:][::-1]
    return [{"name": le.inverse_transform([i])[0], "confidence": float(probs[i])} for i in top3]

def render_results(results):
    top = results[0]
    st.metric("ผลไม้ที่น่าจะเป็น", top["name"],
              f"ความมั่นใจ {top['confidence']*100:.1f}%")
    st.write("**Top 3:**")
    for i, r in enumerate(results):
        emoji = ["🥇","🥈","🥉"][i]
        st.progress(r["confidence"],
                    text=f"{emoji} {r['name']} — {r['confidence']*100:.1f}%")

# ── UI ────────────────────────────────────────────────────
st.title("🍎 Fruit Classifier — ML vs NN")
st.caption("เปรียบเทียบ Random Forest (ML) กับ MobileNetV2 (Neural Network)")
st.divider()

with st.spinner("กำลังโหลด model..."):
    try:
        nn_model, class_names, rf_model, le = load_models()
        st.success(f"โหลด model สำเร็จ! รองรับผลไม้ {len(class_names)} ชนิด")
    except Exception as e:
        st.error(f"โหลด model ไม่สำเร็จ: {e}")
        st.stop()

uploaded = st.file_uploader("📷 อัปโหลดรูปผลไม้", type=["jpg","jpeg","png","webp"])

if uploaded:
    image = Image.open(uploaded)

    # แสดงรูปตรงกลาง
    col_img, _ = st.columns([1, 2])
    with col_img:
        st.image(image, caption="รูปที่อัปโหลด", use_column_width=True)

    st.divider()

    # ผลจากสองโมเดลแบบ side-by-side
    col_ml, col_nn = st.columns(2)

    with col_ml:
        st.subheader("🌲 Random Forest (ML)")
        with st.spinner("กำลังวิเคราะห์..."):
            results_ml = predict_ml(image, rf_model, le)
        render_results(results_ml)

    with col_nn:
        st.subheader("🧠 MobileNetV2 (NN)")
        with st.spinner("กำลังวิเคราะห์..."):
            results_nn = predict_nn(image, nn_model, class_names)
        render_results(results_nn)

    # สรุปเปรียบเทียบ
    st.divider()
    st.subheader("📊 สรุปเปรียบเทียบ")
    same = results_ml[0]["name"] == results_nn[0]["name"]
    if same:
        st.success(f"✅ ทั้งสองโมเดลเห็นตรงกัน: **{results_nn[0]['name']}**")
    else:
        st.warning(
            f"⚠️ ผลต่างกัน — ML: **{results_ml[0]['name']}** | NN: **{results_nn[0]['name']}**"
        )

    comp_data = {
        "โมเดล":        ["Random Forest", "MobileNetV2"],
        "ผลที่ได้":      [results_ml[0]["name"], results_nn[0]["name"]],
        "ความมั่นใจ":   [f"{results_ml[0]['confidence']*100:.1f}%",
                          f"{results_nn[0]['confidence']*100:.1f}%"],
    }
    st.table(comp_data)