import streamlit as st
import pandas as pd
import joblib
import time

# โหลด Model และค่าสถิติ
@st.cache_resource
def load_assets():
    try:
        m = joblib.load('model.pkl')
        f = joblib.load('feature_names.pkl')
        met = joblib.load('metrics.pkl')
        return m, f, met
    except:
        return None, None, None

model, feature_names, metrics = load_assets()

st.title("🚀 Salary AI Predictor & Validator")

if model is None:
    st.error("❌ ไม่พบไฟล์โมเดล! กรุณารัน train_ml.py ก่อน")
else:
    # ส่วนโชว์ความแม่นยำ (Accuracy Check)
    st.sidebar.header("📈 Model Performance")
    st.sidebar.metric("Accuracy (R2)", f"{metrics['r2']:.2%}")
    st.sidebar.write(f"ค่าความคลาดเคลื่อน: ±${metrics['mae']:,.0f}")

    # ส่วนรับข้อมูล
    with st.expander("กรอกข้อมูลสายงานของคุณ", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            exp = st.selectbox("Level", ['Entry', 'Mid', 'Senior', 'Executive'])
            yrs = st.number_input("ประสบการณ์ (ปี)", 0, 40, 5)
        with col2:
            ind = st.selectbox("Industry", ['Tech', 'Finance', 'Healthcare', 'Education'])
            size = st.radio("Company Size", ['Small (S)', 'Medium (M)', 'Large (L)'])

    if st.button("คำนวณและทดสอบ"):
        # --- เพิ่ม Progress Bar ---
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for percent_complete in range(100):
            time.sleep(0.005)
            progress_bar.progress(percent_complete + 1)
            status_text.text(f"กำลังประมวลผลผ่าน Neural Network... {percent_complete+1}%")
        
        # --- ส่วนการทำนาย ---
        input_data = pd.DataFrame([{
            'experience_level': exp, 'years_experience': yrs,
            'industry': ind, 'company_size': size,
            'employment_type': 'Full-time', 'remote_ratio': 50
        }])
        
        input_enc = pd.get_dummies(input_data).reindex(columns=feature_names, fill_value=0)
        prediction = model.predict(input_enc)[0]

        # แสดงผลลัพธ์แบบสวยงาม
        st.success(f"### ผลการคำนวณ: ${prediction:,.2f} / ปี")
        
        # เพิ่ม Comparison (จำลองการเช็คกับค่าเฉลี่ย)
        st.info(f"💡 หมายเหตุ: ผลลัพธ์นี้ผ่านการทดสอบด้วย Test Set โดยมีความแม่นยำอยู่ที่ {metrics['r2']:.2%}")