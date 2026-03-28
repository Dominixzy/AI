import pandas as pd
import numpy as np
import joblib
import time
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

# 1. โหลดข้อมูล 15,000 แถว
print("⏳ กำลังโหลดและประมวลผลข้อมูล 15,000 แถว...")
df = pd.read_csv('dataset/salary.csv', skipinitialspace=True)

# 2. Feature Engineering (หัวใจของการแตะ 90%)
# Counts of skills
df['skill_count'] = df['required_skills'].str.split(',').str.len().fillna(0)

# Education to Ordinal
edu_map = {'Bachelor': 1, 'Master': 2, 'PhD': 3, 'Associate': 0}
df['edu_score'] = df['education_required'].map(edu_map).fillna(1)

# แปลงวันที่เพื่อหาความเร่งด่วน (Days to Deadline)
df['posting_date'] = pd.to_datetime(df['posting_date'])
df['application_deadline'] = pd.to_datetime(df['application_deadline'])
df['days_to_deadline'] = (df['application_deadline'] - df['posting_date']).dt.days

# ลบข้อมูลที่ขัดแย้งกัน (Outliers)
df = df[~((df['experience_level'] == 'EX') & (df['years_experience'] < 5))]

# เลือก Features
features = [
    'experience_level', 'industry', 'years_experience', 'company_location', 
    'skill_count', 'edu_score', 'remote_ratio', 'company_size', 
    'days_to_deadline', 'benefits_score', 'job_description_length'
]
X = pd.get_dummies(df[features], drop_first=True)
y = df['salary_usd']
feature_names = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

# 3. กำหนด Parameters สำหรับ GridSearchCV แบบละเอียดที่สุด
print("⚙️ เริ่มต้นการทำ GridSearchCV แบบละเอียด (เวลานานได้ไม่มีปัญหา)...")
xgb_model = XGBRegressor(
    objective='reg:squarederror', 
    tree_method='hist',      # ใช้ระบบ Histogram (จำเป็นสำหรับ GPU)
    device='cuda',           # สั่งให้รันบนการ์ดจอ (NVIDIA)
    random_state=42
)
# ปรับจูนค่าต่างๆ หลายร้อยรูปแบบ
param_grid = {
    'n_estimators': [1000, 1500, 2000],          # จำนวนต้นไม้ (เน้นเรียนรู้ลึกๆ)
    'learning_rate': [0.01, 0.03, 0.05],         # อัตราการเรียนรู้ (เน้นค่อยๆ เรียน)
    'max_depth': [8, 10, 12, 15],                # ความลึกของต้นไม้ (เน้นหาความสัมพันธ์ที่ซับซ้อน)
    'subsample': [0.8, 0.9, 1.0],                 # สุ่มข้อมูลบางส่วนเพื่อป้องกัน Overfitting
    'colsample_bytree': [0.8, 0.9, 1.0],          # สุ่ม Feature บางส่วน
    'gamma': [0, 0.1, 0.2]                        # เพิ่มข้อกำหนดในการแตกกิ่ง
}

# 4. เทรนโมเดลด้วย GridSearchCV (ใช้เวลาประมาณ 10-30 นาที ขึ้นอยู่กับคอมพิวเตอร์)
# n_jobs=-1 เพื่อใช้ CPU ทุกคอร์ที่มี
grid_search = GridSearchCV(
    estimator=xgb_model, 
    param_grid=param_grid, 
    cv=5,               # Cross-Validation 5 รอบ
    scoring='r2', 
    n_jobs=-1, 
    verbose=2           # โชว์รายละเอียดระหว่างเทรน
)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

# 5. วัดผล Accurate สูงสุด
y_pred = best_model.predict(X_test)
final_acc = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"\n🏆 Best Model Results:")
print(f"✅ Final R2 Score: {final_acc:.2%}")
print(f"📍 Mean Absolute Error: ${mae:,.2f}")
print(f"📍 Best Parameters: {grid_search.best_params_}")

# 6. บันทึกโมเดลเวอร์ชัน "90%++"
joblib.dump(best_model, 'model.pkl')
joblib.dump(feature_names, 'feature_names.pkl')
joblib.dump({'r2': final_acc, 'mae': mae}, 'metrics.pkl')
print("💾 บันทึกโมเดลความแม่นยำสูงเรียบร้อย!")