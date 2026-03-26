import os, numpy as np, joblib, time
from PIL import Image
from skimage.feature import hog
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

TRAIN_DIR = "dataset/Training"
IMG_SIZE  = (64, 64)
MODEL_DIR = "model"

# ══════════════════════════════════════════
# Helper: Progress Bar
# ══════════════════════════════════════════
def progress_bar(current, total, label="", bar_len=30):
    filled = int(bar_len * current / total)
    bar    = "█" * filled + "░" * (bar_len - filled)
    pct    = current / total * 100
    print(f"\r  [{bar}] {pct:5.1f}%  {label}", end="", flush=True)

def section(title):
    print(f"\n{'═'*55}")
    print(f"  {title}")
    print(f"{'═'*55}")

def step(msg):
    print(f"\n  ▶  {msg}")

def done(msg, elapsed=None):
    t = f"  ({elapsed:.1f}s)" if elapsed else ""
    print(f"\r  ✅ {msg}{t}")

# ══════════════════════════════════════════
# Feature Extraction (HOG + Color)
# ══════════════════════════════════════════
def extract_features(img_array):
    hog_feat = hog(
        img_array,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        channel_axis=-1
    )
    color_hist = []
    for ch in range(3):
        hist, _ = np.histogram(img_array[:,:,ch], bins=32, range=(0,1))
        color_hist.extend(hist / hist.sum())
    return np.concatenate([hog_feat, color_hist])

# ══════════════════════════════════════════
# โหลดรูปภาพ + Progress Bar
# ══════════════════════════════════════════
def load_images(base_dir):
    X, y = [], []
    labels = sorted([l for l in os.listdir(base_dir)
                     if os.path.isdir(os.path.join(base_dir, l))])

    print(f"\n  พบ {len(labels)} label: {', '.join(labels)}\n")

    total_files = sum(
        len(os.listdir(os.path.join(base_dir, l))) for l in labels
    )
    processed = 0
    t0 = time.time()

    for label in labels:
        folder = os.path.join(base_dir, label)
        files  = os.listdir(folder)
        ok     = 0

        for fname in files:
            try:
                img = Image.open(os.path.join(folder, fname)).convert("RGB")
                arr = np.array(img.resize(IMG_SIZE)) / 255.0
                X.append(extract_features(arr))
                y.append(label)
                ok += 1
            except:
                pass
            processed += 1
            progress_bar(processed, total_files,
                         f"{label} ({ok}/{len(files)})")

        elapsed = time.time() - t0
        rate    = processed / elapsed if elapsed > 0 else 0
        remain  = (total_files - processed) / rate if rate > 0 else 0
        print(f"\n  └─ {label}: {ok} รูป  |  "
              f"ผ่านมา {elapsed:.0f}s  |  "
              f"เหลืออีก ~{remain:.0f}s")

    print()
    return np.array(X), np.array(y)

# ══════════════════════════════════════════
# STEP 1 — โหลด Dataset
# ══════════════════════════════════════════
section("STEP 1/5 — โหลด Dataset")
t_start = time.time()
X, y = load_images(TRAIN_DIR)
elapsed = time.time() - t_start
done(f"โหลดเสร็จ: {len(X)} รูป  |  Feature size: {X.shape[1]}", elapsed)

# ══════════════════════════════════════════
# STEP 2 — Preprocessing
# ══════════════════════════════════════════
section("STEP 2/5 — Preprocessing")

step("Label Encoding...")
le    = LabelEncoder()
y_enc = le.fit_transform(y)
done(f"Classes: {list(le.classes_)}")

step("Standard Scaling...")
scaler = StandardScaler()
X_sc   = scaler.fit_transform(X)
done("Scale เสร็จ")

step("แบ่ง Train / Test (80/20)...")
X_train, X_test, y_train, y_test = train_test_split(
    X_sc, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)
done(f"Train: {len(X_train)} รูป  |  Test: {len(X_test)} รูป")

# ══════════════════════════════════════════
# STEP 3 — สร้าง Ensemble Stack
# ══════════════════════════════════════════
section("STEP 3/5 — สร้าง Ensemble Stack")

step("ตั้งค่า Base Models...")
base_models = [
    ("rf", RandomForestClassifier(
        n_estimators=300,
        max_features="sqrt",
        n_jobs=-1,
        random_state=42
    )),
    ("xgb", XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        verbosity=0,
        n_jobs=-1,
        random_state=42
    )),
    ("svm", SVC(
        kernel="rbf",
        C=10,
        gamma="scale",
        probability=True,
        random_state=42
    )),
]
meta_model = LogisticRegression(max_iter=1000, C=5, n_jobs=-1)
done("Random Forest  +  XGBoost  +  SVM  →  Logistic Regression")

step("สร้าง StackingClassifier (5-fold CV)...")
stack = StackingClassifier(
    estimators=base_models,
    final_estimator=meta_model,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    passthrough=True,
    n_jobs=-1
)
done("พร้อมเทรน")

# ══════════════════════════════════════════
# STEP 4 — เทรน
# ══════════════════════════════════════════
section("STEP 4/5 — เทรน Ensemble Stack")
print("\n  ⏳ กำลังเทรน... (ประมาณ 10-20 นาที)")
print("  การทำงาน: เทรน RF + XGB + SVM แต่ละ fold → เทรน Meta-model\n")

# แสดง progress แต่ละโมเดล
model_names = ["Random Forest", "XGBoost", "SVM", "Meta-model (LR)"]
for i, name in enumerate(model_names, 1):
    progress_bar(i - 1, len(model_names), f"รอเทรน {name}...")
    time.sleep(0.1)

print()
t_train = time.time()
stack.fit(X_train, y_train)
train_elapsed = time.time() - t_train

done(f"เทรนเสร็จทั้งหมด", train_elapsed)

# ══════════════════════════════════════════
# STEP 5 — ประเมินผล
# ══════════════════════════════════════════
section("STEP 5/5 — ประเมินผล")

step("กำลัง predict test set...")
y_pred = stack.predict(X_test)
acc    = accuracy_score(y_test, y_pred)

print(f"""
  ┌─────────────────────────────────┐
  │  🎯 Test Accuracy : {acc*100:6.2f}%       │
  │  📦 จำนวน test   : {len(X_test):5d} รูป     │
  │  🏷️  จำนวน class  : {len(le.classes_):5d} label   │
  │  ⏱️  เวลาเทรน    : {train_elapsed:5.0f}s         │
  └─────────────────────────────────┘
""")

print("  Classification Report:")
print("  " + "-"*50)
report = classification_report(y_test, y_pred, target_names=le.classes_)
for line in report.split("\n"):
    print(f"  {line}")

# ══════════════════════════════════════════
# บันทึก Model
# ══════════════════════════════════════════
section("บันทึก Model")

os.makedirs(MODEL_DIR, exist_ok=True)

files_to_save = [
    (stack,  "rf_model.pkl",       "Ensemble Stack"),
    (le,     "label_encoder.pkl",  "Label Encoder"),
    (scaler, "scaler.pkl",         "Scaler"),
]

for obj, fname, label in files_to_save:
    step(f"บันทึก {label}...")
    path = os.path.join(MODEL_DIR, fname)
    joblib.dump(obj, path)
    size = os.path.getsize(path) / (1024*1024)
    done(f"{fname}  ({size:.1f} MB)")

total_elapsed = time.time() - t_start
print(f"""
{'═'*55}
  ✅ เสร็จสมบูรณ์!
  📁 ไฟล์อยู่ที่ : {MODEL_DIR}/
  ⏱️  เวลาทั้งหมด : {total_elapsed/60:.1f} นาที
  🎯 Accuracy    : {acc*100:.2f}%
{'═'*55}
""")