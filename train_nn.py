import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ── 1. แก้ไขปัญหา Compatibility สำหรับ RTX 50 Series ───────────
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices=false"
os.environ["XLA_FLAGS"] = "--xla_gpu_enable_triton_gemm=false"
os.environ["TF_CUDNN_USE_AUTOTUNE"] = "0"  # ป้องกัน Error ตอนเริ่ม Conv2D
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async" # การจอง VRAM แบบรวดเร็ว

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ── 2. ตั้งค่า GPU Memory & Mixed Precision ────────────────────
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        
        # เปิดใช้ Mixed Precision (ทำให้ RTX เทรนไวขึ้นมาก)
        from tensorflow.keras import mixed_precision
        mixed_precision.set_global_policy('mixed_float16')
        
        print(f"[GPU] ✓ กำลังใช้งาน: {tf.config.experimental.get_device_details(gpus[0])['device_name']}")
    except RuntimeError as e:
        print(f"[GPU] Error: {e}")

# =============================================================
# 1. CONFIGURATION
# =============================================================
DATASET_DIR  = Path("dataset/Fruit")
IMG_SIZE     = (128, 128)
BATCH_SIZE   = 64  # RTX 5060 Ti แรงพอจะใช้ 64-128 ได้สบายครับ
EPOCHS       = 30
MODEL_SAVE   = "fruit_cnn_model.keras"
CLASS_SAVE   = "class_names.json"

# =============================================================
# 2. DATA PIPELINE (ใช้โค้ดเดิมของคุณที่เขียนไว้ดีอยู่แล้ว)
# =============================================================
def build_generators():
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=0.2,
        rotation_range=20,
        horizontal_flip=True,
        fill_mode="nearest",
    )
    
    # ดึง Data จาก Directory
    train_gen = train_datagen.flow_from_directory(
        DATASET_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        subset="training", class_mode="categorical", shuffle=True
    )
    val_gen = train_datagen.flow_from_directory(
        DATASET_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        subset="validation", class_mode="categorical"
    )
    return train_gen, val_gen

# =============================================================
# 3. MODEL (ปรับโครงสร้างเล็กน้อยเพื่อรองรับ Mixed Precision)
# =============================================================
def build_model(num_classes):
    inputs = keras.Input(shape=(128, 128, 3))
    
    x = layers.Conv2D(32, 3, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Flatten()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    
    # สำคัญ: Layer สุดท้ายต้องใช้ dtype='float32' เสมอเมื่อใช้ Mixed Precision
    outputs = layers.Dense(num_classes, activation="softmax", dtype='float32')(x)
    
    return keras.Model(inputs, outputs)

# =============================================================
# 4. START TRAINING
# =============================================================
def main():
    train_gen, val_gen = build_generators()
    class_names = list(train_gen.class_indices.keys())
    
    # เซฟชื่อ Class ไว้ใช้กับ Streamlit
    with open(CLASS_SAVE, "w", encoding="utf-8") as f:
        json.dump(class_names, f, ensure_ascii=False)

    model = build_model(len(class_names))
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    # เพิ่ม Custom Callback เพื่อหยุดเมื่อ Accuracy > 90%
    class StopAt90(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if logs.get('val_accuracy') >= 0.90:
                print("\n[INFO] 🎯 Accuracy แตะ 90% แล้ว! หยุดเทรนและบันทึกโมเดล...")
                self.model.stop_training = True

    callbacks = [
        StopAt90(),
        keras.callbacks.ModelCheckpoint(MODEL_SAVE, save_best_only=True, monitor="val_accuracy")
    ]

    print("\n🚀 เริ่มการเทรนบน GPU...")
    model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS, callbacks=callbacks)
    print(f"✅ เทรนเสร็จสิ้น! บันทึกโมเดลที่: {MODEL_SAVE}")

if __name__ == "__main__":
    main()