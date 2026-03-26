import os, numpy as np, tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

TRAIN_DIR = "/kaggle/input/fruits/fruits-360_dataset/fruits-360/Training"
IMG_SIZE  = (100, 100)
BATCH     = 32

# Data generators
gen = ImageDataGenerator(rescale=1./255, rotation_range=15,
                         zoom_range=0.1, horizontal_flip=True,
                         validation_split=0.2)

train_gen = gen.flow_from_directory(TRAIN_DIR, target_size=IMG_SIZE,
                                    batch_size=BATCH, class_mode='categorical',
                                    subset='training')
val_gen   = gen.flow_from_directory(TRAIN_DIR, target_size=IMG_SIZE,
                                    batch_size=BATCH, class_mode='categorical',
                                    subset='validation')

# Build model (Transfer Learning)
base = MobileNetV2(input_shape=(100,100,3), include_top=False, weights='imagenet')
base.trainable = False

model = models.Sequential([
    base,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(train_gen.num_classes, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_gen, epochs=10, validation_data=val_gen,
          callbacks=[tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)])

# Save
os.makedirs("model", exist_ok=True)
model.save("model/fruit_classifier.h5")
np.save("model/class_names.npy", list(train_gen.class_indices.keys()))
print("✅ NN saved!")