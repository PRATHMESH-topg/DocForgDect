import os
import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50
from utils import get_data_generators

# ================================================
# 🔧 CONFIGURATION
# ================================================
DATASET_PATH = r"C:\Users\liasp\OneDrive\Desktop\DocForgedDec\dataset"
MODEL_PATH = "forgery_model.keras"
IMG_SIZE = 224
BATCH_SIZE = 8
EPOCHS = 40

# ================================================
# 📦 LOAD DATA
# ================================================
print("📦 Loading dataset and applying augmentations...")
train_gen, val_gen = get_data_generators(DATASET_PATH, img_size=IMG_SIZE, batch_size=BATCH_SIZE)

# ================================================
# 🧩 MODEL: Transfer Learning with ResNet50
# ================================================
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
base_model.trainable = False  # freeze early layers

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.4)(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# ================================================
# 🧠 CALLBACKS (no early stop)
# ================================================
callbacks = [
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1),
    ModelCheckpoint(
        MODEL_PATH,
        save_best_only=True,
        monitor='val_accuracy',
        mode='max',
        verbose=1
    )
]

# ================================================
# 🚀 TRAIN MODEL (Run all epochs)
# ================================================
print(f"🚀 Training started for {EPOCHS} epochs (no early stop)...")
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

# ================================================
# ✅ SAVE FINAL & BEST MODEL
# ================================================
best_val_acc = max(history.history['val_accuracy']) * 100
model.save(MODEL_PATH)
print(f"\n✅ Training finished — ran all {EPOCHS} epochs!")
print(f"🏆 Best Validation Accuracy: {best_val_acc:.2f}%")
print(f"💾 Best model saved as: {MODEL_PATH}")

