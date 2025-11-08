import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = 224


# ===============================
# 🧠 Preprocess Single Image (for prediction)
# ===============================
def preprocess_image(image_path, img_size=IMG_SIZE):
    """
    Loads and preprocesses an image for inference using ResNet preprocessing.
    """
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(img_size, img_size))
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    # Normalize using ResNet preprocessing
    img = tf.keras.applications.resnet50.preprocess_input(img)
    return img


# ===============================
# 🔍 Blur Detection
# ===============================
def calculate_blur_score(image_path):
    """
    Calculates blur score (variance of Laplacian) to estimate image sharpness.
    Lower = blurrier image.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return 0
    return round(cv2.Laplacian(image, cv2.CV_64F).var(), 2)


# ===============================
# ⚙️ Data Generators (with strong augmentation)
# ===============================
def get_data_generators(base_dir, img_size=IMG_SIZE, batch_size=8, val_split=0.2):
    """
    Creates training and validation generators directly from dataset directory.
    Uses strong augmentation to generalize small datasets.
    """
    datagen = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.resnet50.preprocess_input,
        rotation_range=40,
        width_shift_range=0.25,
        height_shift_range=0.25,
        shear_range=0.2,
        zoom_range=0.3,
        horizontal_flip=True,
        brightness_range=[0.6, 1.4],
        fill_mode='nearest',
        validation_split=val_split
    )

    # Training generator
    train_gen = datagen.flow_from_directory(
        base_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='binary',
        subset='training',
        shuffle=True
    )

    # Validation generator
    val_gen = datagen.flow_from_directory(
        base_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='binary',
        subset='validation',
        shuffle=False
    )

    return train_gen, val_gen
