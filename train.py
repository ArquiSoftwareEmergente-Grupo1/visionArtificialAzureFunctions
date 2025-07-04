import os
import cv2
import numpy as np
import json
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, models

# ========== CONFIG ==========
DATASET_PATH = r"C:\Users\user\.cache\kagglehub\datasets\ashishmotwani\tomato\versions\1\train"
IMG_SIZE = (128, 128)
BALANCE_METHOD = "oversample"  # o "undersample"

# ========== ETIQUETADO ==========
label_map = {label: idx for idx, label in enumerate(sorted(os.listdir(DATASET_PATH)))}

X = []
y = []

for label in os.listdir(DATASET_PATH):
    class_path = os.path.join(DATASET_PATH, label)
    if not os.path.isdir(class_path):
        continue
    for img_file in os.listdir(class_path):
        img_path = os.path.join(class_path, img_file)
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.resize(img, IMG_SIZE)
        X.append(img)
        y.append(label_map[label])

X = np.array(X) / 255.0
y = np.array(y)

print(f"Dataset original: {X.shape}, Labels: {np.unique(y, return_counts=True)}")

# ========== BALANCEO ==========
def balance_dataset(X, y, method="oversample"):
    data = defaultdict(list)
    for img, label in zip(X, y):
        data[label].append(img)

    max_count = max(len(imgs) for imgs in data.values())
    min_count = min(len(imgs) for imgs in data.values())

    new_X, new_y = [], []

    for label, imgs in data.items():
        if method == "oversample":
            balanced_imgs = resample(imgs, replace=True, n_samples=max_count, random_state=42)
        elif method == "undersample":
            balanced_imgs = resample(imgs, replace=False, n_samples=min_count, random_state=42)
        else:
            raise ValueError("method must be 'oversample' or 'undersample'")
        
        new_X.extend(balanced_imgs)
        new_y.extend([label] * len(balanced_imgs))

    return np.array(new_X), np.array(new_y)

X, y = balance_dataset(X, y, method=BALANCE_METHOD)
print(f"Dataset balanceado: {X.shape}, Labels: {np.unique(y, return_counts=True)}")

# ========== SEPARAR DATOS ==========
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)

# ========== MODELO ==========
model = models.Sequential([
    layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(label_map), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# ========== ENTRENAMIENTO ==========
model.fit(X_train, y_train_cat, epochs=10, validation_data=(X_test, y_test_cat))

# ========== GUARDADO ==========
model.save("modelo_tomate.h5")

with open("label_map.json", "w") as f:
    json.dump(label_map, f)

print("✅ Modelo y etiquetas guardados correctamente.")
