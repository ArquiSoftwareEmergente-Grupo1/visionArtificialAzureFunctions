from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
import json
import cv2

# ========== CONFIG ==========
MODEL_PATH = "modelo_tomate.h5"
LABELS_PATH = "label_map.json"
IMG_SIZE = (128, 128)

# ========== CARGAR MODELO Y MAPA DE ETIQUETAS ==========
model = load_model(MODEL_PATH)

with open(LABELS_PATH, "r") as f:
    label_map = json.load(f)

inv_label_map = {v: k for k, v in label_map.items()}

# ========== API ==========
app = Flask(__name__)

@app.route("/")
def home():
    return "API para clasificación de hojas de tomate - CNN"

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No se encontró la imagen con la clave 'image'"}), 400

    file = request.files["image"]
    img_bytes = file.read()

    # Leer imagen desde bytes con OpenCV
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "No se pudo leer la imagen"}), 400

    img = cv2.resize(img, IMG_SIZE)
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    preds = model.predict(img)[0]
    class_idx = int(np.argmax(preds))
    confidence = float(np.max(preds))
    predicted_class = inv_label_map[class_idx]

    return jsonify({
        "class": predicted_class,
        "confidence": round(confidence, 4)
    })

if __name__ == "__main__":
    app.run(debug=True)
