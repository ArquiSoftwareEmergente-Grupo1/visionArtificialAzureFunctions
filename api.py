from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from tensorflow.keras.models import load_model
import numpy as np
import json
import cv2
import os
import datetime
from werkzeug.utils import secure_filename

# ========== CONFIG ==========
MODEL_PATH = "modelo_tomate.h5"
LABELS_PATH = "label_map.json"
IMG_SIZE = (128, 128)
UPLOAD_FOLDER = "imagenes_guardadas"  # Carpeta donde se guardarán las imágenes
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# Crear carpeta si no existe
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ========== CARGAR MODELO Y MAPA DE ETIQUETAS ==========
model = load_model(MODEL_PATH)

with open(LABELS_PATH, "r") as f:
    label_map = json.load(f)

inv_label_map = {v: k for k, v in label_map.items()}

# ========== API ==========
app = Flask(__name__)
CORS(app)  # <-- Habilita CORS globalmente

@app.route("/")
def home():
    return "API para clasificación de hojas de tomate - CNN"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No se encontró la imagen con la clave 'image'"}), 400

        file = request.files["image"]
        
        # Validar que el archivo tiene un nombre
        if file.filename == '':
            return jsonify({"error": "No se seleccionó ningún archivo"}), 400
        
        # Validar extensión del archivo
        if not allowed_file(file.filename):
            return jsonify({"error": "Tipo de archivo no soportado. Use: png, jpg, jpeg, gif, bmp"}), 400
        
        # Leer la imagen primero para procesamiento
        img_bytes = file.read()
        
        # Verificar que tenemos datos
        if len(img_bytes) == 0:
            return jsonify({"error": "El archivo de imagen está vacío"}), 400
        
        print(f"Imagen recibida: {file.filename}, tamaño: {len(img_bytes)} bytes")
        
        # Generar nombre único para la imagen
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        cultivoId = request.form.get('cultivoId', 'unknown')
        original_filename = secure_filename(file.filename)
        filename = f"analisis_{cultivoId}_{timestamp}_{original_filename}"
        
        # Guardar la imagen original
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        with open(file_path, 'wb') as f:
            f.write(img_bytes)
        
        print(f"Imagen guardada en: {file_path}")
        
        # Procesar la imagen para análisis
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            # Si falla la decodificación, intentar leer desde el archivo guardado
            print("Falló cv2.imdecode, intentando cv2.imread...")
            img = cv2.imread(file_path)
            if img is None:
                return jsonify({"error": "No se pudo leer la imagen. Formato no soportado."}), 400

        print(f"Imagen procesada con éxito, dimensiones: {img.shape}")

        img_processed = cv2.resize(img, IMG_SIZE)
        img_processed = img_processed.astype("float32") / 255.0
        img_processed = np.expand_dims(img_processed, axis=0)

        preds = model.predict(img_processed)[0]
        class_idx = int(np.argmax(preds))
        confidence = float(np.max(preds))
        predicted_class = inv_label_map[class_idx]

        print(f"Predicción: {predicted_class}, Confianza: {confidence}")

        return jsonify({
            "class": predicted_class,
            "confidence": round(confidence, 4),
            "saved_image_path": filename,
            "saved_image_url": f"/images/{filename}"
        })
        
    except Exception as e:
        print(f"Error en predict: {str(e)}")
        return jsonify({"error": f"Error interno del servidor: {str(e)}"}), 500

@app.route("/images/<filename>", methods=["GET"])
def get_image(filename):
    """Endpoint para servir las imágenes guardadas"""
    try:
        # Asegurar que el filename es seguro
        safe_filename = secure_filename(filename)
        file_path = os.path.join(UPLOAD_FOLDER, safe_filename)
        
        # Verificar que el archivo existe
        if not os.path.exists(file_path):
            return jsonify({"error": "Imagen no encontrada"}), 404
            
        # Servir el archivo
        return send_file(file_path, as_attachment=False)
    except Exception as e:
        return jsonify({"error": f"Error al servir la imagen: {str(e)}"}), 500

@app.route("/images", methods=["GET"])
def list_images():
    """Endpoint para listar todas las imágenes guardadas"""
    try:
        if not os.path.exists(UPLOAD_FOLDER):
            return jsonify({"images": []})
            
        files = []
        for filename in os.listdir(UPLOAD_FOLDER):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                file_path = os.path.join(UPLOAD_FOLDER, filename)
                file_stats = os.stat(file_path)
                files.append({
                    "filename": filename,
                    "url": f"/images/{filename}",
                    "size": file_stats.st_size,
                    "created": datetime.datetime.fromtimestamp(file_stats.st_ctime).isoformat()
                })
                
        return jsonify({"images": files})
    except Exception as e:
        return jsonify({"error": f"Error al listar imágenes: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
