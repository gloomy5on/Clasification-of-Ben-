from flask import Flask, render_template, request, send_from_directory, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
import uuid

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "model_fixed.h5")
model = load_model(MODEL_PATH, compile=False)

class_labels = [
    "battery",
    "biological",
    "cardboard",
    "clothes",
    "glass",
    "metal",
    "paper",
    "plastic",
    "shoes",
    "trash"
]

UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10MB

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# ==================================
# VALIDASI FILE
# ==================================
def allowed_file(filename):
    return "." in filename and \
           filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def predict_waste(image_path):
    try:
        image_size = model.input_shape[1]

        img = load_img(image_path, target_size=(image_size, image_size))
        img_array = img_to_array(img)

        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array, verbose=0)

        predicted_index = np.argmax(prediction)
        confidence = float(np.max(prediction))

        result = class_labels[predicted_index]

        return result.capitalize(), confidence

    except Exception as e:
        return f"Error: {str(e)}", 0

@app.route("/", methods=["GET", "POST"])
def index():

    if request.method == "GET":
        return render_template("index.html")

    if "file" not in request.files:
        return jsonify({"error": "File tidak ditemukan"})

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "Pilih file terlebih dahulu"})

    if file and allowed_file(file.filename):

        ext = file.filename.rsplit(".", 1)[1].lower()
        filename = f"{uuid.uuid4().hex}.{ext}"

        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        result, confidence = predict_waste(filepath)

        return jsonify({
            "result": result,
            "confidence": f"{confidence * 100:.2f}",
            "file_path": f"/uploads/{filename}"
        })

    return jsonify({"error": "Format file tidak didukung"})


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


if __name__ == "__main__":
    app.run(debug=False, use_reloader=False)