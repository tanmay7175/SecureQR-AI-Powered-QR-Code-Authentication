from flask import Flask, request, render_template, url_for
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import uuid  # To generate unique filenames
from werkzeug.utils import secure_filename  # For safe file handling

# Initialize Flask App
app = Flask(__name__)

# Load trained CNN model
MODEL_PATH = "cnn_qr_model.h5"
model = load_model(MODEL_PATH)

# Allowed file extensions
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

# Ensure 'static/uploads' directory exists
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Function to check allowed file extensions
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# Image Preprocessing Function
def preprocess_image(image_path):
    IMG_SIZE = 128  # Match model input size
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale
    if img is None:
        return None  # Invalid image
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))  # Resize to match model input
    img = img / 255.0  # Normalize pixel values
    img = img.reshape(1, IMG_SIZE, IMG_SIZE, 1)  # Reshape for CNN model
    return img

# Flask Routes
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    filename = None

    if request.method == "POST":
        file = request.files["file"]
        if file and allowed_file(file.filename):
            # Generate a unique filename to prevent overwriting
            ext = file.filename.rsplit(".", 1)[1].lower()
            filename = secure_filename(f"qr_{uuid.uuid4().hex}.{ext}")
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(file_path)  # Save uploaded image

            # Preprocess and classify image
            img = preprocess_image(file_path)
            if img is None:
                result = "⚠️ Invalid Image Format"
            else:
                prediction = model.predict(img)[0][0]  # Model Prediction
                result = "✅ Original QR Code" if prediction < 0.5 else "❌ Counterfeit QR Code"

            return render_template("home.html", result=result, filename=filename)

    return render_template("home.html", result=result)

# Run Flask App
if __name__ == "__main__":
    app.run(debug=True)
