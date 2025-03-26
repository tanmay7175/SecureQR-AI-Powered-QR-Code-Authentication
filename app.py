from flask import Flask, request, render_template
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import uuid
from werkzeug.utils import secure_filename

# Disable GPU to avoid unnecessary TensorFlow CUDA warnings
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Initialize Flask App
app = Flask(__name__)

# Define the upload folder path
UPLOAD_FOLDER = os.path.join(app.root_path, "static/uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Ensure directory exists
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

# Load the trained CNN model with error handling
MODEL_PATH = "cnn_qr_model.h5"
model = None
try:
    model = load_model(MODEL_PATH)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])  # Ensure the model is compiled
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")

# Function to check allowed file extensions
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# Image Preprocessing Function
def preprocess_image(image_path):
    IMG_SIZE = 128  # Match model input size
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Load image in grayscale
        if img is None:
            return None  # Invalid image
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))  # Resize to match model input
        img = img / 255.0  # Normalize pixel values
        img = img.reshape(1, IMG_SIZE, IMG_SIZE, 1)  # Reshape for CNN model
        return img
    except Exception as e:
        print(f"⚠️ Error processing image: {e}")
        return None

# Flask Routes
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    filename = None

    if request.method == "POST":
        if "file" not in request.files or request.files["file"].filename == "":
            return render_template("home.html", result="⚠️ No file uploaded!")

        file = request.files["file"]
        if file and allowed_file(file.filename):
            try:
                # Generate a unique filename to prevent overwriting
                ext = file.filename.rsplit(".", 1)[1].lower()
                filename = secure_filename(f"qr_{uuid.uuid4().hex}.{ext}")
                file_path = os.path.join(UPLOAD_FOLDER, filename)
                file.save(file_path)  # Save uploaded image

                # Preprocess and classify image
                img = preprocess_image(file_path)
                if img is None:
                    result = "⚠️ Invalid Image Format"
                elif model is None:
                    result = "❌ Model not loaded. Please check server logs."
                else:
                    prediction = model.predict(img)[0][0]  # Model Prediction
                    result = "✅ Original QR Code" if prediction < 0.5 else "❌ Counterfeit QR Code"

            except Exception as e:
                result = f"⚠️ Error processing file: {e}"

            return render_template("home.html", result=result, filename=filename)

    return render_template("home.html", result=result)
