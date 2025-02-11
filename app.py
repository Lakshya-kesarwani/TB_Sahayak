from flask import Flask, request, jsonify, json, make_response, render_template
import random
import base64
import numpy as np
import os
import requests
from io import BytesIO
from PIL import Image
from flask_cors import CORS, cross_origin
from keras.models import load_model
import cv2
# Fix for Qt issue with OpenCV
os.environ["QT_QPA_PLATFORM"] = "offscreen"

app = Flask(__name__)
CORS(app)
MODEL_URL = "https://github.com/Lakshya-kesarwani/TB_Sahayak/releases/download/v1.0/tb_classifier.h5"
MODEL_PATH = "tb_classifier.h5"

# Global model variable
tb_model = None
def download_model():
    """Download model if not exists"""
    if not os.path.exists(MODEL_PATH):
        print("Downloading model from GitHub Releases...")
        response = requests.get(MODEL_URL, stream=True)
        with open(MODEL_PATH, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        print("Download complete!")

def load_tb_model():
    """Load the model into memory"""
    global tb_model
    if tb_model is None:
        tb_model = load_model(MODEL_PATH)
        print("Model loaded successfully!")

# Run once at startup
download_model()
load_tb_model()


# Load the model after ensuring it's available
from keras.models import load_model
import joblib

# Helper function to convert base64 image to numpy array

#pages
@app.route('/')
@cross_origin()
def home():
    return render_template('index.html')

@app.route('/guide')
@cross_origin()
def guide():
    return render_template('guide.html')

@app.route('/xray')
@cross_origin()
def xray():
    return render_template('xray.html')

@app.route('/symptom')
@cross_origin()
def symptom():
    return render_template('symptoms.html')

@app.route('/find')
@cross_origin()
def find():
    return render_template('find.html',api_key=os.getenv("API_KEY"))

@app.route('/contact')
@cross_origin()
def contact():
    return render_template('contact.html')

#xray prediction
def preprocess_image(image_file):
    """
    Convert uploaded image file to a NumPy array for model prediction.
    """
    try:
        image = Image.open(image_file).convert("RGB")  # Ensure 3-channel RGB
        image = image.resize((224, 224))  # Resize to match model input

        # Convert to NumPy array
        image_array = np.array(image) / 255.0  # Normalize pixel values
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

        print("Image Shape:", image_array.shape)  # Debugging info
        return image_array
    except Exception as e:
        print("Error processing image:", e)
        return None
@app.route("/pred", methods=["GET", "POST"])
@cross_origin()
def upload_xray():
    """
    Handle X-ray image upload and prediction.
    """
    if request.method == "POST":
        if "file" not in request.files:
            return render_template("index.html", error="No file selected")

        file = request.files["file"]
        if file.filename == "":
            return render_template("index.html", error="No file selected")

        image = preprocess_image(file)
        if image is None:
            return render_template("index.html", error="Error processing image")

        # Get model prediction
        prediction = tb_model.predict(image)
        prediction_result = round(float(prediction[0][0]) * 100, 2)  # Convert to percentage

        # return render_template("xray.html", prediction=prediction_result)
        return jsonify({"data":prediction_result})

    return render_template("xray.html")


#symptom prediction
@app.route("/pred-symptom", methods=["GET", "POST"])
@cross_origin()
def predict_symptom():
    """
    Handle symptom form submission and prediction.
    """
    if request.method == "POST":
        data = request.get_json()
        symptoms_pred = data.get("symptoms")
        print("********",symptoms_pred,"********")
        age = 12
        loaded_model = load_model("tb_prediction_model.h5")

        # Load the scaler
        scaler = joblib.load("scaler.pkl")
        # Example new patient data (0s and 1s)
        new_patient = np.array([symptoms_pred])  # Example input
        # Scale input data
        new_patient_scaled = scaler.transform(new_patient)
        # Get TB probability
        tb_probability = float(loaded_model.predict(new_patient_scaled)[0][0])
        print(f"Probability of TB: {tb_probability:.2f}")
        # Generate a random prediction (0 or 1)
        prediction = random.choice(["No Disease", "Possible TB Detected"])

        return jsonify({"Prediction":tb_probability})

    return render_template("symptoms.html")


#cors
def _build_cors_preflight_response():
    response = make_response()
    response.headers.add("Access-Control-Allow-Origin", "https://tb-sahayak.onrender.com")
    # response.headers.add("Access-Control-Allow-Origin", "http://localhost:8000")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type,Authorization")
    response.headers.add("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
    return response, 200  # Explicitly return 200 status

@app.after_request
def after_request(response):
    response.headers.set("Access-Control-Allow-Origin", "https://tb-sahayak.onrender.com")
    # response.headers.set("Access-Control-Allow-Origin", "http://localhost:8000")
    response.headers.set("Access-Control-Allow-Headers", "Content-Type,Authorization")
    response.headers.set("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
    return response

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
