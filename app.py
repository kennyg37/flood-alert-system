import os
import numpy as np
import pickle
import tensorflow as tf
from flask import Flask, request, jsonify
from sklearn.preprocessing import StandardScaler
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Flask app
app = Flask(__name__)
port = int(os.getenv("PORT", 5000))

# Paths to model and scaler
MODEL_PATH = os.getenv("MODEL_PATH", "saved_models/flood_model.keras")
SCALER_PATH = os.getenv("SCALER_PATH", "saved_models/scaler.pkl")

# Load the ML model
model = tf.keras.models.load_model(MODEL_PATH)

# Load the scaler
with open(SCALER_PATH, 'rb') as f:
    scaler = pickle.load(f)

if not isinstance(scaler, StandardScaler):
    raise ValueError("Loaded object is not a valid StandardScaler.")

# Helper functions
def preprocess_input(data, scaler):
    try:
        data_scaled = scaler.transform(data)
        return data_scaled
    except Exception as e:
        raise ValueError(f"Error during preprocessing: {e}")

def make_predictions(model, data_scaled):
    return model.predict(data_scaled)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = request.json.get("data")
        if not input_data:
            return jsonify({"error": "Input data is required."}), 400

        input_array = np.array(input_data)

        if input_array.ndim != 2:
            return jsonify({"error": "Input data must be a 2D array."}), 400

        data_scaled = preprocess_input(input_array, scaler)
        predictions = make_predictions(model, data_scaled)

        response = {
            "predictions": predictions.flatten().tolist(),
            "Will it flood?": ["Yes" if p > 0.5 else "No" for p in predictions.flatten()]
        }
        return jsonify(response), 200

    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400

    except Exception as e:
        return jsonify({"error": "An unexpected error occurred: " + str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=port)