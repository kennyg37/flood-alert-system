import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_model(model_path):
    return tf.keras.models.load_model(model_path)

def preprocess_input(data, scaler_path):
    scaler = np.load(scaler_path, allow_pickle=True)
    
    if isinstance(scaler, StandardScaler):
        data_scaled = scaler.transform(data)
        return data_scaled
    else:
        raise ValueError("Loaded object is not a valid StandardScaler.")


def make_predictions(model, data_scaled):
    return model.predict(data_scaled)


if __name__ == "__main__":
    
    model_path = "saved_models/flood_model.keras"  
    scaler_path = "saved_models/scaler.pkl"  
    
    model = load_model(model_path)
    
    new_data = [
        [15.0, 0.95, 1200.0, 14.8]
    ]
    
    
    new_data_scaled = preprocess_input(np.array(new_data), scaler_path)
    
    predictions = make_predictions(model, new_data_scaled)
    print("Predictions:", predictions)
