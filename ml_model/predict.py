import joblib
import numpy as np

def predict(input_features):
    # Load the model and the scaler
    model = joblib.load('model.pkl')
    scaler = joblib.load('scaler.pkl')
    
    # Scale the input features
    input_features = np.array(input_features).reshape(1, -1)
    input_features = scaler.transform(input_features)
    
    # Predict the cluster
    cluster = model.predict(input_features)
    return cluster[0]