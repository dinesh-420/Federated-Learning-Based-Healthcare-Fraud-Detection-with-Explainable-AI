import joblib

def load_model():
    """
    Load the trained Random Forest model.

    """
    model = joblib.load("models/fraud_detection_model.pkl")
    return model

def load_label_encoders():
    """
    Load saved Label Encoders.
    """
    encoders = joblib.load("saved_objects/label_encoders.pkl")
    return encoders 