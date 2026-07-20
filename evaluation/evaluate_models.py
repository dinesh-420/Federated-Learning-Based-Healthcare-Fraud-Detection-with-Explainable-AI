import os
import sys
# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from federated_learning.global_predict import FederatedPredictor
import json

# Load test dataset
 
X_test = joblib.load("saved_objects/X_test.pkl")
y_test = joblib.load("saved_objects/y_test.pkl")

# Load Random Forest
 
rf_model = joblib.load("models/fraud_detection_model.pkl")

rf_predictions = rf_model.predict(X_test)
rf_probabilities = rf_model.predict_proba(X_test)[:, 1]
 
# Load Federated Model
 
federated_model = FederatedPredictor()

fl_predictions, fl_probabilities, _ = federated_model.predict(X_test)

fl_predictions = fl_predictions
fl_probabilities = fl_probabilities[:, 1]

# Evaluation Function
 
def evaluate(name, y_true, predictions, probabilities):

    metrics = {
        "accuracy": round(accuracy_score(y_true, predictions), 4),
        "precision": round(precision_score(y_true, predictions), 4),
        "recall": round(recall_score(y_true, predictions), 4),
        "f1_score": round(f1_score(y_true, predictions), 4),
        "roc_auc": round(roc_auc_score(y_true, probabilities), 4)
    }

    print("\n==============================")
    print(name)
    print("==============================")

    for key, value in metrics.items():
        print(f"{key}: {value}")

    return metrics

# Compare Models
 
rf_metrics = evaluate(
    "Random Forest",
    y_test,
    rf_predictions,
    rf_probabilities
)

fl_metrics = evaluate(
    "Federated Learning",
    y_test,
    fl_predictions,
    fl_probabilities
)

comparison = {
    "Random Forest": rf_metrics,
    "Federated Learning": fl_metrics
}

with open("saved_objects/model_comparison.json", "w") as f:
    json.dump(comparison, f, indent=4)

print("\nModel comparison saved successfully!")