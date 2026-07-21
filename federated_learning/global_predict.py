import joblib
import numpy as np


class FederatedPredictor:

    def __init__(self):

        self.models = joblib.load(
            "federated_learning/global_model/global_model.pkl"
        )

    def predict(self, X):

        hospital_results = []
        probabilities = []

        threshold = 20  # Same threshold as Random Forest

        for i, model in enumerate(self.models):

            prob = model.predict_proba(X)

            probabilities.append(prob)

            fraud_prob = prob[0][1] * 100

            hospital_results.append({
                "hospital": f"Hospital {chr(65+i)}",
                "fraud_probability": round(fraud_prob, 2),
                "prediction": "🚨 Fraudulent"
                              if fraud_prob >= threshold
                              else "✅ Genuine"
            })

        avg_probability = np.mean(probabilities, axis=0)

        final_prediction = np.argmax(avg_probability, axis=1)

        return (
            final_prediction.tolist(),
            avg_probability,
            hospital_results
        )