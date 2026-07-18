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

        for i, model in enumerate(self.models):

            prob = model.predict_proba(X)

            pred = np.argmax(prob, axis=1)

            probabilities.append(prob)

            hospital_results.append({
                "hospital": f"Hospital {chr(65+i)}",
                "prediction": "🚨 Fraudulent" if pred[0] == 1 else "✅ Genuine",
                "fraud_probability": round(prob[0][1] * 100, 2)
            })

        avg_probability = np.mean(probabilities, axis=0)

        final_prediction = np.argmax(avg_probability, axis=1)

        return (
            final_prediction.tolist(),
            avg_probability,
            hospital_results
        )