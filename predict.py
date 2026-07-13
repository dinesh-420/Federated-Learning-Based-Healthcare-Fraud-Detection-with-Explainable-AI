import pandas as pd

from utils.model_loader import load_model, load_label_encoders
import joblib

 
model = load_model()
label_encoders = load_label_encoders()
feature_columns = joblib.load("saved_objects/feature_columns.pkl")

print("Model loaded successfully!")
print("Label encoders loaded successfully!")
print("Feature columns loaded successfully!")

# Sample healthcare claim
sample_claim = {
    "patient_age": 45,
    "patient_gender": "Male",
    "hospital_type": "Private",
    "treatment_category": "Surgery",
    "diagnosis_code": "D123",
    "claim_amount": 85000,
    "approved_amount": 70000,
    "hospital_stay_days": 5,
    "previous_claims_count": 2,
    "policy_tenure_years": 4,
    "claim_submission_delay_days": 3,
    "high_risk_procedure_flag": 1,
    "document_mismatch_flag": 0,
    "anomaly_score": 0.62
}

print("\nSample Claim Created Successfully!")
print(sample_claim)

 
claim_df = pd.DataFrame([sample_claim])

print("\nClaim converted to DataFrame:")
print(claim_df)

print("\nSaved Encoders:")
print(label_encoders.keys())

for column, encoder in label_encoders.items():
    claim_df[column] = encoder.transform(claim_df[column])

print("\nEncoded Claim:")
print(claim_df)


claim_df = claim_df[feature_columns]

print("\nClaim after arranging feature order:")
print(claim_df)

# Predict fraud
prediction = model.predict(claim_df)

print("\nPrediction Result:")

if prediction[0] == 1:
    print("🚨 Fraudulent Claim")
else:
    print("✅ Genuine Claim")