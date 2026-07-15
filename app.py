from flask import Flask, render_template, request
import pandas as pd
import joblib
from utils.model_loader import load_model, load_label_encoders

app = Flask(__name__)

model = load_model()
label_encoders = load_label_encoders()

feature_columns = joblib.load("saved_objects/feature_columns.pkl")

print("Model loaded successfully!")
print("Label encoders loaded successfully!")
print("Feature columns loaded successfully!")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    patient_age = request.form["patient_age"]
    patient_gender = request.form["patient_gender"]
    hospital_type = request.form["hospital_type"]
    treatment_category = request.form["treatment_category"]
    diagnosis_code = request.form["diagnosis_code"]
    claim_amount = request.form["claim_amount"]
    approved_amount = request.form["approved_amount"]
    hospital_stay_days = request.form["hospital_stay_days"]
    previous_claims_count = request.form["previous_claims_count"]
    policy_tenure_years = request.form["policy_tenure_years"]
    claim_submission_delay_days = request.form["claim_submission_delay_days"]
    high_risk_procedure_flag = request.form["high_risk_procedure_flag"]
    document_mismatch_flag = request.form["document_mismatch_flag"]
    anomaly_score = request.form["anomaly_score"]

    claim_data = {
        "patient_age": [int(patient_age)],
        "patient_gender": [patient_gender],
        "hospital_type": [hospital_type],
        "treatment_category": [treatment_category],
        "diagnosis_code": [diagnosis_code],
        "claim_amount": [float(claim_amount)],
        "approved_amount": [float(approved_amount)],
        "hospital_stay_days": [int(hospital_stay_days)],
        "previous_claims_count": [int(previous_claims_count)],
        "policy_tenure_years": [int(policy_tenure_years)],
        "claim_submission_delay_days": [int(claim_submission_delay_days)],
        "high_risk_procedure_flag": [1 if high_risk_procedure_flag == "Yes" else 0],
        "document_mismatch_flag": [1 if document_mismatch_flag == "Yes" else 0],
        "anomaly_score": [float(anomaly_score)]
    }

    claim_df = pd.DataFrame(claim_data)

    print("\nClaim DataFrame:")
    print(claim_df)
    
    print("Treatment Category Classes:")
    print(label_encoders["treatment_category"].classes_)
    
    for column, encoder in label_encoders.items():
         claim_df[column] = encoder.transform(claim_df[column])

    print("\nEncoded Claim DataFrame:")
    print(claim_df)

    claim_df = claim_df[feature_columns]

    print("\nClaim after arranging feature order:")
    print(claim_df)

    prediction = model.predict(claim_df)

    print("\nPrediction:")
    print(prediction)

    print("Patient Age:", patient_age)
    print("Gender:", patient_gender)
    print("Hospital Type:", hospital_type)
    print("Treatment Category:", treatment_category)
    print("Diagnosis Code:", diagnosis_code)
    print("Claim Amount:", claim_amount)
    print("Approved Amount:", approved_amount)
    print("Hospital Stay Days:", hospital_stay_days)
    print("Previous Claims Count:", previous_claims_count)
    print("Policy Tenure:", policy_tenure_years)
    print("Submission Delay:", claim_submission_delay_days)
    print("High Risk Procedure:", high_risk_procedure_flag)
    print("Document Mismatch:", document_mismatch_flag)
    print("Anomaly Score:", anomaly_score)

    if prediction[0] == 1:
        result = "🚨 Fraudulent Claim"
    else:
        result = "✅ Genuine Claim"

    if prediction[0] == 1:
      result = "🚨 Fraudulent Claim"
      color = "red"
    else:
      result = "✅ Genuine Claim"
      color = "green"

    return render_template( "result.html", result=result,color=color)

if __name__ == "__main__":
    app.run(debug=True)