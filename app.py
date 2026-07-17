from flask import Flask, render_template, request
import pandas as pd
import joblib
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from utils.model_loader import load_model, load_label_encoders
from pdf_report import generate_pdf

app = Flask(__name__)

model = load_model()
explainer = shap.TreeExplainer(model)

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

    # Input Validation
   
    if claim_df["patient_age"][0] < 0 or claim_df["patient_age"][0] > 120:
        return render_template(
            "result.html",
            result="❌ Invalid Patient Age.",
            color="orange",
            confidence="--",
            fraud_probability="--",
            model_name="Random Forest Classifier",
            show_shap=False,
            prediction_success=False
        )

    if claim_df["claim_amount"][0] <= 0:
        return render_template(
            "result.html",
            result="❌ Claim Amount must be greater than 0.",
            color="orange",
            confidence="--",
            fraud_probability="--",
            model_name="Random Forest Classifier",
            show_shap=False,
            prediction_success=False
        )

    if claim_df["approved_amount"][0] < 0:
        return render_template(
            "result.html",
            result="❌ Approved Amount cannot be negative.",
            color="orange",
            confidence="--",
            fraud_probability="--",
            model_name="Random Forest Classifier",
            show_shap=False,
            prediction_success=False
        )

    if claim_df["approved_amount"][0] > claim_df["claim_amount"][0]:
        return render_template(
            "result.html",
            result="❌ Approved Amount cannot exceed Claim Amount.",
            color="orange",
            confidence="--",
            fraud_probability="--",
            model_name="Random Forest Classifier",
            show_shap=False,
            prediction_success=False
        )

    if claim_df["hospital_stay_days"][0] < 0:
        return render_template(
            "result.html",
            result="❌ Hospital Stay Days cannot be negative.",
            color="orange",
            confidence="--",
            fraud_probability="--",
            model_name="Random Forest Classifier",
            show_shap=False,
            prediction_success=False
        )

    if claim_df["previous_claims_count"][0] < 0:
        return render_template(
            "result.html",
            result="❌ Previous Claims Count cannot be negative.",
            color="orange",
            confidence="--",
            fraud_probability="--",
            model_name="Random Forest Classifier",
            show_shap=False,
            prediction_success=False
        )

    if claim_df["policy_tenure_years"][0] < 0:
        return render_template(
            "result.html",
            result="❌ Policy Tenure cannot be negative.",
            color="orange",
            confidence="--",
            fraud_probability="--",
            model_name="Random Forest Classifier",
            show_shap=False,
            prediction_success=False
        )

    if claim_df["claim_submission_delay_days"][0] < 0:
        return render_template(
            "result.html",
            result="❌ Claim Submission Delay cannot be negative.",
            color="orange",
            confidence="--",
            fraud_probability="--",
            model_name="Random Forest Classifier",
            show_shap=False,
            prediction_success=False
        )

    if claim_df["anomaly_score"][0] < 0 or claim_df["anomaly_score"][0] > 1:
        return render_template(
            "result.html",
            result="❌ Anomaly Score must be between 0 and 1.",
            color="orange",
            confidence="--",
            fraud_probability="--",
            model_name="Random Forest Classifier",
            show_shap=False,
            prediction_success=False
        )

    # Encode categorical columns safely
    for column, encoder in label_encoders.items():

         try:
             claim_df[column] = encoder.transform(claim_df[column])

         except ValueError:

             if column == "diagnosis_code":
                  message = "❌ Unknown Diagnosis Code. Please enter a valid diagnosis code."
             else:
                   message = f"❌ Invalid {column.replace('_', ' ').title()}."

             print(message)
         
             return render_template(
                 "result.html",
                  result=message,
                  color="orange", 
                  confidence="--",
                  fraud_probability="--",
                  model_name="Random Forest Classifier",
                  shap_explanations=None,
                  show_shap=False,
                  prediction_success=False
             )

    # Arrange feature order
    claim_df = claim_df[feature_columns]

    # Prediction
    prediction = model.predict(claim_df)
    probability = model.predict_proba(claim_df)

    # SHAP Explainability
    shap_values = explainer.shap_values(claim_df)

    # Generate SHAP Summary Plot
   
    plt.figure()

    # For newer SHAP versions
    shap.summary_plot(
         shap_values[:, :, 1],
         claim_df,
         show=False
    )

    plt.tight_layout()

    plt.savefig("static/shap_summary.png", bbox_inches="tight")

    plt.close()
    
    # Generate SHAP Bar Plot

    plt.figure()

    shap.plots.bar(
        shap.Explanation(
            values=shap_values[0, :, 1],
            base_values=explainer.expected_value[1],
            data=claim_df.iloc[0],
            feature_names=claim_df.columns
        ),
        show=False
    )

    plt.tight_layout()

    plt.savefig("static/shap_bar.png", bbox_inches="tight")

    plt.close()

    # Get SHAP values for the Fraud class (class index = 1)
    feature_impacts = shap_values[0, :, 1]

    shap_explanations = []

    for feature, impact in zip(claim_df.columns, feature_impacts):

        impact = round(float(impact), 4)

        if impact >= 0.10:
           explanation = "Strongly increased fraud probability"

        elif impact > 0:
             explanation = "Slightly increased fraud probability"

        elif impact <= -0.10:
              explanation = "Strongly reduced fraud probability"

        else:
              explanation = "Slightly reduced fraud probability"

        shap_explanations.append(
              (feature, impact, explanation)
        )
    # Sort by absolute impact
    shap_explanations = sorted(
         shap_explanations,
         key=lambda x: abs(x[1]),
         reverse=True
    )

    # Keep only Top 5 features
    shap_explanations = shap_explanations[:5]

    fraud_probability = round(probability[0][1] * 100, 2)
    model_name = "Random Forest Classifier"

    if prediction[0] == 1:
        result = "🚨 Fraudulent Claim"
        color = "red"
        confidence = round(probability[0][1] * 100, 2)
    else:
        result = "✅ Genuine Claim"
        color = "green"
        confidence = round(probability[0][0] * 100, 2)

    # Clean terminal output
    print(f"Prediction: {result}")
    print(f"Confidence: {confidence}%")
    print(f"Fraud Probability: {fraud_probability}%")

    # Generate PDF Report
    generate_pdf(
        "static/prediction_report.pdf",
        result,
        confidence,
        fraud_probability,
        model_name,
        shap_explanations
    )

    return render_template(
        "result.html",
        result=result,
        color=color,
        confidence=confidence,
        fraud_probability=fraud_probability,
        model_name=model_name,
        shap_explanations=shap_explanations,
        show_shap=True,
        prediction_success=True
    )


if __name__ == "__main__":
    app.run(debug=True)