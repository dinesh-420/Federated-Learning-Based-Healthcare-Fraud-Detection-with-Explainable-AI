import pandas as pd
import joblib

# Load dataset
df = pd.read_csv("data/AI_Based_Medical_Insurance_Claim_Fraud_Detection_Dataset.csv")

# Load model
model = joblib.load("models/fraud_detection_model.pkl")

# Load encoders
encoders = joblib.load("saved_objects/label_encoders.pkl")

# Load feature order
feature_columns = joblib.load("saved_objects/feature_columns.pkl")

# Prepare features
X = df.drop(columns=["claim_id", "fraud_label"])

# Encode categorical features
for col, encoder in encoders.items():
    X[col] = encoder.transform(X[col])

# Match training feature order
X = X[feature_columns]

print("="*80)
print("First 10 Fraud cases predicted by the model")
print("="*80)

count = 0

for i in range(len(X)):

    fraud_prob = model.predict_proba(X.iloc[[i]])[0][1] * 100

    if fraud_prob >= 20:

        print(f"\nDataset Row : {i}")
        print(f"Fraud Probability : {fraud_prob:.2f}%")
        print(f"Actual Label : {df.iloc[i]['fraud_label']}")
        print(df.iloc[i])

        print("-"*80)

        count += 1

        if count == 10:
            break