import pandas as pd
import joblib

# Load dataset
df = pd.read_csv(
    "data/AI_Based_Medical_Insurance_Claim_Fraud_Detection_Dataset.csv"
)

# Load trained model
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

# Arrange columns exactly like training
X = X[feature_columns]

print("=" * 80)
print("First 10 Genuine cases predicted by the trained model")
print("=" * 80)

count = 0

for i in range(len(X)):

    probability = model.predict_proba(X.iloc[[i]])[0][1] * 100

    if probability < 20:       # Your fraud threshold

        print(f"\nDataset Row : {i}")
        print(f"Fraud Probability : {probability:.2f}%")
        print(f"Actual Label : {df.iloc[i]['fraud_label']}")

        print(df.iloc[i])

        print("-" * 80)

        count += 1

        if count == 10:
            break

if count == 0:
    print("\nNo Genuine cases found below the 20% fraud threshold.")