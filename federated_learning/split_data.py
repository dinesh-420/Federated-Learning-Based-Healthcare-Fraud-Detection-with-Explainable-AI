import pandas as pd
import os

# Load the original dataset
df = pd.read_csv("data/AI_Based_Medical_Insurance_Claim_Fraud_Detection_Dataset.csv")

# Shuffle the dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Split into three equal hospitals
hospital_a = df.iloc[:3333]
hospital_b = df.iloc[3333:6666]
hospital_c = df.iloc[6666:]

# Create folder if it doesn't exist
os.makedirs("federated_learning/hospital_data", exist_ok=True)

# Save datasets
hospital_a.to_csv(
    "federated_learning/hospital_data/hospital_A.csv",
    index=False
)

hospital_b.to_csv(
    "federated_learning/hospital_data/hospital_B.csv",
    index=False
)

hospital_c.to_csv(
    "federated_learning/hospital_data/hospital_C.csv",
    index=False
)

print("Federated datasets created successfully!\n")

print("Hospital A:", hospital_a.shape)
print("Hospital B:", hospital_b.shape)
print("Hospital C:", hospital_c.shape)