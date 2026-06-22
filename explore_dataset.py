import pandas as pd

DATA_PATH = "data/AI_Based_Medical_Insurance_Claim_Fraud_Detection_Dataset.csv"
df = pd.read_csv(DATA_PATH)

print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns)

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())