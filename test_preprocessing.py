from utils.data_loader import load_dataset
from preprocessing.preprocess import preprocess_data

DATA_PATH = "data/AI_Based_Medical_Insurance_Claim_Fraud_Detection_Dataset.csv"

df = load_dataset(DATA_PATH)

X_train, X_test, y_train, y_test = preprocess_data(df)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)