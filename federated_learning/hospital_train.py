import os
import sys
import pandas as pd
import joblib

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.preprocess import preprocess_data
from models.random_forest_model import train_random_forest

# Folder containing hospital datasets
DATA_FOLDER = "federated_learning/hospital_data"

# Folder to save local models
MODEL_FOLDER = "federated_learning/local_models"

os.makedirs(MODEL_FOLDER, exist_ok=True)

hospital_files = [
    "hospital_A.csv",
    "hospital_B.csv",
    "hospital_C.csv"
]

for hospital in hospital_files:

    print(f"\n==============================")
    print(f"Training {hospital}")
    print("==============================")

    # Load hospital dataset
    df = pd.read_csv(os.path.join(DATA_FOLDER, hospital))

    # Preprocess
    X_train, X_test, y_train, y_test = preprocess_data(df)

    # Train Random Forest
    model = train_random_forest(
        X_train,
        X_test,
        y_train,
        y_test
    )

    # Save model
    model_name = hospital.replace(".csv", "_model.pkl")

    joblib.dump(
        model,
        os.path.join(MODEL_FOLDER, model_name)
    )

    print(f"{model_name} saved successfully!")

print("\nAll hospital models trained successfully!")