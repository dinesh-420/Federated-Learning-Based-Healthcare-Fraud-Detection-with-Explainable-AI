import os
import joblib

LOCAL_MODEL_FOLDER = "federated_learning/local_models"
GLOBAL_MODEL_FOLDER = "federated_learning/global_model"

os.makedirs(GLOBAL_MODEL_FOLDER, exist_ok=True)

# Load local hospital models
hospital_models = []

model_files = [
    "hospital_A_model.pkl",
    "hospital_B_model.pkl",
    "hospital_C_model.pkl"
]

for model_file in model_files:

    model = joblib.load(
        os.path.join(
            LOCAL_MODEL_FOLDER,
            model_file
        )
    )

    hospital_models.append(model)

    print(f"{model_file} loaded successfully.")

# Save all models together as the Global Federated Model
joblib.dump(
    hospital_models,
    os.path.join(
        GLOBAL_MODEL_FOLDER,
        "global_model.pkl"
    )
)

print("\nGlobal Federated Model created successfully!")