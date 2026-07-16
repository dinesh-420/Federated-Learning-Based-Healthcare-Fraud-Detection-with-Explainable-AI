import joblib
import pandas as pd
import shap
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -----------------------------
# Load Model
# -----------------------------
model = joblib.load("models/fraud_detection_model.pkl")

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv(
    "data/AI_Based_Medical_Insurance_Claim_Fraud_Detection_Dataset.csv"
)

# -----------------------------
# Separate Features
# -----------------------------
X = df.drop("fraud_label", axis=1)

# -----------------------------
# Load Label Encoders
# -----------------------------
label_encoders = joblib.load("saved_objects/label_encoders.pkl")

# Encode categorical columns
for column, encoder in label_encoders.items():
    X[column] = encoder.transform(X[column])

# -----------------------------
# Load Feature Order
# -----------------------------
feature_columns = joblib.load("saved_objects/feature_columns.pkl")
X = X[feature_columns]

# -----------------------------
# Use only 500 samples
# -----------------------------
X = X.sample(n=500, random_state=42)

# -----------------------------
# SHAP Explainer
# -----------------------------
explainer = shap.TreeExplainer(model)

# Calculate SHAP values
shap_values = explainer.shap_values(X)

# -----------------------------
# Global SHAP Summary Plot
# -----------------------------
plt.figure(figsize=(10, 6))

shap.summary_plot(
    shap_values[:, :, 1],
    X,
    show=False
)

plt.tight_layout()

plt.savefig(
    "global_shap_summary.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# -----------------------------
# Global SHAP Bar Plot
# -----------------------------
plt.figure(figsize=(10, 6))

shap.plots.bar(
    shap.Explanation(
        values=shap_values[:, :, 1].mean(axis=0),
        feature_names=X.columns
    ),
    show=False
)

plt.tight_layout()

plt.savefig(
    "global_shap_bar.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Global SHAP plots generated successfully!")