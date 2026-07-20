import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from collections import Counter 
import joblib

def preprocess_data(df):
    """
    Preprocess the healthcare fraud dataset.
    """
    df = df.drop(columns=["claim_id"])

    X = df.drop(columns=["fraud_label"])
    y = df["fraud_label"]

    label_encoders = {}

    for column in X.select_dtypes(include=["object"]).columns:
       encoder = LabelEncoder()
       X[column] = encoder.fit_transform(X[column])
       label_encoders[column] = encoder

    joblib.dump(label_encoders, "saved_objects/label_encoders.pkl")
    print("Label encoders saved successfully!")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Save test dataset for model comparison (Phase 13)
    joblib.dump(X_test, "saved_objects/X_test.pkl")
    joblib.dump(y_test, "saved_objects/y_test.pkl")

    print("Test dataset saved successfully!")
    
    print("Before SMOTE:", Counter(y_train))

    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    print("After SMOTE:", Counter(y_train))
    
    return X_train, X_test, y_train, y_test

 