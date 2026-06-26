import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def preprocess_data(df):
    """
    Preprocess the healthcare fraud dataset.
    """
    df = df.drop(columns=["claim_id"])

    X = df.drop(columns=["fraud_label"])
    y = df["fraud_label"]

    label_encoder = LabelEncoder()

    for column in X.select_dtypes(include=["object"]).columns:
        X[column] = label_encoder.fit_transform(X[column])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test