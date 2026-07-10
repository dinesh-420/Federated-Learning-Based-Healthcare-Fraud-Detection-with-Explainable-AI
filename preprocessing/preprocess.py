import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from collections import Counter 

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
    
    print("Before SMOTE:", Counter(y_train))

    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    print("After SMOTE:", Counter(y_train))
    
    return X_train, X_test, y_train, y_test