from sklearn.metrics import (confusion_matrix, classification_report, precision_score, recall_score, f1_score,)

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model using multiple metrics.
    """

    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("\nPrecision:", precision)
    print("Recall:", recall)
    print("F1-Score:", f1)

    print("\nFeature Importance:")

    feature_importance = model.feature_importances_

    for feature, importance in zip(X_test.columns, feature_importance):
        print(f"{feature}: {importance:.4f}")

 