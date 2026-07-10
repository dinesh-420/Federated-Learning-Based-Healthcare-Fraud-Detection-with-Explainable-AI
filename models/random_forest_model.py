from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def train_random_forest(X_train, X_test, y_train, y_test):
    """
    Train a Random Forest model and evaluate its accuracy.
    """
    
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced", min_samples_leaf=5)
    
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print("Model Accuracy:", accuracy)

    return model