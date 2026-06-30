from utils.data_loader import load_dataset
from preprocessing.preprocess import preprocess_data
from models.random_forest_model import train_random_forest
from models.evaluate_model import evaluate_model

DATA_PATH = "data/AI_Based_Medical_Insurance_Claim_Fraud_Detection_Dataset.csv"

def main():

    df = load_dataset(DATA_PATH)

    X_train, X_test, y_train, y_test = preprocess_data(df)
    
    model = train_random_forest(X_train, X_test, y_train, y_test)
    
    evaluate_model(model, X_test, y_test)
    
if __name__ == "__main__":
    main()
    