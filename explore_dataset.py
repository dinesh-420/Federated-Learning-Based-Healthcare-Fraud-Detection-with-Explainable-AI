from utils.data_loader import load_dataset

DATA_PATH = "data/AI_Based_Medical_Insurance_Claim_Fraud_Detection_Dataset.csv"


def main():

    df = load_dataset(DATA_PATH)

    print("\n===== DATASET SHAPE =====")
    print(df.shape)

    print("\n===== FIRST 5 ROWS =====")
    print(df.head())

    print("\n===== COLUMN NAMES =====")
    print(df.columns.tolist())

    print("\n===== DATA TYPES =====")
    print(df.dtypes)

    print("\n===== MISSING VALUES =====")
    print(df.isnull().sum())

    print("\n===== DATASET INFO =====")
    df.info()

    print("\n===== STATISTICAL SUMMARY =====")
    print(df.describe(include="all"))


if __name__ == "__main__":
    main()
