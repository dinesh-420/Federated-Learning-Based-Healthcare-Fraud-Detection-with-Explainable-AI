import pandas as pd

def load_dataset(file_path):
    """
    Load the healthcare fraud dataset.
    """
    return pd.read_csv(file_path)