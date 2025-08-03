import pandas as pd
import numpy as np

def preprocess_data(data: pd.DataFrame, target_column: str):
    """
    Performs basic preprocessing:
    1. Handles missing values
    2. Converts categorical data
    3. Normalizes numerical features
    """
    try:
        # handle missing values
        data = data.fillna(data.mean(numeric_only=True))
        # convert categorical data (simple version)
        data = pd.get_dummies(data, drop_first=True)
        # normalize numerical features
        numeric_cols = data.select_dtypes(include=np.number).columns.drop(target_column)
        data[numeric_cols] = (data[numeric_cols] - data[numeric_cols].mean()) / data[numeric_cols].std()
        print("Preprocessing completed!")
        return data
    except Exception as e:
        print(f"Preprocessing failed: {str(e)}")
        return None
    
if __name__ == "__main__":
    from data_loader import load_data
    raw_data = load_data("sample_data.csv")
    if raw_data is not None:
        clean_data = preprocess_data(raw_data)
        print("Cleaned data shape: ", clean_data.shape)