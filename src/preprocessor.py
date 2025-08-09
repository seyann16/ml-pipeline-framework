import pandas as pd
import numpy as np

def preprocess_data(data: pd.DataFrame, target_column: str, handle_outliers=True) -> pd.DataFrame:
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
        # drop first category to avoid multicollinearity
        data = pd.get_dummies(data, drop_first=True)
        # normalize numerical features
        numeric_cols = data.select_dtypes(include=np.number).columns.drop(target_column)
        # this line standardizes the numeric columns, also called Z-score normalization.
        # useful for SVM, KNN, Logistic Regression, Neural Nets
        data[numeric_cols] = (data[numeric_cols] - data[numeric_cols].mean()) / data[numeric_cols].std()
        # outlier handling
        if handle_outliers:
            # no need to drop the target column because the 0/1 are not outliers
            numeric_cols = data.select_dtypes(include=np.number).columns
            for col in numeric_cols:
                # quantile() returns the value below which a certain percentage of data falls
                q1 = data[col].quantile(0.25)
                q3 = data[col].quantile(0.75)
                # iqr measures the spread of the middle 50% of the data.
                iqr = q3 - q1
                # smallest acceptable value
                lower_bound = q1 - 1.5 * iqr
                # largest acceptable value
                upper_bound = q3 - 1.5 * iqr
                # winsorization / cap outliers
                # replacing extreme values with a maximum or minimum threshold instead of removing them
                # np.where() returns one value if a condition is true, and another if it’s false
                data[col] = np.where(data[col] < lower_bound, lower_bound, data[col])
                data[col] = np.where(data[col] > upper_bound, upper_bound, data[col])
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