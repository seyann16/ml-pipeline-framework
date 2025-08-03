import pandas as pd

def load_data(file_path: str) -> pd.DataFrame:
    # loads data from a csv file
    try:
        data = pd.read_csv(file_path)
        print(f"Successfully loaded data with {data.shape[0]} rows")
        return data
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found")
        return None
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return None

# test the function
if __name__ == "__main__":
    sample_data = load_data("sample_data.csv")
    if sample_data is not None:
        print("First 5 rows:\n", sample_data.head())