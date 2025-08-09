from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import joblib

class ModelTrainer:
    def __init__(self, model_type="random_forest"):
        self.model_type = model_type
        self.model = None
        
    def train(self, data: pd.DataFrame, target_column: str, save_path: str):
        # train model on processed data
        try:
            x = data.drop(columns=[target_column])
            y = data[target_column]
            
            # train and test the x and y data
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
            
            if self.model_type == "random_forest":
                self.model = RandomForestClassifier(n_estimators=100)
                self.model.fit(x_train, y_train)
            
            accuracy = self.model.score(x_test, y_test)
            print(f"Model trained! Test accuracy: {accuracy:.2f}")
            # save the model to pkl file
            if self.model is not None:
                # this method is used to save a trained model (or any Python object) to a file
                joblib.dump(self.model, save_path)
                print(f"Model saved to {save_path}")
            return self.model
        except Exception as e:
            print(f"Training failed: {str(e)}")
            return None  

if  __name__ == "__main__":
    import pandas as pd
    from data_loader import load_data
    from preprocessor import preprocess_data
    
    data = load_data("sample_data.csv")
    clean_data = preprocess_data(data)
    
    if clean_data is not None:
        trainer = ModelTrainer()
        model = trainer.train(clean_data, "target_column_name")