from data_loader import load_data
from preprocessor import preprocess_data
from model_trainer import ModelTrainer
from visualizer import DataVisualizer

class MLPipeline:
    def __init__(self, data_path, target_column):
        self.data_path = data_path
        self.target_column = target_column
        self.data = None
        self.clean_data = None
        self.model = None
    
    def run(self):
        print("Starting pipeline...")
        # phase 1: load data
        self.data = load_data(self.data_path)
        
        if self.data is not None:
            # phase 2: preprocess
            self.clean_data = preprocess_data(self.data, self.target_column)
            
            if self.clean_data is not None:
                self.visualize_data()
                # phase 3: train model
                trainer = ModelTrainer()
                self.model = trainer.train(self.data, self.target_column)
                return self.model
        return None
    
    # visualization setup
    def visualize_data(self):
        print("Visualizing data...")
        # visualize target distribution
        DataVisualizer.plot_feature_distribution(self.clean_data, self.target_column)
        # visualize correlations
        DataVisualizer.plot_correlation_matrix(self.clean_data)
        
# run full pipeline
if __name__ == "__main__":
    pipeline = MLPipeline("sample_data.csv", "target")
    trained_model = pipeline.run()
    print("Pipeline completed!" if trained_model else "Pipeline failed")