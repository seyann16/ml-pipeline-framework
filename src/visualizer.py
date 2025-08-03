import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class DataVisualizer:
    @staticmethod
    def plot_feature_distribution(data: pd.DataFrame, feature: str):
        # Plot distribution of a single feature
        plt.figure(figsize=(10, 6))
        # for binary/categorical
        if data[feature].nunique() <= 5:
            # get counts and percentages
            counts = data[feature].value_counts()
            percents = counts / counts.sum() * 100
            # create bar plot with annotations
            ax = sns.barplot(x=counts.index, y=counts.values)
            plt.title(f"Distribution of {feature}")
            # add count/percentage labels
            for i, (count, percent) in enumerate(zip(counts, percents)):
                ax.text(i, count+0.5, f"{count}\n({percent:.1f}%)", ha='center', va='bottom')
        else:
            # use histogram for continuos features
            sns.histplot(data[feature], kde=True)
            plt.title(f"Distribution of {feature}")
        plt.savefig(f"{feature}_distribution.png")
        plt.close()
        print(f"Saved {feature}_distribution.png")
        
    @staticmethod
    def plot_correlation_matrix(data: pd.DataFrame):
        # Plot correlation matrix of numerical features
        plt.figure(figsize=(12, 8))
        numeric_data = data.select_dtypes(include='number')
        corr = numeric_data.corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title("Feature Correlation Matrix")
        plt.savefig("correlation_matrix.png")
        plt.close()
        print("Saved correlation_matrix.png")
        
# Test the visualizer
if __name__ == "__main__":
    from data_loader import load_data
    from preprocessor import preprocess_data
    
    # load and preprocess data
    data = load_data("sample_data.csv")
    clean_data = preprocess_data(data)
    
    if clean_data is not None:
        # visualize
        DataVisualizer.plot_feature_distribution(clean_data, "age")
        DataVisualizer.plot_correlation_matrix(clean_data)