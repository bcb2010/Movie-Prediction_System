import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class FeatureExplorer:
    def __init__(self, data):
        self.data = data

    def explore_basic(self):
        """Performs basic exploration of the dataset."""
        print("Dataset Overview:\n", self.data.info())
        print("\nSummary Statistics:\n", self.data.describe())

    def visualize_features(self):
        """Creates histograms and scatterplots for numeric features."""
        numeric_columns = self.data.select_dtypes(include=[np.number]).columns

        for column in numeric_columns:
            plt.figure(figsize=(8, 5))
            sns.histplot(self.data[column].dropna(), kde=True)
            plt.title(f"Distribution of {column}")
            plt.xlabel(column)
            plt.ylabel("Frequency")
            plt.show()

        # Scatterplot matrix for numeric columns
        if len(numeric_columns) > 1:
            sns.pairplot(self.data[numeric_columns])
            plt.suptitle("Scatterplot Matrix", y=1.02)
            plt.show()

    def detect_issues(self):
        """Detects data issues such as missing values, outliers, and skewness."""
        print("Missing Values:\n", self.data.isnull().sum())

        # Check skewness for numeric columns
        skewed_features = self.data.select_dtypes(include=[np.number]).apply(lambda x: x.skew())
        print("\nSkewness of Numeric Features:\n", skewed_features)

        # Detect outliers using IQR method
        for column in self.data.select_dtypes(include=[np.number]).columns:
            Q1 = self.data[column].quantile(0.25)
            Q3 = self.data[column].quantile(0.75)
            IQR = Q3 - Q1
            outliers = self.data[(self.data[column] < Q1 - 1.5 * IQR) | (self.data[column] > Q3 + 1.5 * IQR)]
            print(f"\nOutliers detected in {column}: {len(outliers)}")

    def find_relations(self, target_column):
        """Finds relationships between features and the target column."""
        # Correlation matrix
        correlation_matrix = self.data.corr()
        print("\nCorrelation Matrix:\n", correlation_matrix)

        if target_column in correlation_matrix:
            print("\nCorrelation with Target:\n", correlation_matrix[target_column].sort_values(ascending=False))

        # Pairplot with target
        sns.pairplot(self.data, y_vars=target_column, x_vars=self.data.select_dtypes(include=[np.number]).columns)
        plt.suptitle(f"Feature Relations with {target_column}", y=1.02)
        plt.show()

# Example Usage
if __name__ == "__main__":
    # Example dataset
    data = pd.read_csv('TMDB_movie_dataset_v11.csv')

    explorer = FeatureExplorer(data)
    explorer.explore_basic()
    explorer.visualize_features()
    explorer.detect_issues()
    explorer.find_relations(target_column='vote_average')
