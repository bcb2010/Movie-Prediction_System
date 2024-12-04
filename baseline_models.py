import numpy as np
import pandas as pd


# Baseline model functions
class BaselineModels:
    def __init__(self, data):
        # Initialize with the dataset
        self.data = data
        self.mean_vote = data['vote_average'].mean()
        self.median_vote = data['vote_average'].median()

        # Popularity-weighted average
        C = self.mean_vote
        M = data['vote_count'].quantile(0.9)
        self.weighted_scores = (
                                       data['vote_count'] * data['vote_average'] + C * M
                               ) / (data['vote_count'] + M)

        self.simple_regression_model = None

        # Train simple regression model on popularity
        if 'popularity' in data.columns and not data['popularity'].isna().all():
            from sklearn.linear_model import LinearRegression
            X = data[['popularity']].fillna(0)
            y = data['vote_average']
            self.simple_regression_model = LinearRegression()
            self.simple_regression_model.fit(X, y)

    def mean_model(self, row):
        """Predicts the mean vote_average."""
        return self.mean_vote

    def median_model(self, row):
        """Predicts the median vote_average."""
        return self.median_vote

    def popularity_weighted_model(self, row):
        """Predicts the popularity-weighted average score."""
        if 'vote_count' in row:
            C = self.mean_vote
            M = self.data['vote_count'].quantile(0.9)
            return (
                    row.get('vote_count', 0) * row.get('vote_average', C) + C * M
            ) / (row.get('vote_count', 0) + M)
        return self.mean_vote

    def popularity_regression_model(self, row):
        """Predicts using a simple linear regression model on popularity."""
        if self.simple_regression_model is not None and 'popularity' in row:
            return self.simple_regression_model.predict([[row['popularity']]])[0]
        return self.mean_vote


# Example Usage
if __name__ == "__main__":
    # Example dataset
    data = pd.DataFrame({
        'vote_average': [7.5, 6.8, 8.3, 5.9, 7.0],
        'vote_count': [1000, 800, 1200, 400, 500],
        'popularity': [20.1, 15.3, 35.2, 10.5, 18.7]
    })

    models = BaselineModels(data)

    # Example row
    row = {'vote_count': 600, 'popularity': 25.0}

    print("Mean Model Prediction:", models.mean_model(row))
    print("Median Model Prediction:", models.median_model(row))
    print("Popularity-Weighted Prediction:", models.popularity_weighted_model(row))
    print("Simple Regression Prediction:", models.popularity_regression_model(row))
