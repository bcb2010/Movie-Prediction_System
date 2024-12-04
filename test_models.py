import pandas as pd
from sklearn.model_selection import train_test_split
from baseline_models import BaselineModels
from train_v1 import SimpleLinearModel

class ModelTester:
    def __init__(self, data):
        self.data = data
        self.models = BaselineModels(data)
        self.v1_model = SimpleLinearModel(data)

    def test_models(self, test_data):
        results = {
            'mean_model': [],
            'median_model': [],
            'popularity_weighted_model': [],
            'simple_regression_model': []
        }

        for _, row in test_data.iterrows():
            results['mean_model'].append(self.models.mean_model(row))
            results['median_model'].append(self.models.median_model(row))
            results['popularity_weighted_model'].append(self.models.popularity_weighted_model(row))
            results['popularity_regression_model'].append(self.models.popularity_regression_model(row))
            results['test_v1'].append(self.v1_model.predict(row))

        return pd.DataFrame(results)

if __name__ == "__main__":
    # Load the dataset
    train_data = pd.read_csv("training_set.csv")
    test_data = pd.read_csv("test_set.csv")

    # Initialize the tester for training data
    train_tester = ModelTester(train_data)
    train_predictions = train_tester.test_models(train_data)

    # Combine actual and predicted for training comparison
    train_results = train_data.copy()
    train_results = train_results.reset_index(drop=True)
    train_results = pd.concat([train_results, train_predictions], axis=1)

    print("Training Results:")
    print(train_results.head())

    # Initialize the tester for testing data
    test_tester = ModelTester(train_data)  # Use models trained on train_data
    test_predictions = test_tester.test_models(test_data)

    # Combine actual and predicted for testing comparison
    test_results = test_data.copy()
    test_results = test_results.reset_index(drop=True)
    test_results = pd.concat([test_results, test_predictions], axis=1)

    print("Test Results:")
    print(test_results.head())
