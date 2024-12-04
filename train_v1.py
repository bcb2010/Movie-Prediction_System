import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

class SimpleLinearModel:
    def __init__(self, data):
        """Initialize the model with training data."""
        self.features = ['budget', 'popularity', 'runtime']  # Example numeric features
        self.target = 'vote_average'

        # Drop rows with missing values for simplicity
        data = data.dropna(subset=self.features + [self.target])

        # Split data into training and test sets
        X = data[self.features]
        y = data[self.target]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train a simple linear regression model
        self.model = LinearRegression()
        self.model.fit(self.X_train, self.y_train)

    def predict(self, X):
        """Make predictions using the trained model."""
        return self.model.predict(X)

    def evaluate(self):
        """Evaluate the model on both training and test datasets."""
        train_predictions = self.predict(self.X_train)
        test_predictions = self.predict(self.X_test)

        train_mse = mean_squared_error(self.y_train, train_predictions)
        test_mse = mean_squared_error(self.y_test, test_predictions)

        return {
            "Training MSE": train_mse,
            "Test MSE": test_mse
        }

if __name__ == "__main__":
    # Load the dataset
    data = pd.read_csv("training_set.csv")

    # Initialize and train the model
    simple_model = SimpleLinearModel(data)

    # Evaluate the model
    results = simple_model.evaluate()
    print("Linear Regression Model")
    print(f"Training MSE: {results['Training MSE']:.4f}")
    print(f"Test MSE: {results['Test MSE']:.4f}")