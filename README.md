# User Score Prediction Model

This project intends to develop an ML model that predicts **vote_average** in the TSDB 
dataset. To get the dataset, download it at 
[kaggle](https://www.kaggle.com/datasets/asaniczka/tmdb-movies-dataset-2023-930k-movies?resource=download).

To install dependencies, run `pip install -r requirements.txt`.

# Exploratory Analysis

Our goal is to understand which features correlate best with vote_average, and 
how simpler models perform against the training and test data. To this end, 
we have created multiple files:
- **preproccessing.py**: Handles missing data and splits dataset into training and test set.
- **explore_features.py**: Shows correlation between columns of the dataset and vote_average.
It creates visualizations showing cardinality of features, and also detects issues
in the dataset.
- **baseline_models.py**: Creates baseline models against the provided training set. These 
models are supposed to give us a sense of how good our model is against naive numerical 
solutions.
- **test_models.py**: Tests our models against the training and test data set. This file will
generate data for each model, which we can then perform various error calculations on. These
values will tell us how good our models are.
- **train_v1.py**: Trains a naive linear model. This is our v1, which we will improve over time.

## Open Questions
- How will we handle encoding complex information like keywords? One idea is to use a vector
embedding to make the keywords numerical, similar to how LLMs work.
- How will we handle missing data? We can either remove rows with missing data, or fill it in. 
What will lead to a better model?
- What is the best way to handle categorical information like genre? Should we one-hot encode 
each unique genre?

