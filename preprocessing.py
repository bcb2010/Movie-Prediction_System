import pandas as pd
from sklearn.model_selection import train_test_split


# Load the dataset
file_path = 'TMDB_movie_dataset_v11.csv'
df = pd.read_csv(file_path)

# 1. Initial Inspection
print("Initial Dataset Shape:", df.shape)
print("\nDataset Preview:")
print(df.head())
print("\nData Types:")
print(df.dtypes)
print("\nMissing Values:")
print(df.isnull().sum())

# 2. Handle Duplicates
if df.duplicated().sum() > 0:
    print(f"\nFound {df.duplicated().sum()} duplicate rows. Removing them.")
    df = df.drop_duplicates()

# 3. Handle Missing Data
# Identify columns with missing values
missing_cols = df.columns[df.isnull().any()]

# Display percentage of missing data
missing_percentage = df[missing_cols].isnull().mean() * 100
print("\nMissing Data Percentage by Column:")
print(missing_percentage)

# Drop or fill missing data as appropriate
for col in missing_cols:
    if missing_percentage[col] > 50  and col not in ['keywords', 'production_companies']:  # Drop columns with more than 50% missing data and preserve some columns planned for certain tasks
        print(f"Dropping column {col} due to excessive missing data.")
        df = df.drop(columns=[col])
    else:
        if df[col].dtype != 'object':  # Only fill missing values for numerical columns
            df[col] = df[col].fillna(df[col].median())

# 4. Validate Data Integrity
print("\nData Summary After Cleaning:")
print(df.describe())

# 5. Reserve Final Test Set
# Split the dataset into a working dataset and a reserved test set
df_working, df_final_test = train_test_split(df, test_size=0.2, random_state=42)

# Save the reserved test set
test_set_path = 'test_set.csv'
df_final_test.to_csv(test_set_path, index=False)
print(f"\nFinal test set saved to {test_set_path}. Shape: {df_final_test.shape}")

# Save the cleaned working dataset for analysis
working_set_path = 'training_set.csv'
df_working.to_csv(working_set_path, index=False)
print(f"Cleaned working dataset saved to {working_set_path}. Shape: {df_working.shape}")