# The line above writes the following code to a file named preprocessing.py in the code directory.
# Processing Step for Feature Engineering 
# High-Level Overview
# Imports: Import necessary libraries for data manipulation, handling, and preprocessing.
# Column Names and Data Types: Define the column names and data types for the dataset.
# Utility Function: Define a helper function to merge dictionaries.
# Main Script: Read the dataset, preprocess the data, and split it into training, validation, and test sets.
#                        Summary 
# This script performs data preprocessing, including:
#   Reading and merging data.
#   Handling missing values.
#   Scaling numerical features.
#   Encoding categorical features.
#   Splitting the data into training, validation, and test sets.
#   The preprocessing steps ensure that the data is clean and in a suitable format for training machine learning models.


# numpy and pandas for data manipulation 
# scikit-learn for preprocessing 
import argparse
import os
import requests
import tempfile

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder


# Since we get a headerless CSV file, we specify the column names here.
# Defining the column names and data types for the dataset. feature_columns_names contains 
# the names of the feature columns, and label_column contains the name of the target column.
# Feature_columns_dtype and label_column_dtype define the data types for these columns.

feature_columns_names = [
    "sex",
    "length",
    "diameter",
    "height",
    "whole_weight",
    "shucked_weight",
    "viscera_weight",
    "shell_weight",
]
label_column = "rings"

feature_columns_dtype = {
    "sex": str,
    "length": np.float64,
    "diameter": np.float64,
    "height": np.float64,
    "whole_weight": np.float64,
    "shucked_weight": np.float64,
    "viscera_weight": np.float64,
    "shell_weight": np.float64,
}
label_column_dtype = {"rings": np.float64}

# Utility function 
# This function merges two dictionaries. It is used to combine feature_columns_dtype and label_column_dtype.
def merge_two_dicts(x, y):
    z = x.copy()
    z.update(y)
    return z

# Main Script 
# reads the Abalone dataset from a specified directory. It assumes the dataset is in CSV format without headers,
# so it assigns the column names and data types defined earlier.

if __name__ == "__main__":
    base_dir = "/opt/ml/processing"
    

    df = pd.read_csv(
        f"{base_dir}/input/abalone-dataset.csv",
        header=None,
        names=feature_columns_names + [label_column],
        dtype=merge_two_dicts(feature_columns_dtype, label_column_dtype),
    )

# The script separates numeric and categorical features. Numeric features are preprocessed with a pipeline that imputes missing values using the median and scales the features using StandardScaler.
# Categorical features are preprocessed with a pipeline that imputes missing values with "missing" and encodes the categories using OneHotEncoder.
# ColumnTransformer applies these transformations to the respective columns.

    numeric_features = list(feature_columns_names)
    numeric_features.remove("sex")
    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )

    categorical_features = ["sex"]
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

# Splitting and saving the data 
# The target column (rings) is separated from the features.
# The features are transformed using the preprocessing pipeline.
# The target and transformed features are concatenated back together.
# The data is shuffled and split into training (70%), validation (15%), and test (15%) sets.
# These sets are saved as CSV files to the specified directories.

    y = df.pop("rings")
    X_pre = preprocess.fit_transform(df)
    y_pre = y.to_numpy().reshape(len(y), 1)

    X = np.concatenate((y_pre, X_pre), axis=1)

    np.random.shuffle(X)
    train, validation, test = np.split(X, [int(0.7 * len(X)), int(0.85 * len(X))])
# note train, validation test files
    pd.DataFrame(train).to_csv(f"{base_dir}/train/train.csv", header=False, index=False)
    pd.DataFrame(validation).to_csv(
        f"{base_dir}/validation/validation.csv", header=False, index=False
    )
    pd.DataFrame(test).to_csv(f"{base_dir}/test/test.csv", header=False, index=False)
