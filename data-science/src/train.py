# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Trains ML model using training dataset and evaluates using test dataset. Saves trained model.
"""

import argparse
from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import mlflow
import mlflow.sklearn
import os

def select_first_file(path):
    """Selects the first file in a folder, assuming there's only one file.
    Args:
        path (str): Path to the directory or file to choose.
    Returns:
        str: Full path of the selected file.
    """
    for f in os.listdir(path):
        if f.endswith(".csv"):
            return os.path.join(path, f)
    raise FileNotFoundError("No CSV file found")
    
def parse_args():
    '''Parse input arguments'''

    parser = argparse.ArgumentParser("train")
    
    # -------- WRITE YOUR CODE HERE --------
    
    # Step 1: Define arguments for train data, test data, model output, and RandomForest hyperparameters. Specify their types and defaults.  

    parser.add_argument("--train_data", type=str, help="Path to train dataset")
    parser.add_argument("--test_data", type=str, help="Path to test dataset")
    parser.add_argument("--model_output", type=str, help="Path of output model")
    parser.add_argument('--n_estimators', type=int, default=100, help='The number of trees in the forest')
    parser.add_argument('--max_depth', type=int, default=None, help='The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.')

    args = parser.parse_args()

    return args

def main(args):
    '''Read train and test datasets, train model, evaluate model, save trained model'''
    
    TARGET_COL = "price"

    # -------- WRITE YOUR CODE HERE --------

    # Step 2: Read the train and test datasets from the provided paths using pandas. Replace '_______' with appropriate file paths and methods.  
    # Step 3: Split the data into features (X) and target (y) for both train and test datasets. Specify the target column name.  
    # Step 4: Initialize the RandomForest Regressor with specified hyperparameters, and train the model using the training data.  
    # Step 5: Log model hyperparameters like 'n_estimators' and 'max_depth' for tracking purposes in MLflow.  
    # Step 6: Predict target values on the test dataset using the trained model, and calculate the mean squared error.  
    # Step 7: Log the MSE metric in MLflow for model evaluation, and save the trained model to the specified output path.  

    # Load datasets
    train_df = pd.read_csv(select_first_file(args.train_data))
    test_df = pd.read_csv(select_first_file(args.test_data))

    # Split the data into features(X) and target(y)
    y_train = train_df[TARGET_COL]  # Specify the target column
    X_train = train_df.drop(columns=[TARGET_COL])
    y_test = test_df[TARGET_COL]
    X_test = test_df.drop(columns=[TARGET_COL])

    # Initialize and train a RandomForest Regressor
    model = RandomForestRegressor(n_estimators=args.n_estimators, max_depth=args.max_depth, random_state=42)
    model.fit(X_train, y_train)  # Train the model

    # Log model hyperparameters
    mlflow.log_param("model", "azure_random_forest")  # Provide the model name
    mlflow.log_param("n_estimators", args.n_estimators)
    mlflow.log_param("max_depth", args.max_depth)

    # Predict using the RandomForest Regressor on test data
    yhat_test = model.predict(X_test)

    # Compute and log mean squared error for test data
    mse = mean_squared_error(y_test, yhat_test)
    print('Mean Squared Error of RandomForest Regressor on test set: {:.2f}'.format(mse))
    mlflow.log_metric("MSE", float(mse))  # Log the MSE

    # Save the model
    mlflow.sklearn.save_model(sk_model=model, path=args.model_output)  # Save the model

if __name__ == "__main__":

    with mlflow.start_run():
        
         # Parse Arguments
        args = parse_args()
        
        lines = [
            f"Train dataset input path: {args.train_data}",
            f"Test dataset input path: {args.test_data}",
            f"Model output path: {args.model_output}",
            f"Number of Estimators: {args.n_estimators}",
            f"Max Depth: {args.max_depth}"
        ]

        for line in lines:
            print(line)
        
        main(args)
