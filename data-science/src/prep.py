# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Prepares raw data and provides training and test datasets.
"""

import argparse
from pathlib import Path
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import mlflow
import logging

def parse_args():
    '''Parse input arguments'''
    
    parser = argparse.ArgumentParser("prep")  # Create an ArgumentParser object
    parser.add_argument("--raw_data", type=str, help="Path to raw data")  # Specify the type for raw data (str)
    parser.add_argument("--train_data", type=str, help="Path to train dataset")  # Specify the type for train data (str)
    parser.add_argument("--test_data", type=str, help="Path to test dataset")  # Specify the type for test data (str)
    parser.add_argument("--test_train_ratio", type=float, default=0.2, help="Test-train ratio")  # Specify the type (float) and default value (0.2) for test-train ratio
    args = parser.parse_args()

    return args

def main(args):  # Write the function name for the main data preparation logic
    '''Read, preprocess, split, and save datasets'''

    # Reading Data
    df = pd.read_csv(args.raw_data)

    # ------- WRITE YOUR CODE HERE -------

    # Step 1: Perform label encoding to convert categorical features into numerical values for model compatibility.  
    # Step 2: Split the dataset into training and testing sets using train_test_split with specified test size and random state.  
    # Step 3: Save the training and testing datasets as CSV files in separate directories for easier access and organization.  
    # Step 4: Log the number of rows in the training and testing datasets as metrics for tracking and evaluation.  

    # Encode categorical feature
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])  # Encode categorical
    
    # Log the first rows of the dataset
    logging.basicConfig(level=logging.INFO)
    logging.info(f"Transformed Data:\n{df.head(5)}")

    # Split Data into train and test datasets
    train_df, test_df = train_test_split(df, test_size=args.test_train_ratio, random_state=42)  #  Write code to split the data into train and test datasets

    # Save train and test data
    #os.makedirs(args.train_data, exist_ok=True)  # Create directories for train_data
    #os.makedirs(args.test_data, exist_ok=True)  # Create directories for test_data
    train_df.to_csv(os.path.join(args.train_data, "train_data.csv"), index=False)  # Specify the name of the train data file
    test_df.to_csv(os.path.join(args.test_data, "test_data.csv"), index=False)  # Specify the name of the test data file

    # log the metrics
    mlflow.log_metric('train_size', train_df.shape[0])  # Log the train dataset size
    mlflow.log_metric('test_size', test_df.shape[0])  # Log the test dataset size

if __name__ == "__main__":
    with mlflow.start_run():
        args = parse_args()
        
        print(f"Raw data path: {args.raw_data}")
        print(f"Train dataset output path: {args.train_data}")
        print(f"Test dataset path: {args.test_data}")
        print(f"Test-train ratio: {args.test_train_ratio}")
        
        main(args)
