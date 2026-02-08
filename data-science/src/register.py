# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Registers the best-trained ML model from the sweep job.
"""

import argparse
from pathlib import Path
import mlflow
import os 
import json

def parse_args():
    '''Parse input arguments'''

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, help='Name under which model will be registered')  # Hint: Specify the type for model_name (str)
    parser.add_argument('--model_path', type=str, help='Model directory')  # Hint: Specify the type for model_path (str)
    parser.add_argument("--model_info_output_path", type=str, help="Path to write model info JSON")  # Hint: Specify the type for model_info_output_path (str)
    args, _ = parser.parse_known_args()
    print(f'Arguments: {args}')

    return args

def main(args):
    print(f"Registering model: {args.model_name}")

    # Step 1: Load model from artifacts
    model = mlflow.sklearn.load_model(args.model_path)

    # Step 2: Log model as an MLflow artifact
    artifact_path = "model"
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path=artifact_path
    )

    # Step 3: Register the logged model
    run_id = mlflow.active_run().info.run_id
    model_uri = f"runs:/{run_id}/{artifact_path}"

    model_version = mlflow.register_model(
        model_uri=model_uri,
        name=args.model_name
    )

    # Step 4: Write model info JSON
    model_info = {
        "model_name": args.model_name,
        "model_version": model_version.version
    }

    #os.makedirs(args.model_info_output_path, exist_ok=True)
    output_file = Path(args.model_info_output_path) / "model_info.json"

    with open(output_file, "w") as f:
        json.dump(model_info, f)

    print(f"Registered model version: {model_version.version}")

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