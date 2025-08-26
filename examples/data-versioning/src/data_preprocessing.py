import pandas as pd
import numpy as np
import yaml
import os
from pathlib import Path

def load_params():
    """Load parameters from params.yaml"""
    with open('params.yaml', 'r') as f:
        return yaml.safe_load(f)

def preprocess_data(input_path, output_path, params):
    """
    Preprocess raw data with configurable parameters
    """
    # Load data
    df = pd.read_csv(input_path)
    print(f"Loaded dataset with shape: {df.shape}")
    
    # Handle missing values
    missing_strategy = params['preprocessing']['missing_strategy']
    if missing_strategy == 'median':
        df = df.fillna(df.median(numeric_only=True))
    elif missing_strategy == 'mean':
        df = df.fillna(df.mean(numeric_only=True))
    
    # Handle outliers
    outlier_threshold = params['preprocessing']['outlier_threshold']
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - outlier_threshold * IQR
        upper_bound = Q3 + outlier_threshold * IQR
        
        # Cap outliers instead of removing
        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
    
    print(f"Processed dataset with shape: {df.shape}")
    
    # Save processed data
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    return df

if __name__ == "__main__":
    params = load_params()
    preprocess_data(
        input_path='data/raw/dataset.csv',
        output_path='data/processed/clean_dataset.csv',
        params=params
    )