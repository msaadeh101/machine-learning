import pandas as pd
import numpy as np
import yaml
import json
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
import joblib

def load_params():
    """Load parameters from params.yaml"""
    with open('params.yaml', 'r') as f:
        return yaml.safe_load(f)

def engineer_features(input_path, output_dir, params):
    """
    Engineer features with configurable parameters
    """
    # Load processed data
    df = pd.read_csv(input_path)
    print(f"Engineering features for dataset with shape: {df.shape}")
    
    # Assume last column is target
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    
    # Create feature engineering pipeline
    steps = []
    
    # Scaling
    scaling_method = params['feature_engineering']['scaling_method']
    if scaling_method == 'standard':
        steps.append(('scaler', StandardScaler()))
    elif scaling_method == 'minmax':
        steps.append(('scaler', MinMaxScaler()))
    
    # Feature selection
    if params['feature_engineering']['feature_selection']:
        n_features = params['feature_engineering']['n_features']
        steps.append(('selector', SelectKBest(f_classif, k=min(n_features, X.shape[1]))))
    
    # Create and fit pipeline
    pipeline = Pipeline(steps)
    X_transformed = pipeline.fit_transform(X, y)
    
    # Create feature importance information
    feature_importance = {}
    if 'selector' in pipeline.named_steps:
        selector = pipeline.named_steps['selector']
        selected_features = X.columns[selector.get_support()].tolist()
        scores = selector.scores_[selector.get_support()]
        feature_importance = dict(zip(selected_features, scores.tolist()))
    
    # Save engineered features
    os.makedirs(output_dir, exist_ok=True)
    
    # Save features
    feature_df = pd.DataFrame(X_transformed)
    feature_df['target'] = y.values
    feature_df.to_csv(f'{output_dir}/engineered_features.csv', index=False)
    
    # Save feature importance
    with open(f'{output_dir}/feature_importance.json', 'w') as f:
        json.dump(feature_importance, f, indent=2)
    
    print(f"Engineered features saved with shape: {X_transformed.shape}")
    
    return X_transformed, y, pipeline

if __name__ == "__main__":
    params = load_params()
    engineer_features(
        input_path='data/processed/clean_dataset.csv',
        output_dir='data/features',
        params=params
    )