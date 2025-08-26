import pandas as pd
import numpy as np
import yaml
import json
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

def load_params():
    """Load parameters from params.yaml"""
    with open('params.yaml', 'r') as f:
        return yaml.safe_load(f)

def train_model(input_path, output_dir, params):
    """
    Train model with configurable parameters
    """
    # Load engineered features
    df = pd.read_csv(input_path)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    
    print(f"Training model on dataset with shape: {X.shape}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Select algorithm
    algorithm = params['model']['algorithm']
    hyperparams = params['model']['hyperparameters']
    
    if algorithm == 'random_forest':
        model = RandomForestClassifier(**hyperparams)
    elif algorithm == 'gradient_boosting':
        model = GradientBoostingClassifier(**hyperparams)
    elif algorithm == 'logistic_regression':
        model = LogisticRegression(**hyperparams)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    # Train model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted'),
        'recall': recall_score(y_test, y_pred, average='weighted'),
        'f1_score': f1_score(y_test, y_pred, average='weighted'),
        'algorithm': algorithm,
        'train_size': len(X_train),
        'test_size': len(X_test)
    }
    
    print(f"Model trained with accuracy: {metrics['accuracy']:.4f}")
    
    # Save model and metrics
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs('metrics', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    
    # Save model
    joblib.dump(model, f'{output_dir}/model.joblib')
    
    # Save metrics
    with open('metrics/train_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save confusion matrix plot data
    cm = confusion_matrix(y_test, y_pred)
    cm_data = {
        'confusion_matrix': cm.tolist(),
        'labels': sorted(y.unique().tolist())
    }
    with open('plots/confusion_matrix.json', 'w') as f:
        json.dump(cm_data, f, indent=2)
    
    # Save feature importance if available
    if hasattr(model, 'feature_importances_'):
        importance_data = {
            'feature_names': [f'feature_{i}' for i in range(len(model.feature_importances_))],
            'importance_values': model.feature_importances_.tolist()
        }
        with open('plots/feature_importance.json', 'w') as f:
            json.dump(importance_data, f, indent=2)
    
    return model, metrics

if __name__ == "__main__":
    params = load_params()
    train_model(
        input_path='data/features/engineered_features.csv',
        output_dir='models',
        params=params
    )