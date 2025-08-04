"""
- Weights and biases are the "knobs" that help train NN to make predictions.
    - Weight: How much value you place on a feature
    - Biases: Hypothesis adjustement

This script teaches:
- Parameter sensitivity
- How changes effect MRSE
- Optimization tuning
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
import pandas as pd

def linear_prediction(X: np.ndarray, weights: List[float], bias: float) -> np.ndarray:
    """
    Make predictions using linear combination of features.
    (prediction = dot(X * weights) + biases)
    Takes input Data (X) and combines with model's params (weights/bias)
    
    Args:
        X: Input features (n_samples, n_features), 2D numpy array
        weights: List of weights for each feature, 1D array
        bias: Bias term
    
    Returns:
        Predictions array
    """
    # multiply each feature by its corresponding weight, add bias to get final prediction
    return np.dot(X, weights) + bias

def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Root Mean Square Error. RMSE calculates the accuracy of regression models.
    The model's loss function, measures avg magnitude of errors betwqeen y_true and y_pred
    
    Args:
        y_true: True values
        y_pred: Predicted values
    
    Returns:
        RMSE value
    """
    # Subtract each predicted value from corresponding true value, squared, returns the difference
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def weight_bias_rmse_evaluator(X: np.ndarray, y: np.ndarray, 
                              weight_ranges: List[Tuple[float, float]], 
                              bias_range: Tuple[float, float],
                              n_samples: int = 100) -> pd.DataFrame:
    """
    Evaluate RMSE for different combinations of weights and bias.
    Traing loop: iterates over n_samples, randomly generating new w+b within
    predefined ranges. linear_prediction makes new predictions. Calculates
    RMSE for each param. Stores them into a dataframe.
    
    Args:
        X: Input features (n_samples, n_features)
        y: Target values
        weight_ranges: List of (min, max) ranges for each weight
        bias_range: (min, max) range for bias
        n_samples: Number of random combinations to try
    
    Returns:
        DataFrame with weights, bias, and RMSE values
    """
    results = []
    
    for _ in range(n_samples):
        # Generate random weights within specified ranges
        weights = []
        for w_min, w_max in weight_ranges:
            weights.append(np.random.uniform(w_min, w_max))
        
        # Generate random bias within specified range
        bias = np.random.uniform(bias_range[0], bias_range[1])
        
        # Make predictions
        y_pred = linear_prediction(X, weights, bias)
        
        # Calculate RMSE
        rmse = calculate_rmse(y, y_pred)
        
        # Store results
        result = {'bias': bias, 'rmse': rmse}
        for i, w in enumerate(weights):
            result[f'weight_{i}'] = w
        results.append(result)
    
    return pd.DataFrame(results)

def find_best_parameters(results_df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """
    Find the best performing parameter combinations from the weight_bias_rmse_evaluator
    with simple nsmallest() method, identifying the "best" models found
    
    Args:
        results_df: DataFrame from weight_bias_rmse_evaluator
        top_n: Number of top results to return
    
    Returns:
        DataFrame with top performing combinations
    """
    # return the top n rows with smallest RMSE value. drop=True resets the index
    return results_df.nsmallest(top_n, 'rmse').reset_index(drop=True)

def plot_rmse_vs_parameter(results_df: pd.DataFrame, parameter: str, 
                          title: Optional[str] = None):
    """
    Plot RMSE vs a specific parameter.
    
    Args:
        results_df: DataFrame from weight_bias_rmse_evaluator
        parameter: Column name to plot against RMSE
        title: Optional plot title
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(results_df[parameter], results_df['rmse'], alpha=0.6)
    plt.xlabel(parameter.replace('_', ' ').title())
    plt.ylabel('RMSE')
    plt.title(title or f'RMSE vs {parameter.replace("_", " ").title()}')
    plt.grid(True, alpha=0.3)
    plt.show()

def create_sample_data(n_samples: int = 100, n_features: int = 2, 
                      true_weights: List[float] = [2.5, -1.3], 
                      true_bias: float = 0.8, noise_std: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sample dataset for testing.
    
    Args:
        n_samples: Number of samples
        n_features: Number of features
        true_weights: True weights used to generate data
        true_bias: True bias used to generate data
        noise_std: Standard deviation of noise
    
    Returns:
        Tuple of (X, y) arrays
    """
    np.random.seed(42)  # For reproducibility
    X = np.random.randn(n_samples, n_features)
    y = np.dot(X, true_weights) + true_bias + np.random.normal(0, noise_std, n_samples)
    return X, y

# Functions to load real datasets
def load_csv_data(filepath: str, feature_columns: List[str], target_column: str, 
                  normalize: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load data from CSV file.
    
    Args:
        filepath: Path to CSV file
        feature_columns: List of column names to use as features
        target_column: Column name to use as target
        normalize: Whether to normalize features (recommended)
    
    Returns:
        Tuple of (X, y) arrays

    Usage:
        X, y = load_csv_data('your_data.csv', 
                            feature_columns=['age', 'income', 'score'], 
                            target_column='price')
    """
    df = pd.read_csv(filepath)
    X = df[feature_columns].values
    y = df[target_column].values
    
    if normalize:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    
    return X, y

# Usage
if __name__ == "__main__":
    # Create sample data
    print("Creating sample dataset...")
    X, y = create_sample_data(n_samples=200, true_weights=[2.5, -1.3], true_bias=0.8)
    print(f"Dataset shape: X={X.shape}, y={y.shape}")
    print(f"True parameters: weights=[2.5, -1.3], bias=0.8")
    
    # Define search ranges for weights and bias
    weight_ranges = [(-5, 5), (-5, 5)]  # Range for each weight
    bias_range = (-3, 3)  # Range for bias
    
    print("\nEvaluating different weight and bias combinations..")

    # Evaluate different combinations
    results = weight_bias_rmse_evaluator(X, y, weight_ranges, bias_range, n_samples=1000)
    
    # Find best parameters
    print("\nTop 5 best parameter combinations:")
    best_params = find_best_parameters(results, top_n=5)
    print(best_params.round(4))
    
    # Show some statistics to 4 decimal places
    print(f"\nRMSE Statistics:")
    print(f"Best RMSE: {results['rmse'].min():.4f}")
    print(f"Worst RMSE: {results['rmse'].max():.4f}")
    print(f"Mean RMSE: {results['rmse'].mean():.4f}")
    print(f"Std RMSE: {results['rmse'].std():.4f}")
    
    # Plot some of the results
    print("\nGenerating plots...")
    plot_rmse_vs_parameter(results, 'weight_0', 'RMSE vs Weight 0')
    plot_rmse_vs_parameter(results, 'weight_1', 'RMSE vs Weight 1')
    plot_rmse_vs_parameter(results, 'bias', 'RMSE vs Bias')
    
    # Test a specific combination
    print("\nTesting specific parameter combination:")
    test_weights = [2.5, -1.3]
    test_bias = 0.8
    test_pred = linear_prediction(X, test_weights, test_bias)
    test_rmse = calculate_rmse(y, test_pred)
    print(f"True parameters RMSE: {test_rmse:.4f}")
    
    # Compare with best found parameters
    best_row = best_params.iloc[0]
    best_weights = [best_row['weight_0'], best_row['weight_1']]
    best_bias = best_row['bias']
    best_pred = linear_prediction(X, best_weights, best_bias)
    best_rmse = calculate_rmse(y, best_pred)
    print(f"Best found parameters RMSE: {best_rmse:.4f}")
    print(f"Best weights: {best_weights}")
    print(f"Best bias: {best_bias:.4f}")