import pandas as pd
from datetime import datetime, timedelta
import numpy as np

# Create sample data for user_features.parquet
# This matches the schema expected by the Feast FeatureView

# Set random seed for reproducible data
np.random.seed(77)

# Generate sample data - keeping it simple with just 5 users, 1 record each
data = [
    {
        "user_id": 1,
        "event_timestamp": datetime(2024, 1, 15, 10, 30, 0),
        "created_timestamp": datetime(2024, 1, 15, 10, 35, 0),
        "age": 28,
        "income": 65000.00,
        "category": "premium"
    },
    {
        "user_id": 2,
        "event_timestamp": datetime(2024, 1, 16, 14, 20, 0),
        "created_timestamp": datetime(2024, 1, 16, 14, 25, 0),
        "age": 34,
        "income": 78500.50,
        "category": "standard"
    },
    {
        "user_id": 3,
        "event_timestamp": datetime(2024, 1, 17, 9, 45, 0),
        "created_timestamp": datetime(2024, 1, 17, 9, 50, 0),
        "age": 25,
        "income": 45000.75,
        "category": "basic"
    },
    {
        "user_id": 4,
        "event_timestamp": datetime(2024, 1, 18, 16, 10, 0),
        "created_timestamp": datetime(2024, 1, 18, 16, 15, 0),
        "age": 42,
        "income": 92000.25,
        "category": "vip"
    },
    {
        "user_id": 5,
        "event_timestamp": datetime(2024, 1, 19, 11, 0, 0),
        "created_timestamp": datetime(2024, 1, 19, 11, 5, 0),
        "age": 31,
        "income": 58000.00,
        "category": "trial"
    }
]

# Create DataFrame
df = pd.DataFrame(data)

# Display the data that would be in the parquet file
print("Data that will be saved to user_features.parquet:")
print("=" * 60)
print(df.to_string(index=False))
print("\nDataFrame Info:")
print("=" * 30)
print(df.info())
print("\nData Types:")
print("=" * 20)
print(df.dtypes)

# Create the parquet file
df.to_parquet("./data/user_features.parquet", index=False)
print(f"\nCreated user_features.parquet with {len(df)} records")

# Verify by reading it back
df_read = pd.read_parquet("data/user_features.parquet")
print(f"\nVerified: Successfully read back {len(df_read)} records from parquet file")

# Show sample of what Feast would see
print("\nSample of data as Feast would read it:")
print("=" * 40)
print(df_read.head().to_string(index=False))