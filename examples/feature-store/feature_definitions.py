from datetime import timedelta
from feast import Entity, Feature, FeatureView, ValueType, FileSource

# Define the entity
user = Entity(name="user_id", value_type=ValueType.INT64, description="User identifier")

# Data source (historical features live here)
user_source = FileSource(
    path="data/user_features.parquet",  # Dev; in prod point to BigQuery/S3/Snowflake
    event_timestamp_column="event_timestamp",  # Required for point-in-time joins
    created_timestamp_column="created_timestamp",
)

# FeatureView (group of related features)
user_features = FeatureView(
    name="user_features",
    entities=["user_id"],
    ttl=timedelta(days=1),  # how long features remain fresh in online store
    features=[
        Feature(name="age", dtype=ValueType.INT64),
        Feature(name="income", dtype=ValueType.FLOAT),
        Feature(name="category", dtype=ValueType.STRING),
    ],
    online=True,
    source=user_source,
    tags={"team": "ml", "owner": "recommendation-system"},  # 👍 Best practice
)
