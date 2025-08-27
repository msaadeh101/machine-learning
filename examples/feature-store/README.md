# Feature Store

## Features

In machine learning, **features** are individual, measurable properties or characteristics of a phenomenon being observed. They are the **input variables** used by a model to make predictions.

**Key Aspects of Features**
- **Input to the Model**: Features are the independent variables (X) that the machine learning algorithm uses to learn patterns.
- **Representational Power**: The quality and relevance of features are crucial for a model's performance. Good features capture the most important information in the data and allow the model to make accurate predictions. This is why **feature engineering** (process of creating better features based on existing data) is a vital part of the machine learning workflow.
- **Numerical vs. Categorical**: Features can be numerical (e.g., age, temperature, height) or categorical (e.g., color, city, type of vehicle). They often need to be preprocessed and transformed to be in a suitable format for the learning algorithm.
- **Dimensionality**: The number of features in a dataset is called its *dimensionality*. A high number of features can lead to a more complex model and potential issues like the curse of dimensionality, where a model becomes harder to train and more prone to overfitting.

## Feature Store Overview

**Feature stores** are centralized repositories for storing, managing, and serving machine learning features across different models and teams.

**A feature store provides**:
- **Centralized feature management**: Store and version features in a single location.
- **Feature reusability**: Share features across multiple models and teams.
- **Data consistency**: Ensure training and serving data consistency.
- **Feature discovery**: Browse and discover existing features.
- **Real-time and batch serving**: Support both online and offline feature access.

## Feast

**Feast** is an open-source feature store designed for both training and real-time inference.
- **Centralized Feature Management**: Provides feature registry as a centralized catalog for both features and their metadata.
- Dual storage Architecture:
    - **Offline Store**: Store historical feature data used for batch processing, model training, large-scale analytics (like BigQuery, Snowflake)
    - **Online Store**: low-latency, real-time feature serving in production environments to power live model predictions (like Redis/DynamoDB).
- **Data Consistency and Point-in-Time Correctedness**: Feast ensures no data leakage by making feature data available according to timestamps.
- **Scaler and Production Ready**: Feast supports batch and streaming feature ingestion, feature transformations, RBAC, and monitoring.

## Entities

In Feast, an **entity** is the fundamental object that your features describe. It serves as the unique identifier for a row of data. It is the primary key that connects a set of features (age, location, purchase history are features about the *user* entity, making `user_id` the **entity key**).

Feast uses entities to:
- **Group features**: Features are semantically related to an entity. (A `user_age` feature is a property of the `user` entity)
- **Enable joins**: During offline retreival for model training, Feast uses the entity key to join features from multiple data sources. (i.e. `user` features with order features on `user_id` and `order_id` build a complete training set)
- **Facilitate online lookups**: Model provides an entity key (user_123), and Feast online store performs a fast lookup to retrieve all latest features assocated with that key.


## Offline Store

The **offline store** is an interface that allows you to access and work with a historical collection of time-series feature data from a data source. The offline store is not in iteslf a database, but a way for Feast to run queries against existing big data systems.

Main use cases for the *offline store*:

1. **Building training datasets**: When training the ML model, you need a largre amount of historical data. Offline store allows for querying and large joining of datasets to create a single, **Point-in-time correct** training Dataframe.
2. **Materialization**: Process of taking most recent feature values from offline store and loading them into the **online store**, a separate, low-latency database (prepares for real-time model serving).

The offline store (and online store), is configured in the `feature_store.yaml` and can be connected to data warehouses, data lakes, databases (Biguery, Snowflake, Redshift, local parquet). It is designed for high-throughput batch processing.

## Online Store

The Feast **online store** is a specialized database designed to serve the latest feature values at low latency for real-time machine learning predictions.

- The goal is to deliver up-to-date feature values, by **entity key** (user_id), to online model-serving systems in milliseconds.
- The online store holds only the most recent value of each feature per entity.


## Feature Store Configuration

Feast configuration is stored in the `feature_store.yaml`.

```yaml
project: ml_project
provider: aws
online_store:
  type: redis
  connection_string: redis://localhost:6379
offline_store:
  type: postgresql
  connection_string: postgresql://user:pass@localhost/features
registry:
  path: s3://bucket/feature-registry
```

- `project`: logical grouping for features, each is isolated.
- `registry`: tracks feature definitions, data sources, metadata. In prod, these are usually in a shared storage location (blob, s3, GCS).
- `provider`: tells Feast about your environment.
- `offline_store`: where to pull historical features from. In prod, usually a warehouse or lake.
- `online_store`: low-latency DB where features are materialzied for model serving.


## Example Features

Using `./generate_user_features.py`:

```txt
Sample of data as Feast would read it:
========================================
 user_id     event_timestamp   created_timestamp  age   income category
       1 2024-01-15 10:30:00 2024-01-15 10:35:00   28 65000.00  premium
       2 2024-01-16 14:20:00 2024-01-16 14:25:00   34 78500.50 standard
       3 2024-01-17 09:45:00 2024-01-17 09:50:00   25 45000.75    basic
       4 2024-01-18 16:10:00 2024-01-18 16:15:00   42 92000.25      vip
       5 2024-01-19 11:00:00 2024-01-19 11:05:00   31 58000.00    trial
```
