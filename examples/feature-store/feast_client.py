# feast_client.py

from feast import FeatureStore

# This script demonstrates how to connect to a Feast feature store to retrieve features.
# Note: Need feature_store.yaml before running, with access to your feature repo and data store.

def main():
    """
    Main function to run the Feast client operations.
    """
    try:
        # 1. Initialize the FeatureStore object.
        #    The `repo_path` should point to the directory containing
        #    your `feature_store.yaml` file and feature definitions.
        #    For this example, we assume the repository is in the current directory.
        fs = FeatureStore(repo_path="my_feature_repo")

        print("Successfully connected to the Feast feature store.")
        print("-" * 40)

        # 2. List all registered feature views.
        #    This is useful for debugging and understanding the features available.
        print("Registered Feature Views:")
        feature_views = fs.list_feature_views()
        for fv in feature_views:
            print(f"  - {fv.name}")
        print("-" * 40)

        # 3. Get online features for a specific entity (user).
        #    `entity_rows` is a list of dictionaries, where each dictionary
        #    contains the entity key and its value.
        #    `features` is a list of strings in the format "feature_view:feature_name".
        entity_rows_to_get = [{"user_id": 123}]
        features_to_get = ["user_features:age", "user_features:income"]

        print(f"Retrieving online features for entity: {entity_rows_to_get[0]}")
        print(f"Features requested: {features_to_get}")

        # The `get_online_features` call retrieves the latest feature values
        # from the online store (e.g., Redis).
        online_features = fs.get_online_features(
            entity_rows=entity_rows_to_get,
            features=features_to_get
        )

        # 4. Convert the result to a dictionary and print.
        features_dict = online_features.to_dict()
        print("-" * 40)
        print("Retrieved Online Features:")
        print(features_dict)

        # Example of how to access a specific feature value
        user_age = features_dict.get("user_features__age", [None])[0]
        user_income = features_dict.get("user_features__income", [None])[0]
        print(f"\nUser 123's age is: {user_age}")
        print(f"User 123's income is: {user_income}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
