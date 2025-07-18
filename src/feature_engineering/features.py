from src.preprocessing.processing import encode_categorical_columns
from sklearn.preprocessing import StandardScaler
from src.utils.helper import scale_features
from typing import Dict, List, Union
from src.constant.constant import *
from sklearn.cluster import KMeans
import pandas as pd
import logging


logger = logging.getLogger(__name__)


def engineer_company_features(input_dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer company-specific features from raw business data.

    Creates derived features including numeric company size representation and
    user engagement metrics. Transforms categorical company size into numeric
    values and calculates engagement rate as the ratio of daily active users
    to company size. Finally filters to include only the predefined feature set.

    Args:
        input_dataframe: Raw business data containing company metrics.
                        Must include 'company_size' and 'daily_active_users' columns.

    Returns:
        Processed DataFrame with engineered features and filtered columns:
        - company_size_numeric: Numeric mapping of categorical company sizes
        - user_engagement_rate: Daily active users per company size unit
        - Additional columns as defined in FINAL_COLUMNS configuration

    Raises:
        Exception: If feature engineering fails due to missing columns,
                invalid data types, or calculation errors. Original exception
                is logged and re-raised with context.
    """
    feature_engineered_data = input_dataframe.copy()

    try:
        feature_engineered_data["company_size_numeric"] = feature_engineered_data[
            "company_size"
        ].map(COMPANY_SIZE_MAP)

        feature_engineered_data["user_engagement_rate"] = (
            feature_engineered_data["daily_active_users"]
            / feature_engineered_data["company_size_numeric"]
        )

        final_feature_set = feature_engineered_data[FINAL_COLUMNS]

        logger.info(
            f"Successfully engineered features for {len(final_feature_set)} rows of data"
        )
        return final_feature_set

    except Exception as feature_engineering_error:
        logger.error(f"Feature engineering failed: {str(feature_engineering_error)}")
        raise


def create_customer_clusters(
    dataset: pd.DataFrame,
    numerical_features: Union[str, List[str]],
    categorical_mapping: Dict[str, Dict],
) -> tuple[pd.DataFrame, KMeans, StandardScaler]:
    """
    Segments a dataset into clusters using the K-means algorithm.

    This function orchestrates a complete clustering pipeline. It first preprocesses
    the data by standardizing numerical features and encoding categorical features.
    It then applies K-means clustering to group the data. The number of clusters
    is determined by a predefined constant `N_CLUSTER`.

    Note: The 'ai_tool' column is automatically excluded from the feature set
    before clustering. The function prints clustering metrics (Inertia) to the console.

    Args:
        dataset: Input DataFrame containing the raw data for clustering.
        numerical_features: A string or list of strings with the names of
            numerical columns to be scaled (standardized).
        categorical_mapping: A dictionary where keys are the names of categorical
            columns and values are their corresponding encoding maps.

    Returns:
        A tuple containing:
        - pd.DataFrame: The fully preprocessed DataFrame, updated to include
        a final 'cluster label' column.
        - KMeans: The fitted `KMeans` model object. This is useful for
        analyzing cluster centroids and predicting clusters for new data.
        - StandardScaler: The fitted `StandardScaler` object from the
        preprocessing step. This is useful for applying the exact same
        scaling to new, unseen data.

    Raises:
        KeyError: If a column specified in `numerical_features` or
            `categorical_mapping` does not exist in the dataset.
    """

    preprocessed_dataset, feature_scaler = scale_features(dataset, numerical_features)

    encoded_dataset = encode_categorical_columns(
        preprocessed_dataset, categorical_mapping
    )

    encoded_dataset_copy = encoded_dataset.copy()

    if "ai_tool" in encoded_dataset.columns:
        clustering_features = encoded_dataset_copy.drop(columns=["ai_tool"])

    clustering_model = KMeans(
        n_clusters=N_CLUSTER, random_state=RANDOM_STATE, n_init=N_INIT
    )

    clustering_model.fit(clustering_features)

    cluster_labels = clustering_model.predict(clustering_features)

    encoded_dataset["cluster label"] = cluster_labels

    print("\n--- K-Means Clustering Metrics ---")
    print(f"Number of clusters (K): {N_CLUSTER}")
    print(f"Inertia (WCSS): {clustering_model.inertia_:.2f}")

    return encoded_dataset, clustering_model, feature_scaler
